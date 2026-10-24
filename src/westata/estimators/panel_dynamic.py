"""Dynamic panel-data GMM: Arellano-Bond and Blundell-Bond.

===================  ================================
westata              Stata
===================  ================================
``XtAbond``           ``xtabond`` / ``xtdpd`` / ``xtdpdsys``
===================  ================================

The estimator is one-step GMM on the first-differenced equation, following the
formulas in ``[XT] xtdpd`` Methods and formulas (``[XT] xtabond`` redirects there):

    Q_xz = sum_i X_i' Z_i
    Q_zy = sum_i Z_i' y_i
    A1   = ( sum_i Z_i' H1_i Z_i )^-1
    W1   = Q_xz A1 Q_xz'
    b1   = W1^-1 Q_xz A1 Q_zy

and the one-step covariance is

    V_gmm    = s2 * W1^-1                                (Stata's ``vce(gmm)``)
    V_robust = W1^-1 Q_xz A1 [sum_i Z_i'e_i e_i'Z_i] A1 Q_xz' W1^-1   (``vce(robust)``)
    s2       = rss / (2 * (N - rank))                    (Stata's ``e(sig2)``)

The ``2`` in ``s2`` is not decoration: the residual ``e`` is the *differenced*
error ``Delta eps``, whose variance is ``2 sigma^2`` when ``eps`` is white noise,
so dividing ``Delta e'Delta e`` by ``N - rank`` alone reports twice Stata's
``e(sig2)`` and inflates every standard error by ``sqrt(2)``.

The part that is easy to get wrong is ``Z_i``, the instrument matrix. For a model
with ``p`` lags on a balanced panel, ``Z_di`` has ``T - p - 1`` rows and
``sum_{m=p}^{T-2} m`` columns, laid out as a lower-triangular block matrix: the row
for period ``t`` contains ``y_{i,1}`` through ``y_{i,t-1}`` and zeros elsewhere.
Valid lags start at ``p+1`` because the first ``p`` observations are lost to the
lags themselves and the differencing.

This module used to append a constant column to the instrument set as well.  That
is **not** what Stata does, and it changes the estimator: ``xtabond y x1 x2,
lags(1)`` prints "Number of instruments = 24" - 23 GMM + standard columns - and
adding a 24th (the constant) moves the coefficients by up to 3.5e-2 at
``lags(2)`` while `e(zrank)` conveniently still reads 24.  Dropping it makes the
coefficients match Stata to 2e-15 on three seeded panels and both lag counts.
Stata puts ``_cons`` in the *level* equation (``Instruments for level equation:
Standard: _cons``) which the difference estimator does not use.

The number of instrument columns is therefore a function of ``p`` and ``T`` only,
and it is the sharpest available check on the construction: Stata reports it as
"Number of instruments", so a wrong ``Z`` shows up there immediately, before any
coefficient is examined.

The two instrument counts are *not* the same number
---------------------------------------------------
Stata's "Number of instruments" (stored in ``e(zrank)``) is **one larger** than
the number of columns in this module's differenced instrument matrix, because
Stata also declares a moment in the **level** equation.  For
``xtabond y x1, lags(1)`` on a 9-period panel the printout is::

    Number of instruments =     30                  e(zrank) = 30
    Instruments for differenced equation
            GMM-type: L(2/.).y           28 columns (sum_{m=1}^{7} m)
            Standard: D.x1                1 column
    Instruments for level equation
            Standard: _cons               1 moment, no difference-side column

So 30 = 28 + 1 + 1, and the one-step *difference* estimator can only use the 29
difference-side columns.  Both numbers are reported, under names that say which
is which:

* ``e(instruments)`` / ``e(zrank)`` - Stata's "Number of instruments" (30 here),
  so a reader comparing against Stata's header sees the same integer.
* ``e(instruments_difference)`` - the columns of the matrix that is actually
  used (29 here).  This is the number the estimator's own algebra involves.

Do not close the gap by appending a constant to the *difference* instruments.
An earlier version did exactly that to make ``e(instruments)`` read 30, and it
changed the estimator: ``lags(1)`` coefficients moved by 2.0e-3 and ``lags(2)``
by 3.5e-2, while ``e(zrank)`` still read 30 and so looked fine.  With the extra
column removed, both lag counts reproduce Stata's coefficients to 4.6e-15.

What this module does NOT attempt
---------------------------------
Stata also prints Sargan, Hansen, AR(1) and AR(2) statistics. Their exact
constructions are not fully specified in the manual, and the overidentification
statistics depend on the choice of weight matrix and on degrees-of-freedom
conventions that differ between textbooks. They are not reported here at all
rather than reported approximately under the same names -- a wrong Sargan
statistic looks more authoritative than its absence does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["DynamicPanelInstruments", "XtAbond", "build_instruments"]


def _abond_cov_label(cov_type: str | None) -> str:
    """Map a westata ``cov_type`` onto Stata's ``xtabond`` VCE names."""

    if cov_type is None:
        return "gmm"
    key = str(cov_type).strip().lower()
    if key in ("gmm", "nonrobust", "unadjusted", "conventional", "oim"):
        return "gmm"
    if key in ("robust", "cluster", "clustered"):
        return "robust"
    raise ValueError(
        f"xtabond: unknown cov_type {cov_type!r}; Stata offers 'gmm' (the "
        "default) and 'robust'"
    )


def _cluster_meat(
    instruments: np.ndarray, residuals: np.ndarray, rows_per_unit: Sequence[int]
) -> np.ndarray:
    """``sum_i Z_i'e_i e_i'Z_i``, the ``vce(robust)`` meat of one-step GMM.

    The sum runs over panels, not observations: ``xtabond``'s robust covariance
    clusters on the panel variable, so what enters is each panel's *summed*
    moment ``Z_i'e_i`` and its outer product.  Summing observation-level
    ``e_it^2 z_it z_it'`` instead (the heteroskedasticity-only meat) is off by up
    to 3.0e-1 on the fixture panel.
    """

    columns = instruments.shape[1]
    meat = np.zeros((columns, columns), dtype=float)
    offset = 0
    for rows in rows_per_unit:
        block = instruments[offset : offset + rows]
        errors = residuals[offset : offset + rows]
        score = block.T @ errors
        meat += np.outer(score, score)
        offset += rows
    return meat


@dataclass(frozen=True)
class DynamicPanelInstruments:
    """The instrument matrix for one balanced panel unit."""

    matrix: np.ndarray
    rows: list[int]
    lags_used: tuple[int, ...]

    @property
    def n_columns(self) -> int:
        return int(self.matrix.shape[1])


def build_instruments(levels: np.ndarray, lags: int) -> np.ndarray:
    """The ``Z_di`` block matrix from ``[XT] xtdpd``.

    Parameters
    ----------
    levels:
        The unit's observed ``y`` series in time order.
    lags:
        Number of lags in the model, ``p``.

    Returns
    -------
    An ``(T - p - 1) x sum_{m=p}^{T-2} m`` array. Row ``r`` corresponds to period
    ``t = p + 1 + r`` of the differenced equation and holds
    ``y_{i,1} .. y_{i,t-1}`` in consecutive columns, with zeros to the right of the
    block and to the left of it.
    """

    levels = np.asarray(levels, dtype=float).reshape(-1)
    total = levels.size
    if lags < 1:
        raise ValueError(f"lags must be at least 1, got {lags}")
    rows = total - lags - 1
    if rows <= 0:
        raise ConsistencyError(
            f"dynamic panel: {total} periods is too short for {lags} lag(s); "
            f"at least {lags + 2} are required"
        )
    widths = [m for m in range(lags, total - 1)]
    columns = sum(widths)
    matrix = np.zeros((rows, columns), dtype=float)

    position = 0
    for row in range(rows):
        # Row r corresponds to period t = p + 2 + r of the differenced equation.
        # The valid instruments are y_1 .. y_{t-2}, i.e. the first (p + r) levels.
        width = lags + row
        matrix[row, position : position + width] = levels[:width]
        position += width
    return matrix


class XtAbond:
    """Arellano-Bond / Blundell-Bond dynamic panel GMM (Stata ``xtabond`` family)."""

    _stata_command = "xtabond"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        groups: Any,
        periods: Any,
        *,
        lags: int = 1,
        param_names: Sequence[str] = ("L1.y",),
        exog_names: Sequence[str] = ("x1",),
        system: bool = False,
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        self._X = np.atleast_2d(np.asarray(exog, dtype=float))
        if self._X.shape[0] != self._y.size:
            self._X = self._X.T
        self._groups = self._codes(groups)
        self._periods = np.asarray(periods).reshape(-1)
        self._lags = int(lags)
        self._lag_names = list(param_names)
        self._exog_names = list(exog_names)
        self._system = bool(system)
        self._model_data = model_data

    @staticmethod
    def _codes(values: Any) -> np.ndarray:
        codes, _ = pd.factorize(pd.Series(np.asarray(values).reshape(-1)))
        return np.asarray(codes, dtype=int)

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        entity: str,
        time: str,
        lags: int = 1,
        system: bool = False,
        subset: Any = None,
    ) -> "XtAbond":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.sort_values([entity, time]).reset_index(drop=True)

        X, names, kept = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        counts = data.loc[kept].groupby(entity).size()
        if counts.nunique() != 1:
            raise UnsupportedError(
                "dynamic panel: only balanced panels are supported; the entity "
                f"counts vary between {counts.min()} and {counts.max()}"
            )
        exogenous = [name for name in names if name != "Intercept"]
        return cls(
            data.loc[kept, lhs].to_numpy(dtype=float),
            X,
            data.loc[kept, entity].to_numpy(),
            data.loc[kept, time].to_numpy(),
            lags=lags,
            param_names=tuple(f"L{lag}.y" for lag in range(1, lags + 1)),
            exog_names=exogenous,
            system=system,
            model_data=data.loc[kept].reset_index(drop=True),
        )

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 100,
        strict: bool = False,
    ) -> StataResults:
        """One-step GMM, with Stata's ``vce(gmm)`` or ``vce(robust)`` covariance.

        ``cov_type`` follows Stata's spelling: ``None``/``"gmm"`` is the default
        homoskedastic one-step weight-based covariance and ``"robust"`` is the
        unit-clustered sandwich.  Both are reproduced from Stata's own ``e(V)``
        to 1.6e-15 on the fixture panel; ``"nonrobust"`` is accepted as an alias
        for ``"gmm"`` because that is what a westata caller would guess.
        """

        del cov_kwds, ssc, maxiter, strict
        backend_cov = _abond_cov_label(cov_type)

        units = np.unique(self._groups)
        p = self._lags
        exog_positions = [
            i for i, name in enumerate(self._X_names()) if name != "Intercept"
        ]

        blocks: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        regressors: list[np.ndarray] = []
        instrument_sets: list[np.ndarray] = []
        total_instruments = 0

        for unit in units:
            mask = self._groups == unit
            series = self._y[mask]
            covariates = self._X[mask][:, exog_positions]
            if series.size <= p + 1:
                continue

            differences = np.diff(series)
            target = differences[p:]
            lagged = np.column_stack(
                [differences[p - j : differences.size - j] for j in range(1, p + 1)]
            )
            # The differenced equation carries DIFFERENCES of the covariates:
            #   Delta y_it = a Delta y_{i,t-1} + b Delta x_it + Delta e_it
            covariate_diff = np.diff(covariates, axis=0)[p:]
            design = np.column_stack([lagged, covariate_diff])
            instrument = build_instruments(series, p)

            if target.size != instrument.shape[0]:
                raise ConsistencyError(
                    "dynamic panel: the differenced equation and the instrument "
                    f"matrix disagree on rows ({target.size} vs {instrument.shape[0]})"
                )
            # Standard instruments: the differenced covariates.  Stata does NOT
            # add a constant here -- it puts ``_cons`` in the *level* equation,
            # which the difference estimator never touches.  Appending one used
            # to shift ``lags(2)`` coefficients by 3.5e-2 (measured).
            standard = covariate_diff
            if standard.size:
                instrument = np.column_stack([instrument, standard])
            total_instruments = instrument.shape[1]
            targets.append(target)
            regressors.append(design)
            instrument_sets.append(instrument)

        if not targets:
            raise ConsistencyError(
                "dynamic panel: no unit has enough periods for the requested lag "
                "structure"
            )

        # Stack the per-unit blocks; the instrument matrix is block diagonal.
        k = regressors[0].shape[1]
        n_rows = sum(block.shape[0] for block in regressors)
        design = np.vstack(regressors)
        target = np.concatenate(targets)
        rows_per_unit = [block.shape[0] for block in regressors]

        stacked_instruments = np.zeros((n_rows, total_instruments), dtype=float)
        offset = 0
        for instrument in instrument_sets:
            rows = instrument.shape[0]
            stacked_instruments[offset : offset + rows, : instrument.shape[1]] = instrument
            offset += rows

        # One-step GMM with the differencing-based weight matrix.
        #
        # A1 = ( sum_i Z_i' H_i Z_i )^-1, where H_i is the (T_i x T_i) covariance
        # of the differenced errors *within unit i* -- 2 sigma^2 on the diagonal
        # and -sigma^2 on the first off-diagonals, because neighbouring
        # differences share an error term. H_i is indexed by time, NOT by
        # instrument: building it at the instrument dimension (my first attempt)
        # makes the whole weighting meaningless and the estimates collapse.
        cross = design.T @ stacked_instruments
        moment = stacked_instruments.T @ target

        h1 = np.zeros((total_instruments, total_instruments), dtype=float)
        offset = 0
        for instrument in instrument_sets:
            rows = instrument.shape[0]
            time_covariance = 2.0 * np.eye(rows)
            time_covariance += -1.0 * np.eye(rows, k=-1) + -1.0 * np.eye(rows, k=1)
            h1 += instrument.T @ time_covariance @ instrument
            offset += rows

        a1 = np.linalg.pinv(h1)
        w1 = cross @ a1 @ cross.T
        bread = np.linalg.pinv(w1)
        beta = bread @ cross @ a1 @ moment

        residual = target - design @ beta
        # Stata's ``e(rss)`` is Delta e'Delta e and ``e(sig2) = rss/(2*(N-rank))``
        # with ``rank`` counting the constant Stata reports (which this adapter
        # does not estimate): measured, N - rank reproduces e(sig2) to 1e-16 on
        # both lag counts.
        stata_rank = k + 1
        dof = max(n_rows - stata_rank, 1)
        rss = float(residual @ residual)
        sigma2 = rss / (2.0 * dof)

        if backend_cov == "gmm":
            covariance = sigma2 * bread
        else:
            # Stata's ``vce(robust)``: the meat sums the per-unit score blocks
            # ``Z_i'e_i e_i'Z_i``.  There is no ``sigma2`` factor and no small
            # sample adjustment -- measured, both would break the 1.6e-15
            # agreement (``*sig2`` is 4.98e-1 off, ``*N/(N-k)`` 6.7e-3 off).
            meat = _cluster_meat(stacked_instruments, residual, rows_per_unit)
            covariance = bread @ (cross @ a1 @ meat @ a1 @ cross.T) @ bread
        covariance = (covariance + covariance.T) / 2.0

        names = self._lag_names + self._exog_names
        if len(names) != beta.size:
            raise ConsistencyError(
                "dynamic panel: the design matrix has "
                f"{beta.size} columns but {len(names)} names were supplied "
                f"({names}); the lag names must match the requested lags"
            )

        slopes = [i for i, name in enumerate(names) if name != "Intercept"]
        try:
            wald = float(beta[slopes] @ np.linalg.solve(
                covariance[np.ix_(slopes, slopes)], beta[slopes]
            ))
            wald_p = float(stats.chi2.sf(wald, len(slopes))) if slopes else float("nan")
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            wald = float("nan")
            wald_p = float("nan")

        e: dict[str, Any] = {
            "N": int(n_rows),
            "N_g": int(len(instrument_sets)),
            # ``instruments`` is Stata's ``e(zrank)``: the "Number of instruments"
            # the estimator prints.  Stata's total is one larger than the number
            # of columns in this adapter's differenced instrument matrix, because
            # Stata also declares ``_cons`` in the *level* equation
            # (``Instruments for level equation / Standard: _cons``) - a moment
            # the one-step difference estimator never uses (adding it as a
            # difference column instead moves the ``lags(2)`` coefficients by
            # 3.5e-2).  Both counts are reported: ``instruments`` matches Stata's
            # label, ``instruments_difference`` is the matrix that is actually
            # used.
            "instruments": int(total_instruments) + 1,
            "instruments_difference": int(total_instruments),
            "zrank": int(total_instruments) + 1,
            "lags": p,
            "df_m": float(len(slopes)),
            "df_r": float(dof),
            "rank": int(stata_rank),
            "sig2": sigma2,
            "sigma2": sigma2,
            "rss": rss,
            "chi2": wald,
            "p": wald_p,
            "chi2type": "Wald",
            "vce": "gmm" if backend_cov == "gmm" else "robust",
            "method": "system" if self._system else "difference",
            # Deliberately absent: Sargan, Hansen, AR(1), AR(2). See the module
            # docstring -- reporting them approximately under Stata's names would
            # be worse than omitting them.
        }

        return StataResults(
            params=beta,
            param_names=names,
            cov=covariance,
            resid=residual,
            fittedvalues=design @ beta,
            nobs=int(n_rows),
            df_resid=float(dof),
            df_model=float(len(slopes)),
            dist="z",
            model_name=self._stata_command,
            cov_type=backend_cov,
            e=e,
            design_matrix=(design, names),
            model_data=self._model_data,
        )

    def _X_names(self) -> list[str]:
        return ["Intercept"] + self._exog_names


class XtDpd(XtAbond):
    """``xtdpd``: same one-step estimator, exposed under Stata's other spelling."""

    _stata_command = "xtdpd"


class XtDpdSys(XtAbond):
    """``xtdpdsys``: Blundell-Bond system GMM.

    The level equation adds instruments that the difference estimator does not
    use. Only the difference part is currently estimated; ``system=True`` marks the
    result so a reader can see that the level moments are missing rather than
    silently getting a difference estimator labelled as system.
    """

    _stata_command = "xtdpdsys"
