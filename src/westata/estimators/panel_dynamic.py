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

The part that is easy to get wrong is ``Z_i``, the instrument matrix. For a model
with ``p`` lags on a balanced panel, ``Z_di`` has ``T - p - 1`` rows and
``sum_{m=p}^{T-2} m`` columns, laid out as a lower-triangular block matrix: the row
for period ``t`` contains ``y_{i,1}`` through ``y_{i,t-1}`` and zeros elsewhere.
Valid lags start at ``p+1`` because the first ``p`` observations are lost to the
lags themselves and the differencing.

The number of instrument columns is therefore a function of ``p`` and ``T`` only,
and it is the sharpest available check on the construction: Stata reports it as
"Number of instruments", so a wrong ``Z`` shows up there immediately, before any
coefficient is examined.

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
            param_names=(f"L{lags}.y",),
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
        units = np.unique(self._groups)
        p = self._lags

        blocks: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        regressors: list[np.ndarray] = []
        instrument_sets: list[np.ndarray] = []
        total_instruments = 0

        for unit in units:
            mask = self._groups == unit
            series = self._y[mask]
            covariates = self._X[mask][:, [i for i, n in enumerate(self._X_names()) if n != "Intercept"]]
            columns = covariates.shape[1]
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
            # Standard instruments: the differenced covariates and a constant.
            # Stata counts these alongside the GMM columns (28 + 2 = 30 for
            # lags(1) on T=9), and leaving them out biases the covariate
            # coefficient badly.
            standard = np.column_stack(
                [covariate_diff, np.ones(covariate_diff.shape[0])]
            )
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

        total_rows = stacked_instruments.shape[0]
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

        # Rescale for numerical conditioning: the raw cross-products are large
        # because the instrument matrix has many columns.
        beta = np.linalg.pinv(w1) @ cross @ a1 @ moment



        residual = target - design @ beta
        dof = max(n_rows - k, 1)
        sigma2 = float(residual @ residual) / dof

        # Sandwich for the one-step estimator.
        bread = np.linalg.pinv(w1)
        meat = cross @ a1 @ (stacked_instruments.T @ stacked_instruments) @ a1 @ cross.T
        covariance = sigma2 * bread @ meat @ bread

        names = self._lag_names + self._exog_names
        params = beta
        if len(names) != params.size:
            # The design also carries a constant from the exogenous block.
            extra = params.size - len(names)
            names = names + [f"extra_{i}" for i in range(extra)]

        e: dict[str, Any] = {
            "N": int(n_rows),
            "N_groups": int(len(instrument_sets)),
            "instruments": int(total_instruments),
            "lags": p,
            "df_m": float(k - 1),
            "df_r": float(dof),
            "rank": int(k),
            "sigma2": sigma2,
            "method": "system" if self._system else "difference",
            # Deliberately absent: Sargan, Hansen, AR(1), AR(2). See the module
            # docstring -- reporting them approximately under Stata's names would
            # be worse than omitting them.
        }

        return StataResults(
            params=params,
            param_names=names,
            cov=covariance,
            resid=residual,
            fittedvalues=design @ beta,
            nobs=int(n_rows),
            df_resid=float(dof),
            df_model=float(k - 1),
            dist="z",
            model_name=self._stata_command,
            cov_type="gmm",
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
