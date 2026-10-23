"""Complex survey designs (``svyset``) and the ``svy:`` prefix.

===================  ================
westata              Stata
===================  ================
``SurveyDesign``     ``svyset``
``SvyMean``          ``svy: mean``
``SvyTotal``         ``svy: total``
``SvyProportion``    ``svy: proportion``
``SvyRegress``       ``svy: regress``
``SvyLogit``         ``svy: logit``
===================  ================

The point estimates here are ordinary weighted statistics -- one line of algebra
each.  The work is entirely in the **variance**, because survey variance
estimation has a handful of conventions that agree on the point estimate and
diverge only in the third or fourth decimal of a standard error.  Every formula
below was therefore checked against Stata's full-precision ``e(V)``, never
against its printed six-decimal table
(``tests/conformance/stata_fixtures/p12_outputs.txt``).

Linearisation
-------------
For per-observation influence functions ``psi_i`` (the linearised score),

    T_p = sum_{i in PSU p} psi_i
    V   = sum_h  n_h/(n_h - 1) * sum_{p in h} (T_p - mean_h(T))(T_p - mean_h(T))'

with ``n_h`` the number of *PSUs* in stratum ``h``.  With a finite population
correction each stratum's contribution is multiplied by ``(1 - n_h/N_h)``.

The influence functions, each verified against Stata:

===================  ==========================================  =============
command              ``psi_i``                                   max rel gap
===================  ==========================================  =============
``svy: mean``        ``w_i (y_i - ybar) / sum(w)``               3e-16
``svy: total``       ``w_i y_i``                                1e-16
``svy: regress``     ``(X'WX)^-1 w_i x_i e_i``                  2e-15
``svy: logit``       ``I^-1 w_i x_i (y_i - p_i)``               5e-10
===================  ==========================================  =============

Three traps are worth recording, because all three leave the point estimate
untouched:

* ``svy: mean`` must carry its weight **inside** the influence function.
  ``y_i / sum(w)`` -- no weight, no centring, as one plausible reading of the
  literature suggests -- is 65% low on this design, and ``w_i y_i / sum(w)``
  (weight but no centring) is 30% low.  Only ``w_i (y_i - ybar) / sum(w)``
  reproduces Stata (3e-16).  Note that centring *per observation* is what
  matters; the centred total is invariant to it, but the interim PSU sums are
  not.
* ``svy: total`` uses ``psi_i = w_i y_i``.  Centring it on the estimate,
  ``w_i (y_i - T)``, looks harmless and is *not*: on the fixture it inflates
  ``V`` by a factor of 244,000 (1.69e9 instead of 6936.58), because ``T`` is a
  global constant whose interaction with the unequal weights does not cancel
  inside a stratum.
* ``svy: regress`` is not ``x_i e_i``: dropping either ``w_i`` or
  ``(X'WX)^-1`` leaves ``V`` grossly wrong while the coefficients stay exact to
  1e-15.  The sandwich spelling ``(X'WX)^-1 [PSU covariance of sum(w x e)]
  (X'WX)^-1`` is algebraically identical and gives the same numbers.

Degrees of freedom
------------------
``df_r = #PSU - #strata``, reported in ``e(df_r)`` and used as the reference
distribution's degrees of freedom (all five commands report ``dist="t"``).
Clamping it to at least one would silently overstate precision.

Single PSU strata
-----------------
A stratum containing one PSU carries no within-stratum variance.  ``singleunit``
selects what to do instead:

``certainty``
    The lone PSU is a certainty unit: it contributes exactly zero.  This is the
    setting the fixtures were captured under.
``missing``
    ``NaN`` -- the variance cannot be estimated, and a NaN is more honest than a
    number that pretends otherwise.
``scaling``
    Substitute the average per-PSU sum of squares of the strata that do have at
    least two PSUs, scaled to the average stratum size.
``centered``
    Centre the lone PSU's total on the grand mean of all PSU totals.

An unsupported strategy raises :class:`ValueError` at ``svyset`` time.

Conformance status: only ``certainty`` is exercised by the captured Stata
output, so ``scaling`` and ``centered`` carry E1/E2 evidence only -- they are
documented approximations of Stata's internal behaviour, not fixture-verified
arithmetic.  Use them knowingly.

References
----------
* ``[SVY] survey`` and ``[SVY] variance estimation`` -- the linearisation and
  the ``singleunit()`` options.
* Stata ``[U] 20.24`` -- weight semantics, via :mod:`westata.data.weights`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from westata.errors import ConsistencyError, ConvergenceError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = [
    "SurveyDesign",
    "SurveyLayout",
    "SvyLogit",
    "SvyMean",
    "SvyProportion",
    "SvyRegress",
    "SvyTotal",
]

#: Supported ``singleunit()`` strategies.  Only ``certainty`` is fixture-verified.
SINGLEUNIT_STRATEGIES = ("certainty", "scaling", "centered", "missing")


def _canonical_singleunit(value: Any) -> str:
    """Normalise and validate a ``singleunit()`` strategy."""

    if not isinstance(value, str):
        raise ValueError(
            "singleunit must be a string, one of: " + ", ".join(SINGLEUNIT_STRATEGIES)
        )
    strategy = value.strip().lower()
    if strategy not in SINGLEUNIT_STRATEGIES:
        raise ValueError(
            f"unknown singleunit strategy {value!r}; supported strategies are: "
            + ", ".join(SINGLEUNIT_STRATEGIES)
        )
    return strategy


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SurveyDesign:
    """A declared survey design: the westata counterpart of ``svyset``.

    Parameters
    ----------
    psu:
        Column holding the primary sampling unit identifier.  PSU labels are
        only meaningful within a stratum, which is Stata's convention too.
    strata:
        Optional column holding the stratum identifier.  When omitted the whole
        sample forms a single stratum.
    weights:
        Optional column holding the sampling (``pweight``) weights.  When
        omitted every observation counts once.
    fpc:
        Optional column holding the finite population correction.  A value above
        one is the population number of PSUs in that stratum (and must be
        constant within it); a value in ``(0, 1]`` is read as the sampling rate,
        following Stata's ``fpc()`` convention.
    singleunit:
        ``"certainty"``, ``"scaling"``, ``"centered"`` or ``"missing"``; see the
        module docstring.

    Notes
    -----
    The dataclass records *names*, not data, so one design can be applied to any
    frame carrying those columns.  ``__post_init__`` validates the strategy, so
    ``svyset, singleunit(typo)`` fails immediately rather than mid-estimation.
    """

    psu: str
    strata: str | None = None
    weights: str | None = None
    fpc: str | None = None
    singleunit: str = "certainty"

    def __post_init__(self) -> None:
        for label, value in (
            ("psu", self.psu),
            ("strata", self.strata),
            ("weights", self.weights),
            ("fpc", self.fpc),
        ):
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"svyset: {label} must be a column name (str), got {value!r}"
                )
        if not self.psu:
            raise ValueError("svyset: psu must name a column")
        object.__setattr__(self, "singleunit", _canonical_singleunit(self.singleunit))

    # ------------------------------------------------------------------ setup
    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        *,
        psu: str,
        strata: str | None = None,
        weights: str | None = None,
        fpc: str | None = None,
        singleunit: str = "certainty",
    ) -> "SurveyDesign":
        """Declare the design carried by ``frame``.

        Raises
        ------
        ValueError
            If ``singleunit`` is not a supported strategy.
        ConsistencyError
            If a declared column is absent from ``frame``.
        """

        # Validate the strategy before touching the data: `svyset` reports a bad
        # option independently of whether the variables exist.
        strategy = _canonical_singleunit(singleunit)
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("svyset: frame must be a pandas DataFrame")
        missing = [
            name
            for name in (psu, strata, weights, fpc)
            if name is not None and name not in frame.columns
        ]
        if missing:
            raise ConsistencyError(
                "svyset: the following variable(s) are not in the data: "
                + ", ".join(missing)
            )
        return cls(
            psu=psu, strata=strata, weights=weights, fpc=fpc, singleunit=strategy
        )

    # -------------------------------------------------------------- bind data
    def weight_vector(self, frame: pd.DataFrame) -> np.ndarray:
        """Sampling weights for ``frame``; ones when no ``weights`` was declared."""

        if self.weights is None:
            return np.ones(len(frame), dtype=float)
        values = np.asarray(frame[self.weights].to_numpy(), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ConsistencyError(
                f"svyset: weight variable {self.weights!r} has missing or "
                "infinite values"
            )
        if np.any(values <= 0):
            raise ConsistencyError(
                f"svyset: weight variable {self.weights!r} must be strictly "
                "positive (Stata rejects zero and negative pweights)"
            )
        return values

    def stratum_codes(self, frame: pd.DataFrame) -> np.ndarray:
        """Stratum code per observation; all zeros when no ``strata`` was declared."""

        if self.strata is None:
            return np.zeros(len(frame), dtype=int)
        labels = np.asarray(frame[self.strata].to_numpy(), dtype=object)
        if pd.isna(labels).any():
            raise ConsistencyError(
                f"svyset: strata variable {self.strata!r} has missing values"
            )
        codes, _ = pd.factorize(pd.Series(labels))
        return np.asarray(codes, dtype=int)

    def psu_codes(self, frame: pd.DataFrame) -> np.ndarray:
        """PSU code per observation, unique across strata.

        The key is the ``(stratum, psu)`` pair, so a PSU label may repeat across
        strata -- which is the normal Stata setup (``psu`` numbering restarts).
        """

        labels = np.asarray(frame[self.psu].to_numpy(), dtype=object)
        if pd.isna(labels).any():
            raise ConsistencyError(
                f"svyset: psu variable {self.psu!r} has missing values"
            )
        strata = self.stratum_codes(frame)
        keys = list(zip(strata.tolist(), labels.tolist()))
        codes, _ = pd.factorize(pd.Series(keys))
        return np.asarray(codes, dtype=int)

    def fpc_vector(self, frame: pd.DataFrame) -> np.ndarray | None:
        """The raw ``fpc`` column, or ``None`` when no correction was declared."""

        if self.fpc is None:
            return None
        values = np.asarray(frame[self.fpc].to_numpy(), dtype=float)
        if not np.all(np.isfinite(values)):
            raise ConsistencyError(
                f"svyset: fpc variable {self.fpc!r} has missing or infinite values"
            )
        return values

    def layout(self, frame: pd.DataFrame) -> "SurveyLayout":
        """Resolve this design against ``frame``.

        The resulting :class:`SurveyLayout` carries the PSU/stratum bookkeeping
        and owns :meth:`SurveyLayout.linearized_variance`, the single place the
        variance formula is implemented.
        """

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("svyset: frame must be a pandas DataFrame")
        n_obs = int(len(frame))
        if n_obs == 0:
            raise ConsistencyError("svyset: the frame has no observations")

        stratum_codes = self.stratum_codes(frame)
        psu_codes = self.psu_codes(frame)
        n_strata = int(stratum_codes.max()) + 1
        n_psu = int(psu_codes.max()) + 1

        # Each PSU belongs to exactly one stratum by construction of the key.
        psu_stratum = np.empty(n_psu, dtype=int)
        psu_stratum[psu_codes] = stratum_codes
        psus_by_stratum = {
            s: np.flatnonzero(psu_stratum == s) for s in range(n_strata)
        }

        weights = self.weight_vector(frame)
        fpc_factor = self._fpc_factors(frame, stratum_codes, psus_by_stratum, n_strata)

        return SurveyLayout(
            weights=weights,
            psu_codes=psu_codes,
            stratum_codes=stratum_codes,
            psu_stratum=psu_stratum,
            psus_by_stratum=psus_by_stratum,
            fpc_factor=fpc_factor,
            singleunit=self.singleunit,
        )

    def _fpc_factors(
        self,
        frame: pd.DataFrame,
        stratum_codes: np.ndarray,
        psus_by_stratum: Mapping[int, np.ndarray],
        n_strata: int,
    ) -> dict[int, float]:
        """Sampling fraction ``f_h`` per stratum, used as the ``(1 - f_h)`` factor."""

        if self.fpc is None:
            return {s: 0.0 for s in range(n_strata)}

        values = self.fpc_vector(frame)
        assert values is not None  # self.fpc is not None

        factors: dict[int, float] = {}
        for stratum in range(n_strata):
            n_h = int(psus_by_stratum[stratum].size)
            unique = np.unique(values[stratum_codes == stratum])
            if unique.size != 1:
                raise ConsistencyError(
                    f"svyset: fpc variable {self.fpc!r} varies within stratum "
                    f"{stratum}; it must be constant within a stratum"
                )
            population = float(unique[0])
            if population <= 0:
                raise ConsistencyError(
                    f"svyset: fpc value {population!r} in stratum {stratum} must "
                    "be positive"
                )
            if population <= 1.0:
                # Stata reads an fpc at or below one as the sampling rate itself.
                fraction = population
            else:
                if population < n_h:
                    raise ConsistencyError(
                        f"svyset: fpc value {population:g} in stratum {stratum} is "
                        f"smaller than the {n_h} sampled PSUs"
                    )
                fraction = n_h / population
            factors[stratum] = float(fraction)
        return factors

    def variance(self, frame: pd.DataFrame, influence: np.ndarray) -> np.ndarray:
        """Linearised variance of ``influence`` under this design.

        Convenience wrapper around :meth:`SurveyDesign.layout` for callers that
        hold a design and a frame but no layout.
        """

        return self.layout(frame).linearized_variance(influence)


@dataclass(frozen=True)
class SurveyLayout:
    """A :class:`SurveyDesign` resolved against one particular frame.

    Produced by :meth:`SurveyDesign.layout`.  It holds the per-observation PSU
    and stratum codes plus the per-stratum PSU lists that the variance needs, and
    it is immutable, so the same layout can be reused by every ``svy:`` command
    run on that sample.
    """

    weights: np.ndarray
    psu_codes: np.ndarray
    stratum_codes: np.ndarray
    psu_stratum: np.ndarray
    psus_by_stratum: Mapping[int, np.ndarray]
    fpc_factor: Mapping[int, float]
    singleunit: str = "certainty"

    @property
    def n_obs(self) -> int:
        return int(self.psu_codes.size)

    @property
    def n_psu(self) -> int:
        return int(self.psu_stratum.size)

    @property
    def n_strata(self) -> int:
        return int(len(self.psus_by_stratum))

    @property
    def df_r(self) -> int:
        """Design degrees of freedom: ``#PSU - #strata``."""

        return self.n_psu - self.n_strata

    @property
    def n_pop(self) -> float:
        """Estimated population size, i.e. ``sum(weights)``."""

        return float(self.weights.sum())

    def psu_totals(self, scores: np.ndarray) -> np.ndarray:
        """Sum ``scores`` within each PSU: the ``T_p`` of the formula."""

        psi = _as_score_matrix(scores, self.n_obs)
        totals = np.zeros((self.n_psu, psi.shape[1]), dtype=float)
        np.add.at(totals, self.psu_codes, psi)
        return totals

    # --------------------------------------------------------------- variance
    def linearized_variance(
        self, scores: np.ndarray, *, singleunit: str | None = None
    ) -> np.ndarray:
        """Taylor-series variance of the estimator defined by ``scores``.

        Parameters
        ----------
        scores:
            ``(n_obs, k)`` per-observation influence functions ``psi_i``; a 1-D
            array is treated as a single parameter.
        singleunit:
            Override the design's single-PSU strategy (used by ``svydescribe``
            style exploration and by the tests).
        """

        if self.df_r <= 0:
            raise ConsistencyError(
                "svy: the design has no residual degrees of freedom "
                f"({self.n_psu} PSUs in {self.n_strata} strata); at least one "
                "stratum needs two or more PSUs"
            )

        psi = _as_score_matrix(scores, self.n_obs)
        mode = (
            self.singleunit if singleunit is None else _canonical_singleunit(singleunit)
        )
        k = psi.shape[1]
        totals = self.psu_totals(psi)

        # Strata that support a within-stratum estimate.  Their PSU deviations
        # also supply the substitute used by singleunit(scaling).
        multi: list[tuple[int, np.ndarray]] = []
        for stratum, psus in self.psus_by_stratum.items():
            if int(psus.size) < 2:
                continue
            block = totals[psus]
            multi.append((stratum, block - block.mean(axis=0)))

        has_lone = any(int(psus.size) < 2 for psus in self.psus_by_stratum.values())

        substitute: np.ndarray | None = None
        if mode == "scaling" and has_lone:
            if not multi:
                raise ConsistencyError(
                    "svy: singleunit(scaling) needs at least one stratum with two "
                    "or more PSUs to borrow a variance scale from"
                )
            pooled = sum(dev.T @ dev for _, dev in multi)
            n_multi = sum(int(self.psus_by_stratum[s].size) for s, _ in multi)
            average_size = n_multi / len(multi)
            substitute = (average_size / (average_size - 1.0)) * (pooled / n_multi)

        grand_mean = totals.mean(axis=0)
        variance = np.zeros((k, k), dtype=float)

        for stratum, psus in self.psus_by_stratum.items():
            n_h = int(psus.size)
            keep = 1.0 - float(self.fpc_factor[stratum])

            if n_h >= 2:
                block = totals[psus]
                centred = block - block.mean(axis=0)
                variance += (n_h / (n_h - 1.0)) * keep * (centred.T @ centred)
                continue

            if mode == "certainty":
                continue
            if mode == "missing":
                variance += np.full((k, k), np.nan)
            elif mode == "scaling":
                assert substitute is not None  # guaranteed by the guard above
                variance += keep * substitute
            else:  # centered
                deviation = totals[psus[0]] - grand_mean
                variance += keep * np.outer(deviation, deviation)

        return variance


def _as_score_matrix(scores: Any, n_obs: int) -> np.ndarray:
    """Coerce ``scores`` to a validated ``(n_obs, k)`` float matrix."""

    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] != n_obs:
        raise ValueError(
            f"influence functions must have {n_obs} rows (one per observation), "
            f"got shape {matrix.shape}"
        )
    return matrix


# ---------------------------------------------------------------------------
# Shared estimation plumbing
# ---------------------------------------------------------------------------
def _prepare_sample(
    frame: pd.DataFrame, design: SurveyDesign, columns: Sequence[str]
) -> tuple[pd.DataFrame, SurveyLayout]:
    """Listwise-delete on the declared columns, then bind the design."""

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(design, SurveyDesign):
        raise TypeError("design must be a SurveyDesign")

    needed = [design.psu]
    for name in (design.strata, design.weights, design.fpc):
        if name is not None:
            needed.append(name)
    needed.extend(columns)
    unique = list(dict.fromkeys(needed))

    missing = [name for name in unique if name not in frame.columns]
    if missing:
        raise ConsistencyError(
            "svy: the following variable(s) are not in the data: "
            + ", ".join(missing)
        )

    complete = frame.loc[:, unique].notna().all(axis=1).to_numpy()
    if not complete.any():
        raise ConsistencyError(
            "svy: no observations remain after listwise deletion of missing values"
        )
    sample = frame.loc[complete]
    return sample, design.layout(sample)


def _survey_f_stat(
    beta: np.ndarray, covariance: np.ndarray, n_tested: int, df_r: float
) -> tuple[float, float]:
    """Stata's survey-adjusted Wald F test for the first ``n_tested`` slopes.

    ``F = (d - q + 1) / (d q) * W`` on ``F(q, d - q + 1)``, where
    ``W = b_R' [R V R']^-1 b_R`` and ``d`` is the design degrees of freedom.
    Verified against the fixture's ``F(2, 39) = 238.36`` (regress) and
    ``29.68`` (logit).
    """

    if n_tested <= 0 or not np.isfinite(df_r) or df_r - n_tested + 1 <= 0:
        return float("nan"), float("nan")

    selection = np.eye(beta.size)[:n_tested]
    restricted = selection @ beta
    middle = selection @ covariance @ selection.T
    try:
        wald = float(restricted @ np.linalg.solve(middle, restricted))
    except np.linalg.LinAlgError:  # pragma: no cover - singular restriction
        return float("nan"), float("nan")
    statistic = float((df_r - n_tested + 1.0) / (df_r * n_tested) * wald)
    p_value = float(stats.f.sf(statistic, n_tested, df_r - n_tested + 1))
    return statistic, p_value


def _variable_list(variables: Any) -> list[str]:
    """Normalise ``variables`` to a non-empty list of column names."""

    if isinstance(variables, str):
        names = [variables]
    else:
        names = list(variables)
    if not names:
        raise ConsistencyError("svy: at least one variable is required")
    if not all(isinstance(name, str) for name in names):
        raise TypeError("svy: variables must be column names (str)")
    if len(set(names)) != len(names):
        raise ConsistencyError("svy: variables must not repeat")
    return names


class _SvyCommand:
    """State shared by the five ``svy:`` estimators."""

    #: Stata command name; also the ``model_name`` shown in summaries.
    _model_name = ""
    _param_names: list[str]
    _layout: SurveyLayout
    _nobs: int

    def _results(
        self,
        params: np.ndarray,
        covariance: np.ndarray,
        *,
        resid: np.ndarray,
        fitted: np.ndarray,
        df_model: float,
        extra: Mapping[str, Any] | None = None,
    ) -> StataResults:
        e: dict[str, Any] = {
            "N": int(self._nobs),
            "N_strata": self._layout.n_strata,
            "N_psu": self._layout.n_psu,
            "N_pop": self._layout.n_pop,
            "df_m": float(df_model),
            "df_r": float(self._layout.df_r),
            "rank": len(self._param_names),
            "singleunit": self._layout.singleunit,
            "vce": "linearized",
        }
        if extra:
            e.update(dict(extra))
        return StataResults(
            params=np.asarray(params, dtype=float).reshape(-1),
            param_names=list(self._param_names),
            cov=np.asarray(covariance, dtype=float),
            resid=np.asarray(resid, dtype=float).reshape(-1),
            fittedvalues=np.asarray(fitted, dtype=float).reshape(-1),
            nobs=int(self._nobs),
            df_resid=float(self._layout.df_r),
            df_model=float(df_model),
            dist="t",
            model_name=self._model_name,
            cov_type="linearized",
            e=e,
        )


# ---------------------------------------------------------------------------
# svy: mean / total / proportion
# ---------------------------------------------------------------------------
class SvyMean(_SvyCommand):
    """Survey-weighted mean (``svy: mean``).

    ``ybar = sum(w y) / sum(w)`` with influence function
    ``psi_i = w_i (y_i - ybar) / sum(w)``.  Both the weight and the centring are
    part of the linearisation; see the module docstring and
    ``test_mean_requires_the_weight_inside_the_influence``.
    """

    _model_name = "svy: mean"

    def __init__(
        self,
        variables: Any,
        frame: pd.DataFrame,
        design: SurveyDesign,
        *,
        params: np.ndarray,
        layout: SurveyLayout,
        sample: pd.DataFrame,
    ) -> None:
        self._variables = _variable_list(variables)
        self._param_names = list(self._variables)
        self._frame = frame
        self._design = design
        self._params = np.asarray(params, dtype=float).reshape(-1)
        self._layout = layout
        self._sample = sample
        self._nobs = int(layout.n_obs)

    @classmethod
    def from_frame(
        cls, variables: Any, frame: pd.DataFrame, design: SurveyDesign
    ) -> "SvyMean":
        names = _variable_list(variables)
        sample, layout = _prepare_sample(frame, design, names)
        values = sample.loc[:, names].to_numpy(dtype=float)
        params = layout.weights @ values / float(layout.weights.sum())
        return cls(
            names, frame, design, params=params, layout=layout, sample=sample
        )

    def _values(self) -> np.ndarray:
        return self._sample.loc[:, self._variables].to_numpy(dtype=float)

    def influence(self) -> np.ndarray:
        """Per-observation influence functions ``w_i (y_i - ybar) / sum(w)``."""

        values = self._values()
        weights = self._layout.weights
        return (
            weights[:, None] * (values - self._params[None, :])
            / float(weights.sum())
        )

    def fit(self) -> StataResults:
        covariance = self._layout.linearized_variance(self.influence())
        values = self._values()
        # A joint mean over k variables has no single residual vector; the
        # container takes the first variable's deviation.
        return self._results(
            self._params,
            covariance,
            resid=values[:, 0] - self._params[0],
            fitted=np.full(self._nobs, float(self._params[0])),
            df_model=0.0,
        )


class SvyTotal(SvyMean):
    """Survey-weighted total (``svy: total``).

    ``T = sum(w y)`` with influence function ``psi_i = w_i y_i``.  The weights
    are *not* normalised: the total estimates a population quantity, so its scale
    is part of the answer.  Unlike :class:`SvyMean` the influence is **not**
    centred on the estimate -- doing so inflates ``V`` by five orders of
    magnitude on the fixture design.
    """

    _model_name = "svy: total"

    @classmethod
    def from_frame(
        cls, variables: Any, frame: pd.DataFrame, design: SurveyDesign
    ) -> "SvyTotal":
        names = _variable_list(variables)
        sample, layout = _prepare_sample(frame, design, names)
        values = sample.loc[:, names].to_numpy(dtype=float)
        params = layout.weights @ values
        return cls(names, frame, design, params=params, layout=layout, sample=sample)

    def influence(self) -> np.ndarray:
        """Per-observation influence functions ``w_i y_i`` (uncentred)."""

        return self._layout.weights[:, None] * self._values()

    def fit(self) -> StataResults:
        covariance = self._layout.linearized_variance(self.influence())
        return self._results(
            self._params,
            covariance,
            resid=np.zeros(self._nobs, dtype=float),
            fitted=self._layout.weights * self._values()[:, 0],
            df_model=0.0,
        )


def _level_label(variable: str, level: Any) -> str:
    """Row label for a level, in Stata's ``variable=level`` shape."""

    if isinstance(level, (bool, np.bool_)):
        rendered = str(bool(level))
    elif isinstance(level, (int, np.integer)):
        rendered = str(int(level))
    elif isinstance(level, (float, np.floating)):
        value = float(level)
        rendered = str(int(value)) if value.is_integer() else repr(value)
    else:
        rendered = str(level)
    return f"{variable}={rendered}"


def _sorted_levels(column: pd.Series) -> list[Any]:
    """Observed levels in ascending order, falling back to text order."""

    observed = list(pd.unique(column.dropna()))
    try:
        return sorted(observed)
    except TypeError:  # pragma: no cover - mixed types only
        return sorted(observed, key=str)


class SvyProportion(SvyMean):
    """Weighted proportions of a categorical variable (``svy: proportion``).

    One coefficient per observed level, in ``variable=level`` order with the
    joint covariance, matching Stata's table layout: the levels of a binary
    variable get identical standard errors and perfectly negatively correlated
    estimates, because each is the mean of its indicator.
    """

    _model_name = "svy: proportion"

    def __init__(
        self,
        variables: Any,
        frame: pd.DataFrame,
        design: SurveyDesign,
        *,
        params: np.ndarray,
        indicators: np.ndarray,
        layout: SurveyLayout,
        sample: pd.DataFrame,
        names: list[str],
    ) -> None:
        self._variables = _variable_list(variables)
        self._indicators = indicators
        self._param_names = list(names)
        self._frame = frame
        self._design = design
        self._params = np.asarray(params, dtype=float).reshape(-1)
        self._layout = layout
        self._sample = sample
        self._nobs = int(layout.n_obs)

    @classmethod
    def from_frame(
        cls, variables: Any, frame: pd.DataFrame, design: SurveyDesign
    ) -> "SvyProportion":
        declared = _variable_list(variables)
        sample, layout = _prepare_sample(frame, design, declared)

        blocks: list[np.ndarray] = []
        names: list[str] = []
        for variable in declared:
            column = sample[variable]
            for level in _sorted_levels(column):
                blocks.append((column.to_numpy() == level).astype(float))
                names.append(_level_label(variable, level))
        indicators = np.column_stack(blocks)
        params = layout.weights @ indicators / float(layout.weights.sum())
        return cls(
            declared,
            frame,
            design,
            params=params,
            indicators=indicators,
            layout=layout,
            sample=sample,
            names=names,
        )

    def influence(self) -> np.ndarray:
        """Per-observation influence of each level proportion."""

        weights = self._layout.weights
        return (
            weights[:, None]
            * (self._indicators - self._params[None, :])
            / float(weights.sum())
        )

    def fit(self) -> StataResults:
        covariance = self._layout.linearized_variance(self.influence())
        first = self._indicators[:, 0]
        return self._results(
            self._params,
            covariance,
            resid=first - self._params[0],
            fitted=first,
            df_model=0.0,
        )


# ---------------------------------------------------------------------------
# svy: regress / svy: logit
# ---------------------------------------------------------------------------
def _stata_design(
    formula: str, sample: pd.DataFrame
) -> tuple[np.ndarray, list[str], np.ndarray, bool, np.ndarray]:
    """Build the design matrix in Stata's coefficient order.

    Stata reports slopes first and the constant last, named ``_cons``; patsy
    puts ``Intercept`` first.  Returning the wrong *order* makes a numerically
    perfect estimate look wrong to any positional comparison, so the reordering
    happens once, here.

    Returns ``(X, names, y, has_constant, row_positions)`` where
    ``row_positions`` locates the surviving rows inside ``sample`` (patsy drops
    rows whose transformed terms are missing).
    """

    if not isinstance(formula, str) or "~" not in formula:
        raise ConsistencyError(
            "svy: the model must be a two-sided formula such as 'y ~ x1 + x2'"
        )
    lhs = formula.split("~", 1)[0].strip()
    if lhs not in sample.columns:
        raise ConsistencyError(
            f"svy: the left-hand side must be a single column name, got {lhs!r}"
        )

    X, names, kept_index = build_design(formula, sample)
    X = np.asarray(X, dtype=float)
    y = sample.loc[kept_index, lhs].to_numpy(dtype=float)

    positions = sample.index.get_indexer(kept_index)
    if np.any(positions < 0):  # pragma: no cover - patsy always returns subsets
        raise ConsistencyError(
            "svy: could not align the model matrix with the estimation sample"
        )

    has_constant = "Intercept" in names
    if has_constant:
        intercept = names.index("Intercept")
        order = [j for j in range(len(names)) if j != intercept] + [intercept]
        X = X[:, order]
        names = [names[j] for j in order[:-1]] + ["_cons"]
    return X, list(names), y, has_constant, positions


def _align_sample(
    sample: pd.DataFrame,
    layout: SurveyLayout,
    design: SurveyDesign,
    positions: np.ndarray,
) -> tuple[pd.DataFrame, SurveyLayout]:
    """Re-bind the design when a transformed term dropped extra rows."""

    if positions.size == sample.shape[0]:
        return sample, layout
    reduced = sample.iloc[positions]
    return reduced, design.layout(reduced)


class SvyRegress(_SvyCommand):
    """Survey-weighted linear regression (``svy: regress``).

    The coefficients are weighted least squares -- exactly what
    ``ws.OLS(weights=..., weight_type="pweight")`` fits -- so only the variance
    is new.  The influence function is
    ``psi_i = (X'WX)^-1 w_i x_i e_i``: both the weight and the inverse
    cross-product belong to the linearisation.
    """

    _model_name = "svy: regress"

    def __init__(
        self,
        frame: pd.DataFrame,
        formula: str,
        design: SurveyDesign,
        *,
        X: np.ndarray,
        y: np.ndarray,
        names: list[str],
        has_constant: bool,
        weights: np.ndarray,
        layout: SurveyLayout,
        sample: pd.DataFrame,
        params: np.ndarray,
        resid: np.ndarray,
        xtwx_inv_xt: np.ndarray,
    ) -> None:
        self._frame = frame
        self._formula = formula
        self._design = design
        self._X = X
        self._y = y
        self._param_names = list(names)
        self._has_constant = has_constant
        self._weights = weights
        self._layout = layout
        self._sample = sample
        self._params = np.asarray(params, dtype=float).reshape(-1)
        self._resid = np.asarray(resid, dtype=float).reshape(-1)
        self._xtwx_inv_xt = xtwx_inv_xt
        self._nobs = int(layout.n_obs)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        formula: str,
        design: SurveyDesign,
        *,
        subset: Any = None,
    ) -> "SvyRegress":
        if not isinstance(design, SurveyDesign):
            raise TypeError("design must be a SurveyDesign")
        if subset is not None:
            frame = frame.loc[subset]
        lhs = formula.split("~", 1)[0].strip() if isinstance(formula, str) else ""
        sample, layout = _prepare_sample(frame, design, [lhs] if lhs else [])

        X, names, y, has_constant, positions = _stata_design(formula, sample)
        sample, layout = _align_sample(sample, layout, design, positions)

        n, k = X.shape
        if n <= k:
            raise ConsistencyError(
                f"svy: regress: {n} observations but {k} parameters; the model is "
                "not identified"
            )
        if int(np.linalg.matrix_rank(X)) < k:
            raise ConsistencyError(
                "svy: regress: the design matrix is rank deficient; drop "
                "collinear regressors"
            )

        weights = layout.weights
        xtwx = X.T @ (X * weights[:, None])
        try:
            params = np.linalg.solve(xtwx, X.T @ (weights * y))
            # The influence function needs (X'WX)^-1 X'; solving is both cheaper
            # and better conditioned than forming the inverse explicitly.
            xtwx_inv_xt = np.linalg.solve(xtwx, X.T)
        except np.linalg.LinAlgError:  # pragma: no cover - guarded by rank check
            raise ConsistencyError(
                "svy: regress: the weighted cross-product X'WX is singular; drop "
                "collinear regressors"
            ) from None
        resid = y - X @ params

        return cls(
            frame,
            formula,
            design,
            X=X,
            y=y,
            names=names,
            has_constant=has_constant,
            weights=weights,
            layout=layout,
            sample=sample,
            params=params,
            resid=resid,
            xtwx_inv_xt=xtwx_inv_xt,
        )

    def influence(self) -> np.ndarray:
        """Per-observation influence ``(X'WX)^-1 w_i x_i e_i``."""

        weighted_resid = self._weights * self._resid
        return self._xtwx_inv_xt.T * weighted_resid[:, None]

    def _r_squared(self) -> float:
        weights = self._weights
        rss = float((weights * self._resid**2).sum())
        mean = float(weights @ self._y / weights.sum())
        tss = float((weights * (self._y - mean) ** 2).sum())
        return 1.0 - rss / tss if tss > 0 else float("nan")

    def fit(self) -> StataResults:
        covariance = self._layout.linearized_variance(self.influence())
        df_r = float(self._layout.df_r)
        n_tested = self._params.size - (1 if self._has_constant else 0)
        statistic, p_value = _survey_f_stat(self._params, covariance, n_tested, df_r)
        return self._results(
            self._params,
            covariance,
            resid=self._resid,
            fitted=self._X @ self._params,
            df_model=float(n_tested),
            extra={"r2": self._r_squared(), "F": statistic, "p": p_value},
        )


def _weighted_loglikelihood(
    linear_predictor: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> float:
    """Weighted Bernoulli log-likelihood, computed in a numerically safe form."""

    return float(np.sum(weights * (y * linear_predictor - np.logaddexp(0.0, linear_predictor))))


def _fit_weighted_logit(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    maxiter: int = 200,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Pseudo-ML logit with sampling weights, by damped Newton-Raphson.

    For the canonical link the observed and expected information coincide, so
    the Newton step uses ``X' diag(w p(1-p)) X``.  Step halving keeps the
    iteration monotone on near-separated samples, which is where a plain Newton
    loop overshoots into infinity.

    Returns ``(beta, probabilities, loglikelihood, iterations)``.
    """

    n, k = X.shape
    if n <= k:
        raise ConsistencyError(
            f"svy: logit: {n} observations but {k} parameters; the model is not "
            "identified"
        )

    beta = np.zeros(k, dtype=float)
    linear_predictor = X @ beta
    loglik = _weighted_loglikelihood(linear_predictor, y, weights)

    for iteration in range(1, maxiter + 1):
        probabilities = expit(linear_predictor)
        curvature = weights * probabilities * (1.0 - probabilities)
        information = X.T @ (X * curvature[:, None])
        gradient = X.T @ (weights * (y - probabilities))
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            raise ConvergenceError(
                "svy: logit: the information matrix is singular, which normally "
                "means the model is not identified (collinear regressors or a "
                "separated outcome)"
            ) from None
        if not np.all(np.isfinite(step)):
            raise ConvergenceError(
                "svy: logit: the Newton step is not finite; the likelihood is "
                "unbounded, which normally means perfect separation"
            )

        scale = 1.0
        for _ in range(50):
            candidate = beta + scale * step
            candidate_predictor = X @ candidate
            candidate_loglik = _weighted_loglikelihood(candidate_predictor, y, weights)
            if np.isfinite(candidate_loglik) and candidate_loglik >= (
                loglik - 1e-12 * abs(loglik)
            ):
                break
            scale *= 0.5
        else:  # pragma: no cover - only a pathological likelihood
            raise ConvergenceError(
                "svy: logit: the optimiser could not find an improving step; the "
                "likelihood is unbounded, which normally means perfect separation"
            )

        beta = candidate
        linear_predictor = candidate_predictor
        loglik = candidate_loglik
        if np.max(np.abs(scale * step)) <= tol * (1.0 + np.max(np.abs(beta))):
            return beta, expit(linear_predictor), loglik, iteration

    raise ConvergenceError(
        f"svy: logit: no convergence after {maxiter} Newton iterations"
    )


class SvyLogit(_SvyCommand):
    """Survey-weighted logistic regression (``svy: logit``).

    Pseudo-maximum-likelihood with sampling weights, linearised with
    ``psi_i = I^-1 w_i x_i (y_i - p_i)`` where ``I = X' diag(w p(1-p)) X``.

    Conformance note
    ----------------
    On the fixture design the estimates reproduce Stata to about ``5e-10``
    relative.  That residual is **Stata's stopping tolerance, not this code's**:
    evaluating Stata's own score equations at Stata's published coefficients
    leaves a gradient of ``1.7e-8``, while the estimates returned here are the
    exact maximiser (gradient ``8e-15``).  A conformance tolerance tighter than
    ``1e-9`` would therefore assert something untrue about Stata rather than
    about westata.
    """

    _model_name = "svy: logit"

    def __init__(
        self,
        frame: pd.DataFrame,
        formula: str,
        design: SurveyDesign,
        *,
        X: np.ndarray,
        y: np.ndarray,
        names: list[str],
        has_constant: bool,
        weights: np.ndarray,
        layout: SurveyLayout,
        sample: pd.DataFrame,
        params: np.ndarray,
        probabilities: np.ndarray,
        information_inverse_xt: np.ndarray,
        loglik: float,
        iterations: int,
    ) -> None:
        self._frame = frame
        self._formula = formula
        self._design = design
        self._X = X
        self._y = y
        self._param_names = list(names)
        self._has_constant = has_constant
        self._weights = weights
        self._layout = layout
        self._sample = sample
        self._params = np.asarray(params, dtype=float).reshape(-1)
        self._probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
        self._information_inverse_xt = information_inverse_xt
        self._loglik = float(loglik)
        self._iterations = int(iterations)
        self._nobs = int(layout.n_obs)

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        formula: str,
        design: SurveyDesign,
        *,
        subset: Any = None,
    ) -> "SvyLogit":
        if not isinstance(design, SurveyDesign):
            raise TypeError("design must be a SurveyDesign")
        if subset is not None:
            frame = frame.loc[subset]
        lhs = formula.split("~", 1)[0].strip() if isinstance(formula, str) else ""
        sample, layout = _prepare_sample(frame, design, [lhs] if lhs else [])

        X, names, y, has_constant, positions = _stata_design(formula, sample)
        sample, layout = _align_sample(sample, layout, design, positions)

        observed = np.unique(y)
        if not np.all(np.isin(observed, (0.0, 1.0))):
            raise ConsistencyError(
                "svy: logit: the dependent variable must be coded 0/1; found "
                f"values {observed[:5].tolist()}"
            )
        if observed.size < 2:
            raise ConsistencyError(
                "svy: logit: the dependent variable is constant; nothing to estimate"
            )

        weights = layout.weights
        params, probabilities, loglik, iterations = _fit_weighted_logit(X, y, weights)

        curvature = weights * probabilities * (1.0 - probabilities)
        information = X.T @ (X * curvature[:, None])
        try:
            information_inverse_xt = np.linalg.solve(information, X.T)
        except np.linalg.LinAlgError:  # pragma: no cover - guarded inside the fit
            raise ConsistencyError(
                "svy: logit: the information matrix is singular; drop collinear "
                "regressors"
            ) from None

        return cls(
            frame,
            formula,
            design,
            X=X,
            y=y,
            names=names,
            has_constant=has_constant,
            weights=weights,
            layout=layout,
            sample=sample,
            params=params,
            probabilities=probabilities,
            information_inverse_xt=information_inverse_xt,
            loglik=loglik,
            iterations=iterations,
        )

    def influence(self) -> np.ndarray:
        """Per-observation influence ``I^-1 w_i x_i (y_i - p_i)``."""

        weighted_resid = self._weights * (self._y - self._probabilities)
        return self._information_inverse_xt.T * weighted_resid[:, None]

    def fit(self) -> StataResults:
        covariance = self._layout.linearized_variance(self.influence())
        df_r = float(self._layout.df_r)
        n_tested = self._params.size - (1 if self._has_constant else 0)
        statistic, p_value = _survey_f_stat(self._params, covariance, n_tested, df_r)
        return self._results(
            self._params,
            covariance,
            resid=self._y - self._probabilities,
            fitted=self._probabilities,
            df_model=float(n_tested),
            extra={
                "ll": self._loglik,
                "F": statistic,
                "p": p_value,
                "iterations": self._iterations,
            },
        )
