"""Endogeneity score tests: robust (``robwool``) and HAC (``acwool``).

Transcribed from ``ivregress_estat.ado:1187-1263`` (``Endog2sls``).

With ``endog`` the endogenous regressors, ``exog`` the included exogenous ones
and ``addlinst`` the excluded instruments, the procedure is:

.. code-block:: stata

    * uhat: OLS residual of y on (exog + endog), no constant (the constant is
      already in exog when the model has one)
    _regress `depvar' `exog' `endog' `wt' if `touse', `nocons'
    predict double `uhat', residuals

    * endhat_i: each endogenous regressor projected on (exog + addlinst)
    _regress `var' `exog' `addlinst' `wt' if `touse', `nocons'
    predict double `endhat`i''

    * rhat_i: endhat_i residualised on (exog + endog), times uhat
    _regress `var' `exog' `endog' `wt' if `touse', `nocons'
    predict double `rhat`i'', residuals
    replace `rhat`i'' = `rhat`i''*`uhat'

    * robust score
    _regress `one' `rhat' `wt' if `touse', nocons
    robwool = e(N) - e(rss)

    * HAC score: prewhiten rhat by its own lags first
    mvreg `rhat' = L(1/`lags').(`rhat') `wt' if `touse', nocons
    * ... then the same N - rss on the residuals
    acwool = e(N) - e(rss)

Both are chi-squared with ``df`` = number of endogenous regressors tested.

**Which test is reported depends on the VCE** (``ivregress_estat.ado:1128-1145``):

======================  ====================================================
``e(vce)``              tests performed
======================  ====================================================
``unadjusted``          Durbin-Wu
``robust``              ``regF``, ``robwool``
``cluster``             ``regF`` only
``HAC``                 ``acwool``, ``regF``
======================  ====================================================

The shape is deliberately the same as the overidentification score test: regress
a vector of ones on the ``rhat`` columns without a constant and report
``N - rss``.  The prewhitening step is likewise ``mvreg ... nocons`` on the lagged
columns, which drops the first ``lags`` rows.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats

__all__ = ["EndogScoreResult", "endog_score_test"]

_CONSTANT_NAMES = frozenset({"intercept", "intercept_", "_cons", "const"})


def _is_constant(name: str) -> bool:
    return str(name).strip().lower() in _CONSTANT_NAMES


def _ols_residuals(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients


def _ols_fitted(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return design @ coefficients


def _prewhiten(columns: np.ndarray, lags: int) -> np.ndarray:
    """``mvreg rhat = L(1/lags).rhat, nocons`` then keep the residuals.

    ``mvreg`` drops the first ``lags`` rows (they have no lagged values), and with
    ``nocons`` and no cross-equation restriction each equation is one OLS on the
    same regressor matrix.  Padding those rows with zeros instead would inflate
    ``N`` and change the statistic whenever there is more than one ``rhat``
    column -- the mistake round 28 had to fix in the overid score test.
    """

    if lags <= 0:
        return columns
    if columns.shape[0] - lags <= 0:  # pragma: no cover - too short to prewhiten
        return columns
    lagged = np.column_stack(
        [
            columns[lags - 1 - lag : columns.shape[0] - 1 - lag]
            for lag in range(lags)
        ]
    )
    target = columns[lags:]
    out = np.empty_like(target)
    for column in range(target.shape[1]):
        beta, *_ = np.linalg.lstsq(lagged, target[:, column], rcond=None)
        out[:, column] = target[:, column] - lagged @ beta
    return out


def _score_from_rhat(rhat: np.ndarray) -> float:
    """``_regress ones rhat, nocons`` -> ``e(N) - e(rss)``."""

    ones = np.ones(rhat.shape[0])
    beta, *_ = np.linalg.lstsq(rhat, ones, rcond=None)
    residual = ones - rhat @ beta
    return float(rhat.shape[0] - residual @ residual)


class EndogScoreResult:
    """An endogeneity score test.

    Attributes
    ----------
    statistic, df, pvalue:
        The chi-square statistic, its degrees of freedom (the number of
        endogenous regressors tested) and the upper-tail p-value.
    test:
        ``"robwool"`` for the robust variant, ``"acwool"`` for HAC.
    lags:
        HAC truncation used for prewhitening; ``None`` for the robust variant.
    """

    __slots__ = ("statistic", "df", "pvalue", "test", "lags")

    def __init__(
        self,
        *,
        statistic: float,
        df: float,
        test: str,
        lags: int | None = None,
    ) -> None:
        self.statistic = float(statistic)
        self.df = float(df)
        self.test = str(test)
        self.lags = None if lags is None else int(lags)
        self.pvalue = float(stats.chi2.sf(self.statistic, self.df))

    def __repr__(self) -> str:
        return (
            f"<EndogScoreResult {self.test} chi2({self.df:g})="
            f"{self.statistic:.4f} p={self.pvalue:.4f}>"
        )

    def __str__(self) -> str:
        label = (
            "Robust score chi2"
            if self.test == "robwool"
            else f"HAC score chi2 (lags={self.lags})"
        )
        return (
            f"\nTests of endogeneity:\n\n"
            f"{label + '(' + f'{self.df:g}' + ')':<28}"
            f" = {self.statistic:>9.4f}\n"
            f"{'Prob > chi2':<28} = {self.pvalue:>9.4f}"
        )

    def to_dict(self) -> dict[str, float]:
        """Stata's ``r()`` scalars."""

        if self.test == "acwool":
            return {
                "hac_score": self.statistic,
                "p_hac_score": self.pvalue,
                "lags": float(self.lags) if self.lags is not None else float("nan"),
            }
        return {"r_score": self.statistic, "p_r_score": self.pvalue}


def endog_score_test(
    structure: dict[str, Any],
    resid: np.ndarray,
    response: np.ndarray,
    *,
    hac: bool = False,
    lags: int = 0,
    tested: Sequence[str] | None = None,
) -> EndogScoreResult:
    """Run ``robwool`` (robust) or ``acwool`` (HAC).

    Parameters
    ----------
    structure:
        The ``iv_structure`` dict with ``exog``/``endog``/``instruments`` and
        their names.
    resid:
        The IV residuals of the consistent (2SLS) fit.  The ado refits an OLS for
        ``uhat``; on the estimation sample that is the same residual series only
        when ``exog`` spans the same space, so the caller passes the consistent
        residuals explicitly.
    response:
        The dependent variable, used to rebuild ``uhat`` exactly as the ado does.
    hac:
        Use the HAC variant (prewhiten the ``rhat`` columns).
    lags:
        HAC truncation; ignored unless ``hac`` is true.
    tested:
        Endogenous regressors to test.  Stata refuses a subset under a
        robust/cluster/HAC VCE (``ivregress_estat.ado:1070-1076``), so a subset
        longer or shorter than the full list is rejected here too.

    Raises
    ------
    ValueError
        When ``tested`` is not exactly the full set of endogenous regressors, or
        when no endogenous regressor is present.
    """

    exog = np.asarray(structure["exog"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    instruments = np.asarray(structure["instruments"], dtype=float)
    exog_names = [str(name) for name in structure["exog_names"]]
    endog_names = [str(name) for name in structure["endog_names"]]

    n_endog = endog.shape[1]
    if n_endog == 0:
        raise ValueError(
            "estat endogenous needs at least one endogenous regressor"
        )

    # ``ivregress_estat.ado:1070-1076``: a subset cannot be tested under a
    # robust/cluster/HAC VCE.
    if tested is not None:
        requested = [str(v) for v in tested]
        if sorted(requested) != sorted(endog_names):
            raise ValueError(
                "cannot test a subset of endogenous regressors if a robust, "
                "cluster, or HAC VCE was used at estimation; "
                f"tested={requested} but the endogenous set is {endog_names}"
            )

    n = np.asarray(resid, dtype=float).size
    response = np.asarray(response, dtype=float).reshape(-1)
    if response.size != n:
        raise ValueError(
            f"response has {response.size} entries but there are {n} residuals"
        )

    has_constant = any(_is_constant(name) for name in exog_names)
    included = [j for j, name in enumerate(exog_names) if not _is_constant(name)]

    def _design(extra: list[np.ndarray]) -> np.ndarray:
        parts: list[np.ndarray] = []
        if has_constant:
            parts.append(np.ones(n))
        if included:
            parts.append(exog[:, included])
        parts.extend(extra)
        return np.column_stack(parts)

    # uhat: OLS residual of y on (exog + endog).
    uhat = _ols_residuals(response, _design([endog]))

    # endhat_i: each endogenous regressor on (exog + addlinst), where addlinst is
    # the excluded instruments.
    projection_design = _design([instruments])
    endhat = np.column_stack(
        [_ols_fitted(endog[:, c], projection_design) for c in range(n_endog)]
    )

    # rhat_i = residual of endhat_i on (exog + endog), times uhat.
    residualisation_design = _design([endog])
    rhat = np.column_stack(
        [
            _ols_residuals(endhat[:, c], residualisation_design) * uhat
            for c in range(n_endog)
        ]
    )

    if hac:
        rhat = _prewhiten(rhat, int(lags))

    statistic = _score_from_rhat(rhat)
    return EndogScoreResult(
        statistic=statistic,
        df=float(n_endog),
        test="acwool" if hac else "robwool",
        lags=int(lags) if hac else None,
    )
