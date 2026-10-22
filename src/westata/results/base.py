"""statsmodels-compatible results container with Stata ``e()`` scalars.

``StataResults`` is the single results object returned by every westata
estimator.  It implements the surface statsmodels downstream code expects
(``params``/``bse``/``tvalues``/``pvalues``/``conf_int``/``cov_params``/...) and
adds the Stata post-estimation scalars via :attr:`StataResults.e`.

The container is intentionally dumb: it renders what it is given.  Degrees of
freedom, which distribution to use for inference, and the ``e()`` scalars are
decided by the estimator (the Stata-conformance layer), not here.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats

from westata.results.stata_return import EReturn
from westata.results.summary import render_summary

__all__ = ["StataResults"]

_VALID_DISTRIBUTIONS = ("t", "z")


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class StataResults:
    """Estimation results with a statsmodels-compatible API.

    Parameters
    ----------
    params:
        Point estimates, length ``k``.
    param_names:
        Variable names, length ``k``; used as the index of every coefficient
        Series.
    cov:
        Covariance matrix (``k x k``) already including all requested
        small-sample corrections.
    resid, fittedvalues:
        Observation-level residuals and fitted values, length ``nobs``.
    nobs, df_resid, df_model:
        Sample size and degrees of freedom.  ``df_resid`` is the reference
        distribution's degrees of freedom (for clustered VCE it is ``G - 1``,
        not necessarily ``nobs - k``).
    dist:
        ``"t"`` for coefficient t tests / t confidence intervals, ``"z"`` for
        normal tests (logit, probit, Poisson, ...).
    model_name:
        Label shown in summaries (``"OLS"``, ``"Logit"``, ...).
    cov_type:
        Covariance estimator label recorded for display and auditing.
    e:
        Optional Stata ``e()`` scalars.  They are merged over the computed
        defaults ``N``, ``df_m`` and ``df_r`` (an explicit ``e()`` value wins).
    """

    def __init__(
        self,
        *,
        params: np.ndarray,
        param_names: list[str],
        cov: np.ndarray,
        resid: np.ndarray,
        fittedvalues: np.ndarray,
        nobs: int,
        df_resid: float,
        df_model: float,
        dist: str = "t",
        model_name: str = "",
        cov_type: str = "nonrobust",
        e: Mapping[str, Any] | None = None,
    ) -> None:
        names = list(param_names)
        params_arr = np.asarray(params, dtype=float).reshape(-1)
        cov_arr = np.asarray(cov, dtype=float)
        resid_arr = np.asarray(resid, dtype=float).reshape(-1)
        fitted_arr = np.asarray(fittedvalues, dtype=float).reshape(-1)

        if params_arr.size != len(names):
            raise ValueError(
                f"params has {params_arr.size} entries but param_names has {len(names)}"
            )
        if len(set(names)) != len(names):
            raise ValueError("param_names must be unique")
        if cov_arr.shape != (len(names), len(names)):
            raise ValueError(
                f"cov must have shape ({len(names)}, {len(names)}), got {cov_arr.shape}"
            )

        nobs_int = int(nobs)
        if nobs_int != nobs or nobs_int < 1:
            raise ValueError(f"nobs must be a positive integer, got {nobs!r}")
        if resid_arr.size != nobs_int:
            raise ValueError(
                f"resid has {resid_arr.size} entries but nobs is {nobs_int}"
            )
        if fitted_arr.size != nobs_int:
            raise ValueError(
                f"fittedvalues has {fitted_arr.size} entries but nobs is {nobs_int}"
            )

        dist_value = str(dist).strip().lower()
        if dist_value not in _VALID_DISTRIBUTIONS:
            raise ValueError(
                f"dist must be one of {_VALID_DISTRIBUTIONS}, got {dist!r}"
            )

        df_resid_value = float(df_resid)
        df_model_value = float(df_model)

        self._param_names = names
        self._nobs = nobs_int
        self._df_resid = df_resid_value
        self._df_model = df_model_value
        self._dist = dist_value
        self._model_name = str(model_name)
        self._cov_type = str(cov_type)
        self._cov = cov_arr
        self._resid = resid_arr
        self._fittedvalues = fitted_arr

        merged_e: dict[str, Any] = {
            "N": nobs_int,
            "df_m": df_model_value,
            "df_r": df_resid_value,
        }
        if e is not None:
            merged_e.update(dict(e))
        self._e = EReturn(merged_e)

        index = pd.Index(names, dtype=object)
        self._params = pd.Series(params_arr, index=index, name="params")
        self._bse = pd.Series(np.sqrt(np.diag(cov_arr)), index=index, name="bse")
        with np.errstate(divide="ignore", invalid="ignore"):
            tvalues_arr = params_arr / self._bse.to_numpy()
        self._tvalues = pd.Series(tvalues_arr, index=index, name="tvalues")

        if dist_value == "t":
            pvalues_arr = 2.0 * stats.t.sf(np.abs(tvalues_arr), df_resid_value)
        else:
            pvalues_arr = 2.0 * stats.norm.sf(np.abs(tvalues_arr))
        self._pvalues = pd.Series(pvalues_arr, index=index, name="pvalues")

    # ------------------------------------------------------------------
    # Coefficient surface
    # ------------------------------------------------------------------
    @property
    def param_names(self) -> list[str]:
        """Variable names in coefficient order."""

        return list(self._param_names)

    @property
    def params(self) -> pd.Series:
        return self._params

    @property
    def bse(self) -> pd.Series:
        return self._bse

    @property
    def tvalues(self) -> pd.Series:
        return self._tvalues

    @property
    def zvalues(self) -> pd.Series:
        """Normal statistic for :math:`H_0: \\beta_j = 0`.

        Stata reports this as ``z`` for models fitted by maximum likelihood.
        It is numerically identical to :attr:`tvalues` (``params / bse``); the
        alias exists so downstream code can branch on the inference
        distribution without computing it again.
        """

        return self._tvalues

    @property
    def pvalues(self) -> pd.Series:
        return self._pvalues

    def cov_params(self) -> np.ndarray:
        """Return the covariance matrix of the parameter estimates."""

        return self._cov

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Confidence intervals for the coefficients.

        Uses the Student-t reference distribution for ``dist="t"`` and the
        standard normal for ``dist="z"``.
        """

        alpha_value = float(alpha)
        if not 0.0 < alpha_value < 1.0:
            raise ValueError(f"alpha must lie in (0, 1), got {alpha!r}")

        if self._dist == "t":
            critical = float(stats.t.ppf(1.0 - alpha_value / 2.0, self._df_resid))
        else:
            critical = float(stats.norm.ppf(1.0 - alpha_value / 2.0))

        params_arr = self._params.to_numpy()
        bse_arr = self._bse.to_numpy()
        return pd.DataFrame(
            {
                "lower": params_arr - critical * bse_arr,
                "upper": params_arr + critical * bse_arr,
            },
            index=pd.Index(self._param_names, dtype=object),
        )

    # ------------------------------------------------------------------
    # Sample / model metadata
    # ------------------------------------------------------------------
    @property
    def nobs(self) -> int:
        return self._nobs

    @property
    def df_resid(self) -> float:
        return self._df_resid

    @property
    def df_model(self) -> float:
        return self._df_model

    @property
    def resid(self) -> np.ndarray:
        return self._resid

    @property
    def fittedvalues(self) -> np.ndarray:
        return self._fittedvalues

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    @property
    def dist(self) -> str:
        """Inference distribution: ``"t"`` or ``"z"``."""

        return self._dist

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def cov_type(self) -> str:
        return self._cov_type

    # ------------------------------------------------------------------
    # Stata e() scalars
    # ------------------------------------------------------------------
    @property
    def e(self) -> EReturn:
        """Stata ``e()`` scalars (item and attribute access)."""

        return self._e

    def _e_scalar(self, key: str) -> float:
        value = self._e.get(key)
        if value is None:
            return float("nan")
        return _as_float(value)

    @property
    def llf(self) -> float:
        """Log-likelihood from ``e(ll)``; NaN when not recorded."""

        return self._e_scalar("ll")

    @property
    def rsquared(self) -> float:
        """R-squared from ``e(r2)``; NaN when not recorded."""

        return self._e_scalar("r2")

    @property
    def rsquared_adj(self) -> float:
        """Adjusted R-squared from ``e(r2_a)``; NaN when not recorded."""

        return self._e_scalar("r2_a")

    @property
    def fvalue(self) -> float:
        """Model F statistic from ``e(F)``; NaN when not recorded."""

        return self._e_scalar("F")

    @property
    def aic(self) -> float:
        """AIC from ``e(aic)``; NaN when not recorded."""

        return self._e_scalar("aic")

    @property
    def bic(self) -> float:
        """BIC from ``e(bic)``; NaN when not recorded."""

        return self._e_scalar("bic")

    # ------------------------------------------------------------------
    # Presentation
    # ------------------------------------------------------------------
    def summary(self, style: str = "statsmodels") -> str:
        """Render a text summary.

        Parameters
        ----------
        style:
            ``"statsmodels"`` (default) or ``"stata"``.
        """

        return render_summary(self, style)

    def __repr__(self) -> str:
        name = self._model_name or "westata"
        return (
            f"<StataResults {name}: nobs={self._nobs}, "
            f"k={len(self._param_names)}, dist={self._dist}, "
            f"cov_type={self._cov_type}>"
        )
