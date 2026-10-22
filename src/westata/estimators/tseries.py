"""Time-series estimators targeting Stata's time-series commands.

========================  ==================
westata                   Stata
========================  ==================
``ARIMA``                 ``arima``
``VAR``                   ``var``
``VECM``                  ``vec``
``GARCH``                 ``arch``
``unitroot``              ``dfuller`` / ``pperron`` / ``kpss``
========================  ==================

Conformance notes
-----------------
* Stata's ``arima``/``var``/``vec`` report **z** statistics, so their results
  use ``dist="z"`` and print ``Wald chi2``.
* ``var``'s lag-order selection defaults to the same sequential
  likelihood-ratio/AIC/BIC/FPE/HQIC set that Stata prints; ``select_order``
  returns them all so the user can compare, instead of silently picking AIC.
* ``arch`` reports z statistics as well; ``arch`` the package estimates the
  same GARCH(p,q) model but its default optimiser and rescaling differ, so the
  convergence status is surfaced explicitly.
* ``unitroot`` returns the tests themselves (statistic and p-value) rather than
  a fitted model, mirroring ``dfuller``/``pperron``/``kpss``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError, ConvergenceError
from westata.results.base import StataResults

__all__ = [
    "ARIMA",
    "GARCH",
    "VAR",
    "VECM",
    "UnitRootResult",
    "unitroot",
]


def _as_series(endog: Any, name: str = "y") -> pd.Series:
    if isinstance(endog, pd.Series):
        return endog.astype(float)
    values = np.asarray(endog, dtype=float).reshape(-1)
    return pd.Series(values, name=name)


class ARIMA:
    """ARIMA / SARIMA model (Stata ``arima``).

    Parameters
    ----------
    endog:
        Univariate series.
    order:
        ``(p, d, q)``.
    seasonal_order:
        ``(P, D, Q, s)``; ``(0, 0, 0, 0)`` disables seasonality.
    trend:
        ``"c"`` for a constant, ``"n"`` for none, ``"t"`` for a linear trend,
        ``"ct"`` for both.  Stata's default has no constant, so ``"n"`` is the
        default here to match (pass ``"c"`` explicitly for ``arima ..., arima``
        with a constant).
    """

    _stata_command = "arima"

    def __init__(
        self,
        endog: Any,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        *,
        trend: str = "n",
    ) -> None:
        self._y = _as_series(endog)
        if np.isnan(self._y.to_numpy()).any():
            raise ConsistencyError(
                "arima: the series contains missing values; Stata's arima "
                "requires a complete series (use a time-series-aware gap "
                "treatment first)"
            )
        self._order = tuple(order)
        self._seasonal_order = tuple(seasonal_order)
        self._trend = trend

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        from statsmodels.tsa.arima.model import ARIMA as _SMARIMA

        requested = "nonrobust" if cov_type is None else str(cov_type).strip()
        if requested.lower() != "nonrobust":
            raise ValueError(
                f"arima: only cov_type='nonrobust' is implemented; got {cov_type!r}"
            )

        model = _SMARIMA(
            self._y.to_numpy(),
            order=self._order,
            seasonal_order=self._seasonal_order,
            trend=self._trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        # The state-space representation of a differenced model reports
        # parameters that are already in the reported (differenced) scale, so
        # ``trend="n"`` must not add a constant for d>0.
        result = model.fit(method_kwargs={"maxiter": maxiter})

        converged = bool(getattr(result, "mle_retvals", {}).get("converged", True))
        if not converged:
            message = (
                "arima: the optimiser did not converge; estimates may be "
                "unreliable"
            )
            if strict:
                raise ConvergenceError(message)
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=2)

        params = np.asarray(result.params, dtype=float)
        names = list(result.param_names)
        cov = np.asarray(result.cov_params(), dtype=float)
        nobs = int(result.nobs)

        e: dict[str, Any] = {
            "N": nobs,
            "ll": float(result.llf),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "sigma2": (
                float(params[names.index("sigma2")]) if "sigma2" in names else float("nan")
            ),
            "chi2type": "Wald",
            "chi2": float("nan"),
            "p": float("nan"),
            "rank": params.size,
            "df_m": max(params.size - 1, 0),
            "converged": int(converged),
            "order": self._order,
            "seasonal_order": self._seasonal_order,
            "trend": self._trend,
        }

        return StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=np.asarray(result.resid, dtype=float),
            fittedvalues=np.asarray(result.fittedvalues, dtype=float),
            nobs=nobs,
            df_resid=float(nobs),
            df_model=float(e["df_m"]),
            dist="z",
            model_name=self._stata_command,
            cov_type=requested,
            e=e,
        )


class VAR:
    """Vector autoregression (Stata ``var``).

    Parameters
    ----------
    endog:
        Multiple series, shaped ``(n_obs, n_variables)``.
    lags:
        Fixed lag order, or ``None`` to select by information criterion in
        :meth:`select_order`.
    trend:
        ``"c"`` (constant), ``"n"`` (none), ``"ct"``, ``"ctt"``.  Stata's
        default is a constant.
    """

    _stata_command = "var"

    def __init__(
        self,
        endog: Any,
        lags: int | None = 1,
        *,
        trend: str = "c",
        names: Sequence[str] | None = None,
    ) -> None:
        values = np.asarray(endog, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if np.isnan(values).any():
            raise ConsistencyError("var: the series contains missing values")
        self._data = values
        self._lags = lags
        self._trend = trend
        self._names = (
            list(names)
            if names is not None
            else [f"y{i + 1}" for i in range(values.shape[1])]
        )
        if len(self._names) != values.shape[1]:
            raise ValueError(
                f"names has {len(self._names)} entries but endog has "
                f"{values.shape[1]} columns"
            )

    def _make_model(self, lags: int) -> Any:
        from statsmodels.tsa.vector_ar.var_model import VAR as _SMVAR

        return _SMVAR(self._data).fit(lags, trend=self._trend, ic=None)

    def select_order(self, maxlags: int = 4) -> pd.DataFrame:
        """Report AIC/BIC/HQIC/FPE for each candidate lag order.

        Stata's ``varsoc`` prints the same table; returning all criteria keeps
        the choice explicit rather than letting the library pick one.
        """

        from statsmodels.tsa.vector_ar.var_model import VAR as _SMVAR

        model = _SMVAR(self._data)
        selection = model.select_order(maxlags=maxlags, trend=self._trend)
        # ``LagOrderResults`` stores the full criterion paths in ``ics``; the
        # ``aic``/``bic`` attributes are only the minimising lag.
        criteria = dict(getattr(selection, "ics", {}) or {})
        table = pd.DataFrame(
            {
                "AIC": criteria.get("aic", []),
                "BIC": criteria.get("bic", []),
                "HQIC": criteria.get("hqic", []),
                "FPE": criteria.get("fpe", []),
            }
        )
        table.index.name = "lag"
        return table

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
    ) -> "VARResults":
        if self._lags is None:
            raise ConsistencyError(
                "var: no lag order given; call select_order() and pass lags= "
                "explicitly (Stata requires you to choose, too)"
            )
        result = self._make_model(self._lags)
        return VARResults(result, self._names, self._lags, self._trend)


@dataclass
class VARResults:
    """Wrapper over a fitted VAR with Stata-flavoured accessors."""

    _result: Any
    names: list[str]
    lags: int
    trend: str

    @property
    def aic(self) -> float:
        return float(self._result.aic)

    @property
    def bic(self) -> float:
        return float(self._result.bic)

    @property
    def params(self) -> pd.DataFrame:
        """Coefficient table, indexed like Stata's ``var`` output.

        statsmodels returns a bare ``(1 + k*p, k)`` array whose row order is
        ``[const, L1.y1, L1.y2, ..., Lp.y1, Lp.y2]``; this rebuilds the
        ``L1.y1``-style index Stata prints.
        """

        values = np.asarray(self._result.params, dtype=float)
        rows: list[str] = []
        if self.trend.startswith("c"):
            rows.append("const")
        for lag in range(1, self.lags + 1):
            for name in self.names:
                rows.append(f"L{lag}.{name}")
        # Any leftover rows (trend terms) get generic labels rather than a
        # silent misalignment.
        while len(rows) < values.shape[0]:
            rows.append(f"trend{len(rows)}")
        return pd.DataFrame(values, index=rows[: values.shape[0]], columns=self.names)

    def granger_causality(self, caused: str, causing: Sequence[str]) -> Any:
        """Granger causality test (Stata ``vargranger``)."""

        return self._result.test_causality(caused, list(causing), kind="f")

    def irf(self, periods: int = 10) -> Any:
        """Impulse responses (Stata ``irf create``)."""

        return self._result.irf(periods)

    def fevd(self, periods: int = 10) -> Any:
        """Forecast-error variance decomposition (Stata ``fevd``)."""

        return self._result.fevd(periods)

    def forecast(self, steps: int = 1) -> np.ndarray:
        """Out-of-sample forecast (Stata ``fcast``)."""

        return self._result.forecast(self._result.y, steps=steps)


class VECM:
    """Vector error-correction model (Stata ``vec``).

    Parameters
    ----------
    coint_rank:
        Cointegrating rank, from :meth:`johansen` or ``vecrank``.
    """

    _stata_command = "vec"

    def __init__(
        self,
        endog: Any,
        *,
        coint_rank: int = 1,
        lags: int = 1,
        deterministic: str = "ci",
        names: Sequence[str] | None = None,
    ) -> None:
        values = np.asarray(endog, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        self._data = values
        self._rank = coint_rank
        self._lags = lags
        self._deterministic = deterministic
        self._names = (
            list(names)
            if names is not None
            else [f"y{i + 1}" for i in range(values.shape[1])]
        )

    def johansen(self, *, det_order: int = 0, k_ar_diff: int = 1) -> "JohansenResult":
        """Johansen cointegration rank tests (Stata ``vecrank``)."""

        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        outcome = coint_johansen(self._data, det_order, k_ar_diff)
        return JohansenResult(
            trace=outcome.lr1,
            trace_critical=outcome.cvt,
            maxeig=outcome.lr2,
            maxeig_critical=outcome.cvm,
            eigenvectors=outcome.evec,
        )

    def fit(self, cov_type: str | None = None, **_: Any) -> Any:
        from statsmodels.tsa.vector_ar.vecm import VECM as _SMVECM

        model = _SMVECM(
            self._data,
            k_ar_diff=self._lags,
            coint_rank=self._rank,
            deterministic=self._deterministic,
        )
        return model.fit()


@dataclass
class JohansenResult:
    """Johansen trace / max-eigenvalue statistics with critical values."""

    trace: np.ndarray
    trace_critical: np.ndarray
    maxeig: np.ndarray
    maxeig_critical: np.ndarray
    eigenvectors: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trace": self.trace,
                "trace_5pct": self.trace_critical[:, 1],
                "maxeig": self.maxeig,
                "maxeig_5pct": self.maxeig_critical[:, 1],
            }
        )


class GARCH:
    """GARCH / EGARCH volatility model (Stata ``arch``).

    Parameters
    ----------
    endog:
        Returns series.
    p, q:
        ARCH and GARCH orders.
    o:
        Asymmetry order (``o=1`` gives GJR-GARCH).
    vol:
        ``"GARCH"`` or ``"EGARCH"``.
    dist:
        Innovation distribution: ``"normal"`` or ``"t"``.
    """

    _stata_command = "arch"

    def __init__(
        self,
        endog: Any,
        *,
        p: int = 1,
        q: int = 1,
        o: int = 0,
        vol: str = "GARCH",
        dist: str = "normal",
        mean: str = "Constant",
    ) -> None:
        self._y = _as_series(endog)
        if np.isnan(self._y.to_numpy()).any():
            raise ConsistencyError("arch: the series contains missing values")
        if vol not in ("GARCH", "EGARCH"):
            raise ValueError(f"arch: vol must be 'GARCH' or 'EGARCH', got {vol!r}")
        if dist not in ("normal", "t"):
            raise ValueError(f"arch: dist must be 'normal' or 't', got {dist!r}")
        self._p, self._q, self._o = p, q, o
        self._vol = vol
        self._dist = dist
        self._mean = mean

    def fit(self, *, strict: bool = False, maxiter: int = 1000) -> StataResults:
        from arch import arch_model

        # Stata's arch works on the raw series; arch the package rescales by
        # default, which would change the reported omega. Disable that so the
        # parameters stay on the data's own scale.
        model = arch_model(
            self._y.to_numpy(),
            mean=self._mean,
            vol=self._vol,
            p=self._p,
            o=self._o,
            q=self._q,
            dist=self._dist,
            rescale=False,
        )
        result = model.fit(disp="off", options={"maxiter": maxiter})
        converged = bool(getattr(result, "convergence_flag", 1) == 0)
        if not converged:
            message = "arch: the optimiser did not converge; estimates may be unreliable"
            if strict:
                raise ConvergenceError(message)
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=2)

        params = np.asarray(result.params, dtype=float)
        names = list(result.params.index)
        # ``arch`` exposes the parameter covariance as ``param_cov``, not the
        # statsmodels-style ``cov_params()``.
        cov = np.asarray(result.param_cov, dtype=float)
        nobs = int(result.nobs)

        def _pick(prefix: str) -> tuple[float, ...]:
            # arch names the parameters ``alpha[1]``, ``beta[1]``, ``gamma[1]``.
            return tuple(
                float(params[index])
                for index, name in enumerate(names)
                if name.startswith(prefix + "[")
            )

        e: dict[str, Any] = {
            "N": nobs,
            "ll": float(result.loglikelihood),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "chi2type": "Wald",
            "rank": params.size,
            "df_m": max(params.size - 1, 0),
            "converged": int(converged),
            "alpha": _pick("alpha"),
            "beta": _pick("beta"),
            "gamma": _pick("gamma"),
            "omega": (
                float(params[names.index("omega")]) if "omega" in names else float("nan")
            ),
        }

        return StataResults(
            params=params,
            param_names=names,
            cov=cov,
            resid=np.asarray(result.resid, dtype=float),
            fittedvalues=np.asarray(result.conditional_volatility, dtype=float),
            nobs=nobs,
            df_resid=float(nobs),
            df_model=float(e["df_m"]),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
        )


@dataclass
class UnitRootResult:
    """One unit-root test outcome, shaped like Stata's test output."""

    test: str
    statistic: float
    pvalue: float
    lags: int = 0
    observations: int = 0
    critical: Mapping[str, float] = field(default_factory=dict)
    null: str = ""

    def __str__(self) -> str:
        critical = "  ".join(
            f"{key}: {value:.3f}" for key, value in self.critical.items()
        )
        return (
            f"{self.test}\n"
            f"  statistic = {self.statistic:.6g}   p-value = {self.pvalue:.4f}\n"
            f"  lags = {self.lags}   obs = {self.observations}\n"
            f"  critical: {critical}\n"
            f"  H0: {self.null}"
        )


def unitroot(
    endog: Any,
    *,
    test: str = "adf",
    regression: str = "c",
    maxlag: int | None = None,
    autolag: str = "AIC",
) -> UnitRootResult:
    """Unit-root test (Stata ``dfuller`` / ``pperron`` / ``kpss``).

    Parameters
    ----------
    test:
        ``"adf"`` (``dfuller``), ``"pp"`` (``pperron``) or ``"kpss"``.
    regression:
        Deterministic terms: ``"c"``, ``"n"``, ``"ct"``.
    """

    series = _as_series(endog).dropna().to_numpy()

    if test == "adf":
        from statsmodels.tsa.stattools import adfuller

        # ``result_object=True`` is the forward-compatible API; the tuple form
        # is deprecated in statsmodels 0.15 and will change layout.
        outcome = adfuller(
            series,
            maxlag=maxlag,
            regression=regression,
            autolag=autolag,
            result_object=True,
        )
        return UnitRootResult(
            test="Augmented Dickey-Fuller",
            statistic=float(outcome.statistic),
            pvalue=float(outcome.pvalue),
            lags=int(outcome.lags),
            observations=int(outcome.nobs),
            critical={key: float(value) for key, value in outcome.critical_values.items()},
            null="unit root (nonstationary)",
        )

    if test == "pp":
        from arch.unitroot import PhillipsPerron

        outcome = PhillipsPerron(series, trend=regression, lags=maxlag)
        return UnitRootResult(
            test="Phillips-Perron",
            statistic=float(outcome.stat),
            pvalue=float(outcome.pvalue),
            lags=int(outcome.lags),
            observations=int(outcome.nobs),
            critical={
                key: float(value)
                for key, value in outcome.critical_values.items()
            },
            null="unit root (nonstationary)",
        )

    if test == "kpss":
        from statsmodels.tsa.stattools import kpss

        outcome = kpss(
            series,
            regression=regression,
            nlags=maxlag or "auto",
            result_object=True,
        )
        return UnitRootResult(
            test="KPSS",
            statistic=float(outcome.statistic),
            pvalue=float(outcome.pvalue),
            lags=int(outcome.lags),
            observations=int(series.size),
            critical={key: float(value) for key, value in outcome.critical_values.items()},
            null="stationary",
        )

    raise ValueError(f"unitroot: unknown test {test!r}; use 'adf', 'pp' or 'kpss'")
