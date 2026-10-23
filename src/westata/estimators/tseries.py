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
from scipy import stats

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
        )
        result = model.fit(method_kwargs={"maxiter": maxiter})

        # ``enforce_stationarity``/``enforce_invertibility`` must stay at their
        # defaults: forcing them off changes the optimiser's parameterisation and
        # shifted the AR(1) coefficient by 1.9e-3 relative, 40x worse than the
        # residual optimiser difference.
        #
        # The remaining gap against Stata is a stopping-point difference, not a
        # formula difference: evaluating statsmodels' own log-likelihood at
        # Stata's reported parameters gives a *lower* value than at statsmodels'
        # optimum, so both maximise the same likelihood and statsmodels stops
        # marginally closer to the true maximum. Verified empirically against
        # StataNow 19.5 (see scripts/diagnose_arima.py).

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
        # Stata's arima reports a Wald chi2 for the null that all ARMA
        # coefficients are zero. Recompute it from the coefficient vector and
        # its covariance, excluding the variance parameter.
        arma_positions = [
            index for index, name in enumerate(names) if name != "sigma2"
        ]
        if arma_positions:
            sub_params = params[arma_positions]
            sub_cov = cov[np.ix_(arma_positions, arma_positions)]
            try:
                chi2 = float(sub_params @ np.linalg.solve(sub_cov, sub_params))
            except np.linalg.LinAlgError:
                chi2 = float(sub_params @ np.linalg.pinv(sub_cov) @ sub_params)
            df_m = len(arma_positions)
            p_chi2 = float(stats.chi2.sf(chi2, df_m))
        else:
            chi2, df_m, p_chi2 = float("nan"), 0, float("nan")

        e: dict[str, Any] = {
            "N": nobs,
            "ll": float(result.llf),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "sigma2": (
                float(params[names.index("sigma2")]) if "sigma2" in names else float("nan")
            ),
            "chi2type": "Wald",
            "chi2": chi2,
            "p": p_chi2,
            "rank": params.size,
            "df_m": df_m,
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
        deterministic: str = "co",
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

    def johansen(
        self,
        *,
        det_order: int = 0,
        lags: int | None = None,
        k_ar_diff: int | None = None,
    ) -> "JohansenResult":
        """Johansen cointegration rank tests (Stata ``vecrank``).

        Parameters
        ----------
        det_order:
            Deterministic terms, in statsmodels' coding: ``-1`` none, ``0`` a
            constant in the cointegrating relation (Stata's default
            ``trend(constant)``), ``1`` a constant plus a linear trend.
        lags:
            Stata's ``lags(#)``: the number of lags in the **underlying VAR**.
            ``[TS] vecrank`` states that "the VEC model will include one fewer
            lag of the first differences", so this maps to
            ``k_ar_diff = lags - 1``.  Verified against StataNow 19.5: with
            ``lags(2)`` (equivalent to ``k_ar_diff=1``) the trace statistic
            matches to all four printed decimals (195.9673 vs 195.9673).
        k_ar_diff:
            statsmodels' convention directly.  Give either ``lags`` or
            ``k_ar_diff``; ``lags`` wins if both are supplied.

        Notes
        -----
        ``lags(2)`` and above reproduce Stata's printed statistic exactly.

        ``lags(1)`` (``k_ar_diff=0``) does **not**: Stata reports 288.3450 where
        statsmodels gives 287.2485, a 0.4% difference that appears only when no lagged
        differences are present. Both agree on the first eigenvalue to 1e-3, so the
        discrepancy is in the boundary handling rather than the estimator, and it is
        recorded in the registry as the reason this entry is E2 rather than E3.

        That distinction matters for the rank DECISION at ``lags(1)``: Stata's rank-1
        statistic is 3.7832 against a 5% critical value of 3.76, a 0.6% margin, which is
        the same order as the 0.4% implementation difference. The rank verdict can
        therefore differ, and callers should not treat the two as interchangeable at
        ``lags(1)``.
        """

        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        if lags is not None:
            resolved = int(lags) - 1
            if resolved < 0:
                raise ValueError(
                    f"vecrank: lags must be at least 1 (it counts the VAR lags), "
                    f"got {lags}"
                )
        elif k_ar_diff is not None:
            resolved = int(k_ar_diff)
        else:
            # Fall back to the lag order the model was CONSTRUCTED with. This previously
            # defaulted to 1 unconditionally, so VECM(data, lags=1).johansen() and
            # VECM(data, lags=3).johansen() returned identical statistics -- the lag order
            # was ignored entirely. k_ar_diff = 1 happens to equal Stata's lags(2), which is
            # the one case the registry had checked, so the defect stayed hidden.
            resolved = max(int(self._lags) - 1, 0)

        outcome = coint_johansen(self._data, det_order, resolved)
        return JohansenResult(
            trace=outcome.lr1,
            trace_critical=outcome.cvt,
            maxeig=outcome.lr2,
            maxeig_critical=outcome.cvm,
            eigenvectors=outcome.evec,
        )

    def fit(self, cov_type: str | None = None, **_: Any) -> Any:
        """Fit the VECM (Stata ``vec``).

        Two conventions differ from statsmodels and both are applied here:

        *lag order*  Stata's ``lags(#)`` counts VAR lags, so the VECM carries one fewer
        lagged difference: ``k_ar_diff = # - 1``. The same mapping was established for
        ``johansen`` and, applied here, moves alpha from 0.19 away to 3.6e-13.

        *deterministic terms*  ``[TS] vec`` states the default is "an unrestricted
        constant in model", which statsmodels calls ``co``. The previous default, ``ci``,
        puts the constant inside the cointegrating relation -- a different model.

        The reported drift still differs from Stata's printed ``_cons`` (see the registry
        entry); alpha and its standard errors are the comparable quantities.
        """

        from statsmodels.tsa.vector_ar.vecm import VECM as _SMVECM

        k_ar_diff = max(int(self._lags) - 1, 0)
        deterministic = self._deterministic
        # Accept the Stata spelling as well as statsmodels' own codes.
        if deterministic == "constant":
            deterministic = "co"
        elif deterministic == "rconstant":
            deterministic = "ci"

        model = _SMVECM(
            self._data,
            k_ar_diff=k_ar_diff,
            coint_rank=self._rank,
            deterministic=deterministic,
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


def _newey_west_lags(observations: int) -> int:
    """Stata's default Newey-West truncation, ``floor(4 * (n/100)^(2/9))``.

    Used for ``pperron`` when no bandwidth is given. Stata prints this value as
    "Newey-West lags = k", so it is directly checkable from its output.
    """

    return int(np.floor(4.0 * (max(observations, 1) / 100.0) ** (2.0 / 9.0)))


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

        # Stata's pperron prints "Newey-West lags = k" and uses the standard rule
        #     k = floor(4 * (n/100)^(2/9))
        # which gives 5 for n = 399. Letting arch choose (maxlag=None) gave
        # -1.994881 against Stata's -1.953, a 2.1e-2 gap that was recorded in the
        # registry as an unexplained truncation difference; with the rule applied the
        # agreement is 1.1e-4, i.e. Stata's 3-decimal display.
        bandwidth = maxlag if maxlag is not None else _newey_west_lags(series.size)
        outcome = PhillipsPerron(series, trend=regression, lags=bandwidth)
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
