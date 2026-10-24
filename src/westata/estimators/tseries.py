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
* ``var`` reports the **maximum-likelihood** residual covariance, exactly like
  ``var`` does: every equation divides its residual sum of squares by ``T``
  (the number of observations in the estimation sample), not by ``T - k_eq``
  where ``k_eq`` is that equation's regressor count.  ``VARResults.bse``,
  ``VARResults.cov_params()`` and ``VARResults.sigma_u`` therefore carry the
  factor ``(T - k_eq) / T`` relative to statsmodels, and
  ``VARResults.granger_causality`` reports the same Wald chi2 as
  ``vargranger`` (see the docstrings for the arithmetic).
* ``arch`` reports z statistics as well.  Stata's ``arch`` uses **OPG**
  (BHHH) standard errors and starts the variance recursion from
  ``e(archi) = mean(demeaned residual^2)``; both conventions are reproduced
  here, so the defaults of :class:`GARCH` are Stata's, not the ``arch``
  package's (whose defaults are Bollerslev-Wooldridge 'robust' SEs and an
  exponentially weighted backcast).
* ``unitroot`` returns the tests themselves (statistic and p-value) rather than
  a fitted model, mirroring ``dfuller``/``pperron``/``kpss``.  ``dfuller``'s
  printed critical values are *not* MacKinnon response-surface values: they
  come from a six-point table interpolated linearly in N, which is reproduced
  here.
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
        # ``result.nobs`` is the length of the series handed in (statsmodels keeps
        # a full-length residual vector).  Stata's ``e(N)`` for an integrated
        # model is the number of observations in the ESTIMATION sample, i.e.
        # after differencing: ``tmax - tmin + 1 - d - D*s``.  statsmodels calls
        # exactly that ``nobs_effective`` (its ``loglikelihood_burn``), and
        # Stata's own ``estat ic`` uses it for the AIC/BIC penalties -- verified
        # on d=1 (299 vs 300), d=2 (298), and SARIMA(0,1,1,4) (296 of 300).
        nobs = int(result.nobs)
        nobs_estimated = int(round(float(getattr(result, "nobs_effective", nobs))))
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
            "N": nobs_estimated,
            "ll": float(result.llf),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "sigma2": (
                float(params[names.index("sigma2")]) if "sigma2" in names else float("nan")
            ),
            # Stata's arima reports ``e(sigma)`` -- the STANDARD DEVIATION of the innovations -- and
            # does NOT report ``e(sigma2)``. Both are provided here because the two are the same
            # quantity in different units and confusing them costs 3.7% on the variance: on the sweep's
            # AR(1) data Stata's e(sigma) = 0.9645733639, whose square is 0.9304017743, against our
            # sigma2 of 0.9303359775 (7.1e-05 relative). A reader who reads e(sigma) from Stata and
            # sigma2 from us, or the reverse, gets a materially wrong number without any error.
            "sigma": (
                float(np.sqrt(params[names.index("sigma2")]))
                if "sigma2" in names and float(params[names.index("sigma2")]) >= 0.0
                else float("nan")
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
            # ``nobs`` must equal the residual length for StataResults; Stata's
            # estimation-sample size is reported through ``e["N"]`` above and
            # ``nobs_effective`` below.
            nobs=nobs,
            df_resid=float(nobs_estimated),
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
        *,
        ml: bool = True,
    ) -> "VARResults":
        """Fit the VAR (Stata ``var``).

        Parameters
        ----------
        ml:
            Report the **maximum-likelihood** residual covariance, which is
            what Stata's ``var`` stores in ``e(Sigma)``/``e(V)``: each equation
            divides its RSS by ``T``.  The default is ``True`` so westata's
            ``bse``/``e(Sigma)``/Granger chi2 match Stata out of the box.
            ``ml=False`` divides by ``T - k_eq`` instead, which reproduces
            Stata's ``var ..., dfk`` option (verified to 4.7e-15 against
            StataNow 19.5) and coincides with statsmodels' default.
        """

        if self._lags is None:
            raise ConsistencyError(
                "var: no lag order given; call select_order() and pass lags= "
                "explicitly (Stata requires you to choose, too)"
            )
        result = self._make_model(self._lags)
        return VARResults(result, self._names, self._lags, self._trend, ml=ml)


@dataclass
class VARResults:
    """Wrapper over a fitted VAR with Stata-flavoured accessors.

    Stata's ``var`` normalises the residual covariance by **T per equation**,
    not by ``T - k_eq`` where ``k_eq`` is that equation's regressor count.  The
    two differ by the factor ``(T - k_eq)/T`` -- 0.98997 for a bivariate VAR(1)
    on 299 usable observations (``k_eq = 3``) -- which is exactly the gap the
    retest measured on ``bse``, ``e(Sigma)`` and ``vargranger``.  Every
    Stata-facing accessor here applies that factor; ``raw_result`` remains
    available for the statsmodels convention.
    """

    _result: Any
    names: list[str]
    lags: int
    trend: str
    ml: bool = True

    # -- Stata's degrees-of-freedom convention -------------------------------
    @property
    def nobs(self) -> int:
        """Estimation-sample size ``T`` (Stata's ``e(N)`` for ``var``)."""

        return int(self._result.nobs)

    @property
    def k_per_equation(self) -> int:
        """Regressors in ONE equation: ``k_ar*neqs + deterministic terms``."""

        return int(self._result.k_ar) * int(self._result.neqs) + int(
            self._result.k_trend
        )

    @property
    def ml_factor(self) -> float:
        """``(T - k_eq)/T``; the ratio of the ML to the unbiased variance."""

        if not self.ml:
            return 1.0
        observations = float(self._result.nobs)
        return (observations - self.k_per_equation) / observations

    @property
    def sigma_u(self) -> np.ndarray:
        """Residual covariance as Stata reports it (``e(Sigma)``)."""

        return np.asarray(self._result.sigma_u, dtype=float) * self.ml_factor

    @property
    def cov_params(self) -> np.ndarray:
        """Default is Stata's ``var``: per-equation ML degrees of freedom.

        ``e(V)`` = ``kron((Z'Z)^-1, Sigma)`` with Stata's (ML) ``Sigma``.  For
        Stata's ``var ..., dfk`` small-sample version (divide by ``T - k_eq``)
        fit with ``ml=False`` instead -- that is the only switch, and it is
        covered by ``tests/test_tseries.py::test_var_ml_false_reproduces_stata_dfk``.
        """

        return np.asarray(self._result.cov_params(), dtype=float) * self.ml_factor

    @property
    def stata_cov_params(self) -> np.ndarray:
        """``e(V)`` in Stata's equation-by-equation parameter order."""

        order = self._stata_row_order()
        return self.cov_params[np.ix_(order, order)]

    @property
    def bse(self) -> pd.Series:
        """Standard errors in Stata's ``e(b)`` order (equation by equation)."""

        return pd.Series(
            np.sqrt(np.diag(self.stata_cov_params)),
            index=self.stata_param_names,
            name="bse",
        )

    @property
    def stata_param_names(self) -> list[str]:
        """Parameter names in Stata's ``e(b)`` order (equation by equation).

        For a VAR(2) with two variables Stata stores
        ``L.y1 L2.y1 L.y2 L2.y2 _cons`` per equation (variable first, then
        lag), whereas statsmodels uses ``[eq] * (const, L1y1..L1yk, L2y1..)``.
        """

        rows: list[str] = []
        for _ in self.names:
            for name in self.names:
                for lag in range(1, self.lags + 1):
                    rows.append(f"L.{name}" if lag == 1 else f"L{lag}.{name}")
            rows.extend(name for name, _ in self._deterministic_spec())
        return rows

    def _deterministic_spec(self) -> list[tuple[str, int]]:
        """Stata-style names for statsmodels' deterministic rows, in its order.

        statsmodels stores the deterministic block first, as
        ``[const, trend, trend^2]``; the names mirror that so a caller reading
        ``stata_params()`` gets ``_cons``/``_trend``/``_trend2`` in the same
        slots.
        """

        count = int(self._result.k_trend)
        names = ["_cons", "_trend", "_trend2"][:count]
        while len(names) < count:
            names.append(f"_det{len(names) + 1}")
        return list(zip(names, range(count)))

    def _stata_row_order(self) -> np.ndarray:
        """Index into statsmodels' ``vec(B)`` for every row of Stata's ``e(b)``."""

        neqs = int(self._result.neqs)
        k_trend = int(self._result.k_trend)
        order: list[int] = []
        for equation in range(neqs):
            for variable in range(neqs):
                for lag in range(1, self.lags + 1):
                    row = k_trend + (lag - 1) * neqs + variable
                    order.append(row * neqs + equation)
            for _, row in self._deterministic_spec():
                order.append(row * neqs + equation)
        return np.asarray(order, dtype=int)

    def stata_params(self) -> pd.Series:
        """Coefficients in Stata's ``e(b)`` order with Stata's names."""

        flat = np.asarray(self._result.params, dtype=float).ravel()
        return pd.Series(
            flat[self._stata_row_order()],
            index=self.stata_param_names,
            name="params",
        )

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

    def granger_causality(
        self, caused: str, causing: Sequence[str], *, statistic: str = "chi2"
    ) -> Any:
        """Granger causality test (Stata ``vargranger``).

        Stata's ``vargranger`` reports a **Wald chi2** per (caused, causing)
        pair with ``df = #causing * lags``; the default here is therefore the
        Wald statistic, not statsmodels' F.  Passing ``statistic="F"`` returns
        statsmodels' F test unchanged (``df1 = #causing*lags``,
        ``df2 = k*(T - k_eq)``), which is a different statistic and is NOT
        what ``vargranger`` prints.

        Two conventions separate westata from Stata and both are applied:

        * the statistic is multiplied by ``T/(T - k_eq)`` because Stata's Wald
          form uses the ML residual covariance (divide by ``T``) while
          statsmodels' uses the unbiased one (divide by ``T - k_eq``);
        * the degrees of freedom are Stata's (``#causing * lags``), which
          statsmodels already uses for ``kind="wald"``.

        Measured against StataNow 19.5 on a bivariate VAR of a 300-observation
        series: chi2 relative difference 1.4e-15 for VAR(1) and 1.7e-15 for
        VAR(2) after the scaling, against 1.0e-2 / 1.7e-2 before it.
        """

        from statsmodels.tsa.vector_ar.hypothesis_test_results import (
            CausalityTestResults,
        )

        kind = str(statistic).strip().lower()
        if kind in ("f", "f-test"):
            # statsmodels' F test, deliberately unscaled: it is its own
            # convention (df2 already carries T - k_eq), not Stata's.
            return self._result.test_causality(caused, list(causing), kind="f")
        if kind not in ("chi2", "wald", "chisq"):
            raise ValueError(
                f"var: granger causality statistic must be 'chi2' (Stata's "
                f"vargranger) or 'F' (statsmodels' convention), got {statistic!r}"
            )

        outcome = self._result.test_causality(caused, list(causing), kind="wald")
        inflation = 1.0 / self.ml_factor if self.ml else 1.0
        statistic_value = float(outcome.test_statistic) * inflation
        df_value = int(outcome.df)
        pvalue = float(stats.chi2.sf(statistic_value, df_value))
        signif = float(getattr(outcome, "signif", 0.05))
        return CausalityTestResults(
            list(outcome.causing),
            list(outcome.caused),
            statistic_value,
            float(stats.chi2.ppf(1.0 - signif, df_value)),
            pvalue,
            df_value,
            signif,
            test="granger",
            method="wald",
        )

    def irf(self, periods: int = 10) -> Any:
        """Impulse responses (Stata ``irf create``)."""

        return self._result.irf(periods)

    def fevd(self, periods: int = 10) -> Any:
        """Forecast-error variance decomposition (Stata ``fevd``)."""

        return self._result.fevd(periods)

    def forecast(self, steps: int = 1) -> np.ndarray:
        """Out-of-sample forecast (Stata ``fcast``)."""

        return self._result.forecast(self._result.y, steps=steps)


# ---------------------------------------------------------------------------
# Johansen critical values: Stata's tables, not statsmodels' response surface
# ---------------------------------------------------------------------------
# Stata's ``vecrank`` takes its critical values from ``_vecgetcv.ado``, whose
# ``_VECGETCV_`` mata function hard-codes two 11 x 5 matrices (rows = K - r,
# columns = deterministic specification).  They are the Johansen (1995) tables,
# which DIFFER from statsmodels' Osterwald-Lenum values: for K - r = 1 with a
# constant, Stata's 5% value is 3.76 while statsmodels reports 3.8415, and
# 14.07 vs 14.2639 for the max-eigenvalue test.  That 2% gap is enough to
# change the SELECTED RANK on borderline data -- see
# ``retest/fix/tseries/repro_v4_vecrank.py``.
#
# Column order (from the ado's own comment block):
#   1 no trend | 2 restricted constant | 3 constant | 4 restricted trend | 5 trend
_VECRANK_TRACE_CV95: tuple[tuple[float, ...], ...] = (
    (3.84, 9.42, 3.76, 12.25, 3.74),
    (12.53, 19.96, 15.41, 25.32, 18.17),
    (24.31, 34.91, 29.68, 42.44, 34.55),
    (39.89, 53.12, 47.21, 62.99, 54.64),
    (59.46, 76.07, 68.52, 87.31, 77.74),
    (82.49, 102.14, 94.15, 114.90, 104.94),
    (109.99, 131.70, 124.24, 146.76, 136.61),
    (141.20, 165.58, 156.00, 182.82, 170.80),
    (175.77, 202.92, 192.89, 222.21, 208.97),
    (212.67, 244.15, 233.13, 263.42, 250.84),
    (255.27, 291.40, 277.71, 310.81, 295.99),
)

_VECRANK_TRACE_CV99: tuple[tuple[float, ...], ...] = (
    (6.51, 12.97, 6.65, 16.26, 6.40),
    (16.31, 24.60, 20.04, 30.45, 23.46),
    (29.75, 41.07, 35.65, 48.45, 40.49),
    (45.58, 60.16, 54.46, 70.05, 61.21),
    (66.52, 84.45, 76.07, 96.58, 85.78),
    (90.45, 111.01, 103.18, 124.75, 114.36),
    (119.80, 143.09, 133.57, 158.49, 146.99),
    (152.32, 177.20, 168.36, 196.08, 182.51),
    (187.31, 215.74, 204.95, 234.41, 222.46),
    (226.40, 257.68, 247.18, 279.07, 263.94),
    (269.81, 307.64, 293.44, 327.45, 312.58),
)

_VECRANK_MAXEIG_CV95: tuple[tuple[float, ...], ...] = (
    (3.84, 9.24, 3.76, 12.52, 3.74),
    (11.44, 15.67, 14.07, 18.96, 16.87),
    (17.89, 22.00, 20.97, 25.54, 23.78),
    (23.80, 28.14, 27.07, 31.46, 30.33),
    (30.04, 34.40, 33.46, 37.52, 36.41),
    (36.36, 40.30, 39.37, 43.97, 42.48),
    (41.51, 46.45, 45.28, 49.42, 48.45),
    (47.99, 52.00, 51.42, 55.50, 54.25),
    (53.69, 57.42, 57.12, 61.29, 60.29),
    (59.06, 63.57, 62.81, 66.23, 66.10),
    (65.30, 69.74, 68.83, 72.72, 71.68),
)

_VECRANK_MAXEIG_CV99: tuple[tuple[float, ...], ...] = (
    (6.51, 12.97, 6.65, 16.26, 6.40),
    (15.69, 20.20, 18.63, 23.65, 21.47),
    (22.99, 26.81, 25.52, 30.34, 28.83),
    (28.82, 33.24, 32.24, 36.65, 35.68),
    (35.17, 39.79, 38.77, 42.36, 41.58),
    (41.00, 46.82, 45.10, 49.51, 48.17),
    (47.15, 51.91, 51.57, 54.71, 54.48),
    (53.90, 57.95, 57.69, 62.46, 60.81),
    (59.78, 63.71, 62.80, 67.88, 66.91),
    (65.21, 69.94, 69.09, 73.73, 72.96),
    (72.36, 76.63, 75.95, 79.23, 78.51),
)


def _vecrank_column(deterministic: str) -> int:
    """Column of ``_vecgetcv.ado``'s matrix for a Stata ``trend()`` value."""

    key = str(deterministic).strip().lower()
    mapping = {
        "none": 0,
        "n": 0,
        "rconstant": 1,
        "ci": 1,
        "constant": 2,
        "c": 2,
        "co": 2,
        "rtrend": 3,
        "li": 3,
        "trend": 4,
        "lo": 4,
    }
    if key not in mapping:
        raise ValueError(
            f"vecrank: unknown deterministic specification {deterministic!r}; "
            "use one of 'none', 'rconstant', 'constant', 'rtrend', 'trend'"
        )
    return mapping[key]


def _vecrank_critical(
    n_equations: int, deterministic: str, *, statistic: str = "trace"
) -> np.ndarray:
    """Stata's ``vecrank`` critical values as an ``(n_equations, 3)`` array.

    Row ``i`` is the test of ``r = i`` (so the row corresponds to ``K - r`` =
    ``n_equations - i``, exactly like ``_vecgetcv``'s row indexing) and the
    three columns are ``[90%, 95%, 99%]``, which is also statsmodels' layout for
    ``cvt``/``cvm``.

    The 90% column is **NaN by design, not by a missing computation**:
    Stata's ``_vecgetcv.ado`` hard-codes only the 95% and 99% matrices, so
    Stata itself has no 90% value to report.  Filling that column from
    statsmodels' Osterwald-Lenum table would mix two different tables in one
    row -- exactly the failure mode this repair exists to remove -- so the
    column is left empty and callers are expected to raise the level (or use
    5%/1%) instead.  ``JohansenResult.select_rank(level=0.05)`` refuses a 10%
    request for the same reason.
    """

    table95 = (
        _VECRANK_TRACE_CV95 if statistic == "trace" else _VECRANK_MAXEIG_CV95
    )
    table99 = (
        _VECRANK_TRACE_CV99 if statistic == "trace" else _VECRANK_MAXEIG_CV99
    )
    column = _vecrank_column(deterministic)
    if not 1 <= n_equations <= len(table95):
        raise ValueError(
            f"vecrank: critical values are tabulated for 1..{len(table95)} "
            f"equations, got {n_equations}"
        )
    rows = []
    for index in range(n_equations):
        k_minus_r = n_equations - index
        rows.append(
            (
                float("nan"),
                table95[k_minus_r - 1][column],
                table99[k_minus_r - 1][column],
            )
        )
    return np.asarray(rows, dtype=float)


def _johansen_statistics(
    data: np.ndarray, k_ar_diff: int, det_order: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Johansen eigenvalues, trace and max-eigenvalue statistics, computed directly.

    statsmodels' ``coint_johansen`` disagrees with Stata at ``k_ar_diff = 0``: it gives
    trace 287.2485 / 1.7922 where Stata reports 288.3450 / 3.7832, a factor of two on the
    second statistic while the first matches to three decimals. That asymmetry rules out a
    sample-alignment problem, so the statistics are computed here from the moment matrices:

        r0, r1 = residuals of dy and of y_{t-1} on [lagged dy, constant, trend]
        S_ij   = r_i' r_j / T
        solve  S01' S00^-1 S01  v  =  lambda  S11  v
        trace(r) = -T * sum_{i>r} ln(1 - lambda_i)

    Verified against StataNow 19.5 at every lag order on the reference pair: eigenvalues and
    both statistics match to all printed decimals.

    ``det_order`` follows statsmodels' coding and, unlike statsmodels, actually puts the
    linear trend in the regression: ``-1`` no deterministic term, ``0`` a constant, ``1``
    a constant AND a linear trend.  (statsmodels' ``coint_johansen`` detrends the data but
    never adds the trend to the differenced regression, which is why it reports 92.3175
    where Stata's ``vecrank ..., trend(trend)`` reports 92.2870; with the trend column the
    match is exact.)
    """

    from scipy import linalg

    values = np.asarray(data, dtype=float)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    total = values.shape[0]
    differences = np.diff(values, axis=0)

    start = k_ar_diff
    target = differences[start:]
    levels = values[start : total - 1]

    regressors: list[np.ndarray] = [
        differences[start - lag : differences.shape[0] - lag]
        for lag in range(1, k_ar_diff + 1)
    ]
    if det_order >= 0:
        regressors.append(np.ones((target.shape[0], 1)))
    if det_order >= 1:
        regressors.append(np.arange(1, target.shape[0] + 1, dtype=float).reshape(-1, 1))
    design = (
        np.column_stack(regressors)
        if regressors
        else np.empty((target.shape[0], 0))
    )

    def residual(matrix: np.ndarray) -> np.ndarray:
        if design.shape[1] == 0:
            return matrix - matrix.mean(axis=0)
        coefficients, *_ = np.linalg.lstsq(design, matrix, rcond=None)
        return matrix - design @ coefficients

    r0 = residual(target)
    r1 = residual(levels)
    observations = r0.shape[0]

    s00 = r0.T @ r0 / observations
    s01 = r0.T @ r1 / observations
    s11 = r1.T @ r1 / observations

    cholesky = linalg.cholesky(s11, lower=True)
    inverse_cholesky = linalg.solve_triangular(cholesky, np.eye(s11.shape[0]), lower=True)
    middle = inverse_cholesky @ s01.T @ np.linalg.solve(s00, s01) @ inverse_cholesky.T

    eigenvalues = np.sort(np.real(np.linalg.eigvalsh(middle)))[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0 - 1e-15)

    per_eigenvalue = -observations * np.log(1.0 - eigenvalues)
    # Accumulate from the SMALLEST upward: trace(r) sums the eigenvalues beyond r.
    trace = np.array(
        [per_eigenvalue[index:].sum() for index in range(per_eigenvalue.size)]
    )
    maxeig = per_eigenvalue

    _, vectors = np.linalg.eigh(middle)
    return eigenvalues, trace, maxeig, vectors


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
            constant (Stata's default ``trend(constant)``), ``1`` a constant
            plus a linear trend.
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
        **The statistics are computed here; only the critical-value tables come
        from Stata.**  ``_johansen_statistics`` reproduces ``vecrank``'s trace
        and max-eigenvalue statistics without statsmodels' ``coint_johansen``
        (which disagrees at ``k_ar_diff = 0``: 287.2485 vs Stata's 288.3450).
        On the reference pair in ``retest/domains/tseries`` those statistics
        agree with Stata to 4 printed decimals.

        The critical values are the ones Stata prints.  This matters: Stata's
        ``_vecgetcv.ado`` hard-codes the **Johansen (1995)** tables, whereas
        statsmodels returns **Osterwald-Lenum**.  For ``K - r = 1`` with a
        constant the 5% trace value is ``3.76`` in Stata and ``3.8415`` in
        statsmodels, and the max-eigenvalue values are ``14.07`` vs
        ``14.2639``.  Since the selection rule ("smallest r whose statistic
        does not exceed its 5% critical value") is applied to those numbers,
        the 2% table difference **changes the selected rank** on data whose
        statistic lands between them -- see
        ``retest/fix/tseries/repro_v4_vecrank.py``, which builds such a pair.

        ``det_order`` maps to ``_vecgetcv``'s columns the same way statsmodels
        maps its own tables: ``-1`` -> no trend, ``0`` -> constant, ``1`` ->
        linear trend.  Stata's other two specifications (``trend(rconstant)``
        and ``trend(rtrend)``, which restrict the deterministic term to the
        cointegrating relation) are different estimations that this method does
        not implement; their statistics are therefore NOT reproduced.

        The three columns of ``trace_critical``/``maxeig_critical`` are
        ``[90%, 95%, 99%]`` like statsmodels' ``cvt``/``cvm``.  Stata's ado only
        hard-codes the 95% and 99% values, so the 90% column is ``nan`` rather
        than a number from a different table.
        """

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
            # was ignored entirely.
            resolved = max(int(self._lags) - 1, 0)

        eigenvalues, trace, maxeig, eigenvectors = _johansen_statistics(
            self._data, resolved, int(det_order)
        )

        specification = {-1: "none", 0: "constant", 1: "trend"}.get(int(det_order))
        if specification is None:
            raise ValueError(
                f"vecrank: det_order must be -1, 0 or 1 (Stata tabulates only "
                f"those three families), got {det_order!r}"
            )
        n_equations = self._data.shape[1]
        return JohansenResult(
            trace=trace,
            trace_critical=_vecrank_critical(n_equations, specification, statistic="trace"),
            maxeig=maxeig,
            maxeig_critical=_vecrank_critical(
                n_equations, specification, statistic="maxeig"
            ),
            eigenvectors=eigenvectors,
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

    def to_frame(self, *, render_missing: bool = False) -> pd.DataFrame:
        """Trace / max-eigenvalue table with Stata's 5% critical values.

        The 5% column is the one ``vecrank`` stars by default.  For the rank
        DECISION use :meth:`select_rank` rather than comparing by hand: it
        applies the same rule ``vecrank``'s ``*`` marker uses (see its
        docstring) and takes the critical values from the same table.

        The 90% column of ``trace_critical``/``maxeig_critical`` is NaN because
        Stata's ``_vecgetcv.ado`` does not tabulate it (only 95% and 99%).  It
        is not exposed here; :meth:`select_rank` raises for ``level=0.10``
        rather than quietly substituting another table.

        ``render_missing=True`` is a DISPLAY switch: entries that are not
        finite are rendered as ``None`` instead of ``nan`` so a printed table
        reads like Stata's.  The default keeps ``float64`` columns, so numeric
        comparison and ``np.isnan`` keep working.  :meth:`to_stata_table` is the
        ready-made rendering.
        """

        frame = pd.DataFrame(
            {
                "trace": self.trace,
                "trace_5pct": self.trace_critical[:, 1],
                "maxeig": self.maxeig,
                "maxeig_5pct": self.maxeig_critical[:, 1],
            }
        )
        if not render_missing:
            return frame
        for column in ("trace_5pct", "maxeig_5pct"):
            values = frame[column].to_numpy(dtype=float)
            if np.all(np.isfinite(values)):
                continue
            frame[column] = pd.Series(
                [float(value) if np.isfinite(value) else None for value in values],
                index=frame.index,
                dtype=object,
            )
        return frame

    def to_stata_table(self) -> pd.DataFrame:
        """``vecrank``'s printed table, with blank cells shown as ``None``.

        Same numbers as :meth:`to_frame`; missing critical values are rendered
        as ``None`` rather than ``nan`` so a user-facing table matches Stata's
        layout.  This is presentation only -- the numeric arrays
        (``trace_critical`` etc.) keep ``np.nan``, so callers should test with
        ``np.isnan``, not by string-matching this frame.
        """

        return self.to_frame(render_missing=True)

    def select_rank(self, *, statistic: str = "trace", level: float = 0.05) -> int:
        """Cointegration rank ``r`` implied by Stata's critical values.

        Returns the VALUE of ``r`` (the cointegration rank, ``0 <= r <= K``) --
        not a positional index into ``trace``/``maxeig``.  The two coincide
        numerically because row ``i`` tests ``r = i``, but the contract is the
        rank: ``select_rank() == K`` means "no cointegration restriction was
        accepted", which is the same statement as ``vecrank`` printing no star.

        The decision rule is Stata's ``vecrank`` default, applied to Stata's own
        table: the smallest ``r`` whose statistic does not exceed its **5%**
        critical value (``level=0.05``), equivalently the rank ``vecrank``
        marks with ``*``.  ``level=0.01`` reproduces its ``level99`` option.
        A 10% level is refused because Stata's ado does not tabulate it.
        """

        column = _vecrank_level_column(level)
        values = (
            self.trace if statistic == "trace" else self.maxeig
        )
        critical = (
            self.trace_critical if statistic == "trace" else self.maxeig_critical
        )
        for rank, (value, bound) in enumerate(zip(values, critical[:, column])):
            if float(value) <= float(bound):
                return int(rank)
        return int(np.asarray(values).size)


def _vecrank_level_column(level: float) -> int:
    """Column index for a significance level in a ``[90%, 95%, 99%]`` row."""

    mapping = {0.05: 1, 0.01: 2}
    try:
        return mapping[round(float(level), 2)]
    except KeyError:
        raise ValueError(
            f"vecrank: level must be 0.05 or 0.01 (Stata tabulates nothing "
            f"else), got {level!r}"
        ) from None


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
    mean:
        Mean equation; ``"Constant"`` matches Stata's default ``arch ...,
        arch(1) garch(1)``.

    Notes
    -----
    Two of ``arch`` the package's defaults are NOT Stata's, and both are
    overridden here because they change reported numbers:

    *standard errors*  Stata's ``arch`` reports ``e(vcetype) = "OPG"`` (BHHH:
    ``inv(sum_t s_t s_t')``).  The package's default is ``cov_type="robust"``
    (Bollerslev-Wooldridge).  On the 600-observation reference series the two
    differ by up to 52% on a standard error, so ``fit(cov_type="opg")`` is the
    default here.

    *presample variance*  The GARCH recursion needs ``h_0``.  Stata uses
    ``e(archi) = mean(demeaned residual^2)``; the package uses an exponentially
    weighted backcast (``0.94**j`` weights over the first 75 residuals).  The
    two imply slightly different objective functions, so the maximisers differ
    (mu by 1.1e-2 relative).  ``fit(presample="stata")`` iterates the Stata
    convention to a fixed point, which brings all four parameters within 1.8e-4
    of Stata's.

    Neither of these makes the estimates identical to Stata's, and the reason is
    documented rather than absorbed.  The remaining residual is **~1.8e-4
    relative in the parameters and ~1.5e-4 in the standard errors** (measured on
    the reference GARCH(1,1) series against StataNow 19.5).  It is NOT a formula
    difference:

    * evaluating Stata's own variance recursion at STATA's reported parameters
      gives ``ll = -824.3780305032``, while Stata *reports*
      ``e(ll) = -824.3779300106`` -- i.e. Stata's printed likelihood sits
      **1.00e-04 above** what its own recursion yields at its own estimates, so
      Stata's first-observation (``e(condobs) = 0``) handling is slightly
      different from the plain recursion;
    * evaluating the same recursion at westata's estimates gives
      ``-824.3780321``, only **1.6e-06** from Stata's recursion at Stata's
      estimates.

    So both sides sit at essentially the same optimum of the same objective, and
    the gap is Stata's stopping point plus that 1e-4 accounting difference.  A
    1e-9 match would require reproducing Stata's exact conditional-observation
    terms; see ``retest/fix/tseries/repro_v6_arch.py`` for the full
    decomposition and ``retest/fix/tseries/FIXES.md`` for the residual table.
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

    # -- covariance conventions ------------------------------------------
    _COV_TYPES = {
        "opg": "OPG",
        "bhhh": "OPG",
        "robust": "robust",
        "classic": "classic",
        "nonrobust": "classic",
        "hessian": "classic",
    }

    def fit(
        self,
        *,
        strict: bool = False,
        maxiter: int = 1000,
        cov_type: str = "opg",
        presample: str = "stata",
        max_presample_iter: int = 5,
    ) -> StataResults:
        """Estimate the model (Stata ``arch``).

        Parameters
        ----------
        cov_type:
            ``"opg"`` (default, Stata's ``e(vcetype)``), ``"robust"``
            (Bollerslev-Wooldridge, the ``arch`` package's default) or
            ``"classic"`` (inverse Hessian).
        presample:
            ``"stata"`` (default) iterates ``h_0 = mean(demeaned residual^2)``
            to a fixed point, matching ``e(archi)``; ``"arch"`` keeps the
            package's exponentially weighted backcast.
        """

        from arch import arch_model

        covariance = self._COV_TYPES.get(str(cov_type).strip().lower())
        if covariance is None:
            raise ValueError(
                "arch: cov_type must be 'opg' (Stata's vcetype), 'robust' or "
                f"'classic', got {cov_type!r}"
            )
        mode = str(presample).strip().lower()
        if mode not in ("stata", "arch"):
            raise ValueError(
                f"arch: presample must be 'stata' or 'arch', got {presample!r}"
            )

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
        # ``arch``'s fit() only knows 'robust'/'classic'; OPG is computed below.
        fit_cov_type = "classic" if covariance == "OPG" else covariance

        data = self._y.to_numpy(dtype=float)
        backcast: float | None = None
        result = None
        for _ in range(max(1, int(max_presample_iter)) if mode == "stata" else 1):
            result = model.fit(
                disp="off",
                cov_type=fit_cov_type,
                backcast=backcast,
                options={"maxiter": maxiter},
            )
            if mode != "stata":
                break
            base = float(np.asarray(result.params, dtype=float)[0])
            updated = float(np.mean((data - base) ** 2))
            if backcast is not None and abs(updated - backcast) <= 1e-12 * abs(updated):
                backcast = updated
                break
            backcast = updated
        assert result is not None

        converged = bool(getattr(result, "convergence_flag", 1) == 0)
        if not converged:
            message = "arch: the optimiser did not converge; estimates may be unreliable"
            if strict:
                raise ConvergenceError(message)
            import warnings

            warnings.warn(message, RuntimeWarning, stacklevel=2)

        params = np.asarray(result.params, dtype=float)
        names = list(result.params.index)
        if backcast is None:
            # Keep the package's backcast visible in e(archi) instead of a NaN.
            backcast = getattr(model, "_backcast", None)
        if covariance == "OPG":
            cov = _opg_covariance(model, params, backcast)
        else:
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
            "vcetype": covariance,
            # Stata's e(archi): the presample variance the recursion starts from.
            "archi": float(backcast) if backcast is not None else float("nan"),
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
            cov_type=covariance,
            e=e,
        )


def _opg_covariance(model: Any, params: np.ndarray, backcast: float | None) -> np.ndarray:
    """BHHH/OPG covariance: ``inv(sum_t s_t s_t')`` with numerical scores.

    Stata's ``arch`` stores ``e(vcetype) = "OPG"``; the ``arch`` package has no
    equivalent (its ``robust`` estimator is the sandwich
    ``H^-1 (sum s s') H^-1`` with ``H`` the numerical Hessian).  The scores are
    central differences of the INDIVIDUAL log-likelihood contributions, i.e.
    ``s_t = d log f_t / d theta``, summed in the outer product.

    Step choice matters: the objective is extremely flat in ``beta`` and a
    fixed absolute step of 1e-6 loses ~4 digits.  ``1e-5 * |theta|`` (floored
    at 1e-8) reproduces Stata's OPG standard errors to 4e-5 relative on the
    reference series -- far inside the 1e-2 target -- and is stable across
    GARCH(1,1) and ARCH(1).
    """

    values = np.asarray(params, dtype=float)
    residuals = np.asarray(model.resids(values), dtype=float)
    var_bounds = model.volatility.variance_bounds(residuals)
    if backcast is None:
        backcast = model.volatility.backcast(residuals)
    if hasattr(model.volatility, "backcast_transform"):
        backcast = model.volatility.backcast_transform(backcast)

    def individual_loglik(candidate: np.ndarray) -> np.ndarray:
        return np.asarray(
            model._loglikelihood(
                candidate,
                np.zeros_like(residuals),
                backcast,
                var_bounds,
                individual=True,
            ),
            dtype=float,
        )

    size = values.size
    scores = np.empty((residuals.size, size), dtype=float)
    for index in range(size):
        step = max(1e-8, 1e-5 * abs(values[index]))
        up = values.copy()
        down = values.copy()
        up[index] += step
        down[index] -= step
        scores[:, index] = (individual_loglik(up) - individual_loglik(down)) / (2.0 * step)

    information = scores.T @ scores
    try:
        return np.linalg.inv(information)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - singular only if degenerate
        raise ConsistencyError(
            "arch: the OPG information matrix is singular, so the OPG standard "
            "errors Stata reports cannot be reproduced; refit with "
            "cov_type='robust' or 'classic' and read that explicitly"
        ) from exc


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


# Stata's ``dfuller.ado`` (and ``pperron.ado``) do NOT evaluate MacKinnon's
# response surface for the printed critical values.  ``GetCrit`` hard-codes this
# six-point table -- N = 25, 50, 100, 250, 500, infinity -- per deterministic
# case and interpolates LINEARLY in N:
#
#     N <= 25      : column 1
#     N <= 50      : col 1 -> col 2 over  25..50
#     N <= 100     : col 2 -> col 3 over  50..100
#     N <= 250     : col 3 -> col 4 over 100..250
#     N <= 500     : col 4 -> col 5 over 250..500
#     otherwise    : column 6
#
# Reproducing the table verbatim (from ``ado/base/d/dfuller.ado``) is the only
# way to match the values Stata prints; verified against StataNow 19.5 to
# 1.3e-16 relative on five data sets in ``retest/fix/tseries/repro_v3_dfuller.py``.
_DFULLER_TABLE: dict[str, tuple[list[float], ...]] = {
    # case "n": no constant, no trend
    "n": (
        [-2.66, -2.62, -2.60, -2.58, -2.58, -2.58],
        [-1.95, -1.95, -1.95, -1.95, -1.95, -1.95],
        [-1.60, -1.61, -1.61, -1.62, -1.62, -1.62],
    ),
    # case "c": constant (Stata's default)
    "c": (
        [-3.75, -3.58, -3.51, -3.46, -3.44, -3.43],
        [-3.00, -2.93, -2.89, -2.88, -2.87, -2.86],
        [-2.63, -2.60, -2.58, -2.57, -2.57, -2.57],
    ),
    # case "ct": constant plus trend
    "ct": (
        [-4.38, -4.15, -4.04, -3.99, -3.98, -3.96],
        [-3.60, -3.50, -3.45, -3.43, -3.42, -3.41],
        [-3.24, -3.18, -3.15, -3.13, -3.13, -3.12],
    ),
}

_DFULLER_KNOTS: tuple[int, int, int, int, int] = (25, 50, 100, 250, 500)


def _dfuller_critical(case: str, observations: float) -> dict[str, float]:
    """Stata's printed 1%/5%/10% critical values for ``dfuller``.

    ``case`` is one of ``"n"``, ``"c"``, ``"ct"``; ``observations`` is Stata's
    ``r(N)`` (the number of observations in the Dickey-Fuller regression, i.e.
    after lags and differencing).
    """

    if case not in _DFULLER_TABLE:
        raise ValueError(f"dfuller: unknown deterministic case {case!r}")
    table = _DFULLER_TABLE[case]
    n = float(observations)
    values: list[float] = []
    for row in table:
        if n <= _DFULLER_KNOTS[0]:
            value = row[0]
        elif n <= _DFULLER_KNOTS[1]:
            value = row[0] + (n - 25) / 25 * (row[1] - row[0])
        elif n <= _DFULLER_KNOTS[2]:
            value = row[1] + (n - 50) / 50 * (row[2] - row[1])
        elif n <= _DFULLER_KNOTS[3]:
            value = row[2] + (n - 100) / 150 * (row[3] - row[2])
        elif n <= _DFULLER_KNOTS[4]:
            value = row[3] + (n - 250) / 250 * (row[4] - row[3])
        else:
            value = row[5]
        values.append(float(value))
    return {"1%": values[0], "5%": values[1], "10%": values[2]}


#: Map the statsmodels/us regression code onto ``dfuller.ado``'s ``case``.
#: Stata's ``drift`` (case 3) uses a t distribution instead and has no table.
_DFULLER_CASES = {"n": "n", "c": "c", "ct": "ct"}


def _dfuller_case(regression: str) -> str:
    code = str(regression).strip().lower()
    if code not in _DFULLER_CASES:
        raise ValueError(
            f"dfuller: regression must be one of 'n', 'c', 'ct', got {regression!r}"
        )
    return _DFULLER_CASES[code]


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
        Deterministic terms: ``"c"``, ``"n"``, ``"ct"``, ``"ctt"``.
    autolag:
        Lag selection for ``"adf"``; ``None`` uses ``maxlag`` verbatim (Stata's
        ``lags(#)``), a criterion name selects it the statsmodels way.

    Notes
    -----
    For ``"adf"`` the *statistic* and the MacKinnon p-value are the same test
    Stata runs and agree to ~1e-13, but the printed **critical values** are not
    MacKinnon response-surface values: ``dfuller.ado`` interpolates a six-point
    table linearly in N.  This function returns that table's values, so a user
    comparing the printed band sees the same numbers Stata prints.  The p-value
    is unaffected and remains the MacKinnon one.
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
        # Stata's r(cv_*) comes from dfuller.ado's hard-coded table, not from
        # MacKinnon's surface (which is what statsmodels interpolates).
        critical = _dfuller_critical(
            _dfuller_case(regression), float(outcome.nobs)
        )
        return UnitRootResult(
            test="Augmented Dickey-Fuller",
            statistic=float(outcome.statistic),
            pvalue=float(outcome.pvalue),
            lags=int(outcome.lags),
            observations=int(outcome.nobs),
            critical=critical,
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
        # ``arch`` only sets ``nobs`` (= the Dickey-Fuller regression's e(N))
        # once the statistic has been computed, so force that first.
        statistic = float(outcome.stat)
        observations = int(outcome.nobs)
        # pperron.ado uses the SAME GetCrit table as dfuller.ado for Z(t), and
        # indexes it by ``local T = e(N)`` of that regression.  Verified against
        # StataNow 19.5 on a 300-observation series: Stata's 1%/5%/10% values
        # are -3.45608/-2.87804/-2.57, i.e. N = 299 -- which is exactly the
        # ``nobs`` arch reports here, so no offset is needed.
        critical = _dfuller_critical(_dfuller_case(regression), float(observations))
        return UnitRootResult(
            test="Phillips-Perron",
            statistic=statistic,
            pvalue=float(outcome.pvalue),
            lags=int(outcome.lags),
            observations=observations,
            critical=critical,
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
