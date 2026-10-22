"""Conformance tests for the time-series family.

These check structural correctness and recovery of known dynamics (an AR(1)
with a known coefficient, a VAR(1) with known matrices, a GARCH process with
known persistence, and a unit-root series that the tests must detect), rather
than comparing against another implementation of the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.estimators.tseries import (
    ARIMA,
    GARCH,
    VAR,
    VECM,
    unitroot,
)


def _ar1(n: int = 800, phi: float = 0.7, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + shocks[t]
    return y


def _garch_process(n: int = 1200, seed: int = 5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.05, 0.08, 0.88
    variance = np.zeros(n)
    returns = np.zeros(n)
    variance[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = np.sqrt(variance[t]) * rng.normal()
    return returns


# --------------------------------------------------------------------- ARIMA


def test_arima_recovers_ar1_coefficient():
    y = _ar1(n=1500, phi=0.7)
    res = ARIMA(y, order=(1, 0, 0)).fit()
    assert res.params["ar.L1"] == pytest.approx(0.7, abs=0.06)


def test_arima_uses_z_inference():
    res = ARIMA(_ar1(), order=(1, 0, 0)).fit()
    assert res.dist == "z"


def test_arima_e_keys_present():
    res = ARIMA(_ar1(), order=(1, 0, 0)).fit()
    for key in ["N", "ll", "aic", "bic", "converged"]:
        assert key in res.e, key


def test_arima_rejects_missing_values():
    y = _ar1(n=100)
    y[10] = np.nan
    with pytest.raises(Exception, match="missing"):
        ARIMA(y, order=(1, 0, 0)).fit()


def test_arima_prefers_ar1_over_white_noise():
    y = _ar1(n=1500, phi=0.7)
    aic_ar1 = ARIMA(y, order=(1, 0, 0)).fit().aic
    aic_white_noise = ARIMA(y, order=(0, 0, 0)).fit().aic
    # The DGP has real autocorrelation, so AR(1) must beat white noise by a lot.
    # (Comparing AR(1) against AR(3) would only be a probabilistic statement.)
    assert aic_ar1 < aic_white_noise - 100


# ----------------------------------------------------------------------- VAR


def _var1(n: int = 600, seed: int = 7):
    rng = np.random.default_rng(seed)
    A = np.array([[0.5, 0.1], [0.2, 0.4]])
    data = np.zeros((n, 2))
    shocks = rng.normal(size=(n, 2))
    for t in range(1, n):
        data[t] = A @ data[t - 1] + shocks[t]
    return data, A


def test_var_recovers_transition_matrix():
    data, A = _var1()
    result = VAR(data, lags=1, names=["y1", "y2"]).fit()

    # ``params`` rows are [const, L1.y1, L1.y2] and columns are the equations,
    # so the coefficient of lagged variable j in equation i is A[i, j].
    fitted = result.params
    for equation_index, equation in enumerate(["y1", "y2"]):
        for variable_index, lagged in enumerate(["y1", "y2"]):
            row = f"L1.{lagged}"
            assert row in fitted.index
            assert fitted.loc[row, equation] == pytest.approx(
                A[equation_index, variable_index], abs=0.12
            )


def test_var_select_order_returns_all_criteria():
    data, _ = _var1(n=400)
    table = VAR(data, lags=1).select_order(maxlags=3)
    assert set(table.index) == {0, 1, 2, 3}
    assert {"AIC", "BIC", "HQIC", "FPE"}.issubset(set(table.columns))


def test_var_requires_explicit_lag_when_none():
    data, _ = _var1(n=100)
    with pytest.raises(Exception, match="lag order"):
        VAR(data, lags=None).fit()


def test_var_granger_causality_runs():
    data, _ = _var1()
    result = VAR(data, lags=1, names=["y1", "y2"]).fit()
    outcome = result.granger_causality("y1", ["y2"])
    assert outcome is not None


def test_var_irf_and_fevd_shapes():
    data, _ = _var1()
    result = VAR(data, lags=1, names=["y1", "y2"]).fit()
    assert result.irf(5) is not None
    assert result.fevd(5) is not None


# ---------------------------------------------------------------------- VECM


def test_johansen_on_cointegrated_series():
    rng = np.random.default_rng(9)
    n = 400
    random_walk = np.cumsum(rng.normal(size=n))
    # A second series cointegrated with the first.
    other = random_walk + rng.normal(scale=0.2, size=n)
    data = np.column_stack([random_walk, other])

    outcome = VECM(data, coint_rank=1).johansen(k_ar_diff=1)
    frame = outcome.to_frame()
    assert frame.shape[0] >= 1
    # The first null (r=0 against r>=1) should be rejected for cointegrated data.
    assert frame.iloc[0]["trace"] > frame.iloc[0]["trace_5pct"]


def test_vecm_fit_runs():
    rng = np.random.default_rng(11)
    n = 300
    rw = np.cumsum(rng.normal(size=n))
    other = rw + rng.normal(scale=0.3, size=n)
    data = np.column_stack([rw, other])
    fitted = VECM(data, coint_rank=1, lags=1).fit()
    assert fitted is not None


# --------------------------------------------------------------------- GARCH


def test_garch_recovers_persistence():
    returns = _garch_process(n=1500)
    res = GARCH(returns, p=1, q=1).fit()

    alpha = res.e["alpha"]
    beta = res.e["beta"]
    assert len(alpha) == 1 and len(beta) == 1
    assert alpha[0] + beta[0] == pytest.approx(0.96, abs=0.12)
    assert res.e["omega"] > 0


def test_garch_conditional_volatility_is_positive():
    res = GARCH(_garch_process(n=800), p=1, q=1).fit()
    assert np.all(res.fittedvalues > 0)


def test_garch_rejects_bad_volatility_type():
    with pytest.raises(ValueError, match="vol"):
        GARCH(_garch_process(n=100), vol="NOPE")


def test_egarch_runs():
    res = GARCH(_garch_process(n=800), p=1, o=1, q=1, vol="EGARCH").fit()
    assert np.all(np.isfinite(res.params.to_numpy()))


# ------------------------------------------------------------------ unit root


def test_adf_detects_unit_root_in_random_walk():
    rng = np.random.default_rng(13)
    walk = np.cumsum(rng.normal(size=400))
    outcome = unitroot(walk, test="adf")
    # A random walk is nonstationary, so the ADF null is not rejected.
    assert outcome.pvalue > 0.05


def test_adf_rejects_unit_root_for_stationary_series():
    stationary = _ar1(n=600, phi=0.3)
    outcome = unitroot(stationary, test="adf")
    assert outcome.pvalue < 0.05


def test_kpss_has_the_opposite_null():
    rng = np.random.default_rng(17)
    walk = np.cumsum(rng.normal(size=400))
    outcome = unitroot(walk, test="kpss")
    # KPSS's null is stationarity, so a random walk should reject it.
    assert outcome.pvalue < 0.05
    assert "stationary" in outcome.null


def test_pp_runs():
    rng = np.random.default_rng(19)
    walk = np.cumsum(rng.normal(size=300))
    outcome = unitroot(walk, test="pp")
    assert np.isfinite(outcome.statistic)
    assert 0.0 <= outcome.pvalue <= 1.0


def test_unknown_unit_root_test_raises():
    with pytest.raises(ValueError, match="unitroot"):
        unitroot(np.random.default_rng(1).normal(size=50), test="bogus")
