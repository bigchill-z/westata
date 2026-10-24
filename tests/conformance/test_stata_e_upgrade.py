"""E3 conformance for the upgraded panel and time-series commands.

Everything here was captured from LIVE Stata (pystata) at full precision, and every
gap quoted in the assertions was measured before it was written down.

===================  ===========================  ============
command              key parameter                measured gap
===================  ===========================  ============
xtprobit, re         x1 / x2                      2.4e-8 / 1.0e-7
xttobit              x1                           1.3e-9
xtpoisson, re        x1 / _cons                   3.9e-8
xtnbreg, fe          x1                           2.1e-5
xtlogit, pa          x1 / x2 / _cons              9.1e-6 / 9.7e-6 / 7.4e-8
pperron (default)    Z(t)                         1.1e-4 (Stata prints 3 dp)
dfuller (default)    Z(t)                         1.3e-4 (Stata prints 3 dp)
===================  ===========================  ============

``pperron`` deserves a note: its recorded 1.6e-2 gap was never an estimator problem. Stata
prints "Newey-West lags = 5" and uses ``floor(4*(n/100)^(2/9))``; westata previously let
``arch`` choose the bandwidth, which picked something else. Applying the rule gives
-1.953113 against Stata's -1.953.
"""

from __future__ import annotations

import importlib.util

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
REFERENCE = FIXTURES / "e_upgrade_outputs.txt"

if not REFERENCE.exists():
    pytest.skip("upgrade fixtures missing", allow_module_level=True)


def _require(name: str):
    attribute = getattr(ws, name, None)
    if attribute is None:
        pytest.skip(f"{name} is not exposed on westata yet")
    return attribute


# ---- the exact frames the fixtures were captured on ----------------------

def binary_panel(n_firms: int = 120, n_years: int = 8, seed: int = 71) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = n_firms * n_years
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(scale=0.8, size=n_firms)[firm]
    latent = 0.4 * x1 - 0.3 * x2 + effect + rng.logistic(size=n)
    return pd.DataFrame(
        {"firm": firm, "year": year, "x1": x1, "x2": x2, "y": (latent > 0).astype(float)}
    )


def count_panel(n_firms: int = 100, n_years: int = 7, seed: int = 73) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = n_firms * n_years
    x1 = rng.normal(size=n)
    effect = rng.normal(scale=0.4, size=n_firms)[firm]
    mu = np.exp(0.3 + 0.5 * x1 + effect)
    return pd.DataFrame(
        {"firm": firm, "year": year, "x1": x1, "y": rng.poisson(mu).astype(float)}
    )


def censored_panel(n_firms: int = 100, n_years: int = 7, seed: int = 83) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = n_firms * n_years
    x1 = rng.normal(size=n)
    effect = rng.normal(scale=0.5, size=n_firms)[firm]
    latent = 1.0 + 0.9 * x1 + effect + rng.normal(scale=0.6, size=n)
    return pd.DataFrame(
        {"firm": firm, "year": year, "x1": x1, "y": np.maximum(latent, 0.5)}
    )


# ---- Stata's full-precision values --------------------------------------

STATA_XTPROBIT = {"x1": 0.20983396078276506, "x2": -0.17655902464995493}
STATA_XTTOBIT = {"x1": 0.9020131009529725, "_cons": 0.9253712092561762}
STATA_XTPOISSON = {"x1": 0.47955085149744847, "_cons": 0.3485992628857865}
STATA_XTNBREG = {"x1": 0.5041908533303101}
STATA_XTLOGIT_PA = {
    "x1": 0.3007615470018262,
    "x2": -0.2542735657384027,
    "_cons": 0.1488733810746218,
}


def test_xtprobit_re_matches_stata():
    XtProbit = _require("XtProbit")
    result = XtProbit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="re"
    ).fit()

    for name, expected in STATA_XTPROBIT.items():
        assert float(result.params[name]) == pytest.approx(expected, rel=1e-5), name


def test_xttobit_matches_stata():
    XtTobit = _require("XtTobit")
    result = XtTobit.from_formula(
        "y ~ x1", censored_panel(), entity="firm", lower=0.5
    ).fit()

    for name, expected in STATA_XTTOBIT.items():
        assert float(result.params[name]) == pytest.approx(expected, rel=1e-5), name
    # Stata reports both variance components under bare names.
    assert "sigma_u" in result.param_names
    assert "sigma_e" in result.param_names


def test_xtpoisson_re_matches_stata():
    XtPoisson = _require("XtPoisson")
    result = XtPoisson.from_formula(
        "y ~ x1", count_panel(), entity="firm", method="re"
    ).fit()

    for name, expected in STATA_XTPOISSON.items():
        assert float(result.params[name]) == pytest.approx(expected, rel=1e-5), name
    # Stata's spelling is bare, not slashed.
    assert "lnalpha" in result.param_names


def test_xtnbreg_fe_matches_stata():
    XtNbreg = _require("XtNbreg")
    result = XtNbreg.from_formula(
        "y ~ x1", count_panel(), entity="firm", method="fe"
    ).fit()

    # Measured 2.1e-5: the quasi-likelihood converges from a slightly different
    # starting point than Stata's, so the tolerance reflects that rather than
    # pretending it is machine precision.
    assert float(result.params["x1"]) == pytest.approx(
        STATA_XTNBREG["x1"], rel=1e-4
    )


def test_xtlogit_pa_matches_stata():
    """GEE: a quasi-likelihood, so e(ll) is absent by design but the coefficients
    are estimable and now verified."""

    XtLogit = _require("XtLogit")
    result = XtLogit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="pa"
    ).fit()

    for name, expected in STATA_XTLOGIT_PA.items():
        assert float(result.params[name]) == pytest.approx(expected, rel=1e-5), name
    assert "ll" not in result.e


requires_arch = pytest.mark.skipif(
    importlib.util.find_spec("arch") is None,
    reason="needs the optional `arch` dependency (pip install westata[arch])",
)

@requires_arch
def test_pperron_default_bandwidth_follows_newwey_west():
    """The 1.6e-2 gap was a bandwidth rule, not an estimator difference.

    Stata prints "Newey-West lags = 5" for this series and uses
    ``floor(4*(n/100)^(2/9))``. Applying that rule gives -1.953113 against Stata's
    printed -1.953, i.e. agreement to Stata's three decimals.
    """

    rng = np.random.default_rng(41)
    series = np.cumsum(rng.normal(size=400))
    result = ws.unitroot(series, test="pp")

    assert int(result.lags) == 5
    assert float(result.statistic) == pytest.approx(-1.953, abs=1e-3)


def test_dfuller_matches_stata():
    rng = np.random.default_rng(41)
    series = np.cumsum(rng.normal(size=400))
    result = ws.unitroot(series, test="adf")
    assert float(result.statistic) == pytest.approx(-1.941, abs=1e-3)
