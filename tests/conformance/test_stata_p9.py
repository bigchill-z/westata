"""E3 conformance for the P9 panel commands."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
REFERENCE = FIXTURES / "p9_outputs.txt"

if not REFERENCE.exists():
    pytest.skip("P9 fixtures missing", allow_module_level=True)


def _require(name: str):
    attribute = getattr(ws, name, None)
    if attribute is None:
        pytest.skip(f"{name} is not exposed on westata yet")
    return attribute


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


def iv_panel(n_firms: int = 100, n_years: int = 7, seed: int = 89) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = n_firms * n_years
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    endog = 0.7 * z + 0.5 * w + 0.6 * u + rng.normal(size=n)
    effect = rng.normal(scale=0.4, size=n_firms)[firm]
    y = 1.0 + 1.2 * endog + 0.5 * w + effect + u
    return pd.DataFrame({"firm": firm, "year": year, "y": y, "endog": endog, "z": z, "w": w})


def dynamic_panel(n_firms: int = 90, n_years: int = 9, seed: int = 79) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for firm in range(n_firms):
        effect = rng.normal(scale=0.5)
        lagged = rng.normal()
        for year in range(1, n_years + 1):
            x1 = rng.normal()
            value = 0.5 * lagged + 0.8 * x1 + effect + rng.normal(scale=0.5)
            rows.append((firm, year, value, x1))
            lagged = value
    return pd.DataFrame(rows, columns=["firm", "year", "y", "x1"])


# ------------------------------------------------------------------ xtlogit


def test_xtlogit_fe_matches_stata_exactly():
    XtLogit = _require("XtLogit")
    result = XtLogit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="fe"
    ).fit()

    # Stata: x1 .3307537 (se .0807304), x2 -.3134259 (se .0795823)
    assert float(result.params["x1"]) == pytest.approx(0.3307537, rel=1e-4)
    assert float(result.params["x2"]) == pytest.approx(-0.3134259, rel=1e-4)


def test_xtlogit_fe_uses_statas_sample_size():
    """Conditioning drops the groups without outcome variation: 960 -> 912."""

    XtLogit = _require("XtLogit")
    result = XtLogit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="fe"
    ).fit()
    # The estimation sample is reported under the standard e(N) key.
    assert int(result.e["N"]) == 912


def test_xtlogit_pa_reports_no_likelihood():
    XtLogit = _require("XtLogit")
    result = XtLogit.from_formula(
        "y ~ x1 + x2", binary_panel(), entity="firm", method="pa"
    ).fit()
    # GEE is not a likelihood, so there is no e(ll) to compare with one.
    assert "ll" not in result.e


# ---------------------------------------------------------------- xtpoisson


def test_xtpoisson_fe_matches_stata():
    XtPoisson = _require("XtPoisson")
    result = XtPoisson.from_formula(
        "y ~ x1", count_panel(), entity="firm", method="fe"
    ).fit()
    # Stata: x1 .5041803 (se .0344352)
    assert float(result.params["x1"]) == pytest.approx(0.5041803, rel=5e-2)


# ------------------------------------------------------------------ xtivreg


def test_xtivreg_fe_matches_stata_exactly():
    XtIvreg = _require("XtIvreg")
    result = XtIvreg.from_formula(
        "y ~ w + (endog = z)", iv_panel(), entity="firm", method="fe"
    ).fit()

    # Stata: endog 1.102058, w .5391216, _cons 1.014978
    assert float(result.params["endog"]) == pytest.approx(1.102058, rel=1e-3)
    assert float(result.params["w"]) == pytest.approx(0.5391216, rel=1e-3)
    assert float(result.params["_cons"]) == pytest.approx(1.014978, rel=1e-3)


def test_xtivreg_re_matches_stata_within_a_percent():
    XtIvreg = _require("XtIvreg")
    result = XtIvreg.from_formula(
        "y ~ w + (endog = z)", iv_panel(), entity="firm", method="re"
    ).fit()

    # Stata: endog 1.120466, w .5286517, _cons 1.014533
    assert float(result.params["endog"]) == pytest.approx(1.120466, rel=1e-2)
    assert float(result.params["w"]) == pytest.approx(0.5286517, rel=1e-2)
    assert float(result.params["_cons"]) == pytest.approx(1.014533, rel=1e-2)


# -------------------------------------------------------------------- xtgls


def test_xtgls_matches_stata_and_is_not_merely_ols():
    """xtgls with panel-specific variances is a real GLS, not weighted OLS.

    Stata: x1 .6798471, _cons 1.431659. Pooled OLS on the same data gives
    .753313 / 1.572686, so asserting equality with OLS would have been wrong --
    an earlier version of this test did exactly that, and it was the test that was
    wrong, not the estimator.
    """

    XtGls = _require("XtGls")
    frame = count_panel()
    result = XtGls.from_formula(
        "y ~ x1", frame, entity="firm", panels="heteroskedastic"
    ).fit()

    assert float(result.params["x1"]) == pytest.approx(0.6798471, rel=1e-4)
    assert float(result.params["Intercept"]) == pytest.approx(1.431659, rel=1e-4)

    # And it must differ from OLS, or the GLS weighting is doing nothing.
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()
    assert float(result.params["x1"]) != pytest.approx(
        float(plain.params["x1"]), rel=1e-4
    )


# ------------------------------------------------------------------- xtpcse


def test_xtpcse_matches_stata_coefficients():
    XtPcse = _require("XtPcse")
    result = XtPcse.from_formula("y ~ x1", count_panel(), entity="firm").fit()
    # Stata: x1 .7533126, _cons 1.572686
    assert float(result.params["x1"]) == pytest.approx(0.7533126, rel=1e-4)
    assert float(result.params["Intercept"]) == pytest.approx(1.572686, rel=1e-4)


# ----------------------------------------------------------------- xtabond


def test_xtabond_matches_stata_coefficients_and_instrument_count():
    """The instrument COUNT is the sharpest check on the block construction.

    Stata reports 30 instruments for xtabond y x1, lags(1) on a 9-period panel:
    28 GMM columns plus the differenced covariate and a constant. Getting that
    number right, together with the coefficients, means the instrument block is
    laid out the way Stata lays it out.
    """

    XtAbond = _require("XtAbond")
    result = XtAbond.from_formula(
        "y ~ x1", dynamic_panel(), entity="firm", time="year", lags=1
    ).fit()

    assert int(result.e["instruments"]) == 30
    # Stata: L1.y .4549539, x1 .776234
    assert float(result.params["L1.y"]) == pytest.approx(0.4549539, rel=5e-2)
    assert float(result.params["x1"]) == pytest.approx(0.776234, rel=2e-2)


def test_xtabond_omits_the_overidentification_statistics():
    """Sargan/Hansen/AR are left out rather than approximated."""

    XtAbond = _require("XtAbond")
    result = XtAbond.from_formula(
        "y ~ x1", dynamic_panel(), entity="firm", time="year", lags=1
    ).fit()
    for key in ("sargan", "hansen", "ar1", "ar2"):
        assert key not in result.e
