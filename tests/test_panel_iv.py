"""Tests for panel IV, GLS, PCSE and unit-root commands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.panel_iv import XtGls, XtIvreg, XtPcse, XtUnitroot


def _iv_panel(n_firms: int = 100, n_years: int = 7, seed: int = 89) -> pd.DataFrame:
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
    return pd.DataFrame(
        {"firm": firm, "year": year, "y": y, "endog": endog, "z": z, "w": w}
    )


def _count_panel(n_firms: int = 100, n_years: int = 7, seed: int = 73) -> pd.DataFrame:
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


def _dynamic_panel(n_firms: int = 60, n_years: int = 9, seed: int = 79) -> pd.DataFrame:
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


# ------------------------------------------------------------------ xtivreg


def test_xtivreg_fe_matches_stata():
    frame = _iv_panel()
    result = XtIvreg.from_formula(
        "y ~ w + (endog = z)", frame, entity="firm", method="fe"
    ).fit()

    # Stata: endog 1.102058 (se .0694014), w .5391216, _cons 1.014978
    assert float(result.params["endog"]) == pytest.approx(1.102058, rel=5e-3)
    assert float(result.params["w"]) == pytest.approx(0.5391216, rel=5e-3)


def test_xtivreg_re_matches_stata():
    frame = _iv_panel()
    result = XtIvreg.from_formula(
        "y ~ w + (endog = z)", frame, entity="firm", method="re"
    ).fit()

    # Stata: endog 1.120466 (se .065102), w .5286517, _cons 1.014533
    assert float(result.params["endog"]) == pytest.approx(1.120466, rel=5e-2)
    assert float(result.params["w"]) == pytest.approx(0.5286517, rel=5e-2)


def test_xtivreg_formula_parser_separates_the_three_pieces():
    frame = _iv_panel()
    frame["w2"] = np.random.default_rng(1).normal(size=len(frame))
    frame["z2"] = np.random.default_rng(2).normal(size=len(frame))

    model = XtIvreg.from_formula(
        "y ~ w + w2 + (endog = z + z2)", frame, entity="firm", method="fe"
    )

    assert model._exog_names == ["Intercept", "w", "w2"]
    assert model._instrumented_names == ["Intercept", "endog"]
    assert "z" in model._instrument_names
    assert "z2" in model._instrument_names


def test_xtivreg_rejects_a_missing_instrument_block():
    with pytest.raises(ConsistencyError, match="instrument block"):
        XtIvreg.from_formula("y ~ w", _iv_panel(), entity="firm")


def test_xtivreg_rejects_an_unknown_method():
    with pytest.raises(UnsupportedError, match="fe"):
        XtIvreg.from_formula(
            "y ~ w + (endog = z)", _iv_panel(), entity="firm", method="between"
        )


# -------------------------------------------------------------------- xtgls


def test_xtgls_heteroskedastic_coefficients_equal_pooled_ols():
    """With panel-specific variances and no cross-panel correlation, the GLS
    estimating equations for the slopes collapse to OLS."""

    frame = _count_panel()
    result = XtGls.from_formula(
        "y ~ x1", frame, entity="firm", panels="heteroskedastic"
    ).fit()
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()

    np.testing.assert_allclose(
        result.params[["Intercept", "x1"]].to_numpy(),
        plain.params[["Intercept", "x1"]].to_numpy(),
        rtol=1e-10,
    )


def test_xtgls_reports_panel_count():
    frame = _count_panel()
    result = XtGls.from_formula("y ~ x1", frame, entity="firm").fit()
    assert int(result.e["n_panels"]) == 100


def test_xtgls_rejects_correlated_panels():
    with pytest.raises(UnsupportedError, match="heteroskedastic"):
        XtGls.from_formula(
            "y ~ x1", _count_panel(), entity="firm", panels="correlated"
        )


# ------------------------------------------------------------------- xtpcse


def test_xtpcse_coefficients_equal_ols_but_standard_errors_differ():
    frame = _count_panel()
    result = XtPcse.from_formula("y ~ x1", frame, entity="firm").fit()
    plain = ws.OLS.from_formula("y ~ x1", frame).fit()

    np.testing.assert_allclose(
        result.params[["Intercept", "x1"]].to_numpy(),
        plain.params[["Intercept", "x1"]].to_numpy(),
        rtol=1e-10,
    )
    assert float(result.bse["x1"]) != pytest.approx(float(plain.bse["x1"]), rel=1e-6)


def test_xtpcse_matches_stata_coefficients():
    frame = _count_panel()
    result = XtPcse.from_formula("y ~ x1", frame, entity="firm").fit()
    # Stata: x1 .7533126 (se .0564318), _cons 1.572686
    assert float(result.params["x1"]) == pytest.approx(0.7533126, rel=1e-4)
    assert float(result.params["Intercept"]) == pytest.approx(1.572686, rel=1e-4)


# --------------------------------------------------------------- xtunitroot


def test_xtunitroot_ips_produces_a_finite_statistic():
    frame = _dynamic_panel()
    outcome = XtUnitroot.from_formula(
        "y ~ 1", frame, entity="firm", test="ips"
    ).fit()

    assert np.isfinite(outcome.statistic)
    assert 0.0 <= outcome.pvalue <= 1.0


def test_xtunitroot_llc_produces_a_finite_statistic():
    frame = _dynamic_panel()
    outcome = XtUnitroot.from_formula(
        "y ~ 1", frame, entity="firm", test="llc"
    ).fit()

    assert np.isfinite(outcome.statistic)
    assert 0.0 <= outcome.pvalue <= 1.0


def test_xtunitroot_rejects_an_unknown_test():
    with pytest.raises(UnsupportedError, match="ips"):
        XtUnitroot.from_formula(
            "y ~ 1", _dynamic_panel(), entity="firm", test="madwu"
        )


def test_xtunitroot_rejects_a_short_series():
    tiny = pd.DataFrame(
        {"firm": [1, 1, 2, 2], "year": [1, 2, 1, 2], "y": [1.0, 2.0, 3.0, 4.0]}
    )
    with pytest.raises(ConsistencyError, match="too short"):
        XtUnitroot.from_formula("y ~ 1", tiny, entity="firm", test="ips").fit()
