"""End-to-end smoke test of the public API across every estimator family.

This is the "does it actually work as a tool" test: it exercises the exact
calls from the design spec's API section, one per estimator, on synthetic data
with a known data-generating process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def cross_section() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 150
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x_exog = rng.normal(size=n)
    v = rng.normal(size=n)
    x_endog = 0.8 * z1 + 0.5 * z2 + 0.6 * v
    y = 1.0 + 2.0 * x_endog + 0.5 * x_exog + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x_endog": x_endog,
            "x_exog": x_exog,
            "z1": z1,
            "z2": z2,
            "group": rng.integers(0, 8, size=n),
            "weight": rng.uniform(0.5, 3.0, size=n),
        }
    )


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(13)
    n_entities, n_periods = 30, 6
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + rng.normal(scale=0.4, size=n)
    return pd.DataFrame(
        {"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time}
    )


def test_ols_with_cluster_and_pweight(cross_section):
    df = cross_section
    res = ws.OLS.from_formula(
        "y ~ x_exog", df, weights=df["weight"], weight_type="pweight"
    ).fit(cov_type="cluster", cov_kwds={"groups": df["group"]})
    assert res.e["N"] == len(df)
    assert res.df_resid == df["group"].nunique() - 1
    assert np.all(np.isfinite(res.bse.to_numpy()))


def test_iv2sls_end_to_end(cross_section):
    df = cross_section
    res = ws.IV2SLS.from_formula("y ~ x_exog + [x_endog ~ z1 + z2]", df).fit()
    assert res.dist == "z"
    # The DGP has a true coefficient of 2.0 on the endogenous regressor.
    assert res.params["x_endog"] == pytest.approx(2.0, abs=0.2)


def test_feols_two_way(cross_section, panel):
    df = panel
    res = ws.FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit()
    assert res.params["x1"] == pytest.approx(1.5, abs=0.1)
    assert res.params["x2"] == pytest.approx(-0.5, abs=0.1)


def test_panel_fe_re_be_fd(panel):
    df = panel
    fe = ws.PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()
    re = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", df, entity="firm", time="year"
    ).fit()
    be = ws.BetweenOLS.from_formula(
        "y ~ x1 + x2", df, entity="firm", time="year"
    ).fit()
    fd = ws.FirstDifferenceOLS.from_formula(
        "y ~ x1 + x2", df, entity="firm", time="year"
    ).fit()

    for res in (fe, re, be, fd):
        assert np.all(np.isfinite(res.params.to_numpy()))
        assert "Coef." in res.summary(style="stata")

    # FE is consistent here; the DGP slope should be recovered closely.
    assert fe.params["x1"] == pytest.approx(1.5, abs=0.1)
    assert fd.params["x1"] == pytest.approx(1.5, abs=0.2)


def test_all_results_are_statsmodels_duck_typed(panel):
    df = panel
    res = ws.PanelOLS.from_formula("y ~ x1 + x2", df, entity="firm", time="year").fit()
    # The attributes statsmodels downstream code relies on.
    assert isinstance(res.params, pd.Series)
    assert res.cov_params().shape == (len(res.params), len(res.params))
    assert res.conf_int().shape == (len(res.params), 2)
    assert res.nobs > 0
    assert isinstance(res.summary(), str)
