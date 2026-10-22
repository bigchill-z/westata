"""Tests for the panel non-linear models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.panel_nonlinear import (
    XtLogit,
    XtNbreg,
    XtPoisson,
    XtProbit,
    XtTobit,
)


def _binary(n_firms: int = 120, n_years: int = 8, seed: int = 71) -> pd.DataFrame:
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


def _count(n_firms: int = 100, n_years: int = 7, seed: int = 73) -> pd.DataFrame:
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


def _censored(n_firms: int = 100, n_years: int = 7, seed: int = 83) -> pd.DataFrame:
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


# ------------------------------------------------------------------ xtlogit


def test_xtlogit_fe_matches_stata():
    result = XtLogit.from_formula(
        "y ~ x1 + x2", _binary(), entity="firm", method="fe"
    ).fit()

    # Stata: x1 .3307537 (se .0807304), x2 -.3134259 (se .0795823)
    assert float(result.params["x1"]) == pytest.approx(0.3307537, rel=1e-4)
    assert float(result.params["x2"]) == pytest.approx(-0.3134259, rel=1e-4)


def test_xtlogit_fe_uses_exactly_912_observations():
    """Groups whose outcome never varies carry no information and are dropped."""

    result = XtLogit.from_formula(
        "y ~ x1 + x2", _binary(), entity="firm", method="fe"
    ).fit()
    assert int(result.e["N_used"]) == 912


def test_xtlogit_fe_drops_a_constant_group():
    frame = _binary(n_firms=40, n_years=6)
    # Force one firm to have an all-zero outcome.
    frame.loc[frame["firm"] == 0, "y"] = 0.0
    result = XtLogit.from_formula(
        "y ~ x1 + x2", frame, entity="firm", method="fe"
    ).fit()
    # The excluded firm had 6 observations, so the used sample must be smaller.
    total_varying = sum(
        6
        for _, group in frame.groupby("firm")
        if group["y"].min() < 0.5 < group["y"].max()
    )
    assert int(result.e["N_used"]) == total_varying


def test_xtlogit_fe_runs_on_data_without_variation():
    frame = _binary(n_firms=30, n_years=4)
    frame["y"] = 1.0
    with pytest.raises(ConsistencyError, match="no group has variation"):
        XtLogit.from_formula(
            "y ~ x1", frame, entity="firm", method="fe"
        ).fit()


def test_xtlogit_re_runs_and_reports_sigma_u():
    result = XtLogit.from_formula(
        "y ~ x1 + x2", _binary(), entity="firm", method="re"
    ).fit()
    assert np.isfinite(float(result.params["x1"]))
    assert float(result.e["sigma_u"]) > 0


def test_xtlogit_pa_reports_no_likelihood():
    """pa is GEE, not a likelihood; e(ll) must be absent."""

    result = XtLogit.from_formula(
        "y ~ x1 + x2", _binary(), entity="firm", method="pa"
    ).fit()
    assert "ll" not in result.e
    assert result.e["estimator"] == "gee"


def test_xtlogit_rejects_an_unknown_method():
    with pytest.raises(UnsupportedError, match="fe"):
        XtLogit.from_formula(
            "y ~ x1", _binary(), entity="firm", method="between"
        )


# ----------------------------------------------------------------- xtprobit


def test_xtprobit_re_runs():
    result = XtProbit.from_formula(
        "y ~ x1 + x2", _binary(), entity="firm", method="re"
    ).fit()
    assert np.isfinite(float(result.params["x1"]))
    assert "ll" in result.e


def test_xtprobit_rejects_fe():
    with pytest.raises(UnsupportedError, match="re"):
        XtProbit.from_formula("y ~ x1", _binary(), entity="firm", method="fe")


# ------------------------------------------------------------------ xttobit


def test_xttobit_runs_and_reports_both_scales():
    result = XtTobit.from_formula(
        "y ~ x1", _censored(), entity="firm", lower=0.5
    ).fit()
    assert np.isfinite(float(result.params["x1"]))
    assert float(result.e["sigma_u"]) > 0
    assert float(result.e["sigma_e"]) > 0


# ---------------------------------------------------------------- xtpoisson


def test_xtpoisson_fe_matches_stata():
    result = XtPoisson.from_formula(
        "y ~ x1", _count(), entity="firm", method="fe"
    ).fit()
    # Stata: x1 .5041803 (se .0344352)
    assert float(result.params["x1"]) == pytest.approx(0.5041803, rel=5e-2)


def test_xtpoisson_fe_intercept_is_not_identified():
    result = XtPoisson.from_formula(
        "y ~ x1", _count(), entity="firm", method="fe"
    ).fit()
    # Conditioning on the group totals absorbs the constant.
    assert np.isnan(float(result.params["Intercept"]))


def test_xtpoisson_re_runs():
    result = XtPoisson.from_formula(
        "y ~ x1", _count(), entity="firm", method="re"
    ).fit()
    assert np.isfinite(float(result.params["x1"]))


def test_xtpoisson_rejects_an_unknown_method():
    with pytest.raises(UnsupportedError, match="fe"):
        XtPoisson.from_formula("y ~ x1", _count(), entity="firm", method="pa")


# ------------------------------------------------------------------ xtnbreg


def test_xtnbreg_fe_runs_and_flags_the_quasi_likelihood():
    result = XtNbreg.from_formula(
        "y ~ x1", _count(), entity="firm", method="fe"
    ).fit()
    assert np.isfinite(float(result.params["x1"]))
    assert result.e["estimator"] == "conditional_quasi_ml"
    assert result.e["quasi_likelihood"] == 1
    assert "/lnalpha" in result.param_names


def test_xtnbreg_rejects_re():
    with pytest.raises(UnsupportedError, match="fe"):
        XtNbreg.from_formula("y ~ x1", _count(), entity="firm", method="re")
