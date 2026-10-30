"""Tests for `reghdfe`'s within-decomposition block.

``reghdfe`` reports three quantities that are defined against a *different*
total than the ordinary R-squared:

    tss_within    = sum of squares of y after removing ONLY the fixed effects
    r2_within     = 1 - rss/tss_within
    r2_a_within   = 1 - (1 - r2_within)(N-1)/(N - rank)

The identity is confirmed by the fixture's own numbers:

    1 - 131.2293504838976/418.6969854497428 = 0.6865767964798271

which is exactly `reghdfe_firm.json`'s ``e(r2_within)``.  So `tss_within` is
the residual sum of squares of the intercept-only absorbed model ``y ~ 1 | fes``
— not the centred total (which gives the ordinary ``e(r2)``) and not the
within-model total of a demeaned *regression*.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURE = Path("tests/conformance/stata_fixtures/reghdfe_firm.json")

ENTITIES, PERIODS = 30, 6


@pytest.fixture()
def panel():
    rng = np.random.default_rng(20261022)
    firm = np.repeat(np.arange(ENTITIES), PERIODS)
    year = np.tile(np.arange(PERIODS), ENTITIES)
    n = firm.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(scale=1.0, size=ENTITIES)[firm]
    y = 2.0 + 1.5 * x1 - 0.7 * x2 + effect + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "year": year})


def test_the_fixture_identity_holds_on_statas_own_numbers() -> None:
    """Arithmetic check independent of our data."""

    s = json.loads(FIXTURE.read_text())["scalars"]

    assert 1.0 - s["rss"] / s["tss_within"] == pytest.approx(
        s["r2_within"], rel=1e-12
    )
    assert 1.0 - s["rss"] / s["tss"] == pytest.approx(s["r2"], rel=1e-12)


def test_tss_within_equals_the_fe_only_residual_sum_of_squares(panel) -> None:
    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()
    tss_within = result.e["tss_within"]

    # Independently: demean y by entity means and take the sum of squares.
    demeaned = panel["y"] - panel.groupby("firm")["y"].transform("mean")
    expected = float((demeaned**2).sum())

    assert tss_within == pytest.approx(expected, rel=1e-10)


def test_tss_within_differs_from_the_centred_total(panel) -> None:
    """A common mistake is to reuse the ordinary total sum of squares."""

    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()

    assert not np.isclose(result.e["tss_within"], result.e["tss"], rtol=1e-6)


def test_r2_within_is_one_minus_rss_over_tss_within(panel) -> None:
    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()
    e = result.e

    assert e["r2_within"] == pytest.approx(
        1.0 - e["rss"] / e["tss_within"], rel=1e-12
    )


def test_r2_within_differs_from_r2(panel) -> None:
    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()

    assert not np.isclose(result.e["r2_within"], result.e["r2"], rtol=1e-6)


def test_r2_a_within_matches_its_definition(panel) -> None:
    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()
    e = result.e

    expected = 1.0 - (1.0 - e["r2_within"]) * (e["N"] - 1.0) / e["df_r"]
    assert e["r2_a_within"] == pytest.approx(expected, rel=1e-10)


def test_two_way_absorption_still_produces_the_block(panel) -> None:
    """Multi-way absorption must not fall back to one demeaning pass."""

    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm + year", panel).fit()

    assert "tss_within" in result.e
    assert np.isfinite(result.e["tss_within"])
    assert result.e["tss_within"] > 0.0


def test_two_way_tss_within_is_below_one_way(panel) -> None:
    """Absorbing more effects can only shrink the residual sum of squares."""

    one_way = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()
    two_way = ws.FEOLS.from_formula("y ~ x1 + x2 | firm + year", panel).fit()

    assert two_way.e["tss_within"] <= one_way.e["tss_within"]


# ------------------------------------------------------- the other added keys
def test_sumweights_mirrors_sum_w(panel) -> None:
    """``reghdfe`` spells ``e(sum_w)`` as ``e(sumweights)`` too."""

    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()

    assert result.e["sumweights"] == result.e["sum_w"]
    assert result.e["sumweights"] == float(result.nobs)


def test_converged_and_ic_are_recorded(panel) -> None:
    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()

    assert result.e["converged"] == 1.0
    assert result.e["ic"] == 1.0


def test_report_constant_is_zero_without_a_reported_constant(panel) -> None:
    """The absorbed effects subsume the constant, so none is reported."""

    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()

    assert result.e["report_constant"] == 0.0


def test_the_absorbed_bookkeeping_block_is_recorded(panel) -> None:
    """These eight keys used to be recorded as unfakeable.  They are not.

    ``reghdfe_firm_year.json`` pins the identities down:

        df_a_initial = 36     (30 firms + 6 years)
        df_a         = 35
        df_a_nested  = 0
        df_a_redundant = 1
        df_a_initial - df_a == df_a_nested + df_a_redundant == 1
        N_hdfe = 2            (two absorbed dimensions)
        num_singletons = 0    (no observation is alone in its level)

    so every value follows from the absorbed specification and the sample.  Round
    23 deferred them believing they needed reghdfe's iterative solver; they do not.
    """

    one_way = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", panel).fit()
    two_way = ws.FEOLS.from_formula("y ~ x1 + x2 | firm + year", panel).fit()

    for key in (
        "N_hdfe",
        "N_hdfe_extended",
        "N_full",
        "df_a_initial",
        "df_a_nested",
        "df_a_redundant",
        "drop_singletons",
        "num_singletons",
    ):
        assert key in one_way.e, key

    # The identities.
    assert one_way.e["N_hdfe"] == 1.0
    assert two_way.e["N_hdfe"] == 2.0
    for result in (one_way, two_way):
        e = result.e
        assert e["df_a"] == e["df_a_initial"] - (
            e["df_a_nested"] + e["df_a_redundant"]
        )
        assert e["num_singletons"] == 0.0
        assert e["N_full"] == e["N"]
    # A two-way absorption has exactly one redundant partition (the shared
    # constant), so df_a is one below the raw level count.
    assert two_way.e["df_a_redundant"] == 1.0
    assert one_way.e["df_a_redundant"] == 0.0
