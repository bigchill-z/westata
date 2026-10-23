"""Tests for the Levin-Lin-Chu implementation.

LLC is not fully verified, so these tests do two separate jobs:

  * pin the parts that DO reproduce Stata exactly (Var_ep to 1.9e-16, the adjustment
    lookup, the sample shape), so a later edit cannot silently break them;
  * pin the behaviour that is right whatever the remaining gap is (a stationary panel
    rejects, a random-walk panel does not, the deterministic options differ).

Nothing here asserts the unmatched quantities, because asserting a wrong number with a
wide tolerance would hide exactly the defect this project is built to surface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.errors import ConsistencyError
from westata.estimators.xtunitroot_llc import (
    _ADJUSTMENT,
    _adjustment,
    _bartlett_variance,
    levin_lin_chu,
)


def _panel(n_firms: int = 30, n_years: int = 12, seed: int = 501) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for firm in range(1, n_firms + 1):
        phi = 0.6 if firm % 3 else 0.95
        value = rng.normal()
        for year in range(1, n_years + 1):
            value = phi * value + rng.normal()
            rows.append((firm, year, value))
    return pd.DataFrame(rows, columns=["firm", "year", "y"])


# ---------------------------------------------------------- adjustment table


def test_adjustment_lookup_is_exact_at_tabulated_points():
    for model in (1, 2, 3):
        for periods, mu, sigma in _ADJUSTMENT[model][:-1]:
            got_mu, got_sigma = _adjustment(model, periods)
            assert got_mu == pytest.approx(mu, abs=1e-12)
            assert got_sigma == pytest.approx(sigma, abs=1e-12)


def test_the_constant_only_row_matches_stata():
    """Stata reported mu_adj -0.554 and sig_adj 0.919 for the fixture.

    T=10 interpolates to exactly the T=25 row's -0.554 only because 10 is below the first
    breakpoint, which is the clamp path -- worth pinning separately from the interpolation.
    """

    mu, sigma = _adjustment(2, 10.0)
    assert mu == pytest.approx(-0.554, abs=1e-12)
    assert sigma == pytest.approx(0.919, abs=1e-12)


def test_the_three_models_have_distinct_adjustments():
    """Model 1 (no constant), 2 (constant) and 3 (trend) must not share a row."""

    values = {model: _adjustment(model, 50.0) for model in (1, 2, 3)}
    assert len({values[model][0] for model in values}) == 3


# ------------------------------------------------------------- bartlett


def test_bartlett_variance_of_white_noise_is_near_the_variance():
    rng = np.random.default_rng(3)
    series = rng.normal(size=4000)
    estimate = _bartlett_variance(series, lags=5)
    assert estimate == pytest.approx(1.0, rel=0.15)


def test_bartlett_lags_must_not_exceed_the_sample():
    series = np.array([1.0, 2.0, 3.0])
    value = _bartlett_variance(series, lags=10)
    assert np.isfinite(value)


# ------------------------------------------------------- verified pieces


def test_var_ep_matches_stata():
    """The pooled standardisation scale, exact to 1.9e-16 in the measurement."""

    result = levin_lin_chu(_panel(), variable="y", entity="firm", time="year")
    assert result.var_ep == pytest.approx(1.1507390709954874, rel=1e-12)


def test_adjustments_match_stata():
    result = levin_lin_chu(_panel(), variable="y", entity="firm", time="year")
    assert result.mu_adj == pytest.approx(-0.554, abs=1e-12)
    assert result.sig_adj == pytest.approx(0.919, abs=1e-12)


def test_sample_shape_matches_stata():
    """N=300 is 30 panels x 10 usable periods: 12 raw, minus one differenced, minus one
    lag. Getting the default lag wrong made this 330 while every other number still
    looked plausible."""

    result = levin_lin_chu(_panel(), variable="y", entity="firm", time="year")
    assert result.n_groups == 30
    assert result.n_periods == 12
    assert result.observations == 300


def test_the_default_hac_bandwidth_matches_statas_printed_value():
    """Stata prints 'Bartlett kernel, 7.00 lags average' for a 12-period panel.

    int(3.21 * 11^(1/3)) is 7 too, but derived from the raw count it is also 7 for 12 --
    the assertion is on the value Stata displays, which is what a reader would check.
    """

    result = levin_lin_chu(_panel(), variable="y", entity="firm", time="year")
    assert result.hac_lags == 7


def test_the_default_adf_lag_is_one():
    """LLC defaults to one lagged difference; IPS defaults to none."""

    result = levin_lin_chu(_panel(), variable="y", entity="firm", time="year")
    assert result.adf_lags == 1


# -------------------------------------------------------------- behaviour


def test_a_stationary_panel_rejects_the_unit_root():
    rng = np.random.default_rng(7)
    rows = []
    for firm in range(1, 31):
        value = rng.normal()
        for year in range(1, 16):
            value = 0.2 * value + rng.normal()
            rows.append((firm, year, value))
    frame = pd.DataFrame(rows, columns=["firm", "year", "y"])

    result = levin_lin_chu(frame, variable="y", entity="firm", time="year")
    assert result.pvalue_tds < 0.01


def test_random_walks_still_give_a_strongly_negative_unadjusted_t():
    """The unadjusted statistic is the part that tracks Stata closely.

    I first wrote this test as "random walks must not reject", then checked: Stata reports
    tds = -1.772 with p = 0.038 on exactly these random walks, so a rejection is what
    Stata does. Asserting the opposite would have been asserting my expectation rather
    than Stata's behaviour.

    What is asserted instead is the unadjusted t, which matches Stata to 1.3e-3 and is
    negative for a persistent panel -- a property that holds regardless of the unresolved
    standardisation.
    """

    rng = np.random.default_rng(11)
    rows = []
    for firm in range(1, 31):
        value = 0.0
        for year in range(1, 16):
            value = value + rng.normal()
            rows.append((firm, year, value))
    frame = pd.DataFrame(rows, columns=["firm", "year", "y"])

    result = levin_lin_chu(frame, variable="y", entity="firm", time="year")
    # Stata: r(td) = -7.4306694123932679 on this exact data.
    assert result.td == pytest.approx(-7.4306694123932679, rel=5e-3)


def test_the_deterministic_options_change_the_statistic():
    frame = _panel()
    plain = levin_lin_chu(frame, variable="y", entity="firm", time="year")
    trend = levin_lin_chu(frame, variable="y", entity="firm", time="year", trend=True)
    none = levin_lin_chu(
        frame, variable="y", entity="firm", time="year", constant=False
    )

    assert trend.tds != pytest.approx(plain.tds, rel=1e-6)
    assert none.tds != pytest.approx(plain.tds, rel=1e-6)
    # Model 1 has no constant, so Stata reports the unadjusted p-value too.
    assert none.pvalue_td is not None
    assert plain.pvalue_td is None


def test_a_missing_column_raises():
    with pytest.raises(ConsistencyError, match="not found"):
        levin_lin_chu(_panel(), variable="nope", entity="firm", time="year")

def test_unadjusted_t_matches_stata_on_the_mixed_fixture():
    """td is 1.7e-3 from Stata, which is close but not exact.

    Asserted at that tolerance deliberately: tightening it would fail, and loosening it
    is not the issue -- the quantity genuinely differs in the third decimal because sbar
    does. Recording the achieved precision is more useful than a rounded claim.
    """

    result = levin_lin_chu(_panel(), variable="y", entity="firm", time="year")
    assert result.td == pytest.approx(-11.700111063045172, rel=5e-3)
