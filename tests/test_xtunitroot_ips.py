"""Tests for the Im-Pesaran-Shin panel unit-root test."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.xtunitroot_ips import (
    _IPS_MEAN_VAR,
    _IPS_TILDE_MEAN_VAR,
    _lookup,
    im_pesaran_shin,
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


# ------------------------------------------------------------ table lookup


def test_lookup_returns_the_tabulated_values_exactly():
    """At a tabulated T the interpolation must be exact, not approximate."""

    for periods, mean_value, var_value in _IPS_MEAN_VAR[:-1]:
        got_mean, got_var = _lookup(_IPS_MEAN_VAR, periods)
        assert got_mean == pytest.approx(mean_value, abs=1e-12)
        assert got_var == pytest.approx(var_value, abs=1e-12)


def test_lookup_interpolates_between_tabulated_periods():
    """T=12 sits between the T=10 and T=15 rows."""

    mean_value, var_value = _lookup(_IPS_MEAN_VAR, 12.0)
    # Linear between -1.504 and -1.514, weighted 2/5 of the way from 10 to 15.
    expected_mean = -1.504 + (12 - 10) / (15 - 10) * (-1.514 - -1.504)
    expected_var = 1.069 + (12 - 10) / (15 - 10) * (0.923 - 1.069)
    assert mean_value == pytest.approx(expected_mean, abs=1e-12)
    assert var_value == pytest.approx(expected_var, abs=1e-12)


def test_lookup_clamps_below_the_first_row_and_above_100():
    low_mean, low_var = _lookup(_IPS_MEAN_VAR, 3.0)
    assert low_mean == pytest.approx(_IPS_MEAN_VAR[0][1])
    assert low_var == pytest.approx(_IPS_MEAN_VAR[0][2])

    high_mean, _ = _lookup(_IPS_MEAN_VAR, 5000.0)
    assert high_mean == pytest.approx(_IPS_MEAN_VAR[-2][1])


def test_both_tables_have_the_same_period_grid():
    left = [row[0] for row in _IPS_MEAN_VAR]
    right = [row[0] for row in _IPS_TILDE_MEAN_VAR]
    assert left == right


# -------------------------------------------------------- conformance


STATA = {
    "tbar": -2.2259105396588366,
    "ttildebar": -1.7875342227529116,
    "zt": -3.9114817625880884,
    "zttildebar": -3.6087323273920875,
    "pvalue": 0.00015384844272522537,
}


def test_ips_tbar_matches_stata():
    result = im_pesaran_shin(_panel(), variable="y", entity="firm", time="year")
    assert result.tbar == pytest.approx(STATA["tbar"], rel=1e-12)


def test_ips_ttildebar_matches_stata():
    """The t-tilde scale is the part that is easy to invert.

    Stata writes ``b / (se/rmse) * sd``, which evaluates as ``(b/se)*(rmse/sd)``. Using
    the reciprocal gives a value 58% larger, so this assertion is the guard.
    """

    result = im_pesaran_shin(_panel(), variable="y", entity="firm", time="year")
    assert result.ttildebar == pytest.approx(STATA["ttildebar"], rel=1e-12)


def test_ips_zt_matches_stata():
    result = im_pesaran_shin(_panel(), variable="y", entity="firm", time="year")
    assert result.zt == pytest.approx(STATA["zt"], rel=1e-12)


def test_ips_zttildebar_and_pvalue_match_stata():
    result = im_pesaran_shin(_panel(), variable="y", entity="firm", time="year")
    assert result.zttildebar == pytest.approx(STATA["zttildebar"], rel=1e-12)
    assert result.pvalue == pytest.approx(STATA["pvalue"], rel=1e-10)


def test_ips_reports_the_sample_shape():
    result = im_pesaran_shin(_panel(), variable="y", entity="firm", time="year")
    assert result.n_groups == 30
    assert result.n_periods == 12
    assert result.observations == 330


# ------------------------------------------------------------- behaviour


def test_a_stationary_panel_rejects_the_unit_root():
    """Strongly mean-reverting panels must give a strongly negative Z."""

    rng = np.random.default_rng(7)
    rows = []
    for firm in range(1, 26):
        value = rng.normal()
        for year in range(1, 16):
            value = 0.2 * value + rng.normal()
            rows.append((firm, year, value))
    frame = pd.DataFrame(rows, columns=["firm", "year", "y"])

    result = im_pesaran_shin(frame, variable="y", entity="firm", time="year")
    assert result.zttildebar < -3.0
    assert result.pvalue < 0.01


def test_random_walks_do_not_reject():
    rng = np.random.default_rng(11)
    rows = []
    for firm in range(1, 26):
        value = 0.0
        for year in range(1, 16):
            value = value + rng.normal()
            rows.append((firm, year, value))
    frame = pd.DataFrame(rows, columns=["firm", "year", "y"])

    result = im_pesaran_shin(frame, variable="y", entity="firm", time="year")
    assert result.pvalue > 0.05


def test_trend_and_demean_options_run():
    frame = _panel()
    trend = im_pesaran_shin(
        frame, variable="y", entity="firm", time="year", trend=True
    )
    demeaned = im_pesaran_shin(
        frame, variable="y", entity="firm", time="year", demean=True
    )
    # Both must differ from the plain specification.
    plain = im_pesaran_shin(frame, variable="y", entity="firm", time="year")
    assert trend.zt != pytest.approx(plain.zt, rel=1e-6)
    assert demeaned.zt != pytest.approx(plain.zt, rel=1e-6)


def test_lags_option_runs():
    result = im_pesaran_shin(
        _panel(), variable="y", entity="firm", time="year", lags=1
    )
    assert np.isfinite(result.zt)


def test_a_missing_column_raises():
    with pytest.raises(ConsistencyError, match="not found"):
        im_pesaran_shin(_panel(), variable="nope", entity="firm", time="year")


def test_the_exact_critical_values_are_refused_rather_than_faked():
    """IPS Table 2 is a large N-by-T grid that has not been transcribed.

    Returning a plausible-looking number would be worse than refusing, because the
    critical value is what a user reads to decide the test.
    """

    with pytest.raises(UnsupportedError, match="Table 2"):
        from westata.estimators.xtunitroot_ips import _critical_values

        _critical_values(30, 12, trend=False)
