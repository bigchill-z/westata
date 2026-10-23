"""Tests for the dynamic panel GMM estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.panel_dynamic import XtAbond, build_instruments


def _dynamic_panel(n_firms: int = 90, n_years: int = 9, seed: int = 79) -> pd.DataFrame:
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


# ------------------------------------------------------------ instruments


def test_instrument_count_matches_the_documented_formula():
    """For p lags and T periods, Z has sum_{m=p}^{T-2} m columns."""

    for periods, lags in [(9, 1), (9, 2), (12, 1), (8, 3)]:
        matrix = build_instruments(np.arange(periods, dtype=float), lags)
        expected = sum(range(lags, periods - 1))
        assert matrix.shape[1] == expected, (periods, lags)
        assert matrix.shape[0] == periods - lags - 1


def test_instrument_columns_account_for_statas_total():
    """Stata reports 30 instruments for xtabond with lags(1) on T=9.

    The GMM block contributes 28 columns and the two standard instruments (the
    differenced covariate and the constant) account for the other two. That exact
    accounting is the check that the block matrix is right.
    """

    matrix = build_instruments(np.arange(9, dtype=float), 1)
    assert matrix.shape[1] == 28
    # The estimator adds D.x1 and a constant, giving Stata's 30.
    assert matrix.shape[1] + 2 == 30


def test_instrument_matrix_is_lower_triangular_in_the_lag_block():
    """Row t holds y_1..y_{t-1}, with zeros to the right."""

    series = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    matrix = build_instruments(series, 1)

    # p=1, T=5 -> 3 rows and 1+2+3 = 6 columns.
    assert matrix.shape == (3, 6)
    # The valid moment conditions are E[y_{i,t-2-j} dEps_it] = 0, so the
    # instruments for period t are y_1..y_{t-2} and they START AT y_1.
    # Row 0 is period 3 and uses y_1 only.
    assert matrix[0, 0] == 10.0
    assert np.all(matrix[0, 1:] == 0.0)
    # Row 1 is period 4 and uses y_1, y_2.
    assert list(matrix[1, 1:3]) == [10.0, 11.0]
    assert np.all(matrix[1, 3:] == 0.0)
    # Row 2 is period 5 and uses y_1, y_2, y_3.
    assert list(matrix[2, 3:6]) == [10.0, 11.0, 12.0]


def test_instrument_matrix_rejects_too_few_periods():
    with pytest.raises(ConsistencyError, match="too short"):
        build_instruments(np.arange(3, dtype=float), 2)


def test_instrument_matrix_rejects_zero_lags():
    with pytest.raises(ValueError, match="lags"):
        build_instruments(np.arange(10, dtype=float), 0)


# ---------------------------------------------------------------- xtabond


def test_xtabond_recovers_a_persistent_series():
    """The DGP has an AR coefficient of 0.5; one-step GMM should land near it."""

    frame = _dynamic_panel()
    result = XtAbond.from_formula(
        "y ~ x1", frame, entity="firm", time="year", lags=1
    ).fit()

    lag_name = result.param_names[0]
    assert lag_name == "L1.y"
    # The within estimator is biased downward on a short panel, and one-step GMM
    # is noisy, so a generous band is appropriate; a wrong instrument matrix would
    # not be anywhere near this.
    assert 0.35 < float(result.params[lag_name]) < 0.60


def test_xtabond_reports_the_instrument_count():
    frame = _dynamic_panel()
    result = XtAbond.from_formula(
        "y ~ x1", frame, entity="firm", time="year", lags=1
    ).fit()

    # 28 GMM columns plus the two standard instruments Stata counts (D.x1 and
    # the constant), matching Stata's 30 exactly.
    assert int(result.e["instruments"]) == 30


def test_xtabond_lag_two_reduces_the_instrument_count():
    frame = _dynamic_panel()
    one = XtAbond.from_formula(
        "y ~ x1", frame, entity="firm", time="year", lags=1
    ).fit()
    two = XtAbond.from_formula(
        "y ~ x1", frame, entity="firm", time="year", lags=2
    ).fit()
    assert int(two.e["instruments"]) < int(one.e["instruments"])


def test_xtabond_does_not_report_sargan_or_hansen():
    """Those statistics are omitted rather than approximated.

    Their constructions are not fully specified in the manual, and a wrong
    overidentification statistic would look more authoritative than its absence.
    """

    frame = _dynamic_panel()
    result = XtAbond.from_formula(
        "y ~ x1", frame, entity="firm", time="year", lags=1
    ).fit()
    for key in ("sargan", "hansen", "ar1", "ar2"):
        assert key not in result.e


def test_xtabond_rejects_an_unbalanced_panel():
    frame = _dynamic_panel(n_firms=20, n_years=6)
    frame = frame.drop(index=frame.index[3]).reset_index(drop=True)
    with pytest.raises(UnsupportedError, match="balanced"):
        XtAbond.from_formula("y ~ x1", frame, entity="firm", time="year")


def test_xtabond_rejects_too_short_a_panel():
    frame = _dynamic_panel(n_firms=20, n_years=3)
    with pytest.raises(ConsistencyError):
        XtAbond.from_formula(
            "y ~ x1", frame, entity="firm", time="year", lags=2
        ).fit()
