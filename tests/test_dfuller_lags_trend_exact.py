"""ADF with lag augmentation and a trend: `dfuller y, lags(9) trend`.

C14-02 scores this at 0.0 while the plain `dfuller` (C14-01) also scored 0.0 and turned out exact. Measured
on macro_swatson's inflation series:

    ws.dfuller(y, trend=True, lags=9)      Z(t) -5.09291757   N 156   lags 9
    quietly dfuller y, lags(9) trend       r(Zt) -5.09291720   r(N) 156   r(lags) 9

The statistics agree to seven significant figures; the residual 3.7e-7 is the two implementations summing
the same regression in a different order, not a specification difference.

THE AUGMENTATION COUNT IS PART OF THE GUARD because it is what the option changes. `lags(9)` consumes nine
observations at the start, so N is 156 rather than the 165 an unaugmented fit would use -- a package that
ignored the option would return the right-looking Z(t) on the wrong sample.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

MACRO = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def series():
    try:
        frame = pd.read_stata(MACRO).dropna(subset=["dinf"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("macro_swatson.dta is not available")
    return frame["dinf"].to_numpy()


def test_the_augmented_statistic_matches_stata(series) -> None:
    result = ws.dfuller(series, trend=True, lags=9)
    assert float(result.statistic) == pytest.approx(-5.09291720, abs=1e-5)


def test_the_observation_count_reflects_the_augmentation(series) -> None:
    """Nine lags consume nine observations: 156, not the 165 of an unaugmented fit."""

    result = ws.dfuller(series, trend=True, lags=9)
    assert int(result.observations) == 156


def test_the_lag_count_is_reported(series) -> None:
    result = ws.dfuller(series, trend=True, lags=9)
    assert int(result.r["lags"]) == 9


def test_more_lags_use_fewer_observations(series) -> None:
    """If they did not, the option would be decorative."""

    short = ws.dfuller(series, trend=True, lags=1)
    long = ws.dfuller(series, trend=True, lags=9)
    assert int(long.observations) < int(short.observations)


def test_the_critical_values_are_ordered(series) -> None:
    """1%, 5% and 10% cutoffs must be the right way round for a left-tailed test."""

    result = ws.dfuller(series, trend=True, lags=9)
    one, five, ten = (
        float(result.r["cv_1"]),
        float(result.r["cv_5"]),
        float(result.r["cv_10"]),
    )
    assert one < five < ten


def test_a_trend_changes_the_statistic(series) -> None:
    """The trend option must enter the regression, not just the critical values."""

    with_trend = ws.dfuller(series, trend=True, lags=1)
    without = ws.dfuller(series, trend=False, lags=1)
    assert float(with_trend.statistic) != pytest.approx(
        float(without.statistic), rel=1e-6
    )
