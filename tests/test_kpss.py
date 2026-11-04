"""Pin ``kpss`` against Stata's stationarity table.

Reference from a live ``kpss inf`` on macro_swatson.dta (n = 167)::

    KPSS test for inf
    Maxlag = 4 chosen by Schwert criterion
    Autocovariances weighted by Bartlett kernel
    Critical values for H0: inf is trend stationary
        10%: 0.119   5%: 0.146   2.5%: 0.176   1%: 0.216
    Lag order    Test statistic
        0           2.14
        1           1.16
        2           .811
        3           .625
        4           .514

Both row sets reproduce.

Two places where the obvious implementation is wrong, and both were wrong in westata before
this change:

1. **The default is TREND stationary.** A bare ``kpss varname`` prints "H0: ... is trend
   stationary" and uses the ``ct`` critical values; ``notrend`` asks for the level case.
   statsmodels defaults to ``'c'``, so the previous wrapper answered the other question and
   returned 0.3197 where Stata's lag-0 value is 2.14.

2. **The bandwidth coefficient is FOUR, not twelve.** KPSS uses
   ``int(4*(n/100)**0.25)``; the ADF/PP Schwert constant of 12 gives 13 for this sample, and
   printing thirteen rows where Stata prints five is the visible symptom.

Stata also reports a ROW PER LAG ORDER rather than one number, and the row it emphasises is
the maximum-lag one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_STATISTICS = {0: 2.14, 1: 1.16, 2: 0.811, 3: 0.625, 4: 0.514}


@pytest.fixture(scope="module")
def inflation():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return frame["inf"].dropna().to_numpy()


def test_every_lag_order_matches_stata(inflation) -> None:
    result = ws.kpss(inflation)
    observed = {row.lag: row.statistic for row in result.rows}
    assert set(observed) == set(STATA_STATISTICS)
    for lag, expected in STATA_STATISTICS.items():
        assert observed[lag] == pytest.approx(expected, abs=5e-3), lag


def test_the_default_is_trend_stationary(inflation) -> None:
    """``kpss inf`` prints "H0: ... is trend stationary"; notrend is the level case."""

    assert ws.kpss(inflation).trend is True
    assert ws.kpss(inflation, trend=False).trend is False


def test_the_bandwidth_uses_a_coefficient_of_four(inflation) -> None:
    """int(4*(n/100)**0.25) = 4; the ADF constant 12 would give 13 rows."""

    assert ws.kpss(inflation).maxlag == 4


def test_the_critical_values_match_stata(inflation) -> None:
    critical = ws.kpss(inflation).critical
    assert critical[0.10] == pytest.approx(0.119)
    assert critical[0.05] == pytest.approx(0.146)
    assert critical[0.025] == pytest.approx(0.176)
    assert critical[0.01] == pytest.approx(0.216)


def test_trend_and_level_differ_on_a_clearly_trending_series() -> None:
    """Guard against the two cases collapsing into one.

    On ``inf`` they happen to be nearly equal (0.5140 against 0.5139), so the two branches
    are compared on a series with an obvious trend, where detrending changes the answer a lot.
    """

    steps = np.arange(1, 201, dtype=float)
    trending = steps + np.random.default_rng(9).normal(size=steps.size)
    trended = ws.kpss(trending).statistic
    level = ws.kpss(trending, trend=False).statistic
    assert abs(trended - level) > 0.1, (trended, level)
