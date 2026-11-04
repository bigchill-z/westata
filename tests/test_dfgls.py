"""Pin ``dfgls`` against Stata in both the demeaned and detrended cases.

References from live runs on macro_swatson.dta (nobs 162)::

    dfgls inf, maxlag(4) notrend        dfgls inf, maxlag(4) trend
    [lags]   DF-GLS mu                  [lags]   DF-GLS tau
        4       -1.715                      4       -2.116
        3       -1.744                      3       -2.139
        2       -1.434                      2       -1.751
        1       -2.067                      1       -2.537

The detail that makes the lower-lag rows come out right: ``dfgls.ado:137`` marks the sample
out with ``L(1/maxlag+1)`` BEFORE looping, so every lag order is estimated on the same
observations. Letting each order use its own start gave -1.751 and -2.090 at lags 3 and 1
against Stata's -1.744 and -2.067 -- the longest lag was already exact, which is what made the
mistake look like a small numerical issue rather than a sample-alignment one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.postestimation.dfgls import dfgls

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_MU = {4: -1.715, 3: -1.744, 2: -1.434, 1: -2.067}
STATA_TAU = {4: -2.116, 3: -2.139, 2: -1.751, 1: -2.537}


@pytest.fixture(scope="module")
def inflation():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return frame["inf"].dropna().to_numpy()


def test_demeaned_statistics_match_stata(inflation) -> None:
    result = dfgls(inflation, maxlag=4, trend=False)
    observed = {cell.lags: cell.statistic for cell in result.cells}
    assert set(observed) == set(STATA_MU)
    for lags, expected in STATA_MU.items():
        assert observed[lags] == pytest.approx(expected, abs=5e-4), f"lag {lags}"


def test_detrended_statistics_match_stata(inflation) -> None:
    result = dfgls(inflation, maxlag=4, trend=True)
    observed = {cell.lags: cell.statistic for cell in result.cells}
    for lags, expected in STATA_TAU.items():
        assert observed[lags] == pytest.approx(expected, abs=5e-4), f"lag {lags}"


def test_every_lag_uses_the_same_sample(inflation) -> None:
    """dfgls.ado:137 marks the sample out with L(1/maxlag+1) before looping."""

    result = dfgls(inflation, maxlag=4)
    assert len({cell.nobs for cell in result.cells}) == 1


def test_the_default_maxlag_is_schwert(inflation) -> None:
    """int(12*(n/100)**0.25) = 13 for 166 observations."""

    result = dfgls(inflation)
    assert max(cell.lags for cell in result.cells) == 13


def test_ordering_runs_from_the_longest_lag_down(inflation) -> None:
    result = dfgls(inflation, maxlag=4)
    assert [cell.lags for cell in result.cells] == [4, 3, 2, 1]
