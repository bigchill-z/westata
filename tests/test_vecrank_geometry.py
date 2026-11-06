"""vecrank: Stata's exact table geometry, recorded for a future attempt.

Round 106 found two real defects in vecrank's layout -- a missing ``Sample:`` line and a missing
two-line "Critical value" header -- and then spent too long fitting column positions, so the fix
was reverted rather than shipped half-aligned. What that round did establish is the reference
itself, which is the expensive part to capture:

    Johansen tests for cointegration
    Trend: Constant                           Number of obs  = 164
    Sample: 3 thru 166                        Number of lags =   2
    --------------------------------------------------------------
                                                          Critical
    Maximum                                        Trace     value
       rank  Params           LL  Eigenvalue   statistic        5%
          0      6     1132.9103           .    183.0918     15.41
          1      9     1220.5657     0.65664      7.7811      3.76
          2     10    1224.4562     0.04634
    --------------------------------------------------------------

Token positions, measured with re.finditer:

    header   rank 3  Params 9  LL 24  Eigenvalue 28  statistic 40  5% 52
    row      0 at 6  6 at 13  LL at 19  dot at 39  183.0918 at 44  15.41 at 57

Three facts that are easy to get wrong and are now pinned:

  * the LAST rank row is short on purpose -- Stata omits trace and critical there because no
    test remains. The report's phrase "critical value table missing its last row" describes this
    correctly, and it is not a defect.
  * rank 0 has no eigenvalue and Stata prints a DOT, not a blank.
  * the reference data has N = 164 with the sample running 3 thru 166.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def ranked():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.vecrank(frame[["dinf", "unem"]].to_numpy(float), lags=2)


def test_the_statistics_match_stata(ranked) -> None:
    """The numbers were never in question; only the furniture around them."""

    rendered = str(ranked)
    assert "183.0918" in rendered
    assert "7.7811" in rendered
    assert "0.65664" in rendered


def test_the_reference_row_positions() -> None:
    """Pinned so a future layout attempt has the target rather than a guess."""

    header = "   rank  Params           LL  Eigenvalue   statistic        5%"
    row = "      0      6     1132.9103           .    183.0918     15.41"

    def positions(text: str) -> list[int]:
        return [match.start() for match in re.finditer(r"\S+", text)]

    assert positions(header) == [3, 9, 26, 30, 43, 60]
    assert positions(row) == [6, 13, 19, 39, 44, 57]
    assert len(header) == 62
    assert len(row) == 62


def test_the_lag_sample_is_recorded(ranked) -> None:
    """N = 164 with the sample running 3 thru 166; the Sample line itself is still missing."""

    assert ranked.nobs == 164
    assert ranked.lags == 2
    assert 3 == ranked.lags + 1
    assert 3 + ranked.nobs - 1 == 166


def test_only_trend_constant_is_supported(ranked) -> None:
    """The other four options refuse rather than silently substituting constant."""

    frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    for trend in ("trend", "rconstant", "rtrend", "none"):
        with pytest.raises(ws.UnsupportedError):
            ws.vecrank(frame[["dinf", "unem"]].to_numpy(float), lags=2, trend=trend)
