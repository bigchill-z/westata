"""vecrank renders Stata's table byte for byte.

Round 106 found the defects and reverted the fix rather than ship a half-aligned table; round 107
solved the column widths arithmetically from the target token positions instead of by trial and
error, and landed it.

Against ``vecrank dinf unem, lags(2)``::

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

eleven of eleven lines identical.

THE WIDTHS, derived rather than guessed (the reason round 106 failed): each field's width is the
distance between consecutive target token ends, not a number to be tuned.

    header    rank 7   Params 8   LL 13   Eigenvalue 12   statistic 12   5% 10
    rows      rank 7   params 7   LL 14   eigenvalue 12   trace 12   crit 10
    last row  rank 7   params 7   LL 13   eigenvalue 12      <- LL is one narrower, as Stata prints

Three facts the table depends on, each of which is easy to "fix" wrongly:

  * the LAST rank row is short on purpose -- no trace, no critical, because no test remains;
  * rank 0 prints a DOT for the eigenvalue, not a blank;
  * the other four trend options still refuse rather than silently substituting constant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_LINES = [
    "Johansen tests for cointegration",
    "Trend: Constant                           Number of obs  = 164",
    "Sample: 3 thru 166                        Number of lags =   2",
    "-" * 62,
    " " * 54 + "Critical",
    "Maximum" + " " * 40 + "Trace" + " " * 5 + "value",
    "   rank  Params           LL  Eigenvalue   statistic        5%",
    "-" * 62,
    "      0      6     1132.9103           .    183.0918     15.41",
    "      1      9     1220.5657     0.65664      7.7811      3.76",
    "      2     10    1224.4562     0.04634",
    "-" * 62,
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return str(ws.vecrank(frame[["dinf", "unem"]].to_numpy(float), lags=2))


def test_every_line_matches_stata(rendered) -> None:
    lines = rendered.splitlines()
    for expected in STATA_LINES:
        assert expected in lines, repr(expected)


def test_the_table_rows_are_sixty_two_characters(rendered) -> None:
    """Except the last rank row, which Stata prints short at 39 characters."""

    for line in rendered.splitlines():
        if line.startswith("Johansen") or not line:
            continue
        if line.startswith("      2     10"):
            assert len(line) == 39, (len(line), line)
            continue
        assert len(line) == 62, (len(line), line)


def test_the_short_last_row_has_no_trace_or_critical(rendered) -> None:
    """Stata omits them because no test remains -- not a truncation bug."""

    last = rendered.splitlines()[-2]
    assert last.startswith("      2     10")
    assert len(last) == 39
    assert "0.04634" in last


def test_rank_zero_prints_a_dot_for_the_eigenvalue(rendered) -> None:
    row = next(line for line in rendered.splitlines() if line.strip().startswith("0 "))
    assert " . " in row


def test_the_sample_line_states_the_lag_window(rendered) -> None:
    assert "Sample: 3 thru 166" in rendered


def test_trend_is_now_supported_and_the_rest_still_refuse() -> None:
    """Round 136 implemented ``trend(trend)``; the restricted variants still refuse.

    ``trend`` was in this loop until round 136 because the deterministic block was hard-coded to a
    column of ones. Adding a linear trend inside the cointegrating relation reproduces Stata's
    eigenvalues, trace, likelihood, parameter counts AND critical values exactly, so it moved out
    of the refusal list. ``rconstant``, ``rtrend`` and ``none`` each use a different tabulation and
    still change which rank would be chosen, so they stay refused.
    """

    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].to_numpy(float)

    # Supported now.
    supported = ws.vecrank(values, lags=2, trend="trend")
    assert supported.deterministic == "trend"
    assert supported.rows[0].params == 8  # Stata: 8 / 11 / 12, against 6 / 9 / 10 for constant

    # Still refused, with a reason.
    for trend in ("rconstant", "rtrend", "none"):
        with pytest.raises(ws.UnsupportedError):
            ws.vecrank(values, lags=2, trend=trend)
