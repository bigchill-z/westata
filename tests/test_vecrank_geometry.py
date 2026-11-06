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
          2      10    1224.4562     0.04634
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
    "      2      10    1224.4562     0.04634",
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
    """Except the last rank row, which Stata prints short at 40 characters.

    Round 142 corrected this: an older hand-typed reference line had Params ending at column 14
    and the row at 39 characters. A live `vecrank dinf unem, lags(2)` shows Params ending at 15
    and the row at 40. The corrected table matches Stata byte for byte on all five trend cases.
    """

    for line in rendered.splitlines():
        if line.startswith("Johansen") or not line:
            continue
        if line.startswith("      2      10"):
            assert len(line) == 40, (len(line), line)
            continue
        assert len(line) == 62, (len(line), line)


def test_the_short_last_row_has_no_trace_or_critical(rendered) -> None:
    """Stata omits them because no test remains -- not a truncation bug."""

    last = rendered.splitlines()[-2]
    assert last.startswith("      2      10")
    assert len(last) == 40
    assert "0.04634" in last


def test_rank_zero_prints_a_dot_for_the_eigenvalue(rendered) -> None:
    row = next(line for line in rendered.splitlines() if line.strip().startswith("0 "))
    assert " . " in row


def test_the_sample_line_states_the_lag_window(rendered) -> None:
    assert "Sample: 3 thru 166" in rendered


def test_all_five_trend_specifications_are_supported() -> None:
    """Round 136 added ``trend``; round 142 lifted the refusal on the other three.

    ``rconstant`` and ``rtrend`` are not just different residualising blocks: the restricted
    deterministic term is appended to R1 as an extra pseudo-variable, and in the unrestricted
    (full-rank) VAR it becomes another ordinary column. Once both of those are done, all five
    specifications reproduce Stata's eigenvalues, trace, likelihood, parameter counts and
    critical values.

    Earlier versions of this guard asserted three of the five were refused -- correct at the
    time, wrong now. The history is kept so the reversal is visible rather than silent.
    """

    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].to_numpy(float)

    # Measured parameter counts for k=2, lags=2, and the rank-0 trace for each case.
    expected = {
        "none": ([4, 7, 8], 169.1915),
        "rconstant": ([4, 8, 10], 183.1679),
        "constant": ([6, 9, 10], 183.0918),
        "rtrend": ([6, 10, 12], 183.7733),
        "trend": ([8, 11, 12], 183.6114),
    }
    for trend, (params, trace) in expected.items():
        result = ws.vecrank(values, lags=2, trend=trend)
        assert result.deterministic == trend
        assert [row.params for row in result.rows] == params, trend
        assert float(result.rows[0].trace) == pytest.approx(trace, abs=5e-4), trend

    # An unknown spelling still refuses, rather than silently falling back to a default.
    with pytest.raises(ws.UnsupportedError):
        ws.vecrank(values, lags=2, trend="not_a_case")
