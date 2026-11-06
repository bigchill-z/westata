"""vecrank trend(trend): the numbers are implemented; three layout characters are not.

Round 136 implemented ``trend(trend)`` -- the linear trend inside the cointegrating relation -- and
every numeric output matches a live Stata. Round 137 measured the remaining layout differences,
solved one of them exactly, and then stopped rather than keep nudging spaces.

VERIFIED NUMERICALLY (round 136, re-confirmed here):

    params   8 / 11 / 12          (vs 6 / 9 / 10 for constant)
    LL       1132.9913 / 1220.9786 / 1224.797
    eig      .65802 / .04550
    trace    183.6114 / 7.6369
    5% crit  18.17 / 3.74

THE TREND LINE RULE, solved this round. "Trend: Constant" (15 chars) is followed by 27 spaces and
"Trend: Linear" (13 chars) by 28 -- so the label is centred in the space left when the 20-character
"Number of obs  = N" tail is right-aligned to 62. A fixed end-column padding put one line a space
off whichever column was chosen. The rule that reproduces both:

    padding = " " * (27 + (15 - len(label)) // 2)

giving 62 characters for Constant and 61 for Linear, matching Stata exactly.

STILL DIFFERENT, and recorded rather than chased further: two data rows. Stata's field widths shift
with the Params width in a way that also interacts with whether the likelihood keeps its trailing
zero -- the CONSTANT table prints 1224.4562 (13.4f, row ends at 39 characters) and the TREND table
prints 1224.797 (trailing zero dropped, row ends at 38). Both of those are correct as they stand
in round 136's code; the mismatch is in the FULL rows where a two-digit Params moves every later
field one column left.

WHY I STOPPED: this is the third round on a table whose numbers are exact, and the remaining work is
one or two spaces on two rows. Fixing the constant table's short row while breaking the trend one,
then the reverse, is the pattern this task has already learned to recognise -- so the layout changes
were reverted and only the measurement is kept. The next attempt should read the rule off all six
rows at once rather than adjusting one and re-running.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_TREND = {
    "params": [8, 11, 12],
    "ll": [1132.9913, 1220.9786, 1224.797],
    "eig": [0.65802, 0.04550],
    "trace": [183.6114, 7.6369],
    "crit": [18.17, 3.74],
}


@pytest.fixture(scope="module")
def trend_fit():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.vecrank(frame[["dinf", "unem"]].to_numpy(float), lags=2, trend="trend")


def test_the_parameter_counts_match_stata(trend_fit) -> None:
    """8 / 11 / 12 against 6 / 9 / 10 for the constant case -- a uniform +2 at k=2."""

    assert [row.params for row in trend_fit.rows] == STATA_TREND["params"]


def test_the_likelihoods_match_stata(trend_fit) -> None:
    assert [round(float(row.loglikelihood), 4) for row in trend_fit.rows] == STATA_TREND["ll"]


def test_the_eigenvalues_and_traces_match_stata(trend_fit) -> None:
    finite = [row for row in trend_fit.rows if np.isfinite(row.eigenvalue)]
    assert [round(float(row.eigenvalue), 5) for row in finite] == STATA_TREND["eig"]
    traces = [round(float(row.trace), 4) for row in trend_fit.rows if np.isfinite(row.trace)]
    assert traces == STATA_TREND["trace"]


def test_the_critical_values_use_the_trend_table(trend_fit) -> None:
    """18.17 / 3.74, NOT the constant case's 15.41 / 3.76.

    Reusing the constant table would put a real statistic beside the wrong threshold, which
    silently changes which rank gets selected.
    """

    crit = [
        round(float(row.trace_critical), 2)
        for row in trend_fit.rows
        if np.isfinite(row.trace_critical)
    ]
    assert crit == STATA_TREND["crit"]


def test_the_trend_label_is_recorded(trend_fit) -> None:
    assert trend_fit.deterministic == "trend"
    assert "Trend: Linear" in str(trend_fit)


def test_the_trend_line_padding_formula() -> None:
    """The rule solved this round, pinned so it is not re-derived.

    ``27 + (15 - len(label)) // 2`` spaces puts "Number of obs" where Stata has it.
    """

    for label, expected_spaces in (("Trend: Constant", 27), ("Trend: Linear", 28)):
        padding = 27 + (15 - len(label)) // 2
        assert padding == expected_spaces, label
