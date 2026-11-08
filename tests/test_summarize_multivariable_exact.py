"""A multi-variable `summarize` matches Stata line for line, blank opening line included.

Captured from a live `summarize lnw s expr tenure` on grilic.dta:

    (blank)
        Variable |        Obs        Mean    Std. dev.       Min        Max
    -------------+---------------------------------------------------------
             lnw |        758    5.686739    .4289494      4.605      7.051
               s |        758    13.40501    2.231828          9         18
            expr |        758    1.735429    2.105542          0     11.444
          tenure |        758    1.831135     1.67363          0         10

Seven lines, all equal.

THE BLANK FIRST LINE IS PART OF THE BLOCK and is easy to lose twice over. Stata's command output opens
with it, and a comparison that filters blank lines -- `if line.strip()` -- will report a difference of
zero whatever the block does. My first pass at this comparison filtered them and concluded the package
was emitting an EXTRA blank; the full diff showed the two agreed. Both directions of that mistake are
worth remembering: filtering blanks hides a real difference, and trusting a filtered diff invents one.

A second guard checks the alignment survives a row whose Min is an exponent, which is the case that
broke in round 156.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"

STATA_LINES = [
    "",
    "    Variable |        Obs        Mean    Std. dev.       Min        Max",
    "-------------+---------------------------------------------------------",
    "         lnw |        758    5.686739    .4289494      4.605      7.051",
    "           s |        758    13.40501    2.231828          9         18",
    "        expr |        758    1.735429    2.105542          0     11.444",
    "      tenure |        758    1.831135     1.67363          0         10",
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "tenure"]
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")
    return str(ws.summarize(frame[["lnw", "s", "expr", "tenure"]]))


def test_the_block_opens_with_a_blank_line(rendered) -> None:
    assert rendered.splitlines()[0] == ""


@pytest.mark.parametrize("line", STATA_LINES)
def test_each_line_matches_stata(rendered, line) -> None:
    assert line in rendered.splitlines(), line


def test_the_row_count_is_exactly_seven(rendered) -> None:
    """Four variables, a header, a rule and the opening blank -- no more, no fewer."""

    assert len(rendered.splitlines()) == 7


def test_an_exponent_in_the_minimum_keeps_the_columns_aligned(rendered) -> None:
    """Round 156's failure: an eight-character exponent presses the cell one column left."""

    frame = pd.DataFrame({"v": np.array([2.4e-6, 0.5, 2.253077, 0.1])})
    line = str(ws.summarize(frame)).splitlines()[3]
    assert line == "           v |          4    .7132699    1.049022   2.40e-06   2.253077"
