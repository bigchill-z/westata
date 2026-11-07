"""estat summarize: the header blank line and the closing rule were both missing.

C11-xx family auditing, round 197. Comparing a live `estat summarize` after
`regress lnw s expr tenure` showed the package printing NINE lines where Stata prints ELEVEN, with
every value -- means, standard deviations, minima, maxima -- already exact:

    (blank)
      Estimation sample regress                Number of obs =        758
    (blank)                                    <- missing
      -------------------------------------------------------------------
          Variable |         Mean      Std. dev.         Min          Max
      -------------+-----------------------------------------------------
               lnw |     5.686739      .4289494        4.605        7.051
                 s |     13.40501      2.231828            9           18
              expr |     1.735429      2.105542            0       11.444
            tenure |     1.831135       1.67363            0           10
      -------------------------------------------------------------------   <- missing

Also checked in the same round: `estat hettest` matches Stata's nine lines CHARACTER FOR CHARACTER;
its only difference was a trailing newline in the capture, which no splitlines() comparison shows
and which the package does not need to reproduce.

`estat vce` and `estat ovtest` were already byte-identical.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: From `regress lnw s expr tenure` then `estat summarize`.
STATA_LINES = [
    "",
    "  Estimation sample regress                Number of obs =        758",
    "",
    "  " + "-" * 67,
    "      Variable |         Mean      Std. dev.         Min          Max",
    "  " + "-" * 13 + "+" + "-" * 53,
    "           lnw |     5.686739      .4289494        4.605        7.051",
    "             s |     13.40501      2.231828            9           18",
    "          expr |     1.735429      2.105542            0       11.444",
    "        tenure |     1.831135       1.67363            0           10",
    "  " + "-" * 67,
]


@pytest.fixture(scope="module")
def summary():
    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.estat_summarize(ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit())


def test_the_block_matches_stata_byte_for_byte(summary) -> None:
    assert str(summary).splitlines() == STATA_LINES


def test_there_is_a_blank_line_after_the_header(summary) -> None:
    lines = str(summary).splitlines()
    assert lines[1].startswith("  Estimation sample")
    assert lines[2] == ""
    assert lines[3].startswith("  ---")


def test_the_table_is_closed_by_a_rule(summary) -> None:
    lines = str(summary).splitlines()
    assert lines[-1] == "  " + "-" * 67


def test_the_values_are_unchanged(summary) -> None:
    rendered = str(summary)
    for value in ("5.686739", ".4289494", "13.40501", "2.231828", "1.831135", "1.67363"):
        assert value in rendered, value
