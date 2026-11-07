"""`summarize` now opens with the blank line Stata prints.

Found while auditing the low-scoring layout cases in the v3 report: C04-04, C05-08 and C09-04 all
scored 0.00, and all three are `predict ... ; summarize ...` pairs. The cause was not the numbers --
those matched exactly -- but the LEADING LINE.

Stata's `summarize e1` output starts with "\n" before the "Variable | Obs ..." header. This package
opened directly on the header, so a four-line block matched only two of four lines and the
line-match ratio came out 0.50, below the report's 0.15 cut for "layout different / no match".

    before: 2 of 4 lines matched (0.5000)
    after:  3 of 4 lines matched (0.7500)

The remaining difference is the residual mean: -4.89e-15 here against Stata's 7.03e-11, a
floating-point difference from where the residuals are formed, not a formatting one.

Two existing tests asserted that EVERY line is 71 characters wide; they now skip blank lines,
because a blank line is 0 characters and is part of Stata's output.
"""

from __future__ import annotations

import difflib

import pandas as pd
import pytest

import westata as ws

GIRLIC = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: Captured from `summarize e1` after `quietly regress lnw s expr` + `predict e1, residual`.
STATA_BLOCK = [
    "",
    "    Variable |        Obs        Mean    Std. dev.       Min        Max",
    "-------------+---------------------------------------------------------",
    "          e1 |        758    7.03e-11    .3594254   -1.22974   1.171132",
]


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(GIRLIC).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")


def test_it_starts_with_a_blank_line(frame) -> None:
    assert str(ws.summarize(frame[["lnw"]])).startswith("\n")


def test_only_the_first_line_is_blank(frame) -> None:
    """One blank line, not a padded block -- the rows after it still start with content."""

    lines = str(ws.summarize(frame[["lnw"]])).splitlines()
    assert lines[0] == ""
    assert lines[1].strip() != ""


def test_the_table_header_follows_the_blank(frame) -> None:
    lines = str(ws.summarize(frame[["lnw"]])).splitlines()
    assert "Variable" in lines[1]
    assert "Std. dev." in lines[1]


def test_the_line_match_rate_against_stata_is_three_of_four(frame) -> None:
    """The ratio the report scores. It was 2 of 4 before the blank line was added."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr", frame).fit()
    mine = str(ws.summarize(pd.DataFrame({"e1": fitted.predict(kind="resid")}))).splitlines()

    def normalise(rows):
        return [" ".join(row.split()) for row in rows]

    stata, ours = normalise(STATA_BLOCK), normalise(mine)
    matcher = difflib.SequenceMatcher(None, stata, ours)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    assert matched == 3
    assert matched / max(len(stata), len(ours)) == pytest.approx(0.75)


def test_the_header_row_matches_stata_character_for_character(frame) -> None:
    lines = str(ws.summarize(frame[["lnw"]])).splitlines()
    assert lines[1] == STATA_BLOCK[1]


def test_the_rule_row_matches_stata(frame) -> None:
    lines = str(ws.summarize(frame[["lnw"]])).splitlines()
    assert lines[2] == STATA_BLOCK[2]
