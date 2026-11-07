"""A factor coefficient renders as a GROUP HEADER plus a level row, not one long name.

C13-08 (seasonal dummy regression, 0.17 / 0.07) is the weakest row of chapter 13. Round 183 checked
it and found every number already exact -- the seasonal fit matches Stata on every coefficient and
standard error -- with the whole gap in how a ``C(var)[T.level]`` term is laid out.

WHAT WAS WRONG: the stored name ``C(foreign)[T.Foreign]`` is 22 characters and went into the label
column verbatim, so the table was 87 characters wide where Stata's is 78 and every column sat to the
right of its heading. Stata instead prints a blank pipe line, the term right-aligned, then the bare
level:

                 |
           foreign |
         Foreign  |  -1.650029   1.075994    -1.53   0.130      -3.7955    .4954422

Two changes were needed: size the label column from the two PIECES rather than the joined spelling,
and emit the group header without a separating rule (the rule is for equation blocks like zip's).

THE NUMBERS were never in question, which is worth stating: this row's coefficient, standard error,
t, p and interval all already matched.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

AUTO = "tests/data/auto.dta"

#: From `regress mpg weight i.fx` on auto.dta (fx = encode(foreign)).
STATA_WIDTH = 78
# CORRECTED in round 217: the level label is INDENTED by (name_width - 1 - len(text)) and then
# padded two spaces before the pipe, not right-aligned. Re-captured from `regress mpg weight i.fxc`
# on auto.dta, which prints '    Foreign  |  ...' -- four leading spaces and two before the pipe.
# The old string right-aligned it (five leading, none trailing), which the live run never showed.
STATA_COEF_ROW = "    Foreign  |  -1.650029   1.075994    -1.53   0.130      -3.7955    .4954422"
STATA_BLANK_PIPE = " " * 13 + "|"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(AUTO)
    except FileNotFoundError:  # pragma: no cover - the file ships with the repo
        pytest.skip("auto.dta is missing")
    return ws.OLS.from_formula("mpg ~ weight + C(foreign)", frame).fit()


@pytest.fixture(scope="module")
def rendered(fitted):
    return str(fitted.summary(style="stata")).splitlines()


def test_the_table_is_seventy_eight_wide(rendered) -> None:
    """It was 87 while the factor spelling sat in the label column."""

    rules = [line for line in rendered if set(line) == {"-"} and len(line) > 20]
    assert rules
    assert all(len(line) == STATA_WIDTH for line in rules)
    header = next(line for line in rendered if "Coefficient" in line)
    assert len(header) == STATA_WIDTH


def test_the_level_row_matches_stata(rendered) -> None:
    assert STATA_COEF_ROW in rendered


def test_the_group_header_is_a_blank_pipe_line_then_the_term(rendered) -> None:
    index = rendered.index(STATA_COEF_ROW)
    assert rendered[index - 2] == STATA_BLANK_PIPE
    assert rendered[index - 1].endswith("|") and "foreign" in rendered[index - 1]


def test_no_rule_sits_between_the_slopes_and_the_factor_block(rendered) -> None:
    """Stata puts the blank pipe line there instead; a rule was emitted by the equation path."""

    index = rendered.index(STATA_COEF_ROW)
    assert set(rendered[index - 2]) <= {" ", "|"}


def test_the_numeric_coefficients_are_unchanged(fitted) -> None:
    """The layout work must not have moved any estimate."""

    params = dict(fitted.params)
    assert params["weight"] == pytest.approx(-0.00658789, abs=1e-8)
    assert params["C(foreign)[T.Foreign]"] == pytest.approx(-1.650029, abs=1e-6)
    assert params["Intercept"] == pytest.approx(41.679702, abs=1e-6)
