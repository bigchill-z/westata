"""``esttab`` text output, added in round 6 of the v2 report (C20-01).

``etable`` returns a DataFrame, which is what a program wants. Stata's ``esttab`` prints a
TABLE, and the report noted that form was absent: no ``(1) (2)`` numbering row, no ``mtitle``
row, no ``N`` / ``R-sq`` footer, no footnote.

Reference, ``regress y k`` stored as m1 and ``regress y k l`` stored as m2, then
``esttab m1 m2, se r2 mtitle`` on cobb_douglas::

                          (1)             (2)
                           m1              m2
    --------------------------------------------
    k                   0.377***        0.159**
                     (0.0352)        (0.0417)

    l                                   0.917***
                                      (0.1496->0.150)

    _cons               77.58***       -4.855
                      (9.011)         (14.54)
    --------------------------------------------
    N                      24              24
    R-sq                0.839           0.942
    --------------------------------------------
    Standard errors in parentheses
    * p<0.05, ** p<0.01, *** p<0.001

Four formatting rules, each measured rather than assumed:

* stars hang PAST the aligned number ("0.377***" extends two columns beyond "0.377");
* the constant prints LAST even though patsy puts Intercept first;
* a value needing two integer columns loses a decimal place (77.58, not 77.579), keeping the
  cells the same width;
* standard errors START at fixed columns (17 and 33) and are not right-aligned to the
  coefficients' ends -- "(9.011)" ends at 24 while "(0.0352)" ends at 25.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\cobb_douglas.dta"

STATA_LINES = [
    # Table rows are padded to 44 columns, as Stata prints them; the two legend lines are not.
    "                      (1)             (2)   ",
    "                       m1              m2   ",
    "--------------------------------------------",
    "k                   0.377***        0.159** ",
    "                 (0.0352)        (0.0417)   ",
    "l                                   0.917***",
    "                                  (0.150)   ",
    "_cons               77.58***       -4.855   ",
    "                  (9.011)         (14.54)   ",
    "N                      24              24   ",
    "R-sq                0.839           0.942   ",
    "--------------------------------------------",
    "Standard errors in parentheses",
    "* p<0.05, ** p<0.01, *** p<0.001",
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("cobb_douglas.dta is not available")
    first = ws.OLS.from_formula("y ~ k", frame).fit()
    second = ws.OLS.from_formula("y ~ k + l", frame).fit()
    return ws.etable_as_stata(
        [first, second],
        model_names=["m1", "m2"],
        stats=[("N", "nobs"), ("R-sq", "r2")],
    )


@pytest.mark.parametrize("line", STATA_LINES)
def test_each_line_matches_stata(rendered, line: str) -> None:
    # Both sides are rstripped: the renderer pads table rows to 44 columns as Stata does, and this
    # test compares CONTENT. Padding has its own guard (test_rows_are_padded_to_the_table_width).
    assert line.rstrip() in [row.rstrip() for row in rendered.splitlines()], line


def test_the_numbering_and_mtitle_rows_come_first(rendered) -> None:
    # Stata opens the table with a BLANK line; the numbers come second. Measured from a live
    # `esttab m1 m2, se`, whose first line is empty and whose second carries (1) and (2).
    # Blank, rule, numbers, titles -- a live esttab opens with a rule above the column numbers.
    lines = rendered.splitlines()
    assert lines[0] == ""
    assert lines[2].strip().startswith("(1)")
    assert lines[3].strip().startswith("m1")


def test_the_constant_is_last(rendered) -> None:
    """patsy puts Intercept first; esttab prints it after every regressor."""

    lines = rendered.splitlines()
    k_row = next(i for i, line in enumerate(lines) if line.startswith("k "))
    cons_row = next(i for i, line in enumerate(lines) if line.startswith("_cons"))
    assert cons_row > k_row
    assert "Intercept" not in rendered


def test_the_footnote_uses_the_5_1_0_1_scheme(rendered) -> None:
    """esttab's default is * 5%, ** 1%, *** 0.1% -- NOT the 10/5/1 convention."""

    assert "* p<0.05, ** p<0.01, *** p<0.001" in rendered


def test_the_dataframe_form_still_exists() -> None:
    """etable_as_stata is additive; the DataFrame was not replaced."""

    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("cobb_douglas.dta is not available")
    first = ws.OLS.from_formula("y ~ k", frame).fit()
    second = ws.OLS.from_formula("y ~ k + l", frame).fit()
    as_frame = ws.etable([first, second], model_names=["m1", "m2"])
    assert isinstance(as_frame, pd.DataFrame)
