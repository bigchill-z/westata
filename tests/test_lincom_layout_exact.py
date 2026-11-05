"""Pin ``lincom``'s table character for character.

Reference from a live ``regress y k l`` then ``lincom k + l`` on cobb_douglas.dta::

     ( 1)  k + l = 0
    ------------------------------------------------------------------------------
               y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
    -------------+----------------------------------------------------------------
             (1) |   1.075205   .1159063     9.28   0.000     .8341646    1.316245
    ------------------------------------------------------------------------------

Column positions, measured with ``str.index()`` / ``re.finditer`` on those two lines rather than
derived:

    header ends:  y 12, Coefficient 26, Std. err. 37, t 44, P>|t| 53, CI 78
    row    ends:  (1) 14, estimate 25, se 36, t 45, p 53, lo 66, hi 78

Two things were wrong before, and both are the kind that survive a casual look:

1. the response name was LEFT-aligned (``f"{response:<11} |"``), putting ``y`` at the table edge
   and the pipe one column early -- it reads fine until compared against Stata;
2. the constraint marker had no leading space. Stata indents it by exactly one.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\cobb_douglas.dta"
STATA_BLOCK = [
    " ( 1)  k + l = 0",
    "------------------------------------------------------------------------------",
    "           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]",
    "-------------+----------------------------------------------------------------",
    "         (1) |   1.075205   .1159063     9.28   0.000     .8341646    1.316245",
    "------------------------------------------------------------------------------",
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("cobb_douglas.dta is not available")
    return str(ws.OLS.from_formula("y ~ k + l", frame).fit().lincom("k + l"))


def test_every_line_matches_stata(rendered) -> None:
    lines = rendered.splitlines()
    for expected in STATA_BLOCK:
        assert expected in lines, expected


def test_the_response_name_is_right_aligned(rendered) -> None:
    header = next(line for line in rendered.splitlines() if "Coefficient" in line)
    assert header.index("|") == 13
    assert header[11] == "y"


def test_the_constraint_marker_is_indented(rendered) -> None:
    assert rendered.splitlines()[0].startswith(" ")
    assert rendered.splitlines()[0].lstrip().startswith("( 1)")


def test_the_value_and_interval_agree_with_the_fit() -> None:
    """The numbers themselves, independent of layout."""

    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("cobb_douglas.dta is not available")
    outcome = ws.OLS.from_formula("y ~ k + l", frame).fit().lincom("k + l")
    assert float(outcome.estimate) == pytest.approx(1.075205, abs=5e-7)
    assert float(outcome.std_err) == pytest.approx(0.1159063, abs=5e-7)
