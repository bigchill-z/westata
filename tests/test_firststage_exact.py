"""`estat firststage` matches Stata line for line.

C10-02 scores it at 0.2778. Measured on grilic.dta after `ivregress 2sls lnw expr (s = rns mrt)`:

      First-stage regression summary statistics
      ------------------------------------------------------------------------
                   |            Adjusted      Partial
          Variable |   R-sq.       R-sq.        R-sq.      F(2,754)   Prob > F
      -------------+----------------------------------------------------------
                 s |  0.1049      0.1013       0.0493       19.5592    0.0000
      ------------------------------------------------------------------------

Every figure agrees, and so do the header, the rule widths and the column alignment.

MEASURED LENGTHS, printed with len() rather than counted by eye -- my first transcription had the
rules at 72 characters when they are 76, which is the "hand-transcribed reference row" mistake that
error.md already lists:

    43  '  First-stage regression summary statistics'
    76  '  ' + '-' * 74
    49  '               |            Adjusted      Partial'
    74  '      Variable |   R-sq.       R-sq.        R-sq.      F(2,754)   Prob > F'
    76  '  ' + '-' * 13 + '+' + '-' * 60
    73  '             s |  0.1049      0.1013       0.0493       19.5592    0.0000'

THE THREE R-SQUARED COLUMNS ARE DIFFERENT QUANTITIES and the guard checks all three separately: the
plain R-squared of the first stage, the df-adjusted one, and the PARTIAL R-squared of the excluded
instruments after the included ones are projected out. On this data they are 0.1049, 0.1013 and 0.0493
-- close enough that a package returning the same number three times would still look plausible.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"

STATA_LINES = [
    "  First-stage regression summary statistics",
    "  " + "-" * 74,
    "               |            Adjusted      Partial",
    "      Variable |   R-sq.       R-sq.        R-sq.      F(2,754)   Prob > F",
    "  " + "-" * 13 + "+" + "-" * 60,
    "             s |  0.1049      0.1013       0.0493       19.5592    0.0000",
    "  " + "-" * 74,
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "rns", "mrt"]
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")
    fitted = ws.IV2SLS.from_formula("lnw ~ expr + [s ~ rns + mrt]", frame).fit()
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print(str(ws.estat_firststage(fitted)))
    return buffer.getvalue()


@pytest.mark.parametrize("line", STATA_LINES)
def test_each_line_matches_stata(rendered, line) -> None:
    assert line in rendered.splitlines(), line


def test_the_three_r_squareds_are_distinct(rendered) -> None:
    """Adjusted and partial are not the plain one, and none of them may be a relabelled copy."""

    row = next(line for line in rendered.splitlines() if line.strip().startswith("s |"))
    values = row.split("|")[1].split()
    plain, adjusted, partial = values[0], values[1], values[2]
    assert len({plain, adjusted, partial}) == 3


def test_the_f_statistic_carries_its_degrees_of_freedom(rendered) -> None:
    assert "F(2,754)" in rendered
