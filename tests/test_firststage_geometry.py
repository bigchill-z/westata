"""estat firststage renders Stata's summary block byte for byte.

Round 109 found the block sat two columns left of Stata's everywhere and reverted, having failed to
reconcile two data rows with different name lengths. Round 110 solved it by measuring the offsets
RELATIVE TO THE PIPE rather than at absolute columns.

    '          expr |  0.1064 ...'     73    pipe 15
    '          endog |  0.6288 ...'    74    pipe 16

Both names start at column 10. The pipe follows the name's length, and every numeric field after it
is right-aligned at the SAME offset from the pipe:

    first field  +9      second +21     third +34     F +48      p +58

identical for both rows. An absolute-column scheme cannot express that -- six position sets were
tried in round 109, each matching one row and breaking the other.

Captured block from ``ivregress 2sls lnw (expr = s tenure)`` / ``estat firststage``::

      First-stage regression summary statistics
      --------------------------------------------------------------------------
                   |            Adjusted      Partial
          Variable |   R-sq.       R-sq.        R-sq.      F(2,755)   Prob > F
      -------------+------------------------------------------------------------
              expr |  0.1064      0.1041       0.1064        44.963    0.0000
      --------------------------------------------------------------------------
      Minimum eigenvalue statistic = 44.963

The eigenvalues/critical-values half already matched and still does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"

STATA_LINES = [
    "  First-stage regression summary statistics",
    "  " + "-" * 74,
    "               |            Adjusted      Partial",
    "      Variable |   R-sq.       R-sq.        R-sq.      F(2,755)   Prob > F",
    "  " + "-" * 13 + "+" + "-" * 60,
    "          expr |  0.1064      0.1041       0.1064        44.963    0.0000",
    "  " + "-" * 74,
    "  Minimum eigenvalue statistic = 44.963      ",
    "  Critical Values                      # of endogenous regressors:    1",
    "  H0: Instruments are weak             # of excluded instruments:     2",
    "  2SLS size of nominal 5% Wald test  |  19.93   11.59    8.75    7.25",
    "  LIML size of nominal 5% Wald test  |   8.68    5.33    4.42    3.92",
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    fit = ws.IV2SLS.from_formula("lnw ~ [expr ~ s + tenure]", frame).fit()
    return str(fit.estat_firststage())


def test_every_line_matches_stata(rendered) -> None:
    lines = rendered.splitlines()
    for expected in STATA_LINES:
        assert expected in lines, repr(expected)


def test_the_numeric_fields_sit_at_fixed_offsets_from_the_pipe(rendered) -> None:
    """The rule that makes both name lengths work."""

    row = next(line for line in rendered.splitlines() if "expr |" in line)
    pipe = row.index("|")
    for value, offset in (
        ("0.1064", 9),
        ("0.1041", 21),
        ("44.963", 48),
        ("0.0000", 58),
    ):
        assert row.index(value) + len(value) - pipe == offset, value


def test_a_longer_name_shifts_every_field_with_the_pipe(rendered) -> None:
    """'endog' is one character longer than 'expr', so the pipe moves one column right."""

    expr_row = next(line for line in rendered.splitlines() if "expr |" in line)
    assert expr_row.index("|") == 15
    assert len(expr_row) == 73


def test_the_statistics_are_unchanged(rendered) -> None:
    """The numbers were never in question."""

    for number in ("0.1064", "0.1041", "44.963", "19.93", "11.59", "8.68"):
        assert number in rendered, number
