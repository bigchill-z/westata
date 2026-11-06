"""estat firststage: the exact Stata geometry, recorded for a future attempt.

Round 109 established that the SUMMARY BLOCK is shifted two columns left of Stata's, and captured
the reference. The fix was reverted after spending too long on the row alignment rather than
shipped half-correct.

Captured from ``ivregress 2sls lnw (expr = s tenure)`` then ``estat firststage``::

      First-stage regression summary statistics
      --------------------------------------------------------------------------
                   |            Adjusted      Partial
          Variable |   R-sq.       R-sq.        R-sq.      F(2,755)   Prob > F
      -------------+------------------------------------------------------------
              expr |  0.1064      0.1041       0.1064        44.963    0.0000
      --------------------------------------------------------------------------
      Minimum eigenvalue statistic = 44.963

   43 / 76 / 49 / 74 / 76 / 73 / 76 / 45 characters respectively.

THE ALIGNMENT PUZZLE, which is why it was reverted. Two data rows from two datasets:

    '          expr |  0.1064      0.1041       0.1064        44.963    0.0000'   73
    '          endog |  0.6288      0.6260       0.6267       332.355    0.0000'   74

Both names start at column 10. The pipe sits at 15 for the 4-character name and 16 for the
5-character one. But the numbers AFTER the pipe start at 18 for "expr" and at 19 for "endog" --
so the first column is placed relative to the pipe, and every later column shifts with it. That
couples all six fields to the name's length, which the single ``_place(end_column)`` scheme in the
renderer cannot express. A fixed-column scheme matches one row and breaks the other.

Whatever the next attempt does, it needs the numeric columns expressed RELATIVE to the pipe, not
at absolute columns. The numbers themselves already match Stata exactly.
"""

from __future__ import annotations

import re

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
]


def test_the_numbers_match_stata() -> None:
    """The statistics were never in question; only the furniture."""

    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    fit = ws.IV2SLS.from_formula("lnw ~ [expr ~ s + tenure]", frame).fit()
    rendered = str(fit.estat_firststage())
    for number in ("0.1064", "0.1041", "44.963", "19.93", "11.59", "8.68"):
        assert number in rendered, number


def test_the_reference_columns_are_recorded() -> None:
    """Pinned so the next attempt starts from the target, not from a guess."""

    def positions(text: str) -> list[tuple[str, int]]:
        return [(match.group(), match.start()) for match in re.finditer(r"\S+", text)]

    expr_row = "          expr |  0.1064      0.1041       0.1064        44.963    0.0000"
    endog_row = "          endog |  0.6288      0.6260       0.6267       332.355    0.0000"

    assert positions(expr_row) == [
        ("expr", 10), ("|", 15), ("0.1064", 18), ("0.1041", 30),
        ("0.1064", 43), ("44.963", 57), ("0.0000", 67),
    ]
    assert positions(endog_row) == [
        ("endog", 10), ("|", 16), ("0.6288", 19), ("0.6260", 31),
        ("0.6267", 44), ("332.355", 57), ("0.0000", 68),
    ]
    assert len(expr_row) == 73
    assert len(endog_row) == 74


def test_the_pipe_follows_the_name_length() -> None:
    """Which is what makes a single fixed-column scheme impossible."""

    expr_pipe = "          expr |".index("|")
    endog_pipe = "          endog |".index("|")
    assert expr_pipe == 15
    assert endog_pipe == 16
    assert endog_pipe - expr_pipe == 1 == len("endog") - len("expr")
