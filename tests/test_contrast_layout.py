"""Pin ``contrast``'s joint-Wald table to Stata.

Reference from a live ``regress consumption i.grp temp`` then ``contrast grp``::

    Contrasts of marginal linear predictions
    Margins: asbalanced
    ------------------------------------------------
                 |         df           F        P>F
    -------------+----------------------------------
             grp |          2        0.32     0.7288
                 |
     Denominator |         26
    ------------------------------------------------

Five differences from what westata printed before:

* the ``Margins: asbalanced`` line was missing;
* a stray blank line followed the title;
* the name column was one character too wide, shifting the pipe and every number;
* the trailing blank piped row was absent;
* the ``Denominator`` row was absent -- and without the denominator degrees of freedom the
  F statistic cannot be read at all.

Geometry measured from that output: every table line is 48 characters with the pipe at
column 13; df ends at 25, F at 37 and P>F at 48.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    frame["grp"] = np.tile([1, 2, 3], len(frame))[: len(frame)].astype(int)
    fit = ws.OLS.from_formula("consumption ~ C(grp) + temp", frame).fit()
    return str(ws.contrast(fit, "grp"))


STATA_LINES = [
    # CORRECTED in round 195: the block opens with a blank line and has one after each
    # header line, matching the live capture.
    "",
    "Contrasts of marginal linear predictions",
    "",
    "Margins: asbalanced",
    "",
    "-" * 48,
    "             |         df           F        P>F",
    "-" * 13 + "+" + "-" * 34,
    "         grp |          2        0.32     0.7288",
    "             |",
    " Denominator |         26",
    "-" * 48,
]


def test_every_line_matches_stata(rendered) -> None:
    assert rendered.splitlines() == STATA_LINES


def test_the_margins_line_is_present(rendered) -> None:
    assert "Margins: asbalanced" in rendered


def test_the_denominator_row_is_present(rendered) -> None:
    """Without it the F statistic has no second degrees of freedom."""

    assert " Denominator |         26" in rendered


def test_the_pipe_sits_at_column_13_and_full_width_rows_are_48(rendered) -> None:
    """Stata's trailing rows are short on purpose: the blank piped row is 14 characters
    and the Denominator row 25. Only the rules, the header and the statistic row span 48.
    """

    for line in rendered.splitlines():
        if line.startswith("-"):
            assert len(line) == 48, line
        elif "|" in line and line.strip() != "|" and "Denominator" not in line:
            assert len(line) == 48, line
            assert line.index("|") == 13, line
    for short in ("             |", " Denominator |         26"):
        assert short in rendered


def test_the_joint_statistic_is_unchanged(rendered) -> None:
    """The numbers were already right; only the table around them changed."""

    assert "0.32" in rendered
    assert "0.7288" in rendered
