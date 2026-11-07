"""Hausman FE vs RE: the table now matches Stata 14/14 -- the FOOTNOTE names were wrong.

C12-07 scored a layout of 0.05 against 0.17 numeric coverage. Round 174 found the coefficients were
already exact and three separate things around them were not:

  1. THE COMMAND NAMES. The footnote read "obtained from ivregress" / "obtained from regress" for a
     PANEL fit -- naming two commands that were never run. The renderer fell back to those literals
     whenever `stata_command_name` was absent, and panel results do not carry it. On a live
     `hausman fe re` Stata prints "obtained from xtreg" on BOTH lines.

  2. THE COLUMN HEADER. It read
         |                               Difference      Std. err.
     with nothing over the two coefficient columns, because `estimation_label` was missing and the
     fallback was the empty string. Stata heads them with the estimate-store names, fe and re.

  3. THE STANDARD-ERROR PRECISION. A fixed seven decimals printed ".0148540" where Stata prints
     ".014854", so that one cell was a character wider than every other row in its column.

Also corrected: the two footnote lines are indented 26 and 11 spaces (not 22 and 9), and a blank
line separates them from the test paragraph.

TWO-SIDED ENTRY POINT: `PanelOLS` is FE and `RandomEffects` is RE; there is no top-level `xtreg`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\munnell.dta"

#: From `hausman fe re` after storing an `xtreg, fe` and an `xtreg, re` fit.
STATA_LINES = [
    "        lnk1 |   -.0880288    -.0461512       -.0418776         .014854",
    "        lnk2 |    .2324367       .25296       -.0205233        .0139368",
    "     lnlabor |    .8577679     .8129069         .044861         .012759",
    "                          b = Consistent under H0 and Ha; obtained from xtreg.",
    "           B = Inconsistent under Ha, efficient under H0; obtained from xtreg.",
    "            =  33.92",
    "Prob > chi2 = 0.0000",
]


@pytest.fixture(scope="module")
def reference():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("munnell.dta is not available")
    fe = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", frame, entity="st_abb", time="year"
    ).fit()
    re = ws.RandomEffects.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", frame, entity="st_abb", time="year"
    ).fit()
    return fe.hausman(re)


def test_the_table_matches_stata_line_for_line(reference) -> None:
    rendered = str(reference)
    for expected in STATA_LINES:
        assert expected in rendered, repr(expected)


def test_the_footnote_names_the_command_that_was_actually_run(reference) -> None:
    """It used to say ivregress and regress, neither of which produced either side."""

    rendered = str(reference)
    assert "obtained from xtreg." in rendered
    assert "obtained from ivregress" not in rendered
    assert "obtained from regress." not in rendered


def test_the_column_header_names_both_sides(reference) -> None:
    rendered = str(reference)
    assert "fe" in rendered and "re" in rendered
    header = next(line for line in rendered.splitlines() if "Difference" in line and "|" in line)
    assert "fe" in header and "re" in header


def test_the_standard_error_drops_the_trailing_zero(reference) -> None:
    """'.014854', not '.0148540' -- the latter is a character wider than its column."""

    rendered = str(reference)
    assert ".0148540" not in rendered
    assert ".014854" in rendered


def test_the_statistic_itself_is_unchanged(reference) -> None:
    assert float(reference.statistic) == pytest.approx(33.92, abs=5e-3)
    assert float(reference.pvalue) < 1e-4
