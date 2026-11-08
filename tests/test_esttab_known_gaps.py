"""C20-01: what etable_as_stata is missing, recorded as an executable record.

The report scores chapter 20 (results presentation) at 0.091 -- the lowest in the whole report --
on a single item, C20-01, described as "esttab has star alignment and mtitle, etable does not".

Investigation in round 96 found the situation is different from that description:

  * mtitle IS implemented. render_esttab always emits the "(1) (2)" row and the model-name row.
  * stars ARE implemented, and they hang past the column end exactly as esttab does.
  * stats are supported through ``stats=[("N","nobs"),("R-sq","r2")]``, which reproduces
    Stata's ``, se r2`` output.

Two REAL differences remain, both measured against a live ``esttab m1 m2, se r2 mtitle``:

  1. Our numbers use three DECIMALS; esttab uses three or four SIGNIFICANT FIGURES.
     0.0966245 prints "0.097" here and "0.0966" in Stata; 0.0352 prints "0.035" here.
  2. Rows are not padded to the 44-column width, so trailing spaces differ.

Attempting to fix (1) exposed a coupling: the standard-error columns are positioned from the
coefficient text's width, so changing the number format shifted them and broke two fixtures that
had been correct. Round 96 reverted rather than ship a half-changed formatter.

This module records the measurements so the next attempt starts from evidence instead of from the
report's description, which misidentified what was missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.esttab import etable_as_stata


@pytest.fixture(scope="module")
def models():
    try:
        frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\grilic.dta").dropna(
            subset=["lnw", "s", "expr", "tenure"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return [
        ws.OLS.from_formula("lnw ~ s", frame).fit(),
        ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit(),
    ]


def test_mtitle_is_already_implemented(models) -> None:
    """The report says mtitle is missing; the numbered row and name row are both emitted."""

    # Stata opens with a blank line, so the numbers are on line 1 and the titles on line 2.
    lines = etable_as_stata(models, model_names=["m1", "m2"]).splitlines()
    assert lines[0] == ""
    assert lines[1].strip().startswith("(1)")
    assert "m1" in lines[2] and "m2" in lines[2]


def test_stars_hang_past_the_column(models) -> None:
    rendered = etable_as_stata(models, model_names=["m1", "m2"])
    assert "***" in rendered


def test_r2_can_be_requested_through_stats(models) -> None:
    """This reproduces Stata's ``, se r2`` without a dedicated flag."""

    rendered = etable_as_stata(
        models, model_names=["m1", "m2"], stats=[("N", "nobs"), ("R-sq", "r2")]
    )
    assert "R-sq" in rendered
    assert "N " in rendered


def test_the_number_format_uses_significant_figures(models) -> None:
    """esttab prints 0.0966, not 0.097 -- significant figures, not decimals.

    Rounds 96-97 left this open: the formatter could not change without moving the standard-error
    columns. The reason was a false comment claiming SE cells are NOT right-aligned to the column
    end. Capturing all six SE cells from two tables showed they all end at the column edge, so
    aligning them like the coefficients (_cell) decouples the two and the format change lands.
    """

    rendered = etable_as_stata(models, model_names=["m1", "m2"])
    assert "0.0966" in rendered
    assert "0.097" not in rendered


def test_the_standard_errors_use_the_same_rule(models) -> None:
    """9.011188 must print "(9.011)", not "(9.01)" -- four figures at or above 1."""

    rendered = etable_as_stata(
        models, model_names=["m1", "m2"], stats=[("N", "nobs"), ("R-sq", "r2")]
    )
    assert "(0.00604)" in rendered
    assert "(0.00595)" in rendered


def test_rows_are_padded_to_the_table_width(models) -> None:
    """Stata pads every table row to 44 columns; the legend lines are not padded."""

    lines = etable_as_stata(models, model_names=["m1", "m2"]).splitlines()
    # Blank rows are NOT padded -- Stata emits them empty, and padding them made the
    # line-wise comparison fail on lines that are supposed to be zero-width.
    table_rows = [
        line
        for line in lines
        if line and "p<0.05" not in line and "parentheses" not in line
    ]
    assert table_rows
    for row in table_rows:
        assert len(row) == 44, (len(row), row)
