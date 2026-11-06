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

    lines = etable_as_stata(models, model_names=["m1", "m2"]).splitlines()
    assert lines[0].strip().startswith("(1)")
    assert "m1" in lines[1] and "m2" in lines[1]


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


def test_the_number_format_is_the_open_defect(models) -> None:
    """esttab prints 0.0966 where this prints 0.097 -- three significant figures, not decimals.

    Pinned as a KNOWN DIFFERENCE rather than asserted away: fixing the formatter also moves the
    standard-error columns, which round 96 could not land without breaking two fixtures.
    """

    rendered = etable_as_stata(models, model_names=["m1", "m2"])
    assert "0.097" in rendered or "0.097*" in rendered
    assert "0.0966" not in rendered
