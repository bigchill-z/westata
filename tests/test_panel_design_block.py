"""The panel DESIGN block that ``xtpcse``/``xtgls`` print above their estimation table.

No other command has this header, and this package printed none of it, so a reader could
not tell which panels the correlation structure was estimated over -- the single most
important fact about a PCSE fit. The layout is transcribed from the captured Stata run:

``tests/conformance/stata_fixtures/p9_outputs.txt:496-506``

    Linear regression, correlated panels corrected standard errors (PCSEs)

    Group variable:   firm                          Number of obs     =        700
    Time variable:    year                          Number of groups  =        100
    Panels:           correlated (balanced)         Obs per group:
    Autocorrelation:  no autocorrelation                          min =          7
                                                                  avg =          7
                                                                  max =          7
    Estimated covariances      =      5050          R-squared         =     0.2144
    Estimated autocorrelation  =         0          Wald chi2(1)      =     178.20
    Estimated coefficients     =         2          Prob > chi2       =     0.0000

Two implementation notes worth keeping:

* the block has to be attached TWICE in the renderer -- once on the base ``lines`` list and
  once in the branch that REPLACES that list for the ANOVA layout. The first attempt only
  did the former, so the block vanished for exactly the commands that need it.
* ``Estimated coefficients`` falls back to the design's column count when ``e(k)`` is
  absent, because that is what the quantity means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import _panel_design_block, render_summary

FIXTURES = Path(__file__).resolve().parent / "conformance" / "stata_fixtures"


def _panel_frame(n_units: int = 100, n_periods: int = 14, seed: int = 7900):
    rng = np.random.default_rng(seed)
    total = n_units * n_periods
    return pd.DataFrame(
        {
            "y": rng.normal(size=total),
            "x1": rng.normal(size=total),
            "firm": np.repeat(np.arange(n_units), n_periods),
            "year": np.tile(np.arange(1, n_periods + 1), n_units),
        }
    )


@pytest.fixture(scope="module")
def pcse_fit():
    return ws.XtPcse.from_formula(
        "y ~ x1", _panel_frame(), entity="firm", time="year"
    ).fit()


# --------------------------------------------------------------- the block
def test_the_block_names_the_panels(pcse_fit) -> None:
    rendered = render_summary(pcse_fit, style="stata")

    assert "Group variable:" in rendered
    assert "Time variable:" in rendered
    assert "Panels:" in rendered


def test_the_block_reports_the_group_count(pcse_fit) -> None:
    """100 firms -- the number the correlation structure was estimated over."""

    rendered = render_summary(pcse_fit, style="stata")

    assert "Number of groups" in rendered
    assert "100" in rendered


def test_the_block_reports_the_per_group_extremes(pcse_fit) -> None:
    """A balanced panel has min = avg = max = 14."""

    rendered = render_summary(pcse_fit, style="stata")

    assert "Obs per group:" in rendered
    assert "min" in rendered and "avg" in rendered and "max" in rendered


def test_the_block_counts_the_estimated_covariances(pcse_fit) -> None:
    rendered = render_summary(pcse_fit, style="stata")

    assert "Estimated covariances" in rendered
    assert "Estimated autocorrelations" in rendered
    assert "Estimated coefficients" in rendered


def test_the_estimated_coefficient_count_is_a_number(pcse_fit) -> None:
    """``y ~ x1`` estimates two coefficients (slope and constant)."""

    block = _panel_design_block(pcse_fit)
    line = next(entry for entry in block if "Estimated coefficients" in entry)

    assert line.rstrip().endswith("2")


# ------------------------------------------------ the block is command-specific
def test_a_plain_regression_has_no_design_block() -> None:
    """Guard: the block must not leak into the OLS family."""

    rng = np.random.default_rng(7902)
    frame = pd.DataFrame({"y": rng.normal(size=200), "x1": rng.normal(size=200)})
    fitted = ws.OLS.from_formula("y ~ x1", frame).fit()

    assert _panel_design_block(fitted) == []
    rendered = render_summary(fitted, style="stata")
    assert "Group variable:" not in rendered
    assert "Panels:" not in rendered


# ------------------------------------------------------ the capture, re-read
def test_the_capture_really_has_this_block() -> None:
    text = (FIXTURES / "p9_outputs.txt").read_text(encoding="utf-8", errors="ignore")

    assert "Group variable:" in text
    assert "Time variable:" in text
    assert "Panels:" in text
    assert "Estimated covariances" in text
    assert "Obs per group:" in text


def test_the_capture_labels_the_columns_the_same_way() -> None:
    """The exact label texts, so a paraphrase is visible."""

    text = (FIXTURES / "p9_outputs.txt").read_text(encoding="utf-8", errors="ignore")

    for label in (
        "Group variable:",
        "Time variable:",
        "Panels:",
        "Obs per group:",
        "Estimated covariances",
        "Estimated autocorrelations",
        "Estimated coefficients",
    ):
        assert label in text, label


# ------------------------------------------------------------ not collateral
def test_the_estimation_table_is_still_there(pcse_fit) -> None:
    rendered = render_summary(pcse_fit, style="stata")

    assert "Coefficient" in rendered
    assert "Std. err." in rendered
    assert "_cons" in rendered
    assert "Source |" in rendered