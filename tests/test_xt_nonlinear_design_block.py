"""The panel design block for the xt NON-LINEAR commands, which is SHORTER than xtpcse's.

Two different shapes exist, and unifying them produced a table full of ``.`` when tried.

**xtpcse / xtgls** (``p9_outputs.txt:496-506``) print the full block: group AND time
variable, panel description, per-group extremes, and three "Estimated ..." counts.

**xt non-linear** (``p9_outputs.txt:14-22``, ``xtlogit, fe``) print a shorter one::

    Conditional fixed-effects logistic regression      Number of obs    =    912
    Group variable: firm                               Number of groups =    114

                                                       Obs per group:
                                                                    min =      8
                                                                    avg =      8.0
                                                                    max =      8

                                                       LR chi2(2)       =  32.46
    Log likelihood = -387.92362                        Prob > chi2      = 0.0000

No ``Time variable:``, no ``Panels:``, no "Estimated ..." rows. Round 250 gated the block
on a two-name list (``xtpcse``/``xtgls``), so every ``xtlogit``/``xtprobit``/``xtpoisson``
output dropped the panel structure entirely; round 252 widened the gate and split the two
shapes apart.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import _panel_design_block, render_summary

FIXTURES = Path(__file__).resolve().parent / "conformance" / "stata_fixtures"


def _panel_frame(n_units: int = 100, n_periods: int = 9, seed: int = 8100):
    rng = np.random.default_rng(seed)
    total = n_units * n_periods
    return pd.DataFrame(
        {
            "y": (rng.uniform(size=total) < 0.5).astype(int),
            "x1": rng.normal(size=total),
            "firm": np.repeat(np.arange(n_units), n_periods),
            "year": np.tile(np.arange(1, n_periods + 1), n_units),
        }
    )


@pytest.fixture(scope="module")
def xtlogit_fit():
    return ws.XtLogit.from_formula(
        "y ~ x1", _panel_frame(), entity="firm", time="year", method="fe"
    ).fit()


@pytest.fixture(scope="module")
def xtpcse_fit():
    rng = np.random.default_rng(8102)
    total = 100 * 14
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=total),
            "x1": rng.normal(size=total),
            "firm": np.repeat(np.arange(100), 14),
            "year": np.tile(np.arange(1, 15), 100),
        }
    )
    return ws.XtPcse.from_formula("y ~ x1", frame, entity="firm", time="year").fit()


# ------------------------------------------------------- xt non-linear shape
def test_xtlogit_shows_the_group_count(xtlogit_fit) -> None:
    rendered = render_summary(xtlogit_fit, style="stata")

    assert "Group variable:" in rendered
    assert "Number of groups" in rendered


def test_xtlogit_shows_the_per_group_extremes(xtlogit_fit) -> None:
    rendered = render_summary(xtlogit_fit, style="stata")

    assert "Obs per group:" in rendered
    assert "min" in rendered and "avg" in rendered and "max" in rendered


@pytest.mark.parametrize(
    "label",
    ["Time variable:", "Panels:", "Estimated covariances", "Estimated autocorrelations"],
)
def test_xtlogit_omits_the_xtpcse_only_rows(xtlogit_fit, label: str) -> None:
    """THE split: these four belong to xtpcse, and emitting them for xtlogit printed
    rows of ``.`` -- a table that looked broken rather than brief."""

    rendered = render_summary(xtlogit_fit, style="stata")

    assert label not in rendered


def test_the_block_function_returns_the_short_shape(xtlogit_fit) -> None:
    block = _panel_design_block(xtlogit_fit)

    # SEVEN rows, measured from ``xtlogit ybin x1, fe``: Group variable / Obs per group /
    # min / avg / max / LR chi2 / (Log likelihood + Prob > chi2). The model test and the
    # log likelihood belong to this block -- Stata prints them between "max" and the table
    # rule. The upper bound was 6 when those two rows were missing.
    assert 7 <= len(block) <= 8
    assert any("Group variable:" in entry for entry in block)
    assert any("LR chi2" in entry for entry in block)
    assert not any("Estimated covariances" in entry for entry in block)


# ------------------------------------------------------ xtpcse keeps its own
def test_xtpcse_still_has_the_full_block(xtpcse_fit) -> None:
    """Guard: widening the gate must not shorten the xtpcse block."""

    block = _panel_design_block(xtpcse_fit)

    # NINE lines, not ten. The captured block (p9_outputs.txt:499-505) is
    # Panels / Autocorrelation(min) / avg / max / 3x "Estimated ..." -- seven -- plus the
    # Group and Time rows. Counting a blank separator line as part of the block inflated
    # the expected number by one.
    assert len(block) == 9
    assert any("Time variable:" in entry for entry in block)
    assert any("Estimated covariances" in entry for entry in block)
    assert any("Estimated coefficients" in entry for entry in block)


def test_the_two_shapes_are_different(xtlogit_fit, xtpcse_fit) -> None:
    """A single shared shape cannot satisfy both commands."""

    assert len(_panel_design_block(xtlogit_fit)) < len(_panel_design_block(xtpcse_fit))


def test_a_plain_regression_still_has_none() -> None:
    rng = np.random.default_rng(8104)
    frame = pd.DataFrame({"y": rng.normal(size=200), "x": rng.normal(size=200)})
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()

    assert _panel_design_block(fitted) == []


# ------------------------------------------------------ the capture, re-read
def test_the_capture_shows_the_short_shape_for_xtlogit() -> None:
    text = (FIXTURES / "p9_outputs.txt").read_text(encoding="utf-8", errors="ignore")

    assert "Group variable: firm" in text
    assert "Number of groups =    114" in text
    assert "Obs per group:" in text


def test_the_capture_shows_the_long_shape_for_xtpcse() -> None:
    text = (FIXTURES / "p9_outputs.txt").read_text(encoding="utf-8", errors="ignore")

    assert "Estimated covariances" in text
    assert "Estimated autocorrelations" in text


# ----------------------------------------------------------- not collateral
def test_the_estimation_table_survives(xtlogit_fit) -> None:
    rendered = render_summary(xtlogit_fit, style="stata")

    assert "Coefficient" in rendered
    # ``xtlogit`` is an ORDINARY table: capitalised "Std. err.", no banner row. Only
    # xtpcse's header is the lower-case spelling.
    assert "Std. err." in rendered
    assert "P>|z|" in rendered