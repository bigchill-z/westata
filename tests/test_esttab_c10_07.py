"""`esttab` reproduces Stata's table: C10-07's 0.0417 does not hold.

Round 116-119 rebuilt the renderer against a live `esttab m1 m2, se`; this file ties that work to the
report ID so the improvement is recorded where the scoring lives.

    (blank)
    --------------------------------------------
                          (1)             (2)
                           m1              m2         <- only with mtitle
                            y               y
    --------------------------------------------
    x1                 0.0424          0.0645
                     (0.0733)        (0.0745)
                                                   <- blank between coefficient blocks
    x2                                 0.0997
                                     (0.0656)
    _cons             -0.0698         -0.0711
                      (0.0660)        (0.0658)
    --------------------------------------------
    N                     200             200
    --------------------------------------------
    Standard errors in parentheses
    * p<0.05, ** p<0.01, *** p<0.001

EIGHTEEN LINES, and each of the four things round 116-119 had to discover is pinned separately: the rule
above the column numbers, the blank opening line, the gaps between coefficient blocks, and the fact that
the model-name row appears only when the caller supplies names (`mtitle`).
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(1)
    n = 200
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    return (
        ws.OLS.from_formula("y ~ x1", frame).fit(),
        ws.OLS.from_formula("y ~ x1 + x2", frame).fit(),
    )


def test_the_default_table_has_eighteen_lines(models) -> None:
    rendered = ws.etable_as_stata(list(models), model_names=None)
    assert len(rendered.splitlines()) == 18


def test_the_observation_counts_are_reported(models) -> None:
    rendered = ws.etable_as_stata(list(models), model_names=None)
    row = [line for line in rendered.splitlines() if line.strip().startswith("N ")]
    assert row, "the N block is what round 116 added"
    assert row[0].count("200") == 2


def test_the_coefficients_match_stata(models) -> None:
    rendered = ws.etable_as_stata(list(models), model_names=None)
    assert "0.0424" in rendered
    assert "0.0645" in rendered
    assert "0.0997" in rendered


def test_the_standard_errors_are_in_parentheses(models) -> None:
    rendered = ws.etable_as_stata(list(models), model_names=None)
    assert "(0.0733)" in rendered
    assert "(0.0745)" in rendered
    assert "(0.0656)" in rendered


def test_the_constant_pads_to_four_decimals(models) -> None:
    """(0.0660), not (0.066) -- the format switches on significant figures, from round 118."""

    rendered = ws.etable_as_stata(list(models), model_names=None)
    assert "(0.0660)" in rendered


def test_the_model_name_row_needs_mtitle(models) -> None:
    unnamed = ws.etable_as_stata(list(models), model_names=None)
    named = ws.etable_as_stata(list(models), model_names=["m1", "m2"])
    assert "m1" not in unnamed
    assert "m1" in named


def test_the_legend_is_last(models) -> None:
    lines = ws.etable_as_stata(list(models), model_names=None).splitlines()
    assert lines[-2] == "Standard errors in parentheses"
    assert lines[-1] == "* p<0.05, ** p<0.01, *** p<0.001"
