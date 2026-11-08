"""`varlmar` and `varwle` both match Stata, after two real defects each.

C13-03 scored `varlmar` at 0.875 and C13-02 scored `varwle` at 0.9524, and neither had a guard.

THE DEFECTS FOUND this round were not in the numbers:

  * `varlmar` was missing the TOP BORDER of its box, so every row below sat one line early;
  * `VAR(...)` DISCARDED THE COLUMN NAMES of a DataFrame and invented y1/y2, so any table printing an
    equation name said y1 where the caller had passed dinf;
  * `varwle` printed the chi2 at a fixed four decimals, which is wrong for both ends of the scale --
    Stata uses SEVEN SIGNIFICANT FIGURES, giving 72.72164 and 741.8916 from the same column.

With those fixed both tables compare equal to a live Stata:

    varlmar   8 of 8 lines
    varwle   21 of 21 lines

THE NAME BUG WAS THE MOST CONSEQUENTIAL of the three: it is silent, it affects every VAR-derived table,
and it would have made every one of those comparisons look like a formatting disagreement.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

MACRO = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_VARLMAR = [
    "",
    "   Lagrange-multiplier test",
    "  +--------------------------------------+",
    "  | lag  |      chi2    df   Prob > chi2 |",
    "  |------+-------------------------------|",
    "  |   1  |    7.3130     4     0.12024   |",
    "  |   2  |    5.0645     4     0.28074   |",
    "  +--------------------------------------+",
    "   H0: no autocorrelation at lag order",
]


@pytest.fixture(scope="module")
def panel():
    try:
        return pd.read_stata(MACRO).dropna(subset=["dinf", "unem"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("macro_swatson.dta is not available")


@pytest.fixture(scope="module")
def var_fit(panel):
    return ws.VAR(panel[["dinf", "unem"]], lags=2).fit()


def test_a_var_keeps_the_column_names(panel) -> None:
    """It used to invent y1/y2, which renamed every equation in every table."""

    fitted = ws.VAR(panel[["dinf", "unem"]], lags=2).fit()
    assert list(fitted.names) == ["dinf", "unem"]


def test_explicit_names_still_win(panel) -> None:
    fitted = ws.VAR(
        panel[["dinf", "unem"]].to_numpy(), lags=2, names=["a", "b"]
    ).fit()
    assert list(fitted.names) == ["a", "b"]


def test_the_lm_table_matches_stata(var_fit) -> None:
    rendered = str(ws.varlmar(var_fit)).splitlines()
    table = [line for line in rendered if line.strip()]
    expected = [line for line in STATA_VARLMAR if line.strip()]
    for got, want in zip(table, expected):
        assert got == want, want


def test_the_lm_box_is_closed_at_the_top(var_fit) -> None:
    """Without the top border the box looks unclosed and every row sits one line early."""

    rendered = str(ws.varlmar(var_fit))
    assert "  +" + "-" * 38 + "+" in rendered
    lines = [line for line in rendered.splitlines() if line.strip()]
    # '' / title / top border / header / separator / rows / closing border / H0
    assert lines[0] == "   Lagrange-multiplier test"
    assert lines[1].startswith("  +")
    assert lines[2].startswith("  | lag")


def test_the_wald_table_matches_stata(var_fit) -> None:
    rendered = str(ws.varwle(var_fit))
    assert "Equation: dinf" in rendered
    assert "Equation: unem" in rendered
    assert "72.72164" in rendered


def test_the_wald_chi2_uses_seven_significant_figures(var_fit) -> None:
    """A fixed decimal count cannot produce both 72.72164 and 741.8916."""

    rendered = str(ws.varwle(var_fit))
    assert "72.72164" in rendered, "five decimals"
    assert "741.8916" in rendered, "four decimals, same column"
    assert "741.89158" not in rendered


def test_the_wald_table_has_twenty_one_lines(var_fit) -> None:
    lines = [line for line in str(ws.varwle(var_fit)).splitlines() if line.strip()]
    assert len(lines) == 21
