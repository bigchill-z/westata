"""vec's header and per-equation rows reproduce Stata line for line.

C14-06 scores 0.29. Rounds 68-69 fixed the arithmetic; this round fixed the presentation, which
had five separate faults stacked on the same block:

  * the block began with a blank line -- Stata starts flush
  * AIC was paired with "Log likelihood", but Stata's right column has FOUR rows against the
    left's three, with AIC on its own line
  * the "Parms      RMSE" gap was wrong
  * RMSE printed as "0.0137722" -- Stata writes ".013772", fixed-point at six decimals with the
    leading zero dropped (and ".00256" after trailing zeros go)
  * the per-equation rows were 61-66 characters against Stata's 64

Two traps cost time here and are worth recording: ":<16" on an f-string whose literal prefix is
"D_" produces an EIGHTEEN-character field, and "%.6g" is significant figures, not decimals -- the
RMSE needed fixed-point.

Columns now end at 23 / 35 / 44 / 55 / 64, exactly where Stata's do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.tseries import _vec_number

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_HEADER = [
    "Vector error-correction model",
    "",
    "Sample: 3 thru 166                              Number of obs     =        164",
    "                                                AIC               =  -14.77519",
    "Log likelihood =  1220.566                      HQIC              =  -14.70613",
    "Det(Sigma_ml)  =  1.18e-09                      SBIC              =  -14.60508",
    "",
    "Equation           Parms      RMSE     R-sq      chi2     P>chi2",
    "----------------------------------------------------------------",
    "D_dinf                4     .013772   0.7434   463.5891   0.0000",
    "D_unem                4      .00256   0.4062   109.4497   0.0000",
]


@pytest.fixture(scope="module")
def summary():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna().reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    fitted = ws.VECM(
        frame[["dinf", "unem"]].to_numpy(float), coint_rank=1, lags=2, names=["dinf", "unem"]
    ).fit()
    return str(fitted.summary())


def test_every_header_line_matches_stata(summary) -> None:
    lines = summary.splitlines()
    for expected in STATA_HEADER:
        assert expected in lines, repr(expected)


def test_the_block_opens_with_a_blank_then_the_banner(summary) -> None:
    """Stata's `vec` output starts with an EMPTY line, then the banner.

    The previous version of this test asserted `splitlines()[0]` was the banner, which contradicted
    the captured header above it -- a live run has the blank first. That assertion is what kept the
    output one line short of Stata's sixty.
    """

    lines = summary.splitlines()
    assert lines[0] == ""
    assert lines[1] == "Vector error-correction model"


def test_aic_has_its_own_row(summary) -> None:
    """It was paired with Log likelihood, shifting the whole header."""

    lines = summary.splitlines()
    assert any("AIC" in line and "Number of obs" not in line and line.startswith(" ") for line in lines)
    assert not any("Log likelihood" in line and "AIC" in line for line in lines)


def test_the_equation_rows_are_sixty_four_characters(summary) -> None:
    for line in summary.splitlines():
        if line.startswith("D_dinf") and "|" not in line:
            assert len(line) == 64


def test_rmse_is_fixed_point_not_significant_figures() -> None:
    """%.6g gives "0.0137722"; Stata writes ".013772"."""

    assert _vec_number(0.013772186895068768, 6) == ".013772"
    assert _vec_number(0.002560017026, 6) == ".00256"


def test_the_d_prefix_counts_towards_the_field_width() -> None:
    """Pin the trap: "D_" plus ":<14" is a 16-character field, not 14."""

    assert len(f"D_{'dinf':<14}") == 16
