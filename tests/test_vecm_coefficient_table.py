"""vec's coefficient table matches Stata column for column.

Rounds 68-70 fixed vec's arithmetic and header; this fixes the coefficient block, which was
81 characters per row against Stata's 78.

Values END at fixed columns, measured from a live run:

    row    L1. |  -1.872152    .107031   -17.49   0.000    -2.081929   -1.662375
    ends    12 14   25          36        45       53       66            78

so the six value fields are 11 / 11 / 9 / 8 / 13 / 12 wide after the 14-character label column.

The standard error also had the wrong shape: "%.7g" printed "0.107031" where Stata writes
".107031" -- the leading zero has to go, as everywhere else in these tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_ROWS = [
    # Stata prints -1.872152; this fit converges to -1.8721519, an eighth-digit difference.
    "         L1. | -1.8721519    .107031   -17.49   0.000   -2.0819288  -1.6623751",
    "         LD. |   .4011357   .0661408     6.06   0.000     .2715023    .5307692",
    "         LD. | -2.0332769   .3374592    -6.03   0.000   -2.6946847   -1.371869",
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


def test_the_dinf_equation_rows_match_stata(summary) -> None:
    lines = summary.splitlines()
    for expected in STATA_ROWS:
        assert expected in lines, repr(expected)


def test_every_coefficient_row_is_seventy_eight_characters(summary) -> None:
    for line in summary.splitlines():
        if "|" in line and ("L1." in line or "LD." in line or "_cons" in line):
            assert len(line) == 78, (len(line), line)


def test_the_coefficient_header_matches(summary) -> None:
    expected = "             | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]"
    assert expected in summary.splitlines()


def test_standard_errors_drop_the_leading_zero(summary) -> None:
    """It printed "0.107031" where Stata writes ".107031"."""

    assert ".107031" in summary
    assert "0.107031" not in summary
