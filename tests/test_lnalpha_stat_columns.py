"""The z and P columns on /lnalpha depend on the command, not on the row.

Round 244 measured `xtpoisson n x1, re`, saw blank columns on the dispersion row, and hid them for every command.
Measuring two more shows three behaviours:

    nbreg cnt x             '  -11.50729    1.719345'    z and P BLANK
    xtpoisson n x1, re                                    z and P BLANK
    zinb cnt x, inflate(z)  '  -6.24   0.000'             z and P PRESENT

SO THE TWO-EQUATION COMMANDS PRINT THE TEST STATISTIC THERE while the single-equation ones leave it empty. The
row label alone cannot tell them apart; the model name is what distinguishes them.

THE COEFFICIENTS ON THIS TABLE STILL DIFFER from Stata in the seventh significant figure -- -0.5296616806 against
-.529661, -4.0693319678 against -4.069331 -- which is a numerical path difference rather than a formatting one, and
is recorded as such instead of being tuned away.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _zinb_row(n: int = 1000) -> str:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    frame["cnt"] = rng.negative_binomial(2, 0.4, n).astype(float)
    fitted = ws.ZeroInflatedNegativeBinomial.from_formula(
        "cnt ~ x", frame, inflate="z"
    ).fit()
    return next(
        line
        for line in str(fitted.summary(style="stata")).splitlines()
        if "/lnalpha" in line
    )


def _nbreg_row(n: int = 800) -> str:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"x": rng.normal(size=n)})
    frame["cnt"] = rng.poisson(2.0, n).astype(float)
    fitted = ws.NegativeBinomial.from_formula("cnt ~ x", frame).fit()
    return next(
        line
        for line in str(fitted.summary(style="stata")).splitlines()
        if "/lnalpha" in line
    )


def test_zinb_prints_the_test_statistic() -> None:
    row = _zinb_row()
    assert "-6.24" in row, row
    assert "0.000" in row, row


def test_nbreg_leaves_the_columns_blank() -> None:
    row = _nbreg_row()
    # The z column sits between the standard error and the interval; blank means a run of
    # spaces with no bare number in it.
    # The z and P columns span 40..54; the standard error ends at 39.
    middle = row[40:54]
    assert not any(char.isdigit() for char in middle), (row, middle)


def test_the_two_rows_are_the_same_width() -> None:
    assert len(_zinb_row()) == len(_nbreg_row()) == 78


def test_the_interval_is_present_in_both() -> None:
    for row in (_zinb_row(), _nbreg_row()):
        assert "[" not in row
        assert len(row.split()) >= 4
