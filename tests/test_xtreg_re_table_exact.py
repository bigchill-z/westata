"""`xtreg, re`'s table is now character-for-character identical to Stata's.

Rounds 51-53 fixed the e() surface and the header; this round closed the last three lines. All 24
lines of `xtreg lny lnk1 lnk2 lnlabor, re` on munnell.dta compare equal, including:

  * the `Wald chi2(3) = 18370.44` row, which was printing 6123.48 -- e(F) under a chi2 label;
  * the sigma rows, which read `.081862869` (nine significant figures, dot at column 15) where
    Stata prints `.08186287` (dot at column 16). `%.8g` does not count the leading zero as a
    significant digit; Stata counts a digit inside the fraction.

e(chi2) and e(F) differ by df_m: 18370.441 against 6123.480. Both are correct numbers from the same
model, which is why the mistake survived -- only the label said which one Stata prints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"
FORMULA = "lny ~ lnk1 + lnk2 + lnlabor"

STATA_TABLE = [
    '',
    'Random-effects GLS regression                   Number of obs     =        816',
    'Group variable: sid                             Number of groups  =         48',
    '',
    'R-squared:                                      Obs per group:',
    '     Within  = 0.9387                                         min =         17',
    '     Between = 0.9909                                         avg =       17.0',
    '     Overall = 0.9898                                         max =         17',
    '',
    '                                                Wald chi2(3)      =   18370.44',
    'corr(u_i, X) = 0 (assumed)                      Prob > chi2       =     0.0000',
    '',
    '------------------------------------------------------------------------------',
    '         lny | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]',
    '-------------+----------------------------------------------------------------',
    '        lnk1 |  -.0461512   .0226432    -2.04   0.042     -.090531   -.0017715',
    '        lnk2 |     .25296   .0182082    13.89   0.000     .2172725    .2886475',
    '     lnlabor |   .8129069   .0220387    36.89   0.000     .7697118     .856102',
    '       _cons |   2.611567   .1148796    22.73   0.000     2.386407    2.836726',
    '-------------+----------------------------------------------------------------',
    '     sigma_u |  .08186287',
    '     sigma_e |  .03882161',
    '         rho |  .81639881   (fraction of variance due to u_i)',
    '------------------------------------------------------------------------------',
]


@pytest.fixture(scope="module")
def table():
    try:
        frame = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    frame = frame.copy()
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    frame["year"] = frame["year"].astype(int)
    fitted = ws.RandomEffects.from_formula(
        FORMULA, frame, entity="sid", time="year"
    ).fit()
    return str(fitted.summary(style="stata")).splitlines()


def test_the_line_count_matches(table) -> None:
    assert len(table) == len(STATA_TABLE)


def test_every_line_matches_stata(table) -> None:
    """Byte for byte. A near-miss is what let the e(F)/e(chi2) swap survive three rounds."""

    problems = [
        (index, theirs, ours)
        for index, (theirs, ours) in enumerate(zip(STATA_TABLE, table))
        if theirs != ours
    ]
    assert not problems, problems[:3]


def test_the_wald_row_uses_chi2_not_F(table) -> None:
    """e(chi2) = 18370.441 and e(F) = 6123.480 differ by df_m; the row must print the former."""

    row = next(line for line in table if "Wald chi2" in line)
    assert "18370.44" in row
    assert "6123.48" not in row


def test_the_sigma_rows_use_eight_fraction_digits(table) -> None:
    """`%.8g` emits nine significant figures here because it ignores the leading zero."""

    for name, expected in (("sigma_u", "  .08186287"), ("sigma_e", "  .03882161")):
        row = next(line for line in table if line.strip().startswith(name))
        assert row.split("|")[1] == expected, row
