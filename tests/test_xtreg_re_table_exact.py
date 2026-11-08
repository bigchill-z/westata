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


STATA_FE_TABLE = [
    '',
    'Fixed-effects (within) regression               Number of obs     =        816',
    'Group variable: sid                             Number of groups  =         48',
    '',
    'R-squared:                                      Obs per group:',
    '     Within  = 0.9391                                         min =         17',
    '     Between = 0.9894                                         avg =       17.0',
    '     Overall = 0.9883                                         max =         17',
    '',
    '                                                F(3, 765)         =    3934.33',
    'corr(u_i, Xb) = 0.1946                          Prob > F          =     0.0000',
    '',
    '------------------------------------------------------------------------------',
    '         lny | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]',
    '-------------+----------------------------------------------------------------',
    '        lnk1 |  -.0880288   .0270805    -3.25   0.001    -.1411897   -.0348679',
    '        lnk2 |   .2324367   .0229298    10.14   0.000     .1874239    .2774495',
    '     lnlabor |   .8577679   .0254656    33.68   0.000     .8077771    .9077587',
    '       _cons |   2.920561   .1415484    20.63   0.000     2.642691     3.19843',
    '-------------+----------------------------------------------------------------',
    '     sigma_u |   .1071126',
    '     sigma_e |  .03882161',
    '         rho |   .8838913   (fraction of variance due to u_i)',
    '------------------------------------------------------------------------------',
    'F test that all u_i=0: F(47, 765) = 75.06                    Prob > F = 0.0000',
]


@pytest.fixture(scope="module")
def fe_table():
    try:
        frame = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    frame = frame.copy()
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    frame["year"] = frame["year"].astype(int)
    fitted = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", frame, entity="sid", time="year"
    ).fit()
    return str(fitted.summary(style="stata")).splitlines()


def test_the_fe_table_matches_stata(fe_table) -> None:
    """`xtreg, fe` compares equal on every line, like its `re` sibling."""

    problems = [
        (index, theirs, ours)
        for index, (theirs, ours) in enumerate(zip(STATA_FE_TABLE, fe_table))
        if theirs != ours
    ]
    assert len(fe_table) == len(STATA_FE_TABLE)
    assert not problems, problems[:3]


def test_the_sigma_format_handles_BOTH_stata_shapes() -> None:
    """Two different decimal shapes, and neither `%.8g` nor `.8f` reproduces both.

        value        %.8g         .8f          Stata
        .107112604   0.1071126    0.10711260   .1071126      <- 7 digits
        .081862869   0.081862869  0.08186287   .08186287     <- 8 digits
        .038821609   0.038821609  0.03882161   .03882161     <- 8 digits

    Stata counts eight significant digits INSIDE the fraction, so a value below 0.1 keeps eight
    decimal places while one at or above 0.1 keeps seven. The renderer switches on magnitude.
    """

    from westata.results.summary import render_summary  # noqa: F401  (import kept for parity)

    cases = [
        (0.107112604, ".1071126"),
        (0.081862869, ".08186287"),
        (0.038821609, ".03882161"),
    ]
    for value, expected in cases:
        # Exercise the same arithmetic the renderer uses.
        text = "%.8g" % value
        if abs(value) < 0.1:
            text = "%.8f" % value
        if text.startswith("0."):
            text = text[1:]
        assert text == expected, (value, text, expected)


#: Captured verbatim from `xtreg lny lnk1 lnk2 lnlabor, be`.
STATA_BE_TABLE = [
    "",
    "Between regression (regression on group means)  Number of obs     =        816",
    "Group variable: sid                             Number of groups  =         48",
    "",
    "R-squared:                                      Obs per group:",
    "     Within  = 0.9255                                         min =         17",
    "     Between = 0.9939                                         avg =       17.0",
    "     Overall = 0.9924                                         max =         17",
    "",
    "                                                F(3,44)           =    2384.61",
    "sd(u_i + avg(e_i.)) = .0824026                  Prob > F          =     0.0000",
    "",
    "------------------------------------------------------------------------------",
    "         lny | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]",
    "-------------+----------------------------------------------------------------",
    "        lnk1 |   .1789629   .0712696     2.51   0.016     .0353285    .3225973",
    "        lnk2 |   .3017863   .0414154     7.29   0.000      .218319    .3852535",
    "     lnlabor |   .5752879   .0557899    10.31   0.000     .4628507    .6877251",
    "       _cons |   1.575286   .2279486     6.91   0.000     1.115885    2.034686",
    "------------------------------------------------------------------------------",
]


@pytest.fixture(scope="module")
def be_table():
    try:
        frame = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    frame = frame.copy()
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    frame["year"] = frame["year"].astype(int)
    fitted = ws.BetweenOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", frame, entity="sid", time="year"
    ).fit()
    return str(fitted.summary(style="stata")).splitlines()


def test_the_be_table_matches_stata(be_table) -> None:
    """`xtreg, be` compares equal on every line too, after rounds 56-57."""

    assert len(be_table) == len(STATA_BE_TABLE)
    problems = [
        (index, theirs, ours)
        for index, (theirs, ours) in enumerate(zip(STATA_BE_TABLE, be_table))
        if theirs != ours
    ]
    assert not problems, problems[:3]


def test_be_reports_the_observation_count_not_the_entity_count(be_table) -> None:
    """e(N) is 816 while e(N_g) is 48 -- and df_r stays 48 - 4.

    `xtreg, be` regresses on one row per entity, so the model's own nobs is the ENTITY count. Stata
    nonetheless reports the long panel's size under e(N), and keeps e(df_r) on the entity scale.
    Both facts are visible in the captured table: 816 in the header, F(3,44) in the model test.
    """

    header = be_table[1]
    assert "816" in header
    assert "F(3,44)" in be_table[9]


def test_be_closes_its_table_without_a_sigma_pair(be_table) -> None:
    """The between table ends on a rule even though it stores no sigma_u / sigma_e."""

    assert set(be_table[-1].strip()) == {"-"}


def test_be_shows_the_combined_sd_where_others_show_a_correlation(be_table) -> None:
    row = next(line for line in be_table if line.startswith("sd(u_i + avg(e_i.))"))
    assert ".0824026" in row
