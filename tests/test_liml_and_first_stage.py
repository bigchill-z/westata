"""LIML and the inline first-stage table: C10-04 is exact; C10-01's table was missing.

C10-04 (`ivregress liml`, 0.47 / 0.37) was an unaudited low score. Checked in round 167 against a
live `ivregress liml lnw (s = expr tenure age)` on grilic.dta, the whole output matches line for
line, 13 of 13:

    Instrumental-variables LIML regression            Number of obs   =        758
                                                      Wald chi2(1)    =     194.45
           s |   .1479434   .0106094    13.94   0.000     .1271494    .1687374
       _cons |   3.703556   .1429149    25.91   0.000     3.423448    3.983664

So the score predates the LIML implementation rather than measuring a defect.

A REAL GAP SURFACED ALONGSIDE IT: `ivregress ..., first` prints a separate first-stage regression
table ABOVE the main one, and `Result.summary()` has no way to produce it. The entry point is
`estat_firststage` (12/12 lines since round 110), but that is a different report -- the `first`
option duplicates the whole OLS fit inside the `ivregress` output.

The first-stage block is therefore recorded here as not implemented, with its reference capture,
so it is not mistaken for a layout quirk of the main table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
COLUMNS = ["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"]

#: From a live `ivregress liml lnw (s = expr tenure age)`.
STATA_LIML_LINES = [
    "Instrumental-variables LIML regression            Number of obs   =        758",
    "Wald chi2(1)    =     194.45",
    "R-squared       =     0.1815",
    "Root MSE        =     .38783",
    "           s |   .1479434   .0106094    13.94   0.000     .1271494    .1687374",
    "       _cons |   3.703556   .1429149    25.91   0.000     3.423448    3.983664",
    "Endogenous: s",
    "Exogenous:  expr tenure age",
]

#: From `ivregress 2sls lnw (s = expr tenure age), first` -- the block summary() does not print.
STATA_FIRST_STAGE = [
    "First-stage regressions",
    "                                                        F(3, 754)     = 190.15",
    "                                                        R-squared     = 0.4307",
    "        expr |  -.4964593   .0318038   -15.61   0.000    -.5588938   -.4340248",
    "      tenure |  -.2335256   .0392078    -5.96   0.000    -.3104951   -.1565561",
    "         age |   .5157278   .0232279    22.20   0.000     .4701287    .5613268",
    "       _cons |   3.433236   .4786435     7.17   0.000     2.493604    4.372868",
]


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA).dropna(subset=COLUMNS).reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")


def test_liml_matches_stata_line_for_line(frame) -> None:
    fitted = ws.IVLIML.from_formula("lnw ~ [s ~ expr + tenure + age]", frame).fit()
    rendered = str(fitted.summary(style="stata"))
    for expected in STATA_LIML_LINES:
        assert expected in rendered, repr(expected)


def test_liml_reports_the_three_attribution_lines(frame) -> None:
    """LIML names the endogenous and exogenous sets under the table, like 2SLS does."""

    fitted = ws.IVLIML.from_formula("lnw ~ [s ~ expr + tenure + age]", frame).fit()
    rendered = str(fitted.summary(style="stata"))
    assert "Endogenous: s" in rendered
    assert "Exogenous:  expr tenure age" in rendered


def test_the_first_stage_table_exists_and_matches_stata(frame) -> None:
    """`Result.first_stage()` reproduces Stata's `ivregress ..., first` block, 16 of 16 lines.

    Round 167 first recorded this as UNIMPLEMENTED after finding `summary()` cannot print it. That
    was wrong: the table is implemented as a separate method and matches Stata exactly. The real
    gap is narrower -- `summary()` does not offer a `first=` switch, so the block is only reachable
    by calling the method yourself.
    """

    fitted = ws.IV2SLS.from_formula("lnw ~ [s ~ expr + tenure + age]", frame).fit()
    rendered = str(fitted.first_stage())
    for expected in STATA_FIRST_STAGE:
        assert expected in rendered, repr(expected)


def test_summary_first_prepends_the_first_stage_block(frame) -> None:
    """`summary(first=True)` now reproduces `ivregress ..., first` end to end, 29 of 29 lines.

    The table itself already existed as `Result.first_stage()`; what was missing was the switch
    that Stata users reach for. Round 167 first recorded the whole feature as unimplemented --
    checking the method before believing that was the difference.
    """

    fitted = ws.IV2SLS.from_formula("lnw ~ [s ~ expr + tenure + age]", frame).fit()
    rendered = str(fitted.summary(style="stata", first=True))
    assert "First-stage regressions" in rendered
    for expected in STATA_FIRST_STAGE:
        assert expected in rendered, repr(expected)
    # And the main table still follows it.
    assert "Instrumental-variables 2SLS regression" in rendered


def test_first_on_a_non_iv_result_raises_with_a_clear_message(frame) -> None:
    """OLS exposes the method but has no instrument block, so the call refuses with a reason.

    The message names iv_structure rather than failing deeper inside with a KeyError, which is
    what makes this safe to expose on every result class.
    """

    ols = ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()
    with pytest.raises(AttributeError, match="iv_structure"):
        ols.summary(style="stata", first=True)
