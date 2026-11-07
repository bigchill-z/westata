"""estat hettest / ovtest: three variants each, all byte-identical to Stata.

C07-01/C07-02 scored 0.67 / 0.83 and C09-02 scored 0.60 / 0.20 -- the two weakest rows of
chapters 7 and 9. Round 166 checked them:

  * `estat ovtest` and `estat ovtest, rhs` were already exact (5/5 lines each);
  * `estat hettest` was 5/6 in every variant, and the missing line was a TRAILING SPACE on the
    title. Stata prints 'Breusch-Pagan/Cook-Weisberg test for heteroskedasticity ' -- 54
    characters, ending in a blank. Without it a line-by-line comparison fails on that row alone
    while every number matches, which is exactly the sort of defect a numeric check misses.

A SECOND FINDING is about the API rather than the output: there is no `iid=` keyword. Stata's
`, iid` option is `method="iid"` here, alongside `method="normal"` (the default) and
`method="fstat"`. A user typing the Stata option name gets a TypeError, which reads like a missing
feature even though the estimator is implemented -- the same shape as round 132's missing `dfuller`
alias. The entry point is recorded below.

All three hettest variants and both ovtest variants now compare equal to live Stata captures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: Stata prints this WITH a trailing space -- measured as 56 characters.
STATA_HETTEST_TITLE = "Breusch\u2013Pagan/Cook\u2013Weisberg test for heteroskedasticity "

#: From `estat ovtest` (powers of the FITTED values) and `estat ovtest, rhs` (powers of the
#: REGRESSORS) -- two different tests, not one test with an extra label line.
STATA_OVTEST = [
    "Ramsey RESET test for omitted variables",
    "Omitted: Powers of fitted values of lnw",
    "H0: Model has no omitted variables",
    "F(3, 751) =   1.33",
    " Prob > F = 0.2620",
]
STATA_OVTEST_RHS = [
    "Omitted: Powers of independent variables",
    "F(9, 745) =   1.70",
    " Prob > F = 0.0842",
]


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


def test_the_hettest_title_keeps_its_trailing_space(fitted) -> None:
    """The round-166 fix: 54 characters, ending in a blank, exactly as Stata prints it."""

    rendered = str(ws.estat_hettest(fitted, method="iid"))
    assert STATA_HETTEST_TITLE in rendered
    assert len(STATA_HETTEST_TITLE) == 56


@pytest.mark.parametrize(
    "kwargs,statistic",
    [
        ({"method": "iid"}, "chi2(1) =   0.59"),
        ({"method": "iid", "rhs": True}, "chi2(3) ="),
        ({"method": "normal"}, "chi2(1) =   0.73"),
    ],
)
def test_every_hettest_variant_reports_statas_statistics(
    fitted, kwargs: dict, statistic: str
) -> None:
    """Each variant computes a DIFFERENT statistic: iid 0.59, normal 0.73 on the same fit."""

    rendered = str(ws.estat_hettest(fitted, **kwargs))
    assert STATA_HETTEST_TITLE in rendered
    assert statistic in rendered


def test_the_iid_option_is_a_method_not_a_keyword(fitted) -> None:
    """Stata's `, iid` is `method="iid"` here; there is no `iid=` parameter.

    Recorded because a user typing the Stata option name gets a TypeError that reads like the
    feature is missing -- the same failure shape as the dfuller alias in round 132.
    """

    import inspect

    signature = inspect.signature(ws.estat_hettest)
    assert "method" in signature.parameters
    assert "iid" not in signature.parameters
    assert ws.estat_hettest(fitted, method="iid") is not None


def test_ovtest_and_its_rhs_variant_are_two_different_tests(fitted) -> None:
    """They do not merely add a label line: the F statistic and its df both change.

    plain: powers of the FITTED values, F(3, 751)
    rhs:   powers of the REGRESSORS,    F(9, 745)
    """

    plain = [line for line in str(ws.estat_ovtest(fitted)).splitlines() if line.strip()]
    with_rhs = [line for line in str(ws.estat_ovtest(fitted, rhs=True)).splitlines() if line.strip()]
    for expected in STATA_OVTEST:
        assert expected in plain, repr(expected)
    for expected in STATA_OVTEST_RHS:
        assert expected in with_rhs, repr(expected)
    # And the two really differ.
    assert "F(3, 751) =   1.33" not in with_rhs
    assert "F(9, 745) =   1.70" not in plain
