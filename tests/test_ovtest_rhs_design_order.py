"""`estat ovtest, rhs` builds its auxiliary model in Stata's order, which decides the answer.

`ovtest.ado:420` is:

    `ver' _regress `lhs' `zlist' `rhs' `wgtexp'

so THE GENERATED POWERS ENTER BEFORE THE USER'S OWN REGRESSORS, and `_regress` omits any column
that the earlier ones can reproduce. Testing the powers against the finished model instead gives a
different -- and wrong -- result whenever a user's variable duplicates a power:

    grilic, user adds expr2 = expr^2, model lnw ~ s + expr + expr2 + tenure + smsa + rns

    Stata          F(11, 741) = 1.73   p = 0.0626
    powers last    F(10, 741) = 1.27   p = 0.2421   (drops expr^2 as well as expr2^2)
    powers first   F(11, 741) = 1.73   p = 0.0626   (drops expr2^2, then expr2 on entering)

The package previously produced `F(12, 739) = 1.9e+26` on this model: with `expr2` and `expr^2`
both present the auxiliary design is singular (condition number 2e10) and the Wald form inverts it.

Order is also why the reported df is 11 and not 17: `test zlist` tests only the powers, while the
auxiliary regression carries twelve powers plus six regressors minus the two omissions.

Verified against a live capture on four specifications of grilic: 4 of 4 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(DATA).exists(),
    reason="grilic.dta is not available on this machine",
)


def _frame() -> pd.DataFrame:
    frame = pd.read_stata(DATA)
    frame["expr2"] = frame["expr"] ** 2
    return frame


def _result(formula: str):
    return ws.estat_ovtest(ws.OLS.from_formula(formula, _frame()).fit(), rhs=True)


def test_a_duplicated_power_is_omitted_not_the_fitted_model() -> None:
    """expr2 duplicates expr^2, so only ONE power drops and df is 11 of 12."""

    result = _result("lnw ~ s + expr + expr2 + tenure + smsa + rns")
    assert result.df == 11, result.df
    assert result.df_denom == 741, result.df_denom


def test_the_captured_statistic_matches() -> None:
    result = _result("lnw ~ s + expr + expr2 + tenure + smsa + rns")
    assert round(float(result.statistic), 2) == 1.73
    assert round(float(result.pvalue), 4) == 0.0626


def test_the_auxiliary_design_is_not_singular() -> None:
    """Without the omission the statistic came out as 1.9e+26."""

    result = _result("lnw ~ s + expr + expr2 + tenure + smsa + rns")
    assert np.isfinite(result.statistic)
    assert result.statistic < 100, result.statistic


def test_without_the_duplicate_no_power_is_dropped() -> None:
    """Three continuous regressors give nine powers and nothing is collinear."""

    result = _result("lnw ~ s + expr + tenure + smsa + rns")
    assert result.df == 9, result.df
    assert result.df_denom == 743


@pytest.mark.parametrize(
    "formula,expected_df",
    [
        ("lnw ~ s + expr + tenure", 9),
        ("lnw ~ s + expr2 + tenure", 9),
    ],
)
def test_the_degrees_of_freedom_follow_the_continuous_count(formula, expected_df) -> None:
    assert _result(formula).df == expected_df


def test_smsa_and_rns_are_indicators_and_contribute_nothing() -> None:
    """Both are 0/1, so they add no powers; the df counts continuous regressors only."""

    with_dummies = _result("lnw ~ s + expr + tenure + smsa + rns")
    without = _result("lnw ~ s + expr + tenure")
    assert with_dummies.df == without.df
    assert with_dummies.df_denom < without.df_denom
