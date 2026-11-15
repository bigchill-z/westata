"""`xtreg, re`'s e(rmse) comes from the SECOND internal regression, not the RE residuals.

This was `error.md` #34: `e(rmse)` printed 9472.93 where Stata reports 9570.04, a 1.01% gap that had
survived since 0.3.20. Three rounds of formula-guessing had failed, and the recorded conclusion was
to go and read the data transformation instead.

`xtreg_re.ado:277-299` does exactly this:

    by ivar: replace x = x - thetav*x[_N] if !dup     // x[_N] is the group MEAN row
    drop if dup
    gen double cons = 1 - thetav
    _regress varlist cons, hascons
    scalar rmse = e(rmse)

So the reported rmse is the rmse of a regression on QUASI-DEMEANED data over the FULL expanded design
-- year dummies included -- not of the random-effects model's own residuals.

Two details broke every earlier attempt:

* OMITTING THE YEAR DUMMIES gives 10096.69. Only the continuous regressors were being demeaned.
* `hascons` keeps the `cons = 1 - theta` column without adding a separate intercept, so the rank and
  therefore the divisor are 799, not 798 as a naive count suggests.

Replaying the regression inside Stata with the year dummies present reproduces e(rmse) exactly:
rss = 73085290324.19 over df 798, rmse = 9570.0353629102.

Also pinned here: `xtreg, fe` and `xtreg, be` keep their own formulas and still match Stata, so the
override is confined to the random-effects estimator.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(MUNNELL).exists(), reason="munnell.dta is not available"
)


@pytest.fixture(scope="module")
def panel():
    frame = pd.read_stata(MUNNELL).copy()
    frame["year"] = frame["year"].astype(int)
    return frame


def test_random_effects_rmse_matches_stata(panel) -> None:
    """`y ~ k1 + C(year)`: Stata reports 9570.0353629102."""

    fitted = ws.RandomEffects.from_formula(
        "y ~ k1 + C(year)", panel, entity="st_abb", time="year"
    ).fit()
    assert float(fitted.e["rmse"]) == pytest.approx(9570.0353629102, abs=1e-6)


def test_the_year_dummies_are_part_of_the_demeaned_regression(panel) -> None:
    """Dropping them gives 10096.69, which is the trap earlier attempts fell into."""

    fitted = ws.RandomEffects.from_formula(
        "y ~ k1 + C(year)", panel, entity="st_abb", time="year"
    ).fit()
    assert abs(float(fitted.e["rmse"]) - 10096.69) > 1.0


def test_a_two_regressor_model_also_matches(panel) -> None:
    """`y ~ k1 + k2`: Stata reports 5820.7379599524."""

    fitted = ws.RandomEffects.from_formula(
        "y ~ k1 + k2", panel, entity="st_abb", time="year"
    ).fit()
    assert float(fitted.e["rmse"]) == pytest.approx(5820.7379599524, abs=1e-6)


def test_sigma_e_is_unaffected(panel) -> None:
    """sigma_e comes from the WITHIN regression and was already right -- 9567.7548814077."""

    fitted = ws.RandomEffects.from_formula(
        "y ~ k1 + C(year)", panel, entity="st_abb", time="year"
    ).fit()
    assert float(fitted.e["sigma_e"]) == pytest.approx(9567.7548814077, abs=1e-6)


def test_the_two_are_still_different_quantities(panel) -> None:
    """The whole point of the fix: rmse and sigma_e must not collapse onto each other."""

    fitted = ws.RandomEffects.from_formula(
        "y ~ k1 + C(year)", panel, entity="st_abb", time="year"
    ).fit()
    rmse = float(fitted.e["rmse"])
    sigma_e = float(fitted.e["sigma_e"])
    assert rmse > sigma_e
    assert abs(rmse / sigma_e - 1.0) > 1e-4


# ---------------------------------------------------------------- neighbouring estimators
def test_fixed_effects_rmse_is_unchanged(panel) -> None:
    """`xtreg, fe` keeps its own formula: Stata reports 5035.6356610104 for `y ~ k1 + k2`."""

    fitted = ws.PanelOLS.from_formula(
        "y ~ k1 + k2", panel, entity="st_abb", time="year"
    ).fit()
    assert float(fitted.e["rmse"]) == pytest.approx(5035.6356610104, abs=1e-6)


def test_between_effects_rmse_is_unchanged(panel) -> None:
    """`xtreg, be`: Stata reports 8206.9664237721 for `y ~ k1 + k2`."""

    fitted = ws.BetweenOLS.from_formula(
        "y ~ k1 + k2", panel, entity="st_abb", time="year"
    ).fit()
    assert float(fitted.e["rmse"]) == pytest.approx(8206.9664237721, abs=1e-6)
