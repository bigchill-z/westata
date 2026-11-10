"""`estat ovtest, rhs` fired its "all collinear" guard on any model with three indicators.

`ovtest.ado` refuses `rhs` only when NO continuous regressor remains -- its message is "powers of
all explanatory variables collinear / (typically because all explanatory variables are indicator
variables)". The package decided that by comparing the number of generated power columns (three
per continuous variable) against the count of every non-constant name, dummies included:

    if len(omitted_labels) <= len([n for n in names if not _is_constant(n)]):

With one continuous variable and two dummies that is `3 <= 3`, so the guard fired and the command
raised instead of reporting a result:

    reg y x1 i.grp      Stata  F(3, 293) = 0.17   package  ValueError

THE GUARD NOW COUNTS CONTINUOUS REGRESSORS: each contributes exactly three powers, so dividing the
column count by three says how many were testable, and only zero is a refusal. A model that is
genuinely all indicators still raises, and a live Stata still raises `r(499)` with the same wording.

Verified against a live capture at three sample sizes crossed with two specifications: 6 of 6
byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 300, seed: int = 3, with_x2: bool = False) -> pd.DataFrame:
    """The capture that fixes the expected values draws y, x1, then grp, in that order.

    Adding an unused column changes the RNG stream and therefore every statistic, so the column
    set is a parameter rather than always present.
    """

    rng = np.random.default_rng(seed)
    data = {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    if with_x2:
        data["x2"] = rng.normal(size=n)
    frame = pd.DataFrame(data)
    frame["grp"] = rng.integers(0, 3, n).astype(float)
    return frame


@pytest.mark.parametrize("n", [300, 500])
def test_a_continuous_variable_beside_a_factor_is_testable(n) -> None:
    """One continuous regressor plus a factor: three powers, not a refusal."""

    fitted = ws.OLS.from_formula("y ~ x1 + C(grp)", _frame(n)).fit()
    result = ws.estat_ovtest(fitted, rhs=True)
    assert result.df == 3, result.df
    assert np.isfinite(result.statistic)


def test_two_continuous_variables_give_six_degrees_of_freedom() -> None:
    fitted = ws.OLS.from_formula("y ~ x1 + x2 + C(grp)", _frame(with_x2=True)).fit()
    result = ws.estat_ovtest(fitted, rhs=True)
    assert result.df == 6, result.df


def test_the_captured_values_match() -> None:
    """reg y x1 i.grp then `estat ovtest, rhs` at n=300: F(3, 293) = .17, p = .9155."""

    fitted = ws.OLS.from_formula("y ~ x1 + C(grp)", _frame()).fit()
    result = ws.estat_ovtest(fitted, rhs=True)
    assert round(float(result.statistic), 2) == 0.17
    assert result.df_denom == 293
    assert round(float(result.pvalue), 4) == 0.9155


def test_the_label_and_layout_are_unchanged() -> None:
    fitted = ws.OLS.from_formula("y ~ x1 + C(grp)", _frame()).fit()
    lines = [line for line in str(ws.estat_ovtest(fitted, rhs=True)).splitlines()]
    assert lines[1] == "Ramsey RESET test for omitted variables"
    assert lines[2] == "Omitted: Powers of independent variables"


def test_an_all_indicator_model_still_refuses() -> None:
    """Stata raises r(499) here; so must this, with the same wording."""

    rng = np.random.default_rng(5)
    frame = pd.DataFrame({"y": rng.normal(size=300)})
    frame["g1"] = (rng.normal(size=300) > 0).astype(float)
    frame["g2"] = (rng.normal(size=300) > 0.5).astype(float)
    fitted = ws.OLS.from_formula("y ~ g1 + g2", frame).fit()
    with pytest.raises(ValueError, match="powers of all explanatory variables collinear"):
        ws.estat_ovtest(fitted, rhs=True)


def test_the_default_path_is_unaffected() -> None:
    """`rhs=False` uses powers of the fitted values and must still return three df."""

    fitted = ws.OLS.from_formula("y ~ x1 + C(grp)", _frame()).fit()
    result = ws.estat_ovtest(fitted)
    assert result.df == 3
