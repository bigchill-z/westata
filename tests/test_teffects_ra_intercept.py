"""teffects ra: the arm regressions were the THIRD place the intercept was missing.

Rounds 201 and 202 fixed the missing constant in the treatment logit (IPW) and in AIPW's outcome
model. `teffects ra` fits a separate outcome regression per arm through `_arm_fit`, and that call
had the same omission:

    teffects ra (re78 <covariates>) (treat)

        Stata      1621.583
        westata    1679.0065   (before)   -- 3.5% off
        westata    1621.5831   (after)

THE FIRST ATTEMPT AT THIS FIX BROKE 21 TESTS, and the reason is worth keeping: `from_formula`
builds the design through patsy, which already emits an `Intercept` column of ones, while a caller
passing arrays directly does not. Adding a second intercept to a formula-built design makes the
cross-product singular ("an arm's cross-product matrix is singular"). `_has_intercept` now decides,
so both entry points get exactly one constant.

All three estimators are now exact on nsw_dw: ra 1621.5831, aipw 1619.0528, ipw 1641.3152.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

STATA_RA_ATE = 1621.583
STATA_RA_ATET = 1787.761
STATA_AIPW_ATE = 1619.053
STATA_IPW_ATE = 1641.315


def _frame():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")


@pytest.mark.parametrize("estimand, expected", [("ate", STATA_RA_ATE), ("atet", STATA_RA_ATET)])
def test_regression_adjustment_matches_stata(estimand: str, expected: float) -> None:
    frame = _frame()
    fit = ws.RegressionAdjustment(
        frame["re78"].to_numpy(float),
        frame["treat"].to_numpy(float),
        frame[COVARIATES].to_numpy(float),
        estimand=estimand,
    ).fit()
    assert float(fit.estimate) == pytest.approx(expected, abs=5e-3)


def test_all_three_estimators_are_exact_together() -> None:
    """The constants were fixed in three different rounds; this pins the result of all three."""

    frame = _frame()
    y = frame["re78"].to_numpy(float)
    treated = frame["treat"].to_numpy(float)
    x = frame[COVARIATES].to_numpy(float)
    assert float(ws.RegressionAdjustment(y, treated, x).fit().estimate) == pytest.approx(
        STATA_RA_ATE, abs=5e-3
    )
    assert float(ws.AIPW(y, treated, x).fit().estimate) == pytest.approx(
        STATA_AIPW_ATE, abs=5e-3
    )
    assert float(ws.IPW(y, treated, x).fit().estimate) == pytest.approx(
        STATA_IPW_ATE, abs=5e-3
    )


def test_the_intercept_is_not_doubled_for_a_formula_built_design() -> None:
    """`from_formula` already supplies one; adding another made the cross-product singular."""

    from westata.estimators.treatreg_effects import _has_intercept

    frame = _frame()
    x = frame[COVARIATES].to_numpy(float)
    assert not _has_intercept(x)          # arrays: no constant yet
    assert _has_intercept(np.column_stack([np.ones(x.shape[0]), x]))
    # And a formula-built model's design is recognised as already carrying one.
    model = ws.RegressionAdjustment.from_formula("re78", "treat", frame)
    assert _has_intercept(model._x)
