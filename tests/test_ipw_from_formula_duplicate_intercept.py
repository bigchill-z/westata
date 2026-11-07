"""`IPW.from_formula` failed on nsw_dw: the propensity design had TWO intercepts.

N of the v3 report: `IPW.from_formula` raised

    ConsistencyError: teffects ipw: treatment (logit) model did not converge in 100 iterations

while constructing `IPW(y, treated, x)` directly worked and returned Stata's ATE.

THE CAUSE. `_LogitPropensity.__init__` prepended a constant unconditionally:

    self._design = np.column_stack([np.ones(x.shape[0]), x])

but `_prepare` -- the helper `from_formula` goes through -- builds its design via patsy, which
ALREADY emits an Intercept column. The logit therefore had two identical constant columns: a
singular Hessian, no step the line search would accept, and a convergence failure that had nothing
to do with the iteration budget. Raising maxiter from 100 to 500 changed only the number in the
error message.

With `_has_intercept` guarding the prepend, the same eight covariates converge in FOUR Newton
steps (gradient norm 0.0, intercept 1.177674 against Stata's 1.177674) and both entry points
return ATE 1641.3152.

This is the same class of mistake the module had already fixed for the outcome and RA models --
`_has_intercept` existed and was used there, just not on this path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]
STATA_ATE = 1641.3152


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")


def test_from_formula_converges(frame) -> None:
    """It raised on this exact data before the duplicate constant was removed."""

    fitted = ws.IPW.from_formula("re78", "treat", frame, covariates=COVARIATES).fit()
    assert np.isfinite(float(fitted.estimate))


def test_from_formula_matches_stata(frame) -> None:
    fitted = ws.IPW.from_formula("re78", "treat", frame, covariates=COVARIATES).fit()
    assert float(fitted.estimate) == pytest.approx(STATA_ATE, abs=1e-3)


def test_both_entry_points_agree(frame) -> None:
    """A formula build and an array build must produce the same estimate."""

    direct = ws.IPW(
        frame["re78"].to_numpy(dtype=float),
        frame["treat"].to_numpy(dtype=float),
        frame[COVARIATES].to_numpy(dtype=float),
    ).fit()
    formula = ws.IPW.from_formula("re78", "treat", frame, covariates=COVARIATES).fit()
    assert float(direct.estimate) == pytest.approx(float(formula.estimate), rel=1e-12)


def test_the_propensity_design_has_one_constant(frame) -> None:
    """The defect was collinearity, so check the rank directly rather than its symptom."""

    from westata.estimators.treatreg_effects import _has_intercept, _prepare

    _, _, design = _prepare(
        "re78",
        "treat",
        frame,
        covariates=COVARIATES,
        subset=None,
        command="teffects ipw",
    )
    assert _has_intercept(design)
    # One constant column, plus the eight covariates -- not nine covariates plus one.
    assert design.shape[1] == len(COVARIATES) + 1


def test_a_design_without_a_constant_still_gets_one(frame) -> None:
    """The fallback must not be lost: an array design carries no intercept of its own."""

    from westata.estimators.treatreg_effects import _LogitPropensity

    design = frame[COVARIATES].to_numpy(dtype=float)
    assert not np.allclose(design[:, 0], 1.0)
    propensity = _LogitPropensity(
        design,
        frame["treat"].to_numpy(dtype=float),
        command="teffects ipw",
    )
    # The fitted design must end up with a constant, however it was supplied.
    assert np.allclose(propensity._design[:, 0], 1.0)
    assert propensity._design.shape[1] == design.shape[1] + 1
