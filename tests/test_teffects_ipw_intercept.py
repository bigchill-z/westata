"""teffects ipw: the treatment model was fitted WITHOUT an intercept, a 3.1% error.

C15-06 scored 0.06 layout / 0.77 numeric. Auditing it found a genuine estimation defect, not a
formatting one:

    teffects ipw (re78) (treat age education black hispanic married nodegree re74 re75)

        Stata ATE   1641.315
        westata     1589.9166      <- 3.1% off

THE CAUSE was one missing column. ``_LogitPropensity`` fitted the logit on the covariate matrix
exactly as given, so ``beta`` had 8 entries for 8 covariates -- no constant. A live
``logit treat age ... re75`` reports ``_cons = 1.177674``, identical to the intercept this package's
own ``Logit`` produces on the same design, so the intercept belongs in the treatment model. Without
it the propensity scores were wrong by up to 0.087 and the Hajek weighting inherited the error.

WHAT MADE IT INVISIBLE: the module docstring describes the Hajek formula and the influence function
correctly, and cites verified numbers (1.8882, 1.5641, 0.1001317) from a DIFFERENT dataset where the
missing constant happened not to show up as a percentage-scale discrepancy. Reading the docstring
would suggest everything was checked; the fit itself never was.

After the fix: ATE 1641.3152 and ATET 1806.4169, both matching live Stata to the printed digit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

#: From `teffects ipw (re78) (treat <covariates>)` and the same with `, atet`.
STATA_ATE = 1641.315
STATA_ATET = 1806.417


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")


def test_the_ate_matches_stata(frame) -> None:
    """1589.9166 before the intercept was restored."""

    fit = ws.IPW(
        frame["re78"].to_numpy(float),
        frame["treat"].to_numpy(float),
        frame[COVARIATES].to_numpy(float),
    ).fit()
    assert float(fit.estimate) == pytest.approx(STATA_ATE, abs=5e-3)


def test_the_atet_matches_stata(frame) -> None:
    fit = ws.IPW(
        frame["re78"].to_numpy(float),
        frame["treat"].to_numpy(float),
        frame[COVARIATES].to_numpy(float),
        estimand="atet",
    ).fit()
    assert float(fit.estimate) == pytest.approx(STATA_ATET, abs=5e-3)


def test_the_treatment_model_carries_an_intercept(frame) -> None:
    """beta has one entry per covariate PLUS the constant."""

    from westata.estimators.treatreg_effects import _LogitPropensity

    x = frame[COVARIATES].to_numpy(float)
    model = _LogitPropensity(x, frame["treat"].to_numpy(float), command="teffects ipw")
    assert len(model.beta) == x.shape[1] + 1
    # And it agrees with the package's own Logit intercept on the same design.
    reference = ws.Logit(
        frame["treat"].to_numpy(float), np.column_stack([np.ones(x.shape[0]), x])
    ).fit()
    assert float(model.beta[0]) == pytest.approx(float(reference.params.iloc[0]), abs=1e-5)


def test_the_propensity_scores_match_a_reference_logit(frame) -> None:
    """The difference was up to 0.087 before, which is what moved the estimate."""

    from westata.estimators.treatreg_effects import _LogitPropensity

    x = frame[COVARIATES].to_numpy(float)
    treated = frame["treat"].to_numpy(float)
    model = _LogitPropensity(x, treated, command="teffects ipw")
    reference = np.asarray(
        ws.Logit(treated, np.column_stack([np.ones(x.shape[0]), x])).fit().predict()
    )
    assert float(np.max(np.abs(model.propensity - reference))) < 1e-9
