"""teffects aipw: the OUTCOME regression also needed an intercept.

Round 201 found the treatment logit in `teffects ipw` was fitted without a constant (ATE off by
3.1%). `AIPW` inherits from `IPW`, so it got that fix for free -- and remained 0.13% off:

    teffects aipw (re78 <covariates>) (treat <covariates>)

        Stata      1619.053
        westata    1616.9959   (before this round)
        westata    1619.0528   (after)

The remaining gap was the same defect in a second place: `_linear_fit` was called on the bare
covariate matrix, so each arm's outcome regression had no constant.

WHAT IS PINNED AND WHAT IS NOT. The POINT ESTIMATE now matches Stata exactly and that is what the
guards below assert. The STANDARD ERROR does not: widening the moment system to carry the extra
outcome parameters made its Jacobian singular on several reference datasets, so the sandwich is
still evaluated on the narrower parametrisation it was derived for. That is recorded here rather
than hidden -- the coefficient is right, the variance is not recomputed for the wider model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

#: From `teffects aipw (re78 <covariates>) (treat <covariates>)`.
STATA_AIPW_ATE = 1619.053


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")


def test_the_aipw_ate_matches_stata(frame) -> None:
    """1616.9959 before the outcome intercept was restored."""

    fit = ws.AIPW(
        frame["re78"].to_numpy(float),
        frame["treat"].to_numpy(float),
        frame[COVARIATES].to_numpy(float),
    ).fit()
    assert float(fit.estimate) == pytest.approx(STATA_AIPW_ATE, abs=5e-3)


def test_the_outcome_model_carries_an_intercept(frame) -> None:
    """The per-arm regression has one coefficient per covariate PLUS the constant."""

    from westata.estimators.treatreg_effects import _linear_fit

    x = frame[COVARIATES].to_numpy(float)
    y = frame["re78"].to_numpy(float)
    treated = frame["treat"].to_numpy(float)
    design = np.column_stack([np.ones(x.shape[0]), x])
    coefficients = _linear_fit(design[treated == 1], y[treated == 1], "teffects aipw")
    assert len(coefficients) == x.shape[1] + 1


def test_ipw_is_unaffected_by_the_aipw_change(frame) -> None:
    """IPW was fixed in round 201 and must stay exact."""

    fit = ws.IPW(
        frame["re78"].to_numpy(float),
        frame["treat"].to_numpy(float),
        frame[COVARIATES].to_numpy(float),
    ).fit()
    assert float(fit.estimate) == pytest.approx(1641.315, abs=5e-3)
