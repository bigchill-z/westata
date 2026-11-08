"""Predictive margins reproduce Stata's, including the delta-method standard error.

C11-04 scores `margins, dydx(*)` at 0.9474. Measured on nerlove.dta with the Cobb-Douglas cost
regression:

    quietly reg lntc lnq lnpl lnpk lnpf
    quietly margins
        r(table)[1,1]   Stata 1.7246632826    package 1.72466328
        r(table)[2,1]   Stata 0.0325758651    package 0.03257587

Both agree to eight decimals. `margins()` at the means of the regressors is the linear prediction,
so it equals the linear combination of the coefficients at the sample means, and the standard error
is the delta-method one -- which is what Stata reports.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws

NERLOVE = r"D:\myCodeProject\test_westata\data\nerlove.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(NERLOVE)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nerlove.dta is not available")
    return ws.OLS.from_formula("lntc ~ lnq + lnpl + lnpk + lnpf", frame).fit()


def test_the_margin_matches_stata(fitted) -> None:
    result = fitted.margins()
    assert float(result.estimate) == pytest.approx(1.7246632826, abs=1e-8)


def test_the_standard_error_matches_stata(fitted) -> None:
    """The delta-method standard error, not the residual rmse."""

    result = fitted.margins()
    assert float(result.std_err) == pytest.approx(0.0325758651, abs=1e-9)


def test_the_margin_equals_the_linear_prediction_at_the_means(fitted) -> None:
    """margins() evaluates at the sample means, so it must equal the fitted value there."""

    result = fitted.margins()
    design_means = fitted.model.exog.mean(axis=0)
    expected = float(design_means @ np.asarray(fitted.params, dtype=float))
    assert float(result.estimate) == pytest.approx(expected, abs=1e-8)
