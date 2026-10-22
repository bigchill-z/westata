"""E2 evidence for Stata ``regress`` on the canonical ``auto.dta`` example.

Two independent checks:

1. Full-precision algebraic cross-check against statsmodels (both implement the
   same OLS normal equations, so they must agree to machine precision).
2. Agreement with the *published* Stata output for
   ``regress mpg weight foreign`` on ``auto.dta`` (n=74).  The manual prints
   6-7 significant digits, so this check necessarily uses a rounding-tolerant
   relative tolerance of 5e-5.

Published Stata output (Stata 18, [R] regress, auto.dta):

    Number of obs = 74        F(2, 71) = 69.75
    R-squared     = 0.6627    Adj R-squared = 0.6532
    Root MSE      = 3.4071

    mpg | Coefficient  Std. err.      t    P>|t|   [95% conf. interval]
    weight | -.0065879   .0006371   -10.34   0.000
    foreign| -1.650029   1.075994    -1.53   0.128
    _cons  |   41.6797   2.165547    19.25   0.000
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from westata.estimators.linear import OLS

DATA = Path(__file__).resolve().parents[1] / "data" / "auto.dta"

pytestmark = pytest.mark.skipif(
    not DATA.exists(), reason="auto.dta not downloaded (run scripts/fetch_stata_datasets.py)"
)

# param order: Intercept, weight, foreign
FULL_PRECISION_PARAMS = np.array([41.6797023261, -0.00658788637577, -1.65002910584])
FULL_PRECISION_BSE = np.array([2.16554718966, 0.000637112934854, 1.07599411819])

PUBLISHED_PARAMS = np.array([41.6797, -0.0065879, -1.650029])
PUBLISHED_BSE = np.array([2.165547, 0.0006371, 1.075994])


@pytest.fixture(scope="module")
def fitted():
    df = pd.read_stata(DATA, convert_categoricals=False)
    return df, OLS.from_formula("mpg ~ weight + foreign", df).fit()


def test_matches_full_precision_reference(fitted):
    _, res = fitted
    np.testing.assert_allclose(res.params.to_numpy(), FULL_PRECISION_PARAMS, rtol=1e-10)
    np.testing.assert_allclose(res.bse.to_numpy(), FULL_PRECISION_BSE, rtol=1e-10)


def test_matches_published_stata_output(fitted):
    _, res = fitted
    np.testing.assert_allclose(res.params.to_numpy(), PUBLISHED_PARAMS, rtol=5e-5)
    np.testing.assert_allclose(res.bse.to_numpy(), PUBLISHED_BSE, rtol=5e-5)


def test_published_fit_statistics(fitted):
    _, res = fitted
    assert res.e["N"] == 74
    assert res.e["df_m"] == 2
    assert res.e["df_r"] == 71
    assert res.rsquared == pytest.approx(0.6627, rel=5e-5)
    assert res.rsquared_adj == pytest.approx(0.6532, rel=5e-5)
    assert res.e["rmse"] == pytest.approx(3.4071, rel=5e-5)
    assert res.e["F"] == pytest.approx(69.75, rel=5e-5)
