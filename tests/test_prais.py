"""Pin ``prais`` against Stata's iterated Prais-Winsten output.

Reference values come from a live run on ``icecream.dta`` (30 observations)::

    prais consumption temp price income
    Iteration 40: rho = 0.8002

    temp    .0029541   se .0007109
    price  -1.048854    se .759751
    income -.0008022    se .0020458
    _cons   .5870049    se .2952699
    rho     .8002264

    R-squared 0.6234   Adj R-squared 0.5799   Root MSE .03232   F(3, 26) 14.35
    Durbin-Watson (original)    1.021169
    Durbin-Watson (transformed) 1.846795

Three details were only findable by comparing against those numbers:

* R-squared / Root MSE are on the TRANSFORMED scale (reporting the original-scale sums gave
  0.521716 and .048053);
* the "original" Durbin-Watson comes from the ORDINARY regression's residuals, not from the
  Prais-Winsten residual (which gave 0.482716 instead of 1.021169);
* the first observation is scaled by sqrt(1 - rho^2) rather than dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.estimators.prais import prais

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    frame = frame.sort_values("time").reset_index(drop=True)
    return prais(frame, "consumption ~ temp + price + income")


def test_rho_matches_stata(fitted) -> None:
    assert fitted.rho == pytest.approx(0.8002264, abs=1e-5)


def test_coefficients_match_stata(fitted) -> None:
    expected = {
        "Intercept": 0.5870049,
        "temp": 0.0029541,
        "price": -1.048854,
        "income": -0.0008022,
    }
    for name, value in expected.items():
        # Stata printed these to 7 significant digits, so `income` (-.0008022 against
        # -.00080225) is inside the rounding of the reference, not a computation gap.
        assert fitted.params[name] == pytest.approx(value, rel=8e-5), name


def test_standard_errors_match_stata(fitted) -> None:
    expected = {
        "Intercept": 0.2952699,
        "temp": 0.0007109,
        "price": 0.759751,
        "income": 0.0020458,
    }
    for name, value in expected.items():
        assert fitted.bse[name] == pytest.approx(value, rel=2e-4), name


def test_fit_statistics_are_on_the_transformed_scale(fitted) -> None:
    """R2 .6234 / adj .5799 / RMSE .03232 / F 14.35 -- the transformed scale, not the raw one."""

    assert fitted.rsquared == pytest.approx(0.6234, abs=5e-5)
    assert fitted.rsquared_adj == pytest.approx(0.5799, abs=5e-5)
    assert fitted.rmse == pytest.approx(0.03232, abs=5e-6)
    assert fitted.fvalue == pytest.approx(14.35, abs=5e-3)


def test_both_durbin_watson_values_match_stata(fitted) -> None:
    assert fitted.dw_original == pytest.approx(1.021169, abs=1e-6)
    assert fitted.dw_transformed == pytest.approx(1.846795, abs=5e-5)


def test_the_first_observation_is_scaled_not_dropped(fitted) -> None:
    """Prais-Winsten keeps all 30 observations; Cochrane-Orcutt would keep 29."""

    assert fitted.nobs == 30
    assert fitted.df_resid == 26
