"""Pin ``varstable`` and ``vargranger`` against Stata.

References from live runs after ``var dinf unem, lags(1/2)`` on macro_swatson.dta::

    varstable
          .82182             |    .82182
          .7970223           |    .797022
         -.2258777 + .6292178i |  .668533
         -.2258777 - .6292178i |  .668533

    vargranger
        dinf <- unem   48.123   df 2   p 0.000
        unem <- dinf    9.009   df 2   p 0.011

Two details decide the eigenvalues: they belong to the COMPANION matrix (4 values for a
bivariate VAR(2), not the 2 that ``eigvals(coefs)`` returns), and Stata lists them in
descending order of modulus, so the complex pair comes last rather than being ordered by
real part.

``vargranger`` already matched to 1.4e-15 through the wrapper, but calling it raised
"'dinf' is not in list" because statsmodels labelled the equations ``y0, y1`` -- the names
were never handed to it. Passing the data as a named DataFrame fixed that without touching
any number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.varstable import varstable

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_EIGENVALUES = (
    0.82182 + 0.0j,
    0.7970223 + 0.0j,
    -0.2258777 + 0.6292178j,
    -0.2258777 - 0.6292178j,
)
STATA_MODULUS = (0.82182, 0.797022, 0.668533, 0.668533)


@pytest.fixture(scope="module")
def fitted_var():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()


def test_varstable_reports_one_eigenvalue_per_lag_times_equation(fitted_var) -> None:
    """4 = lags(2) * equations(2); eigvals(coefs) would give 2."""

    stability = varstable(fitted_var)
    assert len(stability.eigenvalues) == 4


def test_varstable_modulus_matches_stata(fitted_var) -> None:
    stability = varstable(fitted_var)
    for got, expected in zip(stability.modulus, STATA_MODULUS):
        assert got == pytest.approx(expected, abs=5e-6), (got, expected)


def test_varstable_eigenvalues_match_stata(fitted_var) -> None:
    stability = varstable(fitted_var)
    for got, expected in zip(stability.eigenvalues, STATA_EIGENVALUES):
        assert complex(got).real == pytest.approx(expected.real, abs=5e-7)
        assert complex(got).imag == pytest.approx(expected.imag, abs=5e-7)


def test_varstable_flags_the_var_as_stable(fitted_var) -> None:
    assert varstable(fitted_var).stable is True


def test_vargranger_matches_stata(fitted_var) -> None:
    expected = {("dinf", "unem"): 48.123, ("unem", "dinf"): 9.009}
    for (caused, causing), value in expected.items():
        test = fitted_var.granger_causality(caused, [causing])
        assert float(test.test_statistic) == pytest.approx(value, abs=5e-4), (caused, causing)
        assert int(test.df) == 2


def test_granger_accepts_the_equation_names(fitted_var) -> None:
    """Regression: this raised "'dinf' is not in list" before the names were passed on."""

    result = fitted_var.granger_causality("dinf", ["unem"])
    assert float(result.test_statistic) > 0.0
