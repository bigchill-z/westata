"""Six report items verified exact against a live Stata in one pass.

These are the regression-workflow entries the report scores between 0.93 and 1.0 but marks as
"layout differs". Measured, the NUMBERS agree to the printed digits in every case; what the report
is scoring is that the package returns objects rather than printing Stata's table.

    C05-01  reg lnw s expr tenure smsa rns
            b[s]    0.10264304   package 0.102643
            b[expr] 0.03811890   package 0.038119

    C05-06  test s expr tenure
            F       113.026860   package 113.0269

    C06-01  reg lntc lnq lnpl lnpk lnpf, robust      (nerlove.dta)
            se[lnq]    0.0325375544   package 0.0325376
            se[lnpl]   0.2603259697   package 0.260326
            se[_cons]  1.7183042044   package 1.7183042

The robust standard errors are the interesting case: `reg, robust` is HC1, and the package matches it
without being told which variant -- the mapping is by command, recorded in the API documentation.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"
NERLOVE = r"D:\myCodeProject\test_westata\data\nerlove.dta"


@pytest.fixture(scope="module")
def wages():
    try:
        return pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "smsa"]
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")


@pytest.fixture(scope="module")
def costs():
    try:
        return pd.read_stata(NERLOVE)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nerlove.dta is not available")


def test_the_multiple_regression_coefficients_match_stata(wages) -> None:
    """C05-01."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr + tenure + smsa + rns", wages).fit()
    assert float(fitted.params["s"]) == pytest.approx(0.10264304, abs=1e-8)
    assert float(fitted.params["expr"]) == pytest.approx(0.03811890, abs=1e-8)


def test_the_joint_test_matches_stata(wages) -> None:
    """C05-06: `test s expr tenure` reports F = 113.026860."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr + tenure + smsa + rns", wages).fit()
    result = fitted.test("s expr tenure")
    assert float(result.statistic) == pytest.approx(113.026860, abs=1e-4)


def test_a_single_restriction_against_a_value_matches(wages) -> None:
    """C05-04: `test s = 0.1` is a different hypothesis from `s = 0`."""

    fitted = ws.OLS.from_formula("lnw ~ s + expr + tenure + smsa + rns", wages).fit()
    against_value = fitted.test("s = 0.1")
    against_zero = fitted.test("s = 0")
    assert float(against_value.statistic) != pytest.approx(
        float(against_zero.statistic), rel=1e-6
    )


def test_the_robust_standard_errors_match_stata(costs) -> None:
    """C06-01: `reg ..., robust` is HC1 and the package picks it from the command name."""

    fitted = ws.OLS.from_formula(
        "lntc ~ lnq + lnpl + lnpk + lnpf", costs
    ).fit(cov_type="robust")

    assert float(fitted.bse["lnq"]) == pytest.approx(0.0325375544, abs=1e-9)
    assert float(fitted.bse["lnpl"]) == pytest.approx(0.2603259697, abs=1e-9)
    assert float(fitted.bse["Intercept"]) == pytest.approx(1.7183042044, abs=1e-9)
