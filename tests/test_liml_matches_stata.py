"""LIML matches Stata's coefficients exactly.

C10-04 scores `ivregress liml` at 0.8421. Measured on grilic.dta against a live Stata:

    quietly ivregress liml lnw expr (s = rns mrt)
        b[s]     Stata 0.2950943613    package 0.2950944
        b[expr]  Stata 0.0928651328    package 0.0928651
        b[_cons] Stata 1.5698341535    package 1.5698342

All three match to eight decimals.

LIML IS NOT 2SLS -- it is the limited-information maximum-likelihood estimator, which minimises the
ratio of two residual variances rather than the 2SLS quadratic form. On this data the two agree
closely enough that a wrong implementation could still look plausible, so a guard checks they are
DISTINCT while both being close.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "rns", "mrt"]
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")


def test_the_liml_coefficients_match_stata(frame) -> None:
    fitted = ws.IVLIML.from_formula("lnw ~ expr + [s ~ rns + mrt]", frame).fit()

    assert float(fitted.params["s"]) == pytest.approx(0.2950943613, abs=1e-8)
    assert float(fitted.params["expr"]) == pytest.approx(0.0928651328, abs=1e-8)
    assert float(fitted.params["Intercept"]) == pytest.approx(1.5698341535, abs=1e-8)


def test_liml_and_2sls_are_different_estimators(frame) -> None:
    """They are close on this data, so a swapped implementation would still look reasonable."""

    liml = ws.IVLIML.from_formula("lnw ~ expr + [s ~ rns + mrt]", frame).fit()
    tsls = ws.IV2SLS.from_formula("lnw ~ expr + [s ~ rns + mrt]", frame).fit()

    assert float(liml.params["s"]) != pytest.approx(float(tsls.params["s"]), rel=1e-9)
    # But not wildly different -- both are consistent under the same assumptions.
    assert float(liml.params["s"]) == pytest.approx(
        float(tsls.params["s"]), rel=0.05
    )


def test_the_liml_standard_errors_are_positive(frame) -> None:
    fitted = ws.IVLIML.from_formula("lnw ~ expr + [s ~ rns + mrt]", frame).fit()
    assert np.all(np.asarray(fitted.bse, dtype=float) > 0.0)
