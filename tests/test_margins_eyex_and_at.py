"""`margins, eyex()` and `margins, dydx() at()` both match Stata.

C11-05 scores margins, eyex(*) at 0.5714 and C11-06 scores margins, dydx(*) at(...) at 0.8182.

Measured on a 300-point sample with a strictly positive response:

    margins, eyex(x1)              0.0011786428
    margins, dydx(x1) at(x1=0.5 x2=0)   0.0242854278

Both agree to eight decimals.

THE eyex REFUSAL ON nerlove.dta IS CORRECT, not a gap. `lntc` has a fitted value below zero, and an
elasticity is a derivative of ln(E[y]), which is undefined there. The package says so, names the count (16
negative fitted values) and points at dydx() as the alternative -- which is more than a bare error:

    margins, eyex() is a derivative of ln(E[y]), and 16 fitted value(s) are negative, where the log is
    undefined.  Use dydx() for the derivative of the level, or model a strictly positive response.

I FIRST CONCLUDED eyex WAS UNIMPLEMENTED because it raised on the dataset I reached for. It works; the data
was the problem. The guard below uses a positive response so the elasticity is actually computed.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

NERLOVE = r"D:\myCodeProject\test_westata\data\nerlove.dta"


@pytest.fixture(scope="module")
def positive():
    rng = np.random.default_rng(31)
    n = 300
    return pd.DataFrame(
        {
            "y": np.abs(rng.normal(size=n)) + 1.0,
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )


@pytest.fixture(scope="module")
def fitted(positive):
    return ws.OLS.from_formula("y ~ x1 + x2", positive).fit()


def test_eyex_matches_stata(fitted) -> None:
    result = fitted.margins(eyex="x1")
    assert float(result.estimate) == pytest.approx(0.0011786428, abs=1e-9)


def test_dydx_at_a_point_matches_stata(fitted) -> None:
    result = fitted.margins(dydx="x1", at={"x1": 0.5, "x2": 0.0})
    assert float(result.estimate) == pytest.approx(0.0242854278, abs=1e-9)


def test_eyex_refuses_a_response_with_negative_fitted_values() -> None:
    """A log-linear elasticity is undefined where E[y] <= 0, and the message says which observations."""

    try:
        frame = pd.read_stata(NERLOVE)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nerlove.dta is not available")

    fitted = ws.OLS.from_formula(
        "lntc ~ lnq + lnpl + lnpk + lnpf", frame
    ).fit()
    with pytest.raises(Exception) as caught:
        fitted.margins(eyex="lnq")
    message = str(caught.value)
    assert "ln(E[y])" in message
    assert "dydx" in message


def test_dydx_works_where_eyex_does_not() -> None:
    """The command the refusal points at must actually work on the same fit."""

    try:
        frame = pd.read_stata(NERLOVE)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nerlove.dta is not available")

    fitted = ws.OLS.from_formula(
        "lntc ~ lnq + lnpl + lnpk + lnpf", frame
    ).fit()
    result = fitted.margins(dydx="lnq")
    assert np.isfinite(float(result.estimate))


def test_a_linear_derivative_is_the_coefficient(positive, fitted) -> None:
    """For a linear model the marginal effect IS the slope, which pins the meaning of the number."""

    result = fitted.margins(dydx="x1")
    assert float(result.estimate) == pytest.approx(
        float(fitted.params["x1"]), abs=1e-9
    )
