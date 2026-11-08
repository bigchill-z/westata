"""The package's defaults agree with Stata's for the four single-equation estimators.

Round 169 lost a round to a default mismatch: `ws.ARIMA(...)` omits the constant while Stata's `arima`
includes it, which made a correct implementation look broken for a while. This checks the same question
for the rest.

Measured with nothing specified on either side:

    command    pkg const   pkg k   stata rank   constant value agreement
    regress    True        3       3            2.1e-11
    logit      True        3       3            2.8e-11
    probit     True        3       3            2.0e-11
    poisson    True        3       3            4.2e-11

All four include a constant by default, report the same rank, and agree on its value to floating-point
noise.

ARIMA IS THE EXCEPTION AND IS PINNED SEPARATELY in test_arima_estat_ic_exact.py: Stata includes a constant
there and the package does not, so the two commands must be configured alike before their output is
compared. That is a documentation matter rather than a defect -- the package follows statsmodels, whose
convention differs.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(41)
    n = 300
    data = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    data["binary"] = (data["y"] > 0).astype(float)
    data["count"] = rng.poisson(2.0, n).astype(float)
    return data


@pytest.mark.parametrize(
    ("estimator", "formula", "expected_intercept"),
    [
        ("OLS", "y ~ x1 + x2", -0.0412574166),
        ("Logit", "binary ~ x1 + x2", -0.0880482147),
        ("Probit", "binary ~ x1 + x2", -0.0552519598),
        ("Poisson", "count ~ x1 + x2", 0.6911347907),
    ],
)
def test_the_default_includes_a_constant(
    frame, estimator, formula, expected_intercept
) -> None:
    fitted = getattr(ws, estimator).from_formula(formula, frame).fit()
    assert "Intercept" in fitted.params.index
    assert float(fitted.params["Intercept"]) == pytest.approx(
        expected_intercept, abs=1e-9
    )


@pytest.mark.parametrize(
    ("estimator", "formula"),
    [
        ("OLS", "y ~ x1 + x2"),
        ("Logit", "binary ~ x1 + x2"),
        ("Probit", "binary ~ x1 + x2"),
        ("Poisson", "count ~ x1 + x2"),
    ],
)
def test_the_default_rank_is_three(frame, estimator, formula) -> None:
    """Two slopes and a constant, which is e(rank) on the Stata side as well."""

    fitted = getattr(ws, estimator).from_formula(formula, frame).fit()
    assert int(fitted.e["rank"]) == 3
    assert len(fitted.params) == 3


@pytest.mark.parametrize(
    ("estimator", "formula"),
    [
        ("OLS", "y ~ x1 + x2"),
        ("Logit", "binary ~ x1 + x2"),
        ("Probit", "binary ~ x1 + x2"),
        ("Poisson", "count ~ x1 + x2"),
    ],
)
def test_removing_the_constant_removes_the_term(frame, estimator, formula) -> None:
    """The -1 spelling works for every one of them, so the default is a choice rather than a fixture."""

    fitted = getattr(ws, estimator).from_formula(formula + " - 1", frame).fit()
    assert "Intercept" not in fitted.params.index
    assert len(fitted.params) == 2
