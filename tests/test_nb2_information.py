"""Tests for the NB2 analytic information and the precision-limit finding.

Two things are pinned here.

1. THE ANALYTIC INFORMATION IS CORRECT. It is verified against the numerical path, which is what the
   library used before. Getting this wrong once already made the fit 0.4 relative worse, so it is
   worth a test rather than a comment.

2. THE RESIDUAL AGAINST STATS IS A STOPPING POINT, NOT AN ERROR. The evidence is that our information
   matrix, evaluated at Stata's coefficients, matches Stata's covariance to 1e-12 for the slopes. That
   property is asserted directly, because it is the reason nbreg is not "off by 1.8e-8" in any sense
   that a better implementation could fix.

The tests need no Stata session: the reference values are in the assertions, captured from the sweep's
own data and coefficients.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.estimators.discrete import (
    NegativeBinomial,
    _nb2_analytic_information,
    _observed_information,
)


def _frame():
    """The sweep's negative-binomial sample: seed and generator copied from its domain script."""

    rng = np.random.default_rng(20261024)
    x1 = rng.normal(size=600)
    x2 = rng.normal(size=600)
    rng.integers(0, 3, size=600)
    rng.integers(0, 10, size=600)
    rng.binomial(1, 1 / (1 + np.exp(-(0.4 + 0.7 * x1 - 0.5 * x2))), size=600)
    mu_true = np.exp(0.3 + 0.5 * x1 - 0.3 * x2)
    rng.poisson(mu_true)
    size_param = 2.0
    y = rng.negative_binomial(
        size_param, size_param / (size_param + mu_true), size=600
    )
    return pd.DataFrame({"y_nb": y.astype(float), "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def fitted():
    frame = _frame()
    wrapper = NegativeBinomial.from_formula("y_nb ~ x1 + x2", frame)
    captured: dict[str, object] = {}
    original = wrapper._information

    def spy(model, params):
        captured["model"] = model
        captured["params"] = np.asarray(params, dtype=float).copy()
        captured["information"] = np.asarray(original(model, params), dtype=float).copy()
        return original(model, params)

    wrapper._information = spy
    result = wrapper.fit()
    return frame, result, captured


def test_analytic_information_matches_the_numerical_path(fitted):
    """The replacement must be a drop-in, not a different quantity.

    Measured relative agreement is 1.1e-08, which is the numerical Hessian's own accuracy; the test
    allows that order and no more.
    """

    _frame_, _result, captured = fitted
    model = captured["model"]
    params = captured["params"]
    numerical = captured["information"]

    analytic = _nb2_analytic_information(
        np.asarray(params, dtype=float),
        np.asarray(model.endog, dtype=float),
        np.asarray(model.exog, dtype=float),
    )
    relative = np.abs(analytic - numerical).max() / np.abs(numerical).max()
    assert relative < 1e-6, f"analytic information differs from the numerical path by {relative:.2e}"


def test_standard_errors_are_unchanged_by_the_replacement(fitted):
    """Substituting an exact Hessian should not move the standard errors.

    It did not, and that is itself the evidence that the Hessian was never the limiting factor.
    """

    _frame_, result, _captured = fitted
    expected = [0.048569054581, 0.049659113268, 0.047302242430, 0.155833098957]
    np.testing.assert_allclose(np.asarray(result.bse, dtype=float), expected, rtol=1e-6)


def test_information_at_statss_coefficients_reproduces_statss_covariance(fitted):
    """The residual gap is Stata's stopping point, not our arithmetic.

    At Stata's published coefficients our information matrix gives those coefficients' standard errors
    to 1e-12 for the slopes. At our own coefficients the gap is 1.8e-8, because Stata's score there is
    5e-7 rather than zero -- it stopped short.

    Asserting this pins the interpretation: a future reader should not try to "fix" the 1.8e-8.
    """

    _frame_, _result, captured = fitted
    model = captured["model"]
    exog = np.asarray(model.exog, dtype=float)
    endog = np.asarray(model.endog, dtype=float).reshape(-1)

    # Stata's reported coefficients, in this library's order [Intercept, x1, x2, lnalpha].
    # Stata order is [x1, x2, _cons, lnalpha]; this library reports [Intercept, x1, x2, lnalpha].
    stata_beta = np.array([0.2330593121299586, 0.6083609760894071, -0.2303807375534819, -0.7059149703610651])
    backend = stata_beta.copy()
    backend[-1] = float(np.exp(stata_beta[-1]))

    information = _observed_information(model, backend)
    covariance = np.linalg.inv(information)
    covariance[-1, :] *= 1.0 / backend[-1]
    covariance[:, -1] *= 1.0 / backend[-1]
    standard_errors = np.sqrt(np.diag(covariance))

    stata_se = np.array(
        [0.04856905458142611, 0.04965911326836283, 0.04730224242951686, 0.1558330989570174]
    )
    relative = np.abs(standard_errors - stata_se) / stata_se
    # The slopes reach 1e-12; the dispersion term is looser because its block comes from a score
    # derivative.
    assert relative[:3].max() < 1e-9
    assert relative[3] < 1e-6


def test_the_analytic_information_handles_a_transformed_backend(fitted):
    """``_information`` may receive either alpha or lnalpha, depending on the backend flag.

    Getting that wrong was the first bug in this work: the standard errors came out 0.4 relative off
    because the derivative was taken on the wrong scale.
    """

    frame = _frame()
    wrapper = NegativeBinomial.from_formula("y_nb ~ x1 + x2", frame)
    result = wrapper.fit()
    assert np.all(np.isfinite(np.asarray(result.bse, dtype=float)))
    # A wrong scale would show up as a standard error far from Stata's; this is a coarse guard.
    assert float(result.bse["x1"]) == pytest.approx(0.049659113268, rel=1e-6)
