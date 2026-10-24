"""Tests for NegativeBinomialNB1, pinned against a live Stata session's numbers.

The model is Stata's `nbreg, dispersion(constant)`: `Var = mu (1 + delta)`, as opposed to the default NB2's
`Var = mu + alpha mu^2`.

THE ERROR THIS GUARDS AGAINST IS SPECIFIC. The likelihood's gamma terms involve `m = mu/delta`, not `mu`
(`nbreg_al.ado` derives `m = exp(xb - lndelta)`). Using `mu` there still converges, still reports plausible
coefficients, and is simply a different model -- the mismatch shows up only when the likelihood is compared at
STAT'S estimates, so a test that only checks "the fit converged" would pass throughout. Measured with the
wrong parameterisation, the likelihood at Stata's coefficients was -947.25 against Stata's own -936.95.

A second trap is pinned too: `e(V)` lives on the `/lndelta` scale, and Stata prints `delta`'s delta-method SE
as a SEPARATE quantity. Applying that Jacobian to `e(V)` put delta's SE under the /lndelta name and was 18.4%
wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.discrete import _nb1_loglike, _nb1_score


def _sample(seed: int = 20261024, n: int = 600):
    """The sweep's negative-binomial sample, so the reference numbers apply to it."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    rng.integers(0, 3, size=n)
    rng.integers(0, 10, size=n)
    rng.binomial(1, 1 / (1 + np.exp(-(0.4 + 0.7 * x1 - 0.5 * x2))), size=n)
    mu = np.exp(0.3 + 0.5 * x1 - 0.3 * x2)
    rng.poisson(mu)
    size_parameter = 2.0
    y = rng.negative_binomial(
        size_parameter, size_parameter / (size_parameter + mu), size=n
    ).astype(float)
    return pd.DataFrame({"y_nb": y, "x1": x1, "x2": x2})


# Stata: nbreg y_nb x1 x2, dispersion(constant)
STATA_B = np.array(
    [0.5524559405156866, -0.21350287321571898, 0.2619664533767144, -0.20370682099400184]
)
STATA_SE = np.array(
    [0.044217201944317365, 0.04504052532308135, 0.05069124599840919, 0.16141972913750582]
)
STATA_DELTA = 0.815701482684229
STATA_LL = -936.9465087153782


def test_nb1_matches_stata():
    frame = _sample()
    result = ws.NegativeBinomialNB1.from_formula("y_nb ~ x1 + x2", frame).fit()

    # Our order is [Intercept, x1, x2, /lndelta]; Stata's is [x1, x2, _cons, /lndelta].
    order = [1, 2, 0, 3]
    # 1e-7 rather than machine precision: the fit stops at its gradient tolerance, the same pattern as the
    # other ML models here (GAP_REGISTER 8.11). Measured gap is 2.9e-09 on the coefficients and 3.4e-10 on the
    # standard errors, so 1e-7 still distinguishes the correct parameterisation from m = mu, which was 1e-02
    # away.
    np.testing.assert_allclose(
        np.asarray(result.params, dtype=float)[order], STATA_B, rtol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(result.bse, dtype=float)[order], STATA_SE, rtol=1e-7
    )


def test_nb1_loglikelihood_matches_stata_at_a_shared_point():
    """The sharpest test of the parameterisation, and the one that caught `m` versus `mu`.

    Comparing the likelihood at *our* estimates can pass by luck when both fits converge somewhere similar.
    Comparing it at STAT'S coefficients tests the likelihood itself.
    """

    frame = _sample()
    X = np.column_stack([np.ones(len(frame)), frame["x1"], frame["x2"]])
    y = frame["y_nb"].to_numpy()
    theta = np.array([STATA_B[2], STATA_B[0], STATA_B[1], STATA_B[3]])
    assert _nb1_loglike(theta, y, X) == pytest.approx(STATA_LL, rel=1e-12)


def test_nb1_score_is_the_gradient_of_the_likelihood():
    """A sign or a missing term in the score would still converge, only more slowly."""

    frame = _sample()
    X = np.column_stack([np.ones(len(frame)), frame["x1"], frame["x2"]])
    y = frame["y_nb"].to_numpy()
    theta = np.array([0.26, 0.55, -0.21, -0.20])

    analytic = _nb1_score(theta, y, X)
    numeric = np.empty_like(theta)
    step = 1e-6
    for column in range(theta.size):
        plus = theta.copy()
        minus = theta.copy()
        plus[column] += step
        minus[column] -= step
        numeric[column] = (
            _nb1_loglike(plus, y, X) - _nb1_loglike(minus, y, X)
        ) / (2 * step)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-6)


def test_nb1_reports_delta_and_its_own_standard_error():
    """`e(V)` is on the /lndelta scale; `e(delta)` and its SE are separate quantities."""

    frame = _sample()
    result = ws.NegativeBinomialNB1.from_formula("y_nb ~ x1 + x2", frame).fit()

    assert float(result.e["delta"]) == pytest.approx(STATA_DELTA, rel=1e-8)
    # The /lndelta standard error in e(b)/e(V) must be the UNTRANSFORMED one.
    lndelta_index = list(result.param_names).index("/lndelta")
    assert float(np.asarray(result.bse, dtype=float)[lndelta_index]) == pytest.approx(
        STATA_SE[3], rel=1e-8
    )
    # And the displayed delta SE is the delta-method one, which is a different number.
    assert float(result.e["sedelta"]) == pytest.approx(0.1316703, rel=1e-6)
    assert float(result.e["sedelta"]) != pytest.approx(STATA_SE[3], rel=1e-3)


def test_nb1_and_nb2_are_different_models():
    """A guard against someone collapsing the two into one class with a flag.

    They fit different likelihoods, so the coefficients must differ materially on the same data.
    """

    frame = _sample()
    nb1 = ws.NegativeBinomialNB1.from_formula("y_nb ~ x1 + x2", frame).fit()
    nb2 = ws.NegativeBinomial.from_formula("y_nb ~ x1 + x2", frame).fit()

    nb1_b = np.asarray(nb1.params, dtype=float)
    nb2_b = np.asarray(nb2.params, dtype=float)
    assert abs(float(nb1.e["ll"]) - float(nb2.e["ll"])) > 1e-3
    assert not np.allclose(nb1_b[:3], nb2_b[:3], rtol=1e-3)
