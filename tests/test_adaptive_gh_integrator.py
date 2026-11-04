"""Guard the adaptive Gauss-Hermite integrator against the constants drifting again.

The bug this protects against shipped in 0.0.5 and earlier: the integrand carried an extra
``-0.5*(tau/sigma)**2`` term, so the fixed-point iteration that re-centres each panel shrank
its scale monotonically towards zero instead of settling. For ``xtlogit, re`` on data with a
small random-effect variance the objective then grew to ~1e8 and the reported ``e(sigma_u)``
was off by two orders of magnitude.

The tests below pin the integrator against closed forms rather than against our own
previous output, because agreeing with an earlier version proves nothing.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from westata.estimators.panel_nonlinear import _adaptive_log_integral_batch, _gauss_hermite


def _reference(log_integrand, sigma: float) -> float:
    """``log int exp(log_integrand(u)) * phi(u; 0, sigma) du`` by adaptive quadrature."""

    def density(u: float) -> float:
        return float(np.exp(log_integrand(np.asarray(u)))) * float(
            np.exp(-0.5 * (u / sigma) ** 2) / (np.sqrt(2.0 * np.pi) * sigma)
        )

    value, _ = integrate.quad(density, -40.0 * sigma, 40.0 * sigma, limit=800)
    return float(np.log(value))


@pytest.mark.parametrize("sigma", [1.5, 1.0, 0.5, 0.01, 1e-4])
def test_a_constant_integrand_returns_the_constant(sigma: float) -> None:
    """The Gaussian must integrate to one, whatever its width.

    This is the check that fails loudest when a normaliser is wrong: the answer is known
    exactly, it does not depend on sigma, and the old code returned -5.0055 (and -1205 at
    sigma = 1e-8) instead of -5.
    """

    nodes, weights = _gauss_hermite(20)

    def constant(tau: np.ndarray) -> np.ndarray:
        return np.full(tau.shape, -5.0)

    got = _adaptive_log_integral_batch(constant, sigma, nodes, weights, 3)
    assert np.allclose(got, -5.0, rtol=0, atol=1e-10), f"{got} at sigma={sigma}"


@pytest.mark.parametrize("sigma", [1.5, 0.5, 0.01])
@pytest.mark.parametrize(
    "log_integrand",
    [
        lambda tau: -0.5 * tau**2,
        lambda tau: -0.5 * (tau - 2.0) ** 2,
        lambda tau: -np.logaddexp(0.0, -tau),
    ],
    ids=["centred-gaussian", "off-centre-gaussian", "logistic"],
)
def test_matches_numerical_integration(log_integrand, sigma: float) -> None:
    """Pin the whole normaliser against scipy, not against a previous version."""

    nodes, weights = _gauss_hermite(20)
    got = _adaptive_log_integral_batch(log_integrand, sigma, nodes, weights, 2)
    expected = _reference(log_integrand, sigma)
    assert got[0] == pytest.approx(expected, rel=1e-8, abs=1e-8)


def test_a_tiny_sigma_does_not_blow_up() -> None:
    """sigma -> 0 is the regime the old code failed in; it must stay finite and correct."""

    nodes, weights = _gauss_hermite(20)

    def log_constant(tau: np.ndarray) -> np.ndarray:
        return np.full(tau.shape, -1.0)

    for sigma in (1e-6, 1e-8, 1e-12):
        got = _adaptive_log_integral_batch(log_constant, sigma, nodes, weights, 2)
        assert np.all(np.isfinite(got)), f"non-finite at sigma={sigma}: {got}"
        assert np.allclose(got, -1.0, rtol=0, atol=1e-9), f"{got} at sigma={sigma}"
