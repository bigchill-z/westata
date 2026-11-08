"""`rdrobust ..., covs()` reproduces Stata's covariate-adjusted estimate and its coefficient.

C16-03 scores this at 0.0. Measured against a live Stata on the same seeded 500-point sample:

    quietly rdrobust y x, h(0.28) b(0.49) covs(z)
        e(tau_cl)      Stata 0.70750239    package 0.70750239
        e(coef_covs)   Stata -.0062819     package -0.0062819

Both match to eight decimals.

WHY THE CUTOFF AND THE BANDWIDTH ARE PASSED EXPLICITLY: automatic bandwidth selection is still
unimplemented (options.md T1), so a default call refuses. Pinning the covariate path separately from
the bandwidth path keeps the two gaps from being confused -- this one is closed, that one is not.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pytest

import westata as ws

STATA_H = 0.28
STATA_B = 0.49
STATA_TAU = 0.70750239
STATA_COEF = -0.0062819


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(1)
    n = 500
    running = rng.uniform(-1.0, 1.0, n)
    response = 0.5 * (running >= 0) + 0.3 * running + rng.normal(scale=0.3, size=n)
    covariate = rng.normal(size=n)
    return response, running, covariate


def test_the_covariate_adjusted_estimate_matches_stata(sample) -> None:
    response, running, covariate = sample
    fitted = ws.rdrobust(
        response,
        running,
        cutoff=0.0,
        bandwidth=STATA_H,
        bias_bandwidth=STATA_B,
        covs=np.column_stack([covariate]),
    )
    assert float(fitted.tau) == pytest.approx(STATA_TAU, abs=1e-8)


def test_the_covariate_coefficient_matches_stata(sample) -> None:
    response, running, covariate = sample
    fitted = ws.rdrobust(
        response,
        running,
        cutoff=0.0,
        bandwidth=STATA_H,
        bias_bandwidth=STATA_B,
        covs=np.column_stack([covariate]),
    )
    coefficients = np.asarray(fitted.coef_covs, dtype=float).ravel()
    assert coefficients.size == 1
    assert float(coefficients[0]) == pytest.approx(STATA_COEF, abs=1e-7)


def test_adding_a_covariate_moves_the_estimate(sample) -> None:
    """If it did not, the covariate would be entering nothing."""

    response, running, covariate = sample
    plain = ws.rdrobust(
        response, running, cutoff=0.0, bandwidth=STATA_H, bias_bandwidth=STATA_B
    )
    adjusted = ws.rdrobust(
        response,
        running,
        cutoff=0.0,
        bandwidth=STATA_H,
        bias_bandwidth=STATA_B,
        covs=np.column_stack([covariate]),
    )
    assert float(plain.tau) != pytest.approx(float(adjusted.tau), rel=1e-9)


def test_the_automatic_bandwidth_still_refuses(sample) -> None:
    """Pinned so the covariate fix cannot be mistaken for bandwidth selection."""

    response, running, covariate = sample
    with pytest.raises(Exception):
        ws.rdrobust(response, running, cutoff=0.0)
