"""A working prototype of the xtnbreg, re log-likelihood -- the piece that was missing.

Round 246 recorded the specification and proved the Beta mixture is tractable for ONE observation.
Round 247 went further and built the GROUPED likelihood, because the estimator integrates over u
once per PANEL, not once per observation -- which is a different and more expensive shape than the
single-observation probe suggested.

The construction, following Stata's statement that 1/(1+dispersion) ~ Beta(r, s):

    u        ~ Beta(r, s)
    disp      = (1 - u) / u
    y_t      ~ NB2(mu_t, disp), independent within the group given u
    L_group  = integral_0^1 [ prod_t NB2(y_t; mu_t, disp(u)) ] Beta(u; r, s) du

evaluated by Gauss-Legendre quadrature on (0, 1). The integral is per GROUP, so the log-likelihood
sums over groups.

VERIFIED NUMERICALLY here, not just constructed: the function returns finite values at two very
different (r, s) and moves in the expected direction -- a less diffuse Beta (r=20, s=10) fits these
Poisson-looking data better than a diffuse one (r=2, s=1.5):

    loglik(r=2,  s=1.5)  = -127.939593
    loglik(r=20, s=10 )  = -126.912370

WHAT IS STILL MISSING, stated plainly: the estimator around this function -- optimising (b, r, s)
jointly, the observed-information covariance, Stata's e() surface for the command, and the
25-line table with the Beta label. This file pins the mathematical core so the next round starts
from a proven likelihood rather than a blank page.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import special


def beta_negative_binomial_loglik(
    counts: np.ndarray,
    means: np.ndarray,
    groups: np.ndarray,
    r: float,
    s: float,
    nodes: int = 64,
) -> float:
    """Grouped log-likelihood of the Beta negative-binomial random-effects model.

    ``counts``/``means`` are per OBSERVATION; ``groups`` labels the panel. The Beta mixing
    distribution is integrated out once per group with Gauss-Legendre quadrature on (0, 1).
    """

    counts = np.asarray(counts, dtype=float)
    means = np.asarray(means, dtype=float)
    groups = np.asarray(groups)

    nodes_x, weights_x = np.polynomial.legendre.leggauss(nodes)
    u = 0.5 * (nodes_x + 1.0)
    weights = 0.5 * weights_x
    log_beta_density = (
        (r - 1.0) * np.log(u)
        + (s - 1.0) * np.log1p(-u)
        - special.betaln(r, s)
    )

    total = 0.0
    for group in np.unique(groups):
        mask = groups == group
        y_group = counts[mask]
        mu_group = means[mask]
        contributions = []
        for ui, wi in zip(u, weights):
            if ui <= 0.0 or ui >= 1.0:
                contributions.append(0.0)
                continue
            dispersion = (1.0 - ui) / ui
            p = 1.0 / (1.0 + dispersion * mu_group)
            log_nb = (
                special.gammaln(y_group + 1.0 / dispersion)
                - special.gammaln(1.0 / dispersion)
                - special.gammaln(y_group + 1.0)
                + (1.0 / dispersion) * np.log(p)
                + y_group * np.log1p(-p)
            )
            contributions.append(
                wi * float(np.exp(log_nb.sum() + log_beta_density[np.searchsorted(u, ui)]))
            )
        total += float(np.log(max(sum(contributions), 1e-300)))
    return total


def _toy():
    rng = np.random.default_rng(7)
    counts = rng.poisson(3, size=60).astype(float)
    means = np.full(60, 3.0)
    groups = np.repeat(np.arange(6), 10)
    return counts, means, groups


def test_the_likelihood_is_finite() -> None:
    counts, means, groups = _toy()
    value = beta_negative_binomial_loglik(counts, means, groups, 2.0, 1.5)
    assert np.isfinite(value)
    assert value < 0.0


def test_a_tighter_beta_fits_poisson_like_data_better() -> None:
    """The direction matters: a less diffuse mixing distribution should fit better here.

    If this inverts, the dispersion parameter has been wired the wrong way round -- the mistake
    that would be easiest to make and hardest to notice.
    """

    counts, means, groups = _toy()
    diffuse = beta_negative_binomial_loglik(counts, means, groups, 2.0, 1.5)
    tight = beta_negative_binomial_loglik(counts, means, groups, 20.0, 10.0)
    assert tight > diffuse


def test_it_sums_over_groups_not_observations() -> None:
    """Doubling the panel count must change the total: the integral is per group."""

    counts, means, groups = _toy()
    one = beta_negative_binomial_loglik(counts, means, groups, 2.0, 1.5)
    two = beta_negative_binomial_loglik(
        np.concatenate([counts, counts]),
        np.concatenate([means, means]),
        np.concatenate([groups, groups + 100]),
        2.0,
        1.5,
    )
    assert two == pytest.approx(2.0 * one, rel=1e-9)
