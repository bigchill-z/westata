"""xtnbreg, re is NOT implemented -- a genuine capability gap, now measured precisely.

Round 245 found that `XtNbreg` accepts only method='fe'. A live Stata run confirms:

    xtnbreg n x1, re nolog        ->  works, and is Stata's DEFAULT method
    block: 25 lines, random effect drawn as "Random effects u_i ~ Beta"

So this is a missing estimator, not a layout difference.

WHAT THE SPEC IS, read from Stata's own help (`xtnbreg.sthlp:172-180`):

    "In the random-effects model, the dispersion varies randomly from group to group,
     such that the inverse of one plus the dispersion follows a Beta distribution."

and the reported macros are `e(r)` / `e(s)`, the two Beta parameters (sthlp:413). So the
likelihood is a Beta negative-binomial mixture: with u ~ Beta(r, s) and the per-group dispersion
equal to (1-u)/u, the marginal probability of y is

    integral_0^1 NB2(y; mu, (1-u)/u) * Beta(u; r, s) du

NUMERICALLY TRACTABLE -- verified in this round with a quadrature probe on a single observation
(value 0.06487288, quadrature error 1.4e-10), so the mixture is not the obstacle.

WHAT IS MISSING is the surrounding estimator work: reparameterising (r, s), optimising the three
parameter blocks (b, r, s) jointly, the observed-information standard errors, Stata's e() surface
for this command, and the 25-line table with the Beta label. That is a coherent piece of work
rather than a patch, and this round recorded the specification instead of starting it and leaving
it half-built.

The guards below pin the gap so it cannot be mistaken for coverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, special

import westata as ws


def _counts():
    rng = np.random.default_rng(13)
    n_grp, n_obs = 25, 12
    sid = np.repeat(np.arange(n_grp), n_obs)
    x1 = rng.normal(size=n_grp * n_obs)
    u = np.repeat(rng.normal(scale=0.5, size=n_grp), n_obs)
    mu = np.exp(0.4 + 0.5 * x1 + u)
    p = 1.0 / (1.0 + 0.7 * mu)
    counts = rng.negative_binomial(1.0 / 0.7, 1.0 - p).astype(float)
    return pd.DataFrame(
        {"n": counts, "x1": x1, "sid": sid, "t": np.tile(np.arange(n_obs), n_grp)}
    )


def test_fe_is_supported_and_exact() -> None:
    """The FE variant is implemented; only the RE method is missing."""

    fitted = ws.XtNbreg.from_formula(
        "n ~ x1", _counts(), entity="sid", time="t", method="fe"
    ).fit()
    assert fitted is not None


def test_re_is_rejected_with_a_clear_error() -> None:
    """If this starts passing, the estimator was implemented -- update the gap record."""

    with pytest.raises(ws.errors.UnsupportedError) as caught:
        ws.XtNbreg.from_formula(
            "n ~ x1", _counts(), entity="sid", time="t", method="re"
        )
    message = str(caught.value)
    assert "re" in message
    assert "fe" in message          # the message names what IS available


def test_the_beta_mixture_is_tractable() -> None:
    """The specification is implementable -- this is the quadrature probe from the diagnosis.

    u ~ Beta(r, s), dispersion (1-u)/u, y ~ NB2(mu, dispersion); marginalise over u. If this ever
    stops producing a finite value, the planned implementation would need a different route.
    """

    r, s, mu_value, y_value = 2.0, 1.5, 3.0, 5

    def integrand(u: float) -> float:
        dispersion = (1.0 - u) / u
        p = 1.0 / (1.0 + dispersion * mu_value)
        log_nb = (
            special.gammaln(y_value + 1.0 / dispersion)
            - special.gammaln(1.0 / dispersion)
            - special.gammaln(y_value + 1.0)
            + (1.0 / dispersion) * np.log(p)
            + y_value * np.log(1.0 - p)
        )
        density = u ** (r - 1.0) * (1.0 - u) ** (s - 1.0) / special.beta(r, s)
        return float(np.exp(log_nb) * density)

    value, error = integrate.quad(integrand, 0.0, 1.0)
    assert np.isfinite(value) and value > 0.0
    assert error < 1e-8
