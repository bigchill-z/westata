"""Guards for the random-effects variance component at small sigma.

The 0.0.5 bug (§8.5 in docs/GAP_REGISTER.md) was an integrand normaliser that was wrong by
a constant and a ``log(scale)`` term. It did NOT show up on data with a large random effect
-- measured, the old and new code agree to all printed digits when sigma_u is ~2.4. It only
appeared when the variance is near zero, where the broken fixed-point iteration drove the
scale towards 0 and the objective blew up.

So these tests deliberately use a frame whose random effect is (nearly) absent, plus the
convergence property checked on a frame where the optimum is interior. Using only the
interior frame would have passed against the broken version, which is why both are here.

Reference values come from Stata on the same frames:

  frame with ~no random effect : Stata sigma_u 2.43e-05, ll -69.197758
  frame with a real one       : Stata sigma_u 2.4040464, ll -162.0797
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.panel_nonlinear import _adaptive_log_integral_batch, _gauss_hermite


@pytest.fixture(scope="module")
def weak_frame() -> pd.DataFrame:
    """The exact frame where the defect was observed.

    ``scripts/scan_all_layouts.py``'s PANEL frame is the one on which Stata reports
    sigma_u = 2.43e-05 -- i.e. the random-effect variance is ~0 and the broken integrator
    drove the estimate to 3.7e-03. Importing it keeps this test pinned to the same data
    rather than to a lookalike that happens to have a healthy variance.
    """

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import scan_all_layouts as scan  # noqa: PLC0415

    return scan.PANEL


@pytest.fixture(scope="module")
def strong_frame() -> pd.DataFrame:
    """Panel with a substantial random effect: the optimum is interior."""

    rng = np.random.default_rng(11)
    n_ent, n_per = 40, 8
    n = n_ent * n_per
    u = np.repeat(rng.normal(scale=1.5, size=n_ent), n_per)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "ybin": (0.8 * x + u + rng.normal(size=n) > 0).astype(int),
            "x": x,
            "id": np.repeat(np.arange(n_ent), n_per),
            "t": np.tile(np.arange(n_per), n_ent),
        }
    )


def test_sigma_u_stays_small_when_the_random_effect_is_absent(
    weak_frame: pd.DataFrame,
) -> None:
    """The regime the bug lived in.

    The broken integrator returned ~3.7e-03 here because its objective exploded as
    sigma -> 0 and the optimiser stopped early; the fixed one stays two orders below that
    and its log-likelihood matches the pooled logit, which is the correct limit.
    """

    result = ws.XtLogit.from_formula(
        "ybin ~ x1", weak_frame, entity="unit", method="re"
    ).fit()
    sigma_u = float(result.e["sigma_u"])
    assert sigma_u < 1e-5, f"sigma_u={sigma_u} (broken version gave ~3.7e-3)"

    pooled = ws.Logit.from_formula("ybin ~ x1", weak_frame).fit()
    # As sigma_u -> 0 the random-effects model collapses onto the pooled one.
    assert float(result.e["ll"]) == pytest.approx(float(pooled.e["ll"]), abs=1e-5)


def test_sigma_u_converges_as_quadrature_points_grow(strong_frame: pd.DataFrame) -> None:
    """Stata's own sequence converges to ~2.40436 on this frame; ours must too."""

    estimates = []
    for points in (12, 20, 40):
        result = ws.XtLogit.from_formula(
            "ybin ~ x", strong_frame, entity="id", method="re"
        ).fit(intpoints=points)
        estimates.append(float(result.e["sigma_u"]))

    assert estimates[-1] == pytest.approx(2.40436, abs=5e-4), estimates
    late_gap = abs(estimates[2] - estimates[1])
    early_gap = abs(estimates[1] - estimates[0])
    assert late_gap < early_gap, (estimates, early_gap, late_gap)


def test_constant_integrand_is_returned_for_every_sigma() -> None:
    """Closed form: a Gaussian integrates to one, so a constant comes back unchanged."""

    nodes, weights = _gauss_hermite(20)
    for sigma in (1.5, 0.5, 1e-2, 1e-6, 1e-10):
        got = _adaptive_log_integral_batch(
            lambda tau: np.full(tau.shape, -2.0), sigma, nodes, weights, 2
        )
        assert np.allclose(got, -2.0, rtol=0, atol=1e-9), f"{got} at sigma={sigma}"
