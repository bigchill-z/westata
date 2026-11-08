"""`rdrobust`'s point estimator is already exact -- the gaps are bandwidth and variance.

Recorded because earlier notes described bandwidth selection as the single missing piece. Measured
against a live Stata on a seeded 500-point sample (uniform running variable, true jump 0.5):

    rdrobust(y, x, bandwidth=0.2808754677, bias_bandwidth=0.4908484209)
    tau = 0.7111570135          Stata tau_cl = 0.7111570135      identical
    se  = nan                   Stata se_tau_cl = 0.1099186153

and Stata's own bandwidths for that sample are h_l = h_r = 0.2808754677, b_l = b_r = 0.4908484209.

So two INDEPENDENT things are missing: the CCT MSE-optimal bandwidth, and the robust variance. The
estimator in between them is correct, which is worth pinning: it means a future change to the
bandwidth machinery cannot quietly alter the point estimate, and a future variance implementation has
a known-good tau to leave untouched.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws

#: Stata's e() values for the seeded sample below.
STATA_H = 0.2808754677
STATA_B = 0.4908484209
STATA_TAU = 0.7111570135
STATA_SE = 0.1099186153


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(1)
    n = 500
    running = rng.uniform(-1.0, 1.0, n)
    response = 0.5 * (running >= 0) + 0.3 * running + rng.normal(scale=0.3, size=n)
    return response, running


def test_the_point_estimate_matches_stata_once_the_bandwidth_is_given(sample) -> None:
    response, running = sample
    fitted = ws.rdrobust(
        response, running, cutoff=0.0, bandwidth=STATA_H, bias_bandwidth=STATA_B
    )
    assert float(fitted.tau) == pytest.approx(STATA_TAU, abs=1e-9)


def test_the_standard_error_is_the_open_gap(sample) -> None:
    """NaN today. When this starts failing, a variance implementation landed -- assert STATA_SE."""

    response, running = sample
    fitted = ws.rdrobust(
        response, running, cutoff=0.0, bandwidth=STATA_H, bias_bandwidth=STATA_B
    )
    assert not np.isfinite(float(fitted.se))
    assert not np.isfinite(float(fitted.ci_lower))
    # ... and the result says so rather than pretending otherwise.
    assert fitted.variance_matches_stata is False


def test_automatic_bandwidth_selection_still_refuses(sample) -> None:
    """CCT's C[1..3] live in `l/lrdrobust.mlib`; nothing in `rdbandwidth.py` computes them."""

    response, running = sample
    with pytest.raises(Exception, match="bandwidth"):
        ws.rdrobust(response, running, cutoff=0.0)


def test_the_pieces_that_do_exist_are_usable(sample) -> None:
    """The combination rules and the pilot bandwidth are implemented and well-behaved."""

    from westata.estimators.rdbandwidth import (
        C_C_TRIANGULAR,
        derivative_scale,
        pilot_bandwidth,
        rd_exponent,
    )

    _, running = sample
    pilot = pilot_bandwidth(running)
    assert np.isfinite(pilot) and pilot > 0
    # A pilot is the starting scale, so it should be the same order as the reported bandwidth.
    assert 0.1 < pilot < 5.0
    assert rd_exponent(1) == pytest.approx(0.2)
    assert derivative_scale(1) == pytest.approx(1.0)
    assert C_C_TRIANGULAR == pytest.approx(2.576)
