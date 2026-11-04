"""Pin ``rdrobust``'s point estimate to Stata, and pin the parts it does NOT reproduce.

Reference from a live ``rdrobust vote margin`` on a synthetic sharp-RD sample (n = 400,
true jump 8.0)::

    Number of obs      224      176        Kernel = Triangular
    Eff. Number of obs  96       69        VCE method = NN
    Order est. (p)       1        1
    BW est. (h)      0.479    0.479
    RD Effect        7.6896             z 20.8565   [6.86877, 8.2963]

with ``e(tau_cl) = 7.6896241``, ``e(h_r) = .47857568``, ``e(se_tau_cl) = .31709743``.

WHAT THIS PINS, because it is exact:

    tau = mu_+(0) - mu_-(0), two local linear fits with a triangular kernel, each on its own
    side of the cutoff -- 7.6896241 against 7.6896241, with 96 and 69 observations inside the
    bandwidth.

WHAT THIS DOCUMENTS BUT DOES NOT PIN, because it is not exact:

    the standard error. The sandwich form and its equivalence-kernel rewrite both give
    0.32163224 against Stata's 0.31709743. The ratio is 0.98581, which is not sqrt(n/(n-1))=0.99875
    nor any simple adjustment; rdrobust uses a variance estimator whose exact weighting this
    port does not match. ``se``/``z``/``p``/``ci`` are therefore NaN, and a test asserts they
    stay NaN so nobody quietly fills them with a nearly-right number.

Bandwidth SELECTION is also unimplemented on purpose: passing ``bandwidth=None`` raises rather
than inventing an MSE-optimal rule that would shift every downstream number.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws

STATA_TAU = 7.6896241
STATA_H = 0.47857568
STATA_N_LEFT, STATA_N_RIGHT = 96, 69
STATA_SE = 0.31709743


def _sample():
    rng = np.random.default_rng(7)
    n = 400
    margin = rng.normal(size=n)
    vote = 50.0 + 8.0 * (margin > 0) + 4.0 * margin + rng.normal(size=n)
    return vote, margin


@pytest.fixture(scope="module")
def fitted():
    vote, margin = _sample()
    return ws.rdrobust(vote, margin, bandwidth=STATA_H)


def test_point_estimate_matches_stata(fitted) -> None:
    assert fitted.tau == pytest.approx(STATA_TAU, abs=5e-7)


def test_effective_sample_sizes_match_stata(fitted) -> None:
    assert fitted.nobs_effective_left == STATA_N_LEFT
    assert fitted.nobs_effective_right == STATA_N_RIGHT


def test_the_order_and_kernel_match_stata(fitted) -> None:
    assert fitted.order == 1
    assert fitted.bias_order == 2
    assert fitted.kernel == "Triangular"


def test_the_standard_error_is_reported_as_unavailable(fitted) -> None:
    """It is 1.4% off, so it must NOT be presented as a result.

    If a later change reproduces Stata's variance estimator, delete this test and pin the
    number instead -- but do not relax it into an approximate assertion.
    """

    assert not fitted.variance_matches_stata
    assert np.isnan(fitted.se)
    assert np.isnan(fitted.z)
    assert np.isnan(fitted.pvalue)


def test_bandwidth_selection_refuses_rather_than_guessing() -> None:
    vote, margin = _sample()
    with pytest.raises(Exception):
        ws.rdrobust(vote, margin)


def test_a_custom_cutoff_is_honoured() -> None:
    """A cutoff away from zero must actually move the estimate."""

    vote, margin = _sample()
    at_zero = ws.rdrobust(vote, margin, bandwidth=0.5, cutoff=0.0).tau
    shifted = ws.rdrobust(vote, margin, bandwidth=0.5, cutoff=0.3).tau
    assert abs(at_zero - shifted) > 1e-3
