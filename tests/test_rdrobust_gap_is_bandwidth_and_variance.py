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


def test_the_standard_error_is_now_hc0(sample) -> None:
    """The package computes vce(hc0); Stata's DEFAULT is vce(nn 3), which is still missing.

    Round 65 read `rdrobust.sthlp:158-171`. `vce(hc0)` there is the plug-in-residual sandwich, and
    the phrase "without weights" refers to the DISTANCE being scaled by the bandwidth rather than to
    a missing weight vector. With `distance / h`, against a live
    `rdrobust y x, h(0.2808754677) b(0.4908484209) vce(hc0)`:

        se  0.1075019935        z  6.6152914062
        p   0.0000000000        ci 0.5004569780 .. 0.9218570489

    so the numbers below are HC0's, not NN's -- which is why `variance_matches_stata` stays False.
    """

    response, running = sample
    fitted = ws.rdrobust(
        response, running, cutoff=0.0, bandwidth=STATA_H, bias_bandwidth=STATA_B
    )
    assert float(fitted.se) == pytest.approx(0.1075019935, abs=1e-9)
    assert float(fitted.z) == pytest.approx(6.6152914062, abs=1e-8)
    assert float(fitted.ci_lower) == pytest.approx(0.5004569780, abs=1e-9)
    assert float(fitted.ci_upper) == pytest.approx(0.9218570489, abs=1e-9)
    # The default nearest-neighbour estimator is still unimplemented, and the flag says so.
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


def _sandwich_standard_error(sample, mode: str) -> float:
    """A plain weighted-sandwich se for the local-linear fit, under three variance estimators.

    These are the three things that look like the right formula and are not. Kept as a helper so
    the test below can show how far each one lands from Stata.
    """

    response, running = sample
    h = STATA_H
    total = 0.0
    for mask in (running < 0, running >= 0):
        distance = np.abs(running[mask])
        values = response[mask]
        weight = np.maximum(1.0 - distance / h, 0.0)
        keep = weight > 0
        distance, values, weight = distance[keep], values[keep], weight[keep]
        n_h = int(keep.sum())

        design = np.column_stack([np.ones(n_h), distance])
        normal = design.T @ (weight[:, None] * design)
        beta = np.linalg.solve(normal, design.T @ (weight * values))
        residual = values - design @ beta

        if mode == "wsum":
            sigma2 = float((weight * residual**2).sum() / weight.sum())
        elif mode == "mean":
            sigma2 = float((residual**2).mean())
        else:
            sigma2 = float((weight * residual**2).sum() / n_h)

        inverse = np.linalg.inv(normal)
        e1 = np.array([1.0, 0.0])
        total += float(
            e1 @ inverse @ (design.T @ ((weight * weight)[:, None] * design))
            @ inverse @ e1
        ) * sigma2
    return float(np.sqrt(total))


def test_stata_uses_nearest_neighbour_variance_not_a_sandwich(sample) -> None:
    """`VCE method = NN` in the Stata header. Three plausible sandwiches all miss.

        mode    computed se      error against 0.1099186153
        wsum    0.1113851870     +1.33%
        mean    0.1142912646     +3.98%
        nh      0.0755949066     -31.3%

    The closest is 1.3% out, which is exactly the size of gap that survives a spot check -- so this
    is pinned rather than rounded towards.

    CCT's NN estimator pairs each observation with a nearest neighbour ON THE OTHER SIDE and weights
    those contributions. It is not a re-weighting of the sandwich above, which is why no choice of
    sigma2 closes the gap.
    """

    observed = {
        mode: _sandwich_standard_error(sample, mode) for mode in ("wsum", "mean", "nh")
    }
    assert observed["wsum"] == pytest.approx(0.1113851870, abs=1e-9)
    assert observed["mean"] == pytest.approx(0.1142912646, abs=1e-9)
    assert observed["nh"] == pytest.approx(0.0755949066, abs=1e-9)

    # None of them is Stata's number, and the closest is still more than 1% away.
    for value in observed.values():
        assert value != pytest.approx(STATA_SE, rel=5e-3)
    closest = min(observed.values(), key=lambda v: abs(v - STATA_SE))
    assert abs(closest - STATA_SE) / STATA_SE > 0.01


def test_the_local_polynomial_itself_is_right(sample) -> None:
    """The two one-sided intercepts differ by exactly Stata's tau -- so only the variance is open."""

    response, running = sample
    h = STATA_H
    intercepts = []
    for mask in (running < 0, running >= 0):
        distance = np.abs(running[mask])
        values = response[mask]
        weight = np.maximum(1.0 - distance / h, 0.0)
        keep = weight > 0
        distance, values, weight = distance[keep], values[keep], weight[keep]
        design = np.column_stack([np.ones(int(keep.sum())), distance])
        normal = design.T @ (weight[:, None] * design)
        beta = np.linalg.solve(normal, design.T @ (weight * values))
        intercepts.append(float(beta[0]))

    assert intercepts[1] - intercepts[0] == pytest.approx(STATA_TAU, abs=1e-9)


def test_no_plausible_reweighting_of_the_sandwich_reaches_stata() -> None:
    """The three sandwich variants are pinned above; both NN readings were tried too.

    Round 65 implemented CCT's nearest-neighbour matching twice -- same-side pairing gave
    0.3929069802 (+257%) and opposite-side pairing gave 0.3373928805 (+207%), against Stata's
    0.1099186153. The two errors differ by a factor of three in variance, so no single
    normalisation reconciles them: the published definition is needed, not a rescaling.

    Those attempts were deleted rather than kept as a helper, because a flawed same-side variant
    produced a NEGATIVE variance and could not even be sqrt-ed. Recording the magnitudes here is
    what stops the next attempt from re-deriving them.
    """

    attempts = {
        "sandwich wsum": 0.1113851870,
        "sandwich mean": 0.1142912646,
        "sandwich nh": 0.0755949066,
        "nn same side": 0.3929069802,
        "nn other side": 0.3373928805,
    }
    for label, value in attempts.items():
        assert abs(value - STATA_SE) / STATA_SE > 1e-2, label
    # The closest failure is still 1.3% out -- the size of gap a spot check would pass.
    closest = min(attempts.values(), key=lambda v: abs(v - STATA_SE))
    assert abs(closest - STATA_SE) / STATA_SE > 0.01


def test_the_nearest_neighbour_estimator_is_insensitive_to_its_neighbour_count() -> None:
    """Recorded so the next attempt does not start from "pick one nearest neighbour".

    Stata, with the bandwidth and tau held fixed, changing only `nnmatch`:

        vce(nn 3)    0.1099186153      <- the default
        vce(nn 10)   0.1105396140
        vce(nn 100)  0.1083379095
        vce(hc0)     0.1075019935      <- implemented, exact

    Two things follow. First, the whole range from 3 to 100 neighbours spans about 1%, so the
    estimator averages matched-residual PRODUCTS rather than depending on any single pairing --
    which is why round 65's "take the nearest one" reading was 200% out: the pairing was wrong, not
    the count. Second, NN and HC0 are only 2.2% apart, so the target is a narrow one.

    These are Stata's numbers, not this package's; the package computes HC0 only.
    """

    stata_values = {
        "nn 3": 0.1099186153,
        "nn 10": 0.1105396140,
        "nn 100": 0.1083379095,
        "hc0": 0.1075019935,
    }

    # The neighbour count moves the answer by roughly one percent, not by orders of magnitude.
    neighbours = [stata_values["nn 3"], stata_values["nn 10"], stata_values["nn 100"]]
    assert (max(neighbours) - min(neighbours)) / min(neighbours) < 0.03

    # And the implemented estimator is within 2.5% of the unimplemented default.
    gap = abs(stata_values["hc0"] - stata_values["nn 3"]) / stata_values["nn 3"]
    assert 0.01 < gap < 0.03


def test_the_package_reports_hc0_while_stata_defaults_to_nn(sample) -> None:
    """The distinction the flag encodes: a right number under the wrong estimator name.

    A default `rdrobust y x` in Stata uses nn 3 and prints 0.1099186153. This package prints
    0.1075019935, which is the hc0 answer. Both are correct for their own estimator, and the
    difference is 2.2% -- enough to matter, small enough to pass unnoticed.
    """

    response, running = sample
    fitted = ws.rdrobust(
        response, running, cutoff=0.0, bandwidth=STATA_H, bias_bandwidth=STATA_B
    )
    assert float(fitted.se) == pytest.approx(0.1075019935, abs=1e-9)
    relative = abs(float(fitted.se) - 0.1099186153) / 0.1099186153
    assert 0.01 < relative < 0.03
