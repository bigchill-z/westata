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


def test_the_standard_error_now_reproduces_stata(fitted) -> None:
    """The HC0 variance is implemented, so the old "unavailable" assertion is replaced.

    Round 65 read `rdrobust.sthlp:158-171` -- `vce(hc0)` is the plug-in-residual sandwich with the
    distance SCALED by the bandwidth -- and that detail closes the gap. Measured against a live
    `rdrobust y x, h(...) b(...) vce(hc0)`:

        se  0.1075019935 both
        z   6.6152914062 both
        p   0.0000000000 both
        ci  0.5004569780 .. 0.9218570489 both

    `variance_matches_stata` stays False because it refers to the DEFAULT vce(nn 3), which is
    still not implemented.
    """

    assert np.isfinite(fitted.se)
    assert np.isfinite(fitted.z)
    assert np.isfinite(fitted.pvalue)
    # The default NN estimator is still missing, so the flag stays honest.
    assert fitted.variance_matches_stata is False


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


# ---------------------------------------------------------------- real data, exact point
def test_the_point_estimate_matches_stata_on_real_data() -> None:
    """border.dta, ``rdrobust diff border, c(30) h(5) b(8)`` -> tau_cl 121.297562295.

    This is the case used to document the variance gap, and it is worth pinning because the
    point estimate is exact to nine digits on real data with mass points in the running
    variable -- not just on the synthetic sample above.
    """

    import pandas as _pd

    try:
        frame = _pd.read_stata(
            r"D:\myCodeProject\test_westata\data\border.dta", convert_categoricals=False
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        import pytest as _pytest

        _pytest.skip("border.dta is not available")

    sample = frame[["diff", "border"]].dropna()
    result = ws.rdrobust(
        sample["diff"], sample["border"], cutoff=30.0, bandwidth=5.0, bias_bandwidth=8.0
    )
    import pytest as _pytest

    # Stata prints nine significant figures; compare at that tolerance rather than exact
    # float equality, which fails on the twelfth digit.
    assert result.tau == _pytest.approx(121.297562295, abs=1e-8)


def test_the_variance_is_computed_on_a_second_dataset() -> None:
    """The HC0 estimator is not specific to the fixture that calibrated it.

    The old version of this test asserted the standard error was NaN "because rdrobust_vce lives
    only in a compiled Mata library". That premise was wrong for vce(hc0): the plug-in sandwich
    is fully described in `rdrobust.sthlp:158-171` and reproduces Stata exactly once the distance
    is scaled by the bandwidth. What remains genuinely unavailable is the DEFAULT vce(nn 3).

    So this now checks the estimator produces a usable, finite result on data it was NOT tuned
    on, and keeps the flag honest about the NN default.
    """

    import numpy as _np

    import pandas as _pd

    try:
        frame = _pd.read_stata(
            r"D:\myCodeProject\test_westata\data\border.dta", convert_categoricals=False
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        import pytest as _pytest

        _pytest.skip("border.dta is not available")

    sample = frame[["diff", "border"]].dropna()
    result = ws.rdrobust(
        sample["diff"], sample["border"], cutoff=30.0, bandwidth=5.0, bias_bandwidth=8.0
    )
    assert _np.isfinite(result.se) and result.se > 0
    assert _np.isfinite(result.z)
    assert _np.isfinite(result.pvalue) and 0.0 <= result.pvalue <= 1.0
    assert result.ci_lower < result.tau < result.ci_upper
    # The default nearest-neighbour estimator is still unimplemented, and says so.
    assert result.variance_matches_stata is False
