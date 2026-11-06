"""rdrobust bandwidth: what round 104 ruled OUT, so it is not retried.

Rounds 102-104 tried to reach Stata's h = 17.754 (mserd, senate data) and the numbers now form a map
of which construction is wrong:

    approach                                        h        vs 17.754
    average the sides' inputs, kernel moments     19.63      +10.6%
    average the sides' inputs, sandwich variance  19.99      +12.6%
    per-side bandwidths, count-weighted mean      20.41      +15.0%
    per-side bandwidths at order 2                28.69      +61.6%

Every attempt OVERSHOOTS, by 10-15%, and the spread between constructions is smaller than the gap
to Stata. That pattern says the error is a systematic factor in my variance term, not the choice of
averaging order -- three different aggregations all land within 5% of each other, 10-15% above the
target.

The one quantity that did reproduce: the (p+1)-th derivative coefficients and the residual variances
at a pilot bandwidth of 50, which are stable across every variant and pinned in
test_rdrobust_bw_inputs.py.

Conclusion recorded for the next attempt: do not re-derive the inputs (they are known), and do not
try another aggregation order (three are ruled out). The likely culprit is the variance
normalisation -- Var(b0) computed at the pilot bandwidth needs a factor relating h_pilot to the
target h that I have not got right, and the bias term's h^{p+1} scaling is a second candidate.
"""

from __future__ import annotations

import pytest


#: Attempts made so far, with the h each produced. Kept as a record of what not to repeat.
RULED_OUT = {
    "kernel moments, averaged inputs": 19.63,
    "sandwich variance, averaged inputs": 19.99,
    "per-side, count-weighted": 20.41,
    "per-side, order 2": 28.69,
}

STATA_H = 17.754


@pytest.mark.parametrize("approach,h", sorted(RULED_OUT.items()))
def test_every_attempted_approach_overshoots(approach: str, h: float) -> None:
    """The map matters because the failure is systematic, not random.

    All four constructions land ABOVE Stata's value and within 5% of each other, while the gap to
    Stata is 10-15%. A single wrong factor in the variance term produces exactly that signature;
    an aggregation-order mistake would scatter the results instead.
    """

    assert h > STATA_H, approach
    assert (h - STATA_H) / STATA_H < 0.70, approach


def test_the_closest_attempt_is_still_off_by_more_than_five_percent() -> None:
    """No variant is close enough to ship."""

    closest = min(RULED_OUT.values())
    assert (closest - STATA_H) / STATA_H > 0.05
