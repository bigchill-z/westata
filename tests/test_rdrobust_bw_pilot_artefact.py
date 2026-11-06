"""rdrobust bandwidth: the pilot=90 agreement is a coincidence, and here is the proof.

Round 105 ran a fine sweep of the pilot bandwidth with order 2 and found h within 0.5% of Stata's
17.754 near pilot = 90-100. That looked like a breakthrough. It is not.

The two sides move in OPPOSITE directions as the pilot grows:

    pilot   h_left   h_right   mean   ratio
     70    14.2249   31.8246   23.02   1.297
     80    14.4802   21.7966   18.14   1.022
     90    14.7860   20.9159   17.85   1.005
    100    15.8036   19.8829   17.84   1.005
    110    17.8157   19.7564   18.79   1.058

h_left climbs from 14.2 to 17.8 while h_right falls from 31.8 to 19.8. Their MEAN happens to cross
17.75 in the middle of that crossing, over a range spanning a factor of two in the inputs. A
construction that were correct would not depend on the pilot this way at all -- an MSE-optimal
bandwidth estimated from a consistent pilot converges, and this does not.

So the match is an artefact of averaging two diverging quantities, and the underlying variance
normalisation is still wrong. Recorded because a later round could easily rediscover "pilot = 90,
order 2, ratio 1.005" and mistake it for a solution.
"""

from __future__ import annotations

import pytest

STATA_H = 17.754

#: (pilot, h_left, h_right) from the order-2 sweep.
SWEEP = [
    (70.0, 14.2249, 31.8246),
    (80.0, 14.4802, 21.7966),
    (90.0, 14.7860, 20.9159),
    (100.0, 15.8036, 19.8829),
    (110.0, 17.8157, 19.7564),
]


def test_the_sides_move_in_opposite_directions() -> None:
    """Which is what makes their mean cross the target by coincidence."""

    lefts = [left for _, left, _ in SWEEP]
    rights = [right for _, _, right in SWEEP]
    assert lefts == sorted(lefts), "h_left should climb with the pilot"
    assert rights == sorted(rights, reverse=True), "h_right should fall with the pilot"


def test_no_pilot_gives_a_stable_answer() -> None:
    """A consistent selector would converge; this wanders by more than 10%."""

    means = [(left + right) / 2.0 for _, left, right in SWEEP]
    assert max(means) / min(means) > 1.10


def test_the_agreement_is_an_artefact_of_averaging() -> None:
    """At the closest pilot the two sides differ from each other by more than 20%.

    Measured: h_left 15.80 against h_right 19.88, a 20.5% spread, whose mean happens to land on
    17.84. Two quantities that disagree with each other by a fifth of their size are not
    corroborating the target -- they bracket it.
    """

    pilot, left, right = min(SWEEP, key=lambda row: abs((row[1] + row[2]) / 2.0 - STATA_H))
    assert pilot in (90.0, 100.0)
    assert abs(left - right) / max(left, right) > 0.20
