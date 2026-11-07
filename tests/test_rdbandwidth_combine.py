"""The three CCT bandwidth COMBINATION rules, transcribed from ``rdbwselect.ado``.

Round 2 of the v3 work read the ado directly instead of re-deriving the rule. It settles what the
module docstring had listed as a guess:

    rdbwselect.ado:537   d_bw_d = ((C_l[1]+C_r[1]) / (C_r[2]-C_l[2])^2) ^ C[4]        mserd
    rdbwselect.ado:527   d_bw_s = ((C_l[1]+C_r[1]) / (C_r[2]+C_l[2])^2) ^ C[4]        msesum
    rdbwselect.ado:499   b_bw_l = ((C_l[1]) / (C_l[2]^2 + regul*C_l[3])) ^ C[4]       msetwo

MSERD and MSESUM differ ONLY in the sign between the two C[2] entries, which is why the module
docstring warned that using the wrong one "gives a plausible bandwidth belonging to a different
estimator".

STILL NOT IMPLEMENTED, and not pretended otherwise: the ``C[1..3]`` vectors themselves. They come
from ``rdrobust_bw()`` inside ``lrdrobust.mlib``, which ``mata which`` confirms is the library --
and that file is 1.9% printable bytes with the function name appearing as an isolated string, i.e.
compiled bytecode on the same footing as ``synth``'s Mata core. These functions are the combination
step only; feeding them requires the missing vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from westata.estimators.rdbandwidth import (
    combine_mserd,
    combine_msesum,
    combine_msetwo,
    rd_exponent,
)

#: A synthetic ``C`` vector: [variance-ish, bias derivative, regularisation, exponent].
C_LEFT = np.array([100.0, 2.0, 0.1, 0.2])
C_RIGHT = np.array([120.0, 3.0, 0.1, 0.2])


def test_mserd_is_the_difference_form() -> None:
    numerator = C_LEFT[0] + C_RIGHT[0]
    denominator = (C_RIGHT[1] - C_LEFT[1]) ** 2
    expected = (numerator / (denominator + C_LEFT[2] + C_RIGHT[2])) ** C_LEFT[3]
    assert combine_mserd(C_LEFT, C_RIGHT) == pytest.approx(expected)


def test_msesum_is_the_sum_form() -> None:
    numerator = C_LEFT[0] + C_RIGHT[0]
    denominator = (C_RIGHT[1] + C_LEFT[1]) ** 2
    expected = (numerator / (denominator + C_LEFT[2] + C_RIGHT[2])) ** C_LEFT[3]
    assert combine_msesum(C_LEFT, C_RIGHT) == pytest.approx(expected)


def test_the_two_forms_differ() -> None:
    """If these ever coincide, the sign has been lost and one of them is wrong."""

    assert combine_mserd(C_LEFT, C_RIGHT) != pytest.approx(
        combine_msesum(C_LEFT, C_RIGHT)
    )


def test_mserd_uses_minus_and_msesum_uses_plus() -> None:
    """Pin the SIGN directly: mserd's denominator is smaller (3-2=1) than msesum's (3+2=5)."""

    numerator = C_LEFT[0] + C_RIGHT[0]
    regularisation = C_LEFT[2] + C_RIGHT[2]
    difference = (numerator / ((C_RIGHT[1] - C_LEFT[1]) ** 2 + regularisation)) ** C_LEFT[3]
    summation = (numerator / ((C_RIGHT[1] + C_LEFT[1]) ** 2 + regularisation)) ** C_LEFT[3]
    assert combine_mserd(C_LEFT, C_RIGHT) == pytest.approx(difference)
    assert combine_msesum(C_LEFT, C_RIGHT) == pytest.approx(summation)
    # And the direction that follows from the arithmetic.
    assert difference > summation


def test_msetwo_is_computed_per_side() -> None:
    left, right = combine_msetwo(C_LEFT, C_RIGHT)
    expected_left = (C_LEFT[0] / (C_LEFT[1] ** 2 + C_LEFT[2])) ** C_LEFT[3]
    expected_right = (C_RIGHT[0] / (C_RIGHT[1] ** 2 + C_RIGHT[2])) ** C_RIGHT[3]
    assert left == pytest.approx(expected_left)
    assert right == pytest.approx(expected_right)


def test_a_degenerate_denominator_returns_nan_not_a_number() -> None:
    """Equal derivatives on both sides make the mserd denominator zero.

    Returning a plausible number there would be the failure mode the whole task exists to avoid.
    """

    equal = np.array([100.0, 2.0, 0.0, 0.2])
    assert np.isnan(combine_mserd(equal, equal))


def test_the_exponent_follows_the_polynomial_order() -> None:
    assert rd_exponent(1) == pytest.approx(1.0 / 5.0)
    assert rd_exponent(2) == pytest.approx(1.0 / 7.0)
