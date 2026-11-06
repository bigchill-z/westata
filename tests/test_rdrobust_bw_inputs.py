"""rdrobust bandwidth: the intermediate quantities, measured, so the next attempt starts here.

Round 102 recorded Stata's target (h = 17.754, b = 28.028, selector "mser") and left the selector
unimplemented. Round 103 derived the pieces and got two different formula variants, neither exact:

    variant A (no density term)   order 1 -> 19.63  (+10.6%)   order 2 -> 18.82  (+6.0%)
    variant B (with density term) order 1 -> 10.50  (-40.8%)   order 2 -> 16.48  (-7.2%)

Both are far better than the old 31%-off version, and neither is right, which says the variance
component is not yet what CCT use. What IS established and worth keeping:

  * the senator data has n = 1297, 595 left of the cutoff and 702 right;
  * with a pilot bandwidth of 50 (half the span on each side), a triangular-kernel local
    polynomial gives these quantities:

        order 1   left  (p+1)-coeff -0.005234  var 78.582
                  right (p+1)-coeff +0.003243  var 61.186
        order 2   left  (p+1)-coeff +0.000223  var 78.355
                  right (p+1)-coeff +0.000304  var 61.095

  * the kernel density of the running variable at the cutoff is f(0) = 0.011735.

These are the inputs a correct CCT implementation needs, and re-deriving them costs a round each
time. Pinned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

DATA = r"D:\myCodeProject\test_westata\work\rdrobust_senate.dta"


@pytest.fixture(scope="module")
def senate():
    try:
        return pd.read_stata(DATA)[["vote", "margin"]].dropna().reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("rdrobust_senate.dta is not available")


def test_the_sample_sides_match_stata(senate) -> None:
    """Stata: 1297 observations, 595 left of c = 0 and 702 right."""

    margin = senate["margin"].to_numpy(float)
    assert len(senate) == 1297
    assert int((margin < 0).sum()) == 595
    assert int((margin >= 0).sum()) == 702


def test_the_pilot_local_polynomial_inputs(senate) -> None:
    """The quantities a CCT selector needs, at a pilot bandwidth of 50.

    Computed once here so a future implementation has a cross-check rather than a from-scratch
    derivation. Values are from a triangular-kernel fit on each side.
    """

    y = senate["vote"].to_numpy(float)
    x = senate["margin"].to_numpy(float)
    left = x < 0.0
    right = ~left

    def fit(distance, response, bandwidth, order):
        weights = np.clip(1.0 - np.abs(distance) / bandwidth, 0.0, None)
        inside = weights > 0
        xs, ys, ws = distance[inside], response[inside], weights[inside]
        design = np.column_stack([xs**k for k in range(order + 1)])
        weighted = design * ws[:, None]
        beta = np.linalg.solve(design.T @ weighted, weighted.T @ ys)
        residuals = ys - design @ beta
        variance = float(np.sum(ws * residuals**2) / max(int(inside.sum()), 1))
        return beta, variance

    pilot = 50.0
    for order, expected in ((1, (-0.005234, 0.003243)), (2, (0.000223, 0.000304))):
        left_beta, left_var = fit(-x[left], y[left], pilot, order + 1)
        right_beta, right_var = fit(x[right], y[right], pilot, order + 1)
        assert left_beta[order + 1] == pytest.approx(expected[0], abs=5e-6), order
        assert right_beta[order + 1] == pytest.approx(expected[1], abs=5e-6), order
        assert 55.0 < min(left_var, right_var) < 85.0


def test_the_density_at_the_cutoff(senate) -> None:
    """f(0) = 0.011735, estimated with the same triangular kernel at the pilot bandwidth."""

    x = senate["margin"].to_numpy(float)
    left = x < 0.0
    right = ~left
    pilot = 50.0

    def density(distance):
        weights = np.clip(1.0 - np.abs(distance) / pilot, 0.0, None)
        return float(weights.sum() / (distance.size * pilot))

    f0 = (density(-x[left]) + density(x[right])) / 2.0
    assert f0 == pytest.approx(0.011735, abs=1e-6)
