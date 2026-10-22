"""Rewrite the contrast E3 assertions to use the verified Stata values.

Reference (StataNow 19.5, ``regress y x i.g`` on a 3-level factor, then
``contrast r.g`` / ``contrast a.g`` / ``contrast g.g`` / ``pwcompare g``):

    contrast r.g   (1 vs 0)  0.8134023  se 0.0699776
                   (2 vs 0) -0.5222208  se 0.0705751
    contrast a.g   (0 vs 1) -0.8134023  se 0.0699776
                   (1 vs 2)  1.335623   se 0.066388
    contrast g.g   (0 vs mean) -0.0970605  se 0.0412958
                   (1 vs mean)  0.7163418  se 0.0389124
                   (2 vs mean) -0.6192813  se 0.0392705
    pwcompare g    1 vs 0  0.8134023
                   2 vs 0 -0.5222208
                   2 vs 1 -1.335623
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import UnsupportedError


def _factor_frame(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 3, size=n)
    x = rng.normal(size=n)
    effect = np.array([0.0, 0.8, -0.5])[group]
    y = 1.0 + 2.0 * x + effect + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x": x, "g": group})


@pytest.fixture(scope="module")
def fitted():
    return ws.OLS.from_formula(
        "y ~ x + C(g, base=0)", _factor_frame()
    ).fit()


def test_contrast_reference_matches_stata(fitted):
    rows = fitted.contrast("C(g)", method="r")
    assert len(rows) == 2

    estimates = [float(row.estimate) for row in rows]
    errors = [float(row.std_err) for row in rows]
    # Stata: (1 vs 0) .8134023 se .0699776 ; (2 vs 0) -.5222208 se .0705751
    np.testing.assert_allclose(estimates, [0.8134023, -0.5222208], rtol=2e-7)
    np.testing.assert_allclose(errors, [0.0699776, 0.0705751], rtol=2e-6)


def test_contrast_adjacent_matches_stata_and_has_the_opposite_sign(fitted):
    rows = fitted.contrast("C(g)", method="a")
    assert len(rows) == 2

    estimates = [float(row.estimate) for row in rows]
    errors = [float(row.std_err) for row in rows]
    # Stata: (0 vs 1) -.8134023 se .0699776 ; (1 vs 2) 1.335623 se .066388
    np.testing.assert_allclose(estimates, [-0.8134023, 1.335623], rtol=2e-6)
    np.testing.assert_allclose(errors, [0.0699776, 0.066388], rtol=2e-5)

    # The direction is deliberate: adjacent reports low-vs-high, the reference
    # contrast reports high-vs-low, so the first pair flips sign.
    reference = fitted.contrast("C(g)", method="r")
    assert float(rows[0].estimate) == pytest.approx(
        -float(reference[0].estimate), rel=1e-12
    )


def test_contrast_grand_matches_stata(fitted):
    rows = fitted.contrast("C(g)", method="g")
    assert len(rows) == 3

    estimates = [float(row.estimate) for row in rows]
    errors = [float(row.std_err) for row in rows]
    np.testing.assert_allclose(
        estimates, [-0.0970605, 0.7163418, -0.6192813], rtol=2e-6
    )
    np.testing.assert_allclose(errors, [0.0412958, 0.0389124, 0.0392705], rtol=2e-5)


def test_pwcompare_matches_stata(fitted):
    rows = fitted.pwcompare("C(g)")
    assert len(rows) == 3

    estimates = [float(row.estimate) for row in rows]
    # Stata: 1 vs 0 .8134023 ; 2 vs 0 -.5222208 ; 2 vs 1 -1.335623
    np.testing.assert_allclose(estimates, [0.8134023, -0.5222208, -1.335623], rtol=2e-6)


def test_contrast_rejects_unimplemented_methods(fitted):
    for method in ("h", "m", "nope"):
        with pytest.raises(UnsupportedError):
            fitted.contrast("C(g)", method=method)


def test_pwcompare_rejects_adjustment(fitted):
    with pytest.raises(UnsupportedError, match="adjust"):
        fitted.pwcompare("C(g)", adjust="bonferroni")
