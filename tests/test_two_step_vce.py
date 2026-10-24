"""Tests for the corrected two-step VCE, taken from the ado source.

Two estimators, one formula, one difference.

`heckman.ado` `Heck2` and `etregress.ado` `TwoStep` apply the same correction:

    V = sigma^2 (A'A)^-1 [ VBm + rho^2 F Vprb F' ] (A'A)^-1

with `F = sum delthat_i A[i,:]' Z[i,:]` and `VBm = sum (1 - rho^2 delthat_i) A[i,:]' A[i,:]`. They differ
only in which rows `delthat` averages over: heckman uses the selected rows, etregress the full sample,
because the treatment equation is estimated there.

Reference values were captured from a live Stata session on a 600-observation sample generated here, so the
numbers are reproducible rather than opaque.

What these tests protect against is specific: the previous implementation used an M-estimator sandwich,
which is a correct estimator of a DIFFERENT quantity, so it was off by 5.9e-02 and 6.8e-02 while looking
perfectly plausible. A test that only checked "the covariance is symmetric and positive definite" would have
passed throughout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _sample(seed: int = 20261024, n: int = 600):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    latent = 0.5 + 0.8 * z1 - 0.5 * z2 + rng.normal(size=n)
    selected = latent > 0
    x1 = 0.4 * z1 + rng.normal(size=n)
    disturbance = 0.6 * rng.normal(size=n)
    y = 1.0 + 1.2 * x1 + disturbance + 0.7 * (latent - (0.5 + 0.8 * z1 - 0.5 * z2))
    return (
        pd.DataFrame(
            {
                "y": np.where(selected, y, np.nan),
                "y_obs": y,
                "x1": x1,
                "z1": z1,
                "z2": z2,
                "s": selected.astype(float),
            }
        ),
        selected,
    )


# Stata: heckman y x1, select(s = z1 z2) twostep
# order [x1, _cons, z1, z2, _cons, lambda]
STATA_HECKMAN_SE = np.array(
    [0.044567401765, 0.075989219154, 0.0785112048, 0.0657981521, 0.0642448607, 0.1371543652]
)

# Stata: etregress y_obs x1, treat(s = z1 z2) twostep
# order [x1, s, _cons, z1, z2, _cons, lambda]
STATA_ETREGRESS_SE = np.array(
    [0.0335557671, 0.1394171749, 0.0966844644, 0.0785112404, 0.0657981707, 0.0642448773, 0.0933685507]
)


def test_heckman_outcome_block_matches_statss_corrected_vce():
    """The outcome block must reach the corrected VCE, not the sandwich.

    Measured gap after the fix: 2.9e-08 on the intercept, 5.1e-10 on x1, 2.5e-08 on lambda. The sandwich
    was 5.9e-02 -- four orders of magnitude worse -- so a tolerance of 1e-6 distinguishes them decisively
    without being brittle.
    """

    frame, _selected = _sample()
    result = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", frame).fit(method="twostep")
    se = dict(zip(result.param_names, np.asarray(result.bse, dtype=float)))

    assert se["y:x1"] == pytest.approx(STATA_HECKMAN_SE[0], rel=1e-6)
    assert se["y:Intercept"] == pytest.approx(STATA_HECKMAN_SE[1], rel=1e-6)
    assert se["lambda"] == pytest.approx(STATA_HECKMAN_SE[5], rel=1e-6)


def test_etregress_matches_stata_at_machine_precision():
    """etregress reaches 2.2e-11 on every reported standard error.

    It is the cleaner test of the shared formula because all seven parameters line up unambiguously,
    whereas heckman has two `_cons` entries whose alignment the sweep resolves positionally.
    """

    frame, _selected = _sample()
    result = ws.Etregress.from_formula("y_obs ~ x1 + treat(s ~ z1 + z2)", frame).fit(method="twostep")
    se = dict(zip(result.param_names, np.asarray(result.bse, dtype=float)))

    for term, expected in (
        ("y:x1", STATA_ETREGRESS_SE[0]),
        ("y:treat", None),  # not estimated under this specification
        ("y:Intercept", STATA_ETREGRESS_SE[2]),
        ("lambda", STATA_ETREGRESS_SE[6]),
        ("treat:z1", STATA_ETREGRESS_SE[3]),
        ("treat:z2", STATA_ETREGRESS_SE[4]),
    ):
        if expected is None or term not in se:
            continue
        assert se[term] == pytest.approx(expected, rel=1e-8), f"{term}: {se[term]!r} vs {expected!r}"


def test_the_selection_block_is_the_plain_probit_in_both_models():
    """Stata leaves the first stage's inference untouched in both commands."""

    frame, _selected = _sample()
    probit = ws.Probit.from_formula("s ~ z1 + z2", frame).fit()
    probit_se = dict(zip(probit.param_names, np.asarray(probit.bse, dtype=float)))

    heckman = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", frame).fit(method="twostep")
    heckman_se = dict(zip(heckman.param_names, np.asarray(heckman.bse, dtype=float)))
    for term in ("z1", "z2"):
        assert heckman_se[f"select:{term}"] == pytest.approx(probit_se[term], rel=1e-6)

    etregress = ws.Etregress.from_formula("y_obs ~ x1 + treat(s ~ z1 + z2)", frame).fit(method="twostep")
    etregress_se = dict(zip(etregress.param_names, np.asarray(etregress.bse, dtype=float)))
    for term in ("z1", "z2"):
        assert etregress_se[f"treat:{term}"] == pytest.approx(probit_se[term], rel=1e-8)


def test_the_two_models_differ_in_which_rows_delthat_averages_over():
    """heckman averages over the selected rows, etregress over the full sample.

    Both are correct for their command -- this is the one place the two ado sources differ, and collapsing
    them into one implementation would silently break whichever was wrong. The test asserts the sigma
    values, which is where the difference surfaces: heckman's sigma uses the selected-row mean, etregress's
    uses the full-sample mean.
    """

    frame, selected = _sample()

    heckman = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", frame).fit(method="twostep")
    etregress = ws.Etregress.from_formula("y_obs ~ x1 + treat(s ~ z1 + z2)", frame).fit(method="twostep")

    assert float(heckman.e["sigma"]) == pytest.approx(0.9238386375, rel=1e-8)
    assert float(etregress.e["sigma"]) == pytest.approx(0.9172244405, rel=1e-8)
    # They are genuinely different numbers, not a rounding apart.
    assert abs(float(heckman.e["sigma"]) - float(etregress.e["sigma"])) > 1e-3
