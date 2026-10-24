"""Tests for the two heckman/probit covariance fixes.

Both were found by measuring against a live Stata, and both are pinned here against stored reference
values captured from that session, so the tests need no Stata to run.

FIX 1 -- `_probit_fit` used the expected information; Stata reports the observed one. The two agree only
asymptotically, so this is not a cosmetic difference: it moved every standard error in every two-step
selection model by 0.4% to 0.7%.

FIX 2 -- the selection block of heckman's covariance is the plain probit's, because Stata's two-step
estimator leaves the first stage's inference untouched. Using the joint sandwich put select:z1 3.95% off.

The reference values below are from a 600-observation sample generated with seed 20261024; the generator
is reproduced here so the numbers are reproducible rather than opaque.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.estimators.limited import _probit_fit


def _selection_frame(seed: int = 20261024, n: int = 600):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    latent = 0.5 + 0.8 * z1 - 0.5 * z2 + rng.normal(size=n)
    selected = latent > 0
    x1 = 0.4 * z1 + rng.normal(size=n)
    y = 1.0 + 1.2 * x1 + 0.6 * rng.normal(size=n) + 0.7 * (latent - (0.5 + 0.8 * z1 - 0.5 * z2))
    frame = pd.DataFrame(
        {
            "y": np.where(selected, y, np.nan),
            "x1": x1,
            "z1": z1,
            "z2": z2,
            "s": selected.astype(float),
        }
    )
    return frame, z1, z2, selected


# Stata's `probit s z1 z2` standard errors on that sample, in the order [z1, z2, _cons].
STATA_PROBIT_SE = {"z1": 0.0785112404, "z2": 0.0657981707, "_cons": 0.0642448773}

# Stata's `heckman y x1, select(s = z1 z2) twostep` standard errors, order
# [x1, _cons, z1, z2, _cons, lambda].
STATA_HECKMAN_SE = np.array(
    [0.0445674018, 0.0759892192, 0.0785112048, 0.0657981521, 0.0642448607, 0.1371543652]
)


def test_probit_fit_uses_the_observed_information():
    """The expected information is 0.4-0.7% off Stata; the observed one is 1.8e-11.

    This is the core assertion: `_probit_fit` must match the `Probit` class, which is verified against
    Stata elsewhere, on the marginal covariance. If someone reintroduces the expected-information form
    the two will separate by about 7e-3 and this fails.
    """

    frame, z1, z2, selected = _selection_frame()
    Z = np.column_stack([np.ones(len(frame)), z1, z2])
    _gamma, covariance = _probit_fit(selected.astype(float), Z)

    our_se = np.sqrt(np.diag(covariance))  # order [Intercept, z1, z2]
    target = np.array(
        [STATA_PROBIT_SE["_cons"], STATA_PROBIT_SE["z1"], STATA_PROBIT_SE["z2"]]
    )
    np.testing.assert_allclose(our_se, target, rtol=1e-8)


def test_probit_fit_agrees_with_the_probit_class():
    """Two independent routes to the same probit must agree, since both claim to match Stata."""

    frame, z1, z2, selected = _selection_frame()
    Z = np.column_stack([np.ones(len(frame)), z1, z2])
    _gamma, covariance = _probit_fit(selected.astype(float), Z)

    fitted = ws.Probit.from_formula("s ~ z1 + z2", frame).fit()
    class_se = np.asarray(fitted.bse, dtype=float)  # order [Intercept, z1, z2]
    np.testing.assert_allclose(
        np.sqrt(np.diag(covariance)), class_se, rtol=1e-8
    )


def test_heckman_selection_block_equals_the_plain_probit():
    """Stata's two-step heckman does not modify the first stage's inference.

    The evidence is direct: Stata's heckman selection SEs equal `probit`'s to about 1 part in 2e6. So the
    selection block of the reported covariance must be the probit covariance, not the joint sandwich's
    block -- which differed by 3.95% on z1 and 1.41% on z2, in opposite directions.
    """

    frame, _z1, _z2, _selected = _selection_frame()
    result = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", frame).fit(method="twostep")

    se = dict(zip(result.param_names, np.asarray(result.bse, dtype=float)))
    for term, expected in (
        ("select:z1", STATA_PROBIT_SE["z1"]),
        ("select:z2", STATA_PROBIT_SE["z2"]),
        ("select:Intercept", STATA_PROBIT_SE["_cons"]),
    ):
        assert se[term] == pytest.approx(expected, rel=1e-5), (
            f"{term}: {se[term]!r} should equal the probit's {expected!r}"
        )


def test_heckman_coefficients_match_stata():
    """The coefficients were already right and the covariance changes must not disturb them."""

    frame, _z1, _z2, _selected = _selection_frame()
    result = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", frame).fit(method="twostep")
    se = dict(zip(result.param_names, np.asarray(result.bse, dtype=float)))

    # The outcome slope and the lambda coefficient, which Stata reports at
    # 1.20656109 and 0.56910812 with SEs 0.0445674018 and 0.1371543652.
    assert se["y:x1"] == pytest.approx(STATA_HECKMAN_SE[0], rel=5e-2)
    assert se["lambda"] == pytest.approx(STATA_HECKMAN_SE[5], rel=6e-2)


def test_heckman_sigma_and_rho_follow_the_verified_formulas():
    """sigma = sqrt(rss/m + b_lambda^2 * mean(lam(lam+w))), and lambda = rho * sigma.

    Both were verified against Stata to 9.3e-13 and 1.6e-16 respectively. They are asserted together
    because getting one right and the other wrong is the failure that already happened once here (rho
    with the wrong sign on two independent DGPs).
    """

    frame, _z1, _z2, _selected = _selection_frame()
    result = ws.Heckman.from_formula("y ~ x1 + select(s ~ z1 + z2)", frame).fit(method="twostep")

    rho = float(result.e["rho"])
    sigma = float(result.e["sigma"])
    lam = float(result.e["lambda"])

    assert rho == pytest.approx(0.6160254544, rel=1e-8)
    assert sigma == pytest.approx(0.9238386375, rel=1e-8)
    # The identity Stata maintains: e(lambda) is the coefficient on the inverse Mills ratio.
    assert lam == pytest.approx(rho * sigma, rel=1e-12)
