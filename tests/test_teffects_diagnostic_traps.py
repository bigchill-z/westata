"""Two lessons from round 38-39, pinned so a future round cannot un-learn them.

LESSON A -- the *obvious* repair is not the repair. The aipw correction term
`mean(t*resid1/p)` evaluates to -6.58 where it should be ~0. Weighting the outcome regression with
`t/p` and `(1-t)/(1-p)` drives that term to exactly 0.0 by construction -- and moves the point
estimate to 1617.18, FURTHER from Stata's 1621.583 than the unweighted 1619.05. Satisfying one
theoretical condition broke a different one. The arithmetic is pinned here so the trap is visible
to whoever next tries that route.

LESSON B -- a diagnostic's baseline can be the wrong estimator. Comparing the package's propensity
scores against a least-squares fit of `treat` made them look badly wrong ([0.195, 0.676] against
[0.543, 0.663]); against a real logit MLE they agree to 4e-05. A linear-probability regression is not
a propensity model. Pinned so the comparison is never made again by accident.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


@pytest.fixture(scope="module")
def data():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nsw_dw.dta is not available")
    return (
        frame[COVARIATES].to_numpy(dtype=float),
        frame["re78"].to_numpy(dtype=float),
        frame["treat"].to_numpy(dtype=float),
    )


@pytest.fixture(scope="module")
def pieces(data):
    from westata.estimators.treatreg_effects import _LogitPropensity

    x, y, treated = data
    n = y.size
    arm = treated > 0.5
    design = np.column_stack([np.ones(n), x])
    propensity = _LogitPropensity(x, treated, command="teipw").checked_propensity()
    beta1 = np.linalg.lstsq(design[arm], y[arm], rcond=None)[0]
    beta0 = np.linalg.lstsq(design[~arm], y[~arm], rcond=None)[0]
    return design, y, treated, propensity, beta1, beta0


def test_the_unweighted_correction_is_not_zero(pieces) -> None:
    """The defect, stated as a number: -6.5792 against a target of ~0."""

    design, y, treated, propensity, beta1, _ = pieces
    residual1 = y - design @ beta1
    correction = float((treated * residual1 / propensity).mean())
    assert correction == pytest.approx(-6.5792, abs=1e-3)


def test_weighting_does_zero_the_correction(pieces) -> None:
    """LESSON A, part 1: the weighted fit satisfies the orthogonality condition exactly."""

    design, y, treated, propensity, _, _ = pieces
    weights = treated / propensity
    root = np.sqrt(weights)
    beta_weighted = np.linalg.lstsq(
        design * root[:, None], y * root, rcond=None
    )[0]
    residual = y - design @ beta_weighted
    correction = float((treated * residual / propensity).mean())
    assert correction == pytest.approx(0.0, abs=1e-6)


def test_but_weighting_makes_the_estimate_WORSE(pieces) -> None:
    """LESSON A, part 2: satisfying that condition breaks the potential-outcome means.

    Stata's ATE is 1621.583. The unweighted route lands at 1619.05; the weighted one at 1617.18 --
    more than twice as far. This is why "zero the correction term" is not the fix.
    """

    design, y, treated, propensity, beta1, beta0 = pieces

    def ate(b1, b0):
        fitted1 = design @ b1
        fitted0 = design @ b0
        return float(fitted1.mean() - fitted0.mean())

    unweighted = ate(beta1, beta0)
    assert unweighted == pytest.approx(1621.583, abs=0.01)

    root = np.sqrt(treated / propensity)
    weighted1 = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
    root0 = np.sqrt((1.0 - treated) / (1.0 - propensity))
    weighted0 = np.linalg.lstsq(design * root0[:, None], y * root0, rcond=None)[0]
    weighted = ate(weighted1, weighted0)

    # Both are below Stata, but the weighted one is FURTHER away.
    assert weighted < unweighted
    assert abs(weighted - 1621.583) > abs(unweighted - 1621.583)


def test_a_linear_probability_fit_is_not_the_propensity_model(data) -> None:
    """LESSON B: the wrong baseline looks like a large defect.

    `lstsq` of treat on the covariates gives a much narrower probability range than the MLE. Using
    it as the comparison once made a correct propensity model look broken.
    """

    from scipy import optimize
    from scipy.special import expit

    from westata.estimators.treatreg_effects import _LogitPropensity

    x, _, treated = data
    design = np.column_stack([np.ones(treated.size), x])

    def negative_loglik(beta):
        eta = design @ beta
        return -np.sum(treated * eta - np.logaddexp(0.0, eta))

    mle = expit(design @ optimize.minimize(
        negative_loglik, np.zeros(design.shape[1]), method="BFGS"
    ).x)
    linear_probability = design @ np.linalg.lstsq(design, treated, rcond=None)[0]
    package = _LogitPropensity(x, treated, command="teipw").checked_propensity()

    # The package agrees with the MLE ...
    assert np.allclose(package, mle, atol=1e-3)
    # ... and NOT with the linear-probability fit, which is the wrong yardstick.
    assert not np.allclose(package, np.clip(linear_probability, 0.0, 1.0), atol=1e-2)
    # The MLE's range is the wide one; a narrow range means the wrong estimator was used.
    assert float(mle.max() - mle.min()) > 0.4
