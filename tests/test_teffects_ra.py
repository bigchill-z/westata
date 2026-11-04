"""Pin ``teffects ra``'s point estimates to Stata, and pin what it does not reproduce.

Reference from live runs on a synthetic sample (n = 500, true effect 1.5)::

    teffects ra (y age educ) (treat)
        ATE        1.592426   .2430606
        POmean(0)  44.25704   .3107156
        POmean(1)  45.84947   .232849

    teffects ra (y age educ) (treat), atet
        ATET       1.635659   .2765799

Every point estimate reproduces: regression adjustment fits one outcome regression per arm
and averages the counterfactuals.

The robust standard errors do NOT. Three influence-function forms were tried -- the basic one,
one adding a coefficient-uncertainty term that turned out to be dimensionally wrong, and a
corrected version -- giving 0.1839496 / 0.1839496 / 0.1827790 against Stata's 0.2430606. The
M-estimation form was not recovered, so the fields stay NaN and a test pins that, the same
decision as ``rdrobust``.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws

STATA = {
    "ate": 1.592426,
    "po_0": 44.25704,
    "po_1": 45.84947,
    "atet": 1.635659,
}
STATA_SE_ATE = 0.2430606


def _sample():
    rng = np.random.default_rng(20261105)
    n = 500
    age = rng.integers(25, 65, size=n).astype(float)
    educ = rng.integers(8, 20, size=n).astype(float)
    treat = (rng.normal(size=n) + 0.05 * age - 0.1 * educ > 0).astype(int)
    outcome = 20.0 + 1.5 * treat + 0.3 * age + 0.8 * educ + rng.normal(scale=2.0, size=n)
    return outcome, treat, np.column_stack([age, educ])


@pytest.fixture(scope="module")
def fitted():
    y, d, x = _sample()
    return ws.teffects_ra(y, d, x)


def test_ate_matches_stata(fitted) -> None:
    assert fitted.ate == pytest.approx(STATA["ate"], abs=5e-7)


def test_potential_outcome_means_match_stata(fitted) -> None:
    assert fitted.po_0 == pytest.approx(STATA["po_0"], abs=5e-6)
    assert fitted.po_1 == pytest.approx(STATA["po_1"], abs=5e-6)


def test_atet_matches_stata() -> None:
    y, d, x = _sample()
    result = ws.teffects_ra(y, d, x, estimand="atet")
    assert result.atet == pytest.approx(STATA["atet"], abs=5e-7)
    assert result.estimand == "ATET"


def test_ate_equals_the_difference_of_the_po_means(fitted) -> None:
    """The identity Stata's own table satisfies."""

    assert fitted.ate == pytest.approx(fitted.po_1 - fitted.po_0, rel=1e-12)


def test_the_standard_error_is_reported_as_unavailable(fitted) -> None:
    """Ours is 0.183 against Stata's 0.243, so it must not be shown as a result."""

    assert not fitted.variance_matches_stata
    assert np.isnan(fitted.se)
    assert np.isnan(fitted.z)
    assert np.isnan(fitted.pvalue)


def test_bad_treatment_coding_is_refused() -> None:
    y, _, x = _sample()
    bad = np.where(np.arange(y.size) % 2 == 0, 1, 2)
    with pytest.raises(Exception):
        ws.teffects_ra(y, bad, x)


def test_unknown_estimands_are_refused() -> None:
    y, d, x = _sample()
    with pytest.raises(Exception):
        ws.teffects_ra(y, d, x, estimand="atc")
