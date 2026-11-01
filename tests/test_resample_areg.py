"""``Bootstrap`` and ``Jackknife`` must be able to resample an ``areg`` fit.

Stata documents both as areg's own SE options (``areg.sthlp:47-49``: "vcetype may be ols,
robust, cluster clustvarlist, bootstrap, jackknife, or hc2"), and the help file carries a
worked example of the bootstrap form.

The reference values below were measured on StataNow 19.5 with a 210-row panel
(30 groups x 7 periods, seed 1), one regressor plus ``absorb(firm)``::

    areg y x1, absorb(firm) vce(jackknife)      se(x1) = .08945882
    areg y x1, absorb(firm) vce(bootstrap)      se(x1) = .08104864

Jackknife is deterministic, so it is asserted to the printed digit. Bootstrapping is
random, so it is asserted to a tolerance that a wrong estimator could not meet.

The intercept is deliberately NOT part of the resampled design: areg's constant IS the
absorbed effect, so an explicit intercept column is collinear with the group dummies and
the within transform annihilates it. Leaving it in produced a coefficient of exactly 0.0
with a 0.0 standard error -- a silently broken result, which is why it is called out here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _panel(n_units: int = 30, per_unit: int = 7, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_units * per_unit
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "firm": np.repeat(np.arange(n_units), per_unit),
        }
    )


@pytest.fixture(scope="module")
def panel():
    return _panel()


# ------------------------------------------------------------- jackknife
def test_jackknife_accepts_an_areg_estimator(panel) -> None:
    fitted = ws.Jackknife.from_formula(
        "y ~ x1", panel, estimator="areg", absorb="firm"
    ).fit()

    assert np.all(np.isfinite(np.asarray(fitted.bse, dtype=float)))


def test_jackknife_matches_stata(panel) -> None:
    """Deterministic, so it must match the printed digit rather than 'be close'."""

    fitted = ws.Jackknife.from_formula(
        "y ~ x1", panel, estimator="areg", absorb="firm"
    ).fit()

    assert float(np.asarray(fitted.bse, dtype=float)[0]) == pytest.approx(
        0.08945882, rel=1e-7
    )


# ------------------------------------------------------------- bootstrap
def test_bootstrap_accepts_an_areg_estimator(panel) -> None:
    fitted = ws.Bootstrap.from_formula(
        "y ~ x1", panel, estimator="areg", absorb="firm", reps=50, seed=42
    ).fit()

    assert np.all(np.isfinite(np.asarray(fitted.bse, dtype=float)))


def test_bootstrap_matches_stata(panel) -> None:
    """Random, so the tolerance is loose enough for resampling and tight enough that a
    wrong estimator (which was off by ~10% before the intercept fix) cannot pass."""

    fitted = ws.Bootstrap.from_formula(
        "y ~ x1", panel, estimator="areg", absorb="firm", reps=50, seed=42
    ).fit()

    assert float(np.asarray(fitted.bse, dtype=float)[0]) == pytest.approx(
        0.08104864, rel=5e-3
    )


def test_bootstrap_is_reproducible_with_a_seed(panel) -> None:
    first = ws.Bootstrap.from_formula(
        "y ~ x1", panel, estimator="areg", absorb="firm", reps=30, seed=7
    ).fit()
    replay = ws.Bootstrap.from_formula(
        "y ~ x1", panel, estimator="areg", absorb="firm", reps=30, seed=7
    ).fit()

    np.testing.assert_allclose(
        np.asarray(first.bse, dtype=float), np.asarray(replay.bse, dtype=float),
        rtol=0.0, atol=0.0,
    )


# --------------------------------------------------- the intercept trap
def test_the_intercept_is_not_doubly_counted(panel) -> None:
    """The bug this guards: a design carrying an intercept column gives a 0.0 estimate.

    areg's constant is the absorbed effect; an explicit intercept is collinear with the
    group dummies, and the within transform drives it to exactly zero. A zero here means
    the column is still there.
    """

    fitted = ws.Bootstrap.from_formula(
        "y ~ x1", panel, estimator="areg", absorb="firm", reps=20, seed=3
    ).fit()

    standard_errors = np.asarray(fitted.bse, dtype=float)
    assert np.all(standard_errors > 1e-8), (
        f"a zero standard error means the intercept was resampled twice: {standard_errors}"
    )


# ---------------------------------------------------------- refusals
def test_areg_without_absorb_is_refused(panel) -> None:
    """Stata's absorb() is required, so the resampler cannot guess it."""

    with pytest.raises(ValueError, match="absorb"):
        ws.Bootstrap.from_formula("y ~ x1", panel, estimator="areg")

    with pytest.raises(ValueError, match="absorb"):
        ws.Jackknife.from_formula("y ~ x1", panel, estimator="areg")


def test_the_existing_estimators_still_work(panel) -> None:
    """Adding 'areg' must not disturb 'regress' -- the common path."""

    fitted = ws.Bootstrap.from_formula(
        "y ~ x1", panel, estimator="regress", reps=20, seed=5
    ).fit()

    assert len(np.asarray(fitted.bse, dtype=float)) == 2  # intercept + x1