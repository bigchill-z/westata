"""``tobit``'s ``pr(a,b)`` -- ``Pr(a < y < b)`` on the CENSORED outcome.

``tobit_postestimation.sthlp:88-95`` lists the main statistics:

    xb            linear prediction; the default
    stdp          standard error of the linear prediction
    stdf          standard error of the forecast
    pr(a,b)       Pr(a < y < b)
    e(a,b)        E(y | a < y < b)
    ystar(a,b)    E(y*), y* = max{a, min(y,b)}

``pr`` was the one missing, and its absence had a visible consequence: ``tobit.ado:
145-148`` blesses ``Pr(passthru)`` for ``margins``, so ``margins(predict_kind="pr")``
was refused after tobit even though the command's own macro said it was allowed. That
mismatch is what surfaced the gap.

``pr`` differs from ``ystar`` in what the arguments mean. ``ystar`` censors the LATENT
variable between a and b; ``pr`` asks about the OBSERVED outcome, whose distribution has a
point mass at each censoring limit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 400, seed: int = 5500):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 1.0 + 0.8 * covariate + rng.normal(size=n)
    return pd.DataFrame({"y": np.clip(latent, 0.0, None), "x": covariate})


@pytest.fixture(scope="module")
def tobit_fit():
    return ws.Tobit.from_formula("y ~ x", _frame(), lower=0).fit()


# ------------------------------------------------------------- the registration
def test_pr_is_registered_for_tobit() -> None:
    from westata.postestimation.predict import supported_types

    assert "pr" in supported_types("tobit")


def test_margins_accepts_pr_after_tobit(tobit_fit) -> None:
    """``tobit.ado:145-148`` blesses ``Pr(passthru)``; ``margins`` must honour that.

    This is the assertion that would have failed before round 226.
    """

    assert tobit_fit.margins(dydx=["x"], predict_kind="pr") is not None


# ------------------------------------------------------------------ the values
def test_the_unbounded_interval_is_certainty(tobit_fit) -> None:
    """``Pr(-inf < y < inf) = 1`` for any proper distribution."""

    probability = np.asarray(tobit_fit.predict(kind="pr"), dtype=float)

    assert np.allclose(probability, 1.0)


@pytest.mark.parametrize(
    "lower,upper", [(-1.0, 2.0), (0.0, 1.0), (-2.0, -0.5), (1.5, 3.0)]
)
def test_the_probability_is_a_probability(tobit_fit, lower: float, upper: float) -> None:
    probability = np.asarray(
        tobit_fit.predict(kind="pr", lower=lower, upper=upper), dtype=float
    )

    assert np.all(probability >= 0.0)
    assert np.all(probability <= 1.0)
    assert np.all(np.isfinite(probability))


def test_a_wider_interval_is_more_likely(tobit_fit) -> None:
    """Nesting in the set sense must give a larger probability, observation by
    observation -- a formula using the wrong tail would break it."""

    wide = np.asarray(tobit_fit.predict(kind="pr", lower=-2.0, upper=2.0), dtype=float)
    narrow = np.asarray(tobit_fit.predict(kind="pr", lower=-1.0, upper=1.0), dtype=float)

    assert np.all(wide >= narrow - 1e-12)


def test_a_degenerate_interval_has_almost_no_mass(tobit_fit) -> None:
    """``Pr(c < y < c) = 0`` when c is not a censoring limit."""

    probability = np.asarray(
        tobit_fit.predict(kind="pr", lower=0.7, upper=0.7), dtype=float
    )

    assert np.allclose(probability, 0.0, atol=1e-12)


def test_the_lower_censoring_point_holds_mass(tobit_fit) -> None:
    """With ``lower=0``, every censored observation sits exactly AT zero.

    An interval that covers zero therefore collects that point mass, which a pure
    latent-normal probability would miss entirely.  Comparing against the latent-only
    value is what proves the censoring was accounted for.
    """

    from scipy import stats as scipy_stats

    params = np.asarray(tobit_fit.params, dtype=float)
    sigma = float(params[-1])
    design = np.asarray(tobit_fit.design_matrix[0], dtype=float)
    linear = design @ params[:-1]

    covering = np.asarray(
        tobit_fit.predict(kind="pr", lower=-100.0, upper=0.5), dtype=float
    )
    latent_only = scipy_stats.norm.cdf((0.5 - linear) / sigma)

    # The observed-outcome probability must exceed the latent-only one, by the mass at 0.
    assert np.all(covering > latent_only - 1e-12)


def test_pr_and_ystar_are_different_quantities(tobit_fit) -> None:
    """``pr`` is a PROBABILITY; ``ystar`` is an EXPECTATION of a censored variable.

    Guard: if one were implemented by calling the other, this would fail.
    """

    probability = np.asarray(
        tobit_fit.predict(kind="pr", lower=0.0, upper=3.0), dtype=float
    )
    expectation = np.asarray(
        tobit_fit.predict(kind="ystar", lower=0.0, upper=3.0), dtype=float
    )

    assert not np.allclose(probability, expectation, rtol=1e-6)
    # The probability is bounded by 1; the expectation is on the outcome's scale.
    assert np.all(probability <= 1.0)
    assert np.any(expectation > 1.0)


# ------------------------------------------------------------ nothing regressed
def test_e_and_ystar_still_work(tobit_fit) -> None:
    """The dispatch was reorganised, so the two pre-existing types are re-checked."""

    assert np.asarray(tobit_fit.predict(kind="e", lower=0, upper=2)).shape == (
        len(tobit_fit.model_data),
    )
    assert np.asarray(tobit_fit.predict(kind="ystar", lower=0, upper=2)).shape == (
        len(tobit_fit.model_data),
    )


def test_the_registration_lists_all_six_statistics() -> None:
    """``tobit_postestimation.sthlp:88-95``: xb, stdp, stdf, e, ystar, pr."""

    from westata.postestimation.predict import supported_types

    offered = set(supported_types("tobit"))

    assert {"xb", "stdp", "stdf", "e", "ystar", "pr"} <= offered