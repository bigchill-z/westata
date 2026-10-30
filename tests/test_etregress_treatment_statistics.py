"""``etregress``'s treatment-specific predictions.

``etregress.ado:423``:

    est local marginsok "XB CTE YCTrt YCNTrt PTrt XBTrt default"

``etregress_postestimation.sthlp:136-167`` defines each:

    ptrt     Pr(treatment=1) = Pr(w gamma + u > 0)
    xbtrt    the treatment equation's linear prediction, w gamma
    yctrt    E(y | treatment=1)
    ycntrt   E(y | treatment=0)
    cte      the conditional treatment effect at the treatment level

Only ``xb``/``stdp``/``stdf``/``fitted`` were registered here, so six of the macro's
entries named capabilities that did not exist.

**``cte`` is derived, not independent.** It is ``yctrt - ycntrt`` by definition
("the difference of potential-outcome means, conditioned on treatment level"), and that
is exactly what makes it checkable: computing it separately from the two potential-outcome
means and comparing catches a sign error in either half. A ``cte`` implemented by its own
formula could be wrong in a way the identity would reveal only if someone thought to look.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

TRUE_EFFECT = 1.5


def _frame(n: int = 400, seed: int = 5800):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    instrument = rng.normal(size=n)
    treatment = (0.3 + 0.9 * instrument + rng.normal(size=n) > 0).astype(int)
    outcome = (
        1.0
        + 0.8 * covariate
        + TRUE_EFFECT * treatment
        + rng.normal(size=n)
    )
    return pd.DataFrame(
        {"y": outcome, "x": covariate, "z": instrument, "t": treatment}
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.Etregress.from_formula("y ~ x + treat(t ~ z)", _frame()).fit()


# ------------------------------------------------------------- the registration
@pytest.mark.parametrize(
    "kind", ["cte", "yctrt", "ycntrt", "ptrt", "xbtrt"]
)
def test_the_five_treatment_statistics_are_registered(kind: str) -> None:
    from westata.postestimation.predict import supported_types

    assert kind in supported_types("etregress")


def test_the_treatment_varlist_is_recorded(fitted) -> None:
    """The probit design must be rebuildable from ``e()`` and ``model_data``."""

    assert str(fitted.e.get("treat_varlist", "")).split() == ["Intercept", "z"]


# --------------------------------------------------------------------- shapes
@pytest.mark.parametrize("kind", ["ptrt", "xbtrt", "yctrt", "ycntrt", "cte"])
def test_every_statistic_is_one_value_per_observation(fitted, kind: str) -> None:
    produced = np.asarray(fitted.predict(kind=kind), dtype=float)

    assert produced.ndim == 1
    assert produced.size == len(fitted.model_data)


# --------------------------------------------------------------------- values
def test_ptrt_is_a_probability(fitted) -> None:
    """``Pr(treatment=1)`` -- stated outright in ``sthlp:164``."""

    produced = np.asarray(fitted.predict(kind="ptrt"), dtype=float)

    assert np.all(produced >= 0.0)
    assert np.all(produced <= 1.0)


def test_xbtrt_is_the_treatment_linear_index(fitted) -> None:
    """``sthlp:167``: "the linear prediction for the treatment equation".

    It must be the normal quantile of ``ptrt``, which is a property no other statistic
    has -- checking it pins that the right design was used.
    """

    from scipy import stats as scipy_stats

    linear = np.asarray(fitted.predict(kind="xbtrt"), dtype=float)
    probability = np.asarray(fitted.predict(kind="ptrt"), dtype=float)

    assert np.allclose(scipy_stats.norm.cdf(linear), probability, rtol=1e-10)


def test_cte_is_exactly_the_difference_of_the_two_paths(fitted) -> None:
    """THE identity: ``cte == yctrt - ycntrt`` (``sthlp:136-137``)."""

    treated = np.asarray(fitted.predict(kind="yctrt"), dtype=float)
    untreated = np.asarray(fitted.predict(kind="ycntrt"), dtype=float)
    effect = np.asarray(fitted.predict(kind="cte"), dtype=float)

    assert np.allclose(effect, treated - untreated, rtol=1e-12)


def test_the_estimated_effect_recovers_the_true_one(fitted) -> None:
    """The data were generated with an effect of 1.5; a wrong sign or scale would show."""

    effect = np.asarray(fitted.predict(kind="cte"), dtype=float)

    assert float(effect.mean()) == pytest.approx(TRUE_EFFECT, abs=0.35)


def test_the_treated_path_lies_above_the_untreated_path(fitted) -> None:
    """With a positive treatment effect, ``yctrt > ycntrt`` observation by observation."""

    treated = np.asarray(fitted.predict(kind="yctrt"), dtype=float)
    untreated = np.asarray(fitted.predict(kind="ycntrt"), dtype=float)

    assert np.all(treated > untreated)


def test_the_two_potential_outcomes_differ_by_more_than_a_constant(
    fitted,
) -> None:
    """The selection correction varies by observation, so the gap is not just ``b_t``.

    A implementation that returned ``xb`` and ``xb + b_t`` would give a CONSTANT gap; the
    observed spread is what proves the Mills-ratio terms were included.
    """

    effect = np.asarray(fitted.predict(kind="cte"), dtype=float)

    assert float(effect.std()) > 1e-3


# ----------------------------------------------------------- nothing regressed
def test_xb_still_works(fitted) -> None:
    assert np.asarray(fitted.predict(kind="xb")).size == int(fitted.nobs)


def test_the_linear_types_are_still_offered() -> None:
    from westata.postestimation.predict import supported_types

    assert {"xb", "stdp", "stdf", "fitted"} <= set(supported_types("etregress"))