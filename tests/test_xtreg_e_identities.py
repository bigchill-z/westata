"""The algebraic identities among ``xtreg``'s stored panel statistics.

``xtreg.sthlp`` documents the returned scalars, and three of them are DEFINED by the
others rather than estimated independently:

* ``e(sigma)``     -- "square root of the sum of e(sigma_e)-squared and e(sigma_u)-squared"
* ``e(rho)``       -- the share of the total variance that is panel-level
* ``e(Tcon)``      -- 1 when T is constant across panels

That makes them checkable without a Stata fixture: if the components are right, the
derived values must follow. A wrong denominator or a squared-versus-unsquared slip
produces a number that still looks like a plausible variance share, which is why the
definitions are asserted rather than eyeballed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _panel(*, balanced: bool, seed: int, entities: int = 40):
    rng = np.random.default_rng(seed)
    if balanced:
        periods = np.full(entities, 6)
    else:
        periods = rng.integers(3, 8, size=entities)
    entity = np.repeat(np.arange(entities), periods)
    n = entity.size
    covariate = rng.normal(size=n)
    effect = rng.normal(size=entities) * 1.5
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + effect[entity] + rng.normal(size=n),
            "x1": covariate,
            "firm": entity,
            "year": np.concatenate([np.arange(p) for p in periods]),
        }
    )
    return frame


@pytest.fixture(scope="module")
def balanced_fit():
    frame = _panel(balanced=True, seed=970)
    return frame, ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()


@pytest.fixture(scope="module")
def unbalanced_fit():
    frame = _panel(balanced=False, seed=972)
    return frame, ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()


def test_sigma_is_the_root_of_the_sum_of_squares(balanced_fit) -> None:
    """``xtreg.sthlp``: ``e(sigma)`` is the square root of ``sigma_e^2 + sigma_u^2``.

    Not the sum of the two standard deviations -- that would be larger and still look
    like a standard deviation.
    """

    e = balanced_fit[1].e
    expected = float(np.sqrt(float(e["sigma_e"]) ** 2 + float(e["sigma_u"]) ** 2))

    assert float(e["sigma"]) == pytest.approx(expected, rel=1e-12)
    # Guard: the wrong formula gives a different number on this fixture.
    assert expected != pytest.approx(float(e["sigma_e"]) + float(e["sigma_u"]))


def test_rho_is_the_panel_share_of_variance(balanced_fit) -> None:
    """``rho = sigma_u^2 / (sigma_u^2 + sigma_e^2)``."""

    e = balanced_fit[1].e
    sigma_u = float(e["sigma_u"])
    sigma_e = float(e["sigma_e"])
    expected = sigma_u ** 2 / (sigma_u ** 2 + sigma_e ** 2)

    assert float(e["rho"]) == pytest.approx(expected, rel=1e-12)


def test_rho_and_sigma_are_consistent(balanced_fit) -> None:
    """Two routes to the same decomposition must agree."""

    e = balanced_fit[1].e
    sigma = float(e["sigma"])
    sigma_u = float(e["sigma_u"])
    rho = float(e["rho"])

    # sigma^2 = sigma_u^2 / rho  -- rearrange the rho definition.
    assert sigma ** 2 == pytest.approx(sigma_u ** 2 / rho, rel=1e-10)


def test_rho_lies_between_zero_and_one(balanced_fit) -> None:
    """A variance share cannot escape the unit interval."""

    rho = float(balanced_fit[1].e["rho"])

    assert 0.0 <= rho <= 1.0


def test_tcon_is_one_for_a_balanced_panel(balanced_fit) -> None:
    """``e(Tcon)``: 1 if T is constant across panels."""

    assert float(balanced_fit[1].e["Tcon"]) == 1.0


def test_tcon_is_zero_for_an_unbalanced_panel(unbalanced_fit) -> None:
    """The other direction -- the flag must be able to say no."""

    assert float(unbalanced_fit[1].e["Tcon"]) == 0.0


def test_the_group_size_extremes_bound_the_mean(balanced_fit) -> None:
    """``g_min <= g_avg <= g_max`` by construction."""

    e = balanced_fit[1].e

    assert float(e["g_min"]) <= float(e["g_avg"]) <= float(e["g_max"])


def test_the_balanced_panel_has_all_groups_equal(balanced_fit) -> None:
    """With a constant T the three size statistics coincide."""

    e = balanced_fit[1].e

    assert float(e["g_min"]) == float(e["g_max"]) == float(e["Tbar"])


def test_the_unbalanced_panel_has_groups_of_different_size(unbalanced_fit) -> None:
    """Guard: if the fixture were actually balanced the previous test is vacuous."""

    e = unbalanced_fit[1].e

    assert float(e["g_min"]) < float(e["g_max"])


def test_n_groups_times_mean_size_is_the_sample(balanced_fit) -> None:
    """``N == N_g * Tbar`` for a balanced panel."""

    e = balanced_fit[1].e

    assert float(e["N"]) == pytest.approx(
        float(e["N_g"]) * float(e["Tbar"]), rel=1e-12
    )


def test_the_within_and_between_r_squared_are_distinct(balanced_fit) -> None:
    """``xtreg`` reports three R-squareds; they are different quantities."""

    e = balanced_fit[1].e

    within = float(e["r2_w"])
    between = float(e["r2_b"])
    overall = float(e["r2_o"])

    assert within != pytest.approx(between, rel=1e-6)
    assert within != pytest.approx(overall, rel=1e-6)


def test_rc_is_a_return_code_not_a_statistic(balanced_fit) -> None:
    """``xtreg.sthlp:1222``: "``e(rc)`` return code".

    Zero means success. It is NOT a correlation and NOT a panel statistic, so reading it
    as missing data or as an estimate would be a mistake.
    """

    assert float(balanced_fit[1].e["rc"]) == 0.0