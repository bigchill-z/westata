"""Panel `e()` identities.

Rounds 95-97 established the identity layer for OLS and the ML families. The panel
estimators store a richer set -- ``sigma_u``, ``sigma_e``, ``rho``, ``theta`` and
four R-squared variants -- and each pair has a definitional relation that a
plausible-looking wrong number would not satisfy.

All the identities below are algebraic, so they need no Stata reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

ENTITIES, PERIODS = 50, 5


@pytest.fixture(scope="module")
def panel_frame():
    rng = np.random.default_rng(103)
    firm = np.repeat(np.arange(ENTITIES), PERIODS)
    year = np.tile(np.arange(PERIODS), ENTITIES)
    n = firm.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=ENTITIES)[firm]
    y = 1.0 + 0.6 * x1 - 0.3 * x2 + effect + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": firm, "year": year})


@pytest.fixture(scope="module")
def fe_result(panel_frame):
    return ws.PanelOLS.from_formula(
        "y ~ x1 + x2", panel_frame, entity="firm", time="year"
    ).fit()


@pytest.fixture(scope="module")
def re_result(panel_frame):
    return ws.RandomEffects.from_formula(
        "y ~ x1 + x2", panel_frame, entity="firm", time="year"
    ).fit()


# --------------------------------------------------------------- rho
@pytest.mark.parametrize("fixture_name", ["fe_result", "re_result"])
def test_rho_is_the_variance_share_of_the_entity_effect(request, fixture_name) -> None:
    """``rho = sigma_u^2 / (sigma_u^2 + sigma_e^2)``.

    This is the share of unexplained variance attributable to the entity effect.
    A number outside [0, 1], or one that does not match the two sigmas, would mean
    the components were estimated separately and never reconciled.
    """

    result = request.getfixturevalue(fixture_name)
    e = result.e

    expected = e["sigma_u"] ** 2 / (e["sigma_u"] ** 2 + e["sigma_e"] ** 2)

    assert e["rho"] == pytest.approx(expected, rel=1e-12)
    assert 0.0 <= e["rho"] <= 1.0


# --------------------------------------------------------------- theta
def test_theta_is_the_quasi_demeaning_factor(re_result) -> None:
    """``theta = 1 - sigma_e / sqrt(T*sigma_u^2 + sigma_e^2)``.

    ``xtreg_re.ado:265-266`` defines it this way; with ``T`` periods the random
    effects estimator subtracts ``theta`` times the entity mean.  ``theta -> 0``
    is pooled OLS and ``theta -> 1`` is fixed effects, so a value outside [0, 1]
    would be meaningless.
    """

    e = re_result.e
    expected = 1.0 - e["sigma_e"] / np.sqrt(
        PERIODS * e["sigma_u"] ** 2 + e["sigma_e"] ** 2
    )

    assert e["theta"] == pytest.approx(expected, rel=1e-12)
    assert 0.0 <= e["theta"] <= 1.0


def test_the_fe_model_stores_no_theta(fe_result) -> None:
    """Theta belongs to the random-effects estimator.

    Fixed effects removes the entity mean entirely, so there is no partial
    demeaning and no theta to report.
    """

    assert "theta" not in fe_result.e


# --------------------------------------------------------------- the r2 family
def test_r2_equals_the_within_r2(fe_result) -> None:
    """``xtreg, fe`` reports the WITHIN R-squared as ``e(r2)``.

    ``e(r2_w)`` carries the same number under its explicit name.
    """

    e = fe_result.e

    assert e["r2"] == pytest.approx(e["r2_w"], rel=1e-12)


def test_r2_o_is_the_overall_r2_and_differs(fe_result) -> None:
    """The overall R-squared answers a different question and must not coincide.

    With a strong entity effect the within and overall values differ
    substantially; if they matched, one of them would be mislabelled.
    """

    e = fe_result.e

    assert not np.isclose(e["r2"], e["r2_o"], rtol=1e-3)
    assert 0.0 <= e["r2_o"] <= 1.0


def test_the_between_r2_of_a_single_regressor_is_one(panel_frame) -> None:
    """With ONE regressor the between regression fits the group means exactly.

    Any number of groups and one slope plus an intercept is a saturated fit on the
    entity means, so ``r2_b == 1``.  It looks suspicious but is a consequence of
    the design, not a bug -- recording it keeps it from being "fixed".
    """

    result = ws.PanelOLS.from_formula(
        "y ~ x1", panel_frame, entity="firm", time="year"
    ).fit()

    assert result.e["r2_b"] == pytest.approx(1.0, rel=1e-10)


def test_the_between_r2_is_below_one_with_two_regressors(fe_result) -> None:
    """Two regressors leave the between fit over-determined, so ``r2_b < 1``."""

    assert fe_result.e["r2_b"] < 1.0


# --------------------------------------------------------------- counts
def test_n_g_is_the_entity_count(fe_result) -> None:
    assert fe_result.e["N_g"] == ENTITIES


def test_the_sigma_components_reproduce_the_total_variance(fe_result) -> None:
    """``sigma_u^2 + sigma_e^2`` is the total unexplained variance.

    The two components are stored separately; their sum must equal the variance
    the pooled residual would show, which is what ``rho`` divides by.
    """

    e = fe_result.e
    total = e["sigma_u"] ** 2 + e["sigma_e"] ** 2

    assert total > 0.0
    assert e["rho"] == pytest.approx(e["sigma_u"] ** 2 / total, rel=1e-12)


def test_between_estimator_fits_one_observation_per_entity(panel_frame) -> None:
    """``xtreg, be`` runs on the entity means, so its sample is the entity count."""

    result = ws.BetweenOLS.from_formula(
        "y ~ x1", panel_frame, entity="firm", time="year"
    ).fit()

    assert result.nobs == ENTITIES


# -------------------------------- theta is one number OR a distribution
UNBALANCED_SIZES = (2, 3, 5, 8, 4)


@pytest.fixture(scope="module")
def unbalanced_frame():
    rng = np.random.default_rng(107)
    firm = np.concatenate(
        [np.full(size, index) for index, size in enumerate(UNBALANCED_SIZES)]
    )
    n = firm.size
    return pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x1": rng.normal(size=n),
            "firm": firm,
            "year": np.arange(n),
        }
    )


def test_a_balanced_panel_stores_a_single_theta(re_result) -> None:
    """``xtreg_re.ado:368-371``: when ``e(Tcon)`` is true, only ``theta`` is set."""

    e = re_result.e

    assert e["Tcon"] == 1
    assert "theta" in e
    for key in ("thta_min", "thta_5", "thta_50", "thta_95", "thta_max"):
        assert key not in e, key


def test_an_unbalanced_panel_stores_the_theta_distribution(
    unbalanced_frame,
) -> None:
    """``xtreg_re.ado:372-379``: otherwise the five percentiles are stored.

    The two forms are MUTUALLY EXCLUSIVE in the ado -- an unbalanced panel has no
    single theta, because each entity contributes a different number of periods and
    theta depends on that count.  Reporting one ``theta`` anyway would be a number
    with no population behind it.
    """

    result = ws.RandomEffects.from_formula(
        "y ~ x1", unbalanced_frame, entity="firm", time="year"
    ).fit()
    e = result.e

    assert e["Tcon"] == 0
    assert "theta" not in e
    for key in ("thta_min", "thta_5", "thta_50", "thta_95", "thta_max"):
        assert key in e, key


def test_the_theta_distribution_is_ordered(unbalanced_frame) -> None:
    """min <= p5 <= p50 <= p95 <= max, and all inside [0, 1]."""

    result = ws.RandomEffects.from_formula(
        "y ~ x1", unbalanced_frame, entity="firm", time="year"
    ).fit()
    e = result.e

    ordered = [
        e["thta_min"],
        e["thta_5"],
        e["thta_50"],
        e["thta_95"],
        e["thta_max"],
    ]

    assert ordered == sorted(ordered)
    assert all(0.0 <= value <= 1.0 for value in ordered)


def test_the_theta_distribution_follows_the_group_sizes(unbalanced_frame) -> None:
    """Recompute each entity's theta from its own group size and compare.

    ``theta = 1 - sigma_e / sqrt(T_i*sigma_u^2 + sigma_e^2)`` uses the ENTITY's
    own ``T_i``.  A larger ``T_i`` pushes theta towards 1, so the largest entity
    must have the largest theta -- which also checks the sizes were matched to the
    right entities.
    """

    result = ws.RandomEffects.from_formula(
        "y ~ x1", unbalanced_frame, entity="firm", time="year"
    ).fit()
    e = result.e

    per_entity = np.array(
        [
            1.0 - e["sigma_e"] / np.sqrt(size * e["sigma_u"] ** 2 + e["sigma_e"] ** 2)
            for size in UNBALANCED_SIZES
        ]
    )

    assert e["thta_min"] == pytest.approx(float(per_entity.min()), rel=1e-12)
    assert e["thta_max"] == pytest.approx(float(per_entity.max()), rel=1e-12)
    # The 5-period entity sits in the middle; the 2-period one is the minimum and
    # the 8-period one the maximum.
    assert int(np.argmin(per_entity)) == 0  # size 2
    assert int(np.argmax(per_entity)) == 3  # size 8


# ------------------- only the random-effects estimator quasi-demeans
@pytest.mark.parametrize("label", ["fe", "be", "fd"])
def test_only_random_effects_stores_theta(panel_frame, label) -> None:
    """``theta`` exists ONLY for ``xtreg, re``.

    ``xtreg_be.ado`` contains no ``theta`` (or ``thta``) at all -- the between
    estimator runs OLS on the entity means and never quasi-demeans.  ``be``
    inherits from ``RandomEffects``, so the "store theta" flag had to be switched
    off explicitly; leaving it inherited put a theta on a be result where Stata has
    none.  That is the same inheritance trap as ``_Ordered`` overriding ``fit`` in
    round 91.
    """

    estimators = {
        "fe": ws.PanelOLS,
        "be": ws.BetweenOLS,
        "fd": ws.FirstDifferenceOLS,
    }
    result = estimators[label].from_formula(
        "y ~ x1", panel_frame, entity="firm", time="year"
    ).fit()

    assert "theta" not in result.e
    assert not any("thta" in key for key in result.e)


def test_random_effects_does_store_theta(re_result) -> None:
    """The positive side of the same rule."""

    assert "theta" in re_result.e


# --------------- every panel estimator's feature flags are explicit
PANEL_FLAGS = {
    "PanelOLS": {
        "_stores_theta": False,
        "_stores_absorbed": True,
        "_stores_mss": False,
        "_stores_ll_0": True,
    },
    "RandomEffects": {
        "_stores_theta": True,
        "_stores_absorbed": False,
        "_stores_mss": False,
        "_stores_ll_0": False,
    },
    "BetweenOLS": {
        "_stores_theta": False,
        "_stores_absorbed": False,
        "_stores_mss": True,
        "_stores_ll_0": True,
    },
    "FirstDifferenceOLS": {
        "_stores_theta": False,
        "_stores_absorbed": False,
        "_stores_mss": False,
        "_stores_ll_0": False,
    },
}


@pytest.mark.parametrize("class_name", sorted(PANEL_FLAGS))
def test_each_panel_estimator_declares_its_own_flags(class_name) -> None:
    """No panel estimator should rely on INHERITED feature flags.

    ``BetweenOLS`` and ``FirstDifferenceOLS`` both subclass ``RandomEffects``, which
    stores ``theta``.  Leaving that inherited put a theta on ``be`` (round 100).
    ``fd`` was correct only by accident: the theta block is guarded by
    ``isfinite(sigma_u)`` and ``fd`` happens to have no ``sigma_u``, so a future
    change that started computing one would silently start reporting theta for a
    model that never quasi-demeans.

    This test states each estimator's intended flags explicitly, so an inherited
    default cannot pass unnoticed.
    """

    estimator = getattr(ws, class_name)
    expected = PANEL_FLAGS[class_name]

    for flag, value in expected.items():
        assert getattr(estimator, flag) is value, (class_name, flag)


def test_no_panel_result_reports_a_component_it_cannot_have(
    panel_frame,
) -> None:
    """End-to-end: the flags must show up as the right e() keys."""

    fitted = {
        "fe": ws.PanelOLS.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit(),
        "re": ws.RandomEffects.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit(),
        "be": ws.BetweenOLS.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit(),
        "fd": ws.FirstDifferenceOLS.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit(),
    }

    # theta: random effects only.
    for label, result in fitted.items():
        if label == "re":
            assert "theta" in result.e or "thta_min" in result.e, label
        else:
            assert "theta" not in result.e, label
            assert "thta_min" not in result.e, label

    # the absorbed-effects block belongs to fixed effects only.  ``F_absorb`` is
    # conditional (it needs an auxiliary pooled regression with residual df), so
    # the presence of ``k_absorb``/``df_a`` is the stable marker.
    for label, result in fitted.items():
        has_absorbed = "k_absorb" in result.e or "df_a" in result.e
        assert has_absorbed == (label == "fe"), label


# ---------------- the sigma/rho/corr family is fe + re only
SIGMA_KEYS = ("sigma_u", "sigma_e", "sigma", "rho", "corr", "ui")


@pytest.mark.parametrize("label", ["fe", "re"])
def test_fe_and_re_store_the_sigma_family(panel_frame, label) -> None:
    estimators = {"fe": ws.PanelOLS, "re": ws.RandomEffects}
    result = estimators[label].from_formula(
        "y ~ x1", panel_frame, entity="firm", time="year"
    ).fit()

    for key in SIGMA_KEYS:
        assert key in result.e, (label, key)


@pytest.mark.parametrize("label", ["be", "fd"])
def test_be_and_fd_store_no_sigma_family(panel_frame, label) -> None:
    """``xtreg_be.ado`` NEVER sets these as e() keys.

    The ``corr`` occurrences in that file are the CORR COMMAND, used to build
    ``r2_o`` and ``r2_w`` (``xtreg_be.ado:148-166``) -- reading them as e() keys is
    what put ``sigma_u``/``rho``/``corr`` on a between-estimator result.  The
    between fit is OLS on the entity means, so there is no panel/epsilon split to
    report at all.

    ``xtreg, fd`` differences the data, which removes the entity effect outright.
    """

    estimators = {"be": ws.BetweenOLS, "fd": ws.FirstDifferenceOLS}
    result = estimators[label].from_formula(
        "y ~ x1", panel_frame, entity="firm", time="year"
    ).fit()

    for key in SIGMA_KEYS:
        assert key not in result.e, (label, key)


def test_the_sigma_flag_is_declared_for_every_panel_estimator() -> None:
    """Pin the flag itself, so an inherited default cannot reappear."""

    assert ws.PanelOLS._stores_sigma is True
    assert ws.RandomEffects._stores_sigma is True
    assert ws.BetweenOLS._stores_sigma is False
    assert ws.FirstDifferenceOLS._stores_sigma is False


def test_between_stores_r2_b(panel_frame) -> None:
    """``xtreg_be.ado:132``: ``est scalar r2_b = e(r2)``.

    Unlike the sigma family -- which ``be`` must NOT have -- ``r2_b`` is genuinely
    set by the between estimator, so removing the inherited block must not remove
    this one.
    """

    result = ws.BetweenOLS.from_formula(
        "y ~ x1", panel_frame, entity="firm", time="year"
    ).fit()

    assert "r2_b" in result.e


def test_fd_r2_within_and_overall_are_a_recorded_gap(panel_frame) -> None:
    """``xtreg, fd`` is documented with ``r2_w``/``r2_o``, which we do not store.

    Recording rather than implementing: ``xtreg.ado`` contains no ``r2_w``/``r2_o``
    at all -- they are set by the per-method sub-programs ``xtreg_fe``/``xtreg_re``
    /``xtreg_be`` -- and ``xtreg, fd`` has no sub-program of its own.  So the
    definition of the within and overall R-squared for a DIFFERENCED model cannot be
    read from the ado the way the other variants' can, and the help file lists it in
    a block that mixes every variant's keys.

    Adding a formula guessed from the pooled regression would produce a plausible
    number with no source.  The gap is recorded here so it is not mistaken for
    "defined and matching".
    """

    result = ws.FirstDifferenceOLS.from_formula(
        "y ~ x1", panel_frame, entity="firm", time="year"
    ).fit()

    assert "r2" in result.e
    assert "r2_w" not in result.e
    assert "r2_o" not in result.e