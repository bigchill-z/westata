"""``reghdfe`` must be able to predict, and its ``xb`` must include the fixed effects.

``FEOLS`` was the only linear estimator with no prediction types registered at all, so
``fit().predict(kind="xb")`` raised ``UnsupportedError`` while ``OLS``, ``IV2SLS``,
``PanelOLS``, ``Logit`` and ``Poisson`` all worked. ``WESTATA_GAPS.md`` P0-2 recorded it.

The authority for the SEMANTICS is ``areg_p.ado`` -- ``areg`` is the same estimator with one
absorbed dimension, and unlike ``reghdfe`` it ships with Stata::

    areg_p.ado:147-158
        xb (the default) = xb + u        <- INCLUDES the absorbed effect
        d                = u             <- the absorbed effect alone
        u = within-group weighted mean of (e(depvar) - xb)

That distinction is the whole risk in this change. Returning only ``Xb`` would yield a
number that looks like a prediction and is wrong by a group constant -- nobody inspecting
the output would see it. So every test below checks the fixed effect is actually in there,
against an independent within-group computation rather than against a recorded number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _panel(n_units: int = 20, per_unit: int = 10, seed: int = 6101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_units * per_unit
    unit_effect = np.repeat(rng.normal(scale=2.0, size=n_units), per_unit)
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * rng.normal(size=n) + unit_effect,
            "x": rng.normal(size=n),
            "firm": np.repeat(np.arange(n_units), per_unit),
            "year": np.tile(np.arange(1, per_unit + 1), n_units),
        }
    )


@pytest.fixture(scope="module")
def panel():
    return _panel()


@pytest.fixture(scope="module")
def fitted(panel):
    return ws.FEOLS.from_formula("y ~ x | firm", panel).fit()


# --------------------------------------------------------------- registration
def test_predict_is_registered(fitted) -> None:
    """The gap itself: this used to raise "no prediction types registered"."""

    assert fitted.predict(kind="xb") is not None


def test_the_declared_types_match_areg(fitted) -> None:
    """``areg_p.ado`` offers xb (default), d, residuals; adding dresiduals is the same set."""

    assert set(fitted._predict_types) == {"xb", "d", "residuals", "dresiduals"}


# -------------------------------------------- xb includes the absorbed effect
def test_xb_equals_the_full_fitted_values(fitted) -> None:
    """``xb`` is ``xb + u``, so it must equal the model's published fitted values."""

    np.testing.assert_allclose(
        np.asarray(fitted.predict(kind="xb"), dtype=float),
        np.asarray(fitted.fittedvalues, dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


def test_xb_is_not_just_the_linear_predictor(fitted, panel) -> None:
    """The failure this change exists to avoid.

    ``Xb`` alone omits the fixed effect. If the two coincided, either the absorbed effects
    are all zero (impossible here, they were drawn with scale 2) or ``xb`` is dropping them
    and every prediction is wrong by a group constant.
    """

    coefficient = float(np.asarray(fitted.params, dtype=float)[0])
    linear_only = coefficient * panel["x"].to_numpy()

    assert not np.allclose(
        np.asarray(fitted.predict(kind="xb"), dtype=float), linear_only, atol=1e-6
    )


def test_xb_decomposes_into_linear_part_plus_absorbed_effect(fitted, panel) -> None:
    coefficient = float(np.asarray(fitted.params, dtype=float)[0])
    linear_only = coefficient * panel["x"].to_numpy()

    np.testing.assert_allclose(
        np.asarray(fitted.predict(kind="xb"), dtype=float),
        linear_only + np.asarray(fitted.predict(kind="d"), dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


# ------------------------------------------------------- d is the within mean
def test_d_equals_the_within_group_mean_of_the_residual(fitted, panel) -> None:
    """``u = sum(w*(y - xb)) / sum(w)`` per absorbed group, per ``areg_p.ado``.

    Recomputed independently, so this checks the estimator against its definition rather
    than against another code path in the same package.
    """

    coefficient = float(np.asarray(fitted.params, dtype=float)[0])
    y = panel["y"].to_numpy()
    within = y - coefficient * panel["x"].to_numpy()

    groups = panel["firm"].to_numpy()
    manual = np.array([within[groups == g].mean() for g in groups])

    np.testing.assert_allclose(
        np.asarray(fitted.predict(kind="d"), dtype=float), manual, rtol=1e-12, atol=0.0
    )


def test_d_is_constant_within_a_group(fitted, panel) -> None:
    """A fixed effect takes one value per unit, by definition."""

    effect = np.asarray(fitted.predict(kind="d"), dtype=float)
    groups = panel["firm"].to_numpy()

    for group in np.unique(groups)[:5]:
        assert np.ptp(effect[groups == group]) == pytest.approx(0.0, abs=1e-12)


def test_d_has_one_value_per_unit(fitted, panel) -> None:
    effect = np.asarray(fitted.predict(kind="d"), dtype=float)

    assert np.unique(np.round(effect, 10)).size == panel["firm"].nunique()


# ------------------------------------------------------------- residuals
def test_residuals_and_xb_add_up_to_y(fitted, panel) -> None:
    """The identity a reader relies on: y = xb + residual."""

    xb = np.asarray(fitted.predict(kind="xb"), dtype=float)
    residuals = np.asarray(fitted.predict(kind="residuals"), dtype=float)

    np.testing.assert_allclose(xb + residuals, panel["y"].to_numpy(), rtol=1e-12, atol=0.0)


def test_residuals_match_the_result_objects_residuals(fitted) -> None:
    np.testing.assert_allclose(
        np.asarray(fitted.predict(kind="residuals"), dtype=float),
        np.asarray(fitted.resid, dtype=float),
        rtol=1e-10,
        atol=1e-12,
    )


def test_dresiduals_is_the_same_as_residuals_here(fitted) -> None:
    """``areg``'s ``dresiduals`` is ``y - xb`` with ``xb`` already including ``u``."""

    np.testing.assert_allclose(
        np.asarray(fitted.predict(kind="dresiduals"), dtype=float),
        np.asarray(fitted.predict(kind="residuals"), dtype=float),
        rtol=1e-12,
        atol=0.0,
    )


# ------------------------------------------------- the estimator is consistent
def test_the_coefficient_equals_the_within_regression(panel) -> None:
    """Guard that the prediction machinery describes the estimator actually fitted.

    The within (demeaned) regression is what ``xtreg, fe`` / ``areg`` compute; if ``xb``
    were built from a different coefficient, this would disagree.
    """

    fitted = ws.FEOLS.from_formula("y ~ x | firm", panel).fit()
    coefficient = float(np.asarray(fitted.params, dtype=float)[0])

    groups = panel["firm"].to_numpy()
    within_x = panel["x"].to_numpy() - np.array(
        [panel["x"].to_numpy()[groups == g].mean() for g in groups]
    )
    within_y = panel["y"].to_numpy() - np.array(
        [panel["y"].to_numpy()[groups == g].mean() for g in groups]
    )

    assert coefficient == pytest.approx(
        float(within_x @ within_y / (within_x @ within_x)), rel=1e-10
    )


# ------------------------------------------------------------ the boundaries
def test_multi_way_absorption_is_refused_with_a_reason(panel) -> None:
    """Multi-way effects are identified only up to a normalisation.

    Refusing is right; refusing with "no prediction types registered" would have been
    misleading, since single-way absorption works.
    """

    fitted = ws.FEOLS.from_formula("y ~ x | firm + year", panel).fit()

    with pytest.raises(ws.UnsupportedError, match="MORE THAN ONE absorbed dimension"):
        fitted.predict(kind="xb")


def test_a_model_without_absorption_registers_nothing(panel) -> None:
    """No absorbed effect means no ``d``; the generic ``xb`` path already applies."""

    fitted = ws.FEOLS.from_formula("y ~ x", panel).fit()

    assert getattr(fitted, "_predict_types", None) is None


def test_an_unknown_kind_is_refused(panel) -> None:
    fitted = ws.FEOLS.from_formula("y ~ x | firm", panel).fit()

    with pytest.raises((ws.UnsupportedError, ValueError)):
        fitted.predict(kind="stdf")


# --------------------------------------------------------------- the columns
def test_the_prediction_has_one_entry_per_observation(fitted, panel) -> None:
    assert np.asarray(fitted.predict(kind="xb"), dtype=float).shape == (len(panel),)


def test_every_prediction_is_finite(fitted) -> None:
    for kind in ("xb", "d", "residuals", "dresiduals"):
        values = np.asarray(fitted.predict(kind=kind), dtype=float)
        assert np.all(np.isfinite(values)), kind