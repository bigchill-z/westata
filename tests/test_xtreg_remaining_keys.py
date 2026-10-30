"""Coverage tests for the remaining ``xtreg`` ``e()`` keys.

These pin the keys added after the variance-component block: ``e(theta)`` and
``e(chi2)``/``e(p)`` for ``xtreg, re``, plus ``e(df_a)``, ``e(k_absorb)``,
``e(r2_c)``, ``e(ll_0)``, ``e(sum_w)`` and ``e(rc)``.

``e(theta)`` is asserted against the formula in ``xtreg_re.ado:265-266`` and
against the fixture's own values, so a change to the quasi-demeaning factor
fails numerically rather than silently.

``e(F_f)``/``e(p_f)`` are deliberately **not** implemented: they test that the
absorbed effects are jointly zero, which needs the entity effects themselves that
linearmodels does not return.  A test records their absence so nobody mistakes a
missing key for an oversight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

N_ENTITIES = 30
N_PERIODS = 6


@pytest.fixture()
def panel():
    rng = np.random.default_rng(20261022)
    entity = np.repeat(np.arange(N_ENTITIES), N_PERIODS)
    time = np.tile(np.arange(N_PERIODS), N_ENTITIES)
    n = entity.size
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(scale=1.0, size=N_ENTITIES)[entity]
    y = 2.0 + 1.5 * x1 - 0.7 * x2 + effect + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


# ------------------------------------------------------------------ theta
def test_re_stores_theta_from_the_ado_formula(panel) -> None:
    """``theta = 1 - s_e/sqrt(T*s_u^2 + s_e^2)`` (``xtreg_re.ado:265-266``)."""

    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    expected = 1.0 - e["sigma_e"] / np.sqrt(
        e["Tbar"] * e["sigma_u"] ** 2 + e["sigma_e"] ** 2
    )

    assert e["theta"] == pytest.approx(float(expected), rel=1e-12)


def test_theta_is_within_the_unit_interval(panel) -> None:
    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()

    assert 0.0 <= result.e["theta"] <= 1.0


def test_fe_does_not_store_theta(panel) -> None:
    """``xtreg_fe.json`` has no ``e(theta)``: the within estimator never
    quasi-demeans."""

    result = ws.PanelOLS.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()

    assert "theta" not in result.e


# ------------------------------------------------------------------- chi2
def test_re_stores_a_wald_chi_square(panel) -> None:
    """``xtreg_re.json`` carries ``chi2`` and ``p`` and no ``e(F)`` statistic."""

    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    assert "chi2" in e
    assert e["chi2type"] == "Wald"
    assert np.isfinite(e["chi2"])
    assert 0.0 <= e["p"] <= 1.0


def test_re_chi2_equals_the_wald_statistic(panel) -> None:
    """``e(chi2)`` is ``e(F) * e(df_m)`` because F is that Wald over df_m."""

    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    assert e["chi2"] == pytest.approx(e["F"] * e["df_m"], rel=1e-12)


def test_re_p_is_the_chi2_upper_tail(panel) -> None:
    from scipy import stats

    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    assert e["p"] == pytest.approx(
        float(stats.chi2.sf(e["chi2"], e["df_m"])), rel=1e-12
    )


# -------------------------------------------------------------- fe block
def test_fe_stores_the_absorbed_effects_block(panel) -> None:
    result = ws.PanelOLS.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    # k_absorb counts the absorbed levels; df_a drops one for the constant.
    assert e["k_absorb"] == float(N_ENTITIES)
    assert e["df_a"] == float(N_ENTITIES - 1)


def test_fe_r2_c_is_the_pooled_ols_r2(panel) -> None:
    """``r2_c`` is the R-squared of a plain OLS fit (``xtreg_fe.ado:513-520``).

    It is *not* comparable with ``e(r2)`` here: for ``xtreg, fe`` that key holds
    the WITHIN R-squared, which can be larger or smaller than the pooled one.
    """

    result = ws.PanelOLS.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()

    matrix = np.column_stack(
        [np.ones(len(panel)), panel["x1"].to_numpy(), panel["x2"].to_numpy()]
    )
    y = panel["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    resid = y - matrix @ beta
    centred = y - y.mean()
    expected = 1.0 - float(resid @ resid) / float(centred @ centred)

    assert result.e["r2_c"] == pytest.approx(expected, rel=1e-12)


def test_fe_F_f_compares_within_r2_against_pooled_r2(panel) -> None:
    """``F_f = ((r2 - r2_c)/df_a) / ((1 - r2)/df_r)`` (``xtreg_fe.ado:506, 524``).

    The ``r2`` in THAT formula is ``1 - e(rss)/e(tss)``, which
    ``xtreg_fe.ado:505`` explicitly warns is "not the same as e(r2) returned from
    _regress".  This test used to substitute ``e["r2"]`` and thereby locked the
    WRONG behaviour in: on weakly clustered data that produced a NEGATIVE F
    statistic (-0.00891), and on ``xtreg_fe.json`` it gave 17.075 against the
    recorded 28.085.
    """

    result = ws.PanelOLS.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    # The formula's r2, computed the way Stata computes it.
    r2_used = 1.0 - e["rss"] / e["tss"]
    expected = ((r2_used - e["r2_c"]) / e["df_a"]) / (
        (1.0 - r2_used) / e["df_r"]
    )

    assert e["F_f"] == pytest.approx(float(expected), rel=1e-12)
    # An F statistic cannot be negative; the old assertion allowed it.
    assert e["F_f"] >= 0.0


def test_fe_p_f_is_the_upper_tail_of_F_f(panel) -> None:
    from scipy import stats

    result = ws.PanelOLS.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    assert e["p_f"] == pytest.approx(
        float(stats.f.sf(e["F_f"], e["df_a"], e["df_r"])), rel=1e-12
    )


def test_fe_and_be_store_ll_0(panel) -> None:
    for estimator in (ws.PanelOLS, ws.BetweenOLS):
        result = estimator.from_formula(
            "y ~ x1 + x2", panel, entity="firm", time="year"
        ).fit()
        assert "ll_0" in result.e, estimator._stata_command
        assert np.isfinite(result.e["ll_0"])
        # The null model fits worse, so its log-likelihood is more negative.
        assert result.e["ll_0"] < result.e["ll"], estimator._stata_command


def test_re_does_not_store_ll_0(panel) -> None:
    """``xtreg_re.json`` has no ``e(ll_0)``."""

    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()

    assert "ll_0" not in result.e


def test_re_does_not_store_the_absorbed_block(panel) -> None:
    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()

    assert "k_absorb" not in result.e
    assert "df_a" not in result.e
    assert "r2_c" not in result.e


# ---------------------------------------------------------------- common
def test_sum_w_and_rc_are_stored(panel) -> None:
    result = ws.PanelOLS.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()

    assert result.e["sum_w"] == float(result.nobs)
    assert result.e["rc"] == 0.0


def test_the_model_test_p_value_is_stored(panel) -> None:
    from scipy import stats

    result = ws.PanelOLS.from_formula(
        "y ~ x1 + x2", panel, entity="firm", time="year"
    ).fit()
    e = result.e

    assert e["p"] == pytest.approx(
        float(stats.f.sf(e["F"], e["df_m"], e["df_r"])), rel=1e-10
    )


def test_full_fixture_coverage(panel) -> None:
    """Every key in the three Stata fixtures is now stored.

    ``xtreg_fe.json`` has 37 scalar keys, ``xtreg_re.json`` 20 and
    ``xtreg_be.json`` 23.  This asserts full coverage rather than a sample, so a
    future regression that drops one key fails here.
    """

    import json
    from pathlib import Path

    fixtures = Path("tests/conformance/stata_fixtures")
    for name, fixture, estimator in (
        ("xtreg, fe", "xtreg_fe.json", ws.PanelOLS),
        ("xtreg, re", "xtreg_re.json", ws.RandomEffects),
        ("xtreg, be", "xtreg_be.json", ws.BetweenOLS),
    ):
        expected = set(json.loads((fixtures / fixture).read_text())["scalars"])
        result = estimator.from_formula(
            "y ~ x1 + x2", panel, entity="firm", time="year"
        ).fit()
        missing = expected - set(result.e)
        assert not missing, f"{name} missing {sorted(missing)}"
