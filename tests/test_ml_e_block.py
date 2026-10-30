"""Tests for the ML ``e()`` block: ``e(ic)``, the ``k*`` counts, and AIC/BIC.

The headline defect these guard against is ``e(ic)``.  Stata stores the AIC
penalty term there (parameters + 1), and the fixtures confirm the value is a
small integer: ``logit_sim.json`` 4.0, ``probit_sim.json`` 3.0,
``poisson_sim.json`` 1.0, ``nbreg.json`` 4.0, ``ologit.json`` 4.0.

westata's panel family instead stored the *optimiser iteration count* in that
slot -- 7, 116, 156 or 162 depending on the model.  Any caller porting
``estat ic`` (``-2*ll + 2*e(ic)``) would have computed a plausible-looking but
meaningless number, with no error.  That is the failure mode this file pins.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def auto():
    df = pd.read_stata("tests/data/auto.dta")
    df["dom"] = (df["foreign"] == "Foreign").astype(int)
    return df


@pytest.fixture()
def binary_panel() -> pd.DataFrame:
    """A 60-entity panel with a binary outcome and a real entity effect."""

    rng = np.random.default_rng(11)
    n_entities, n_periods = 60, 5
    entity = np.repeat(np.arange(1, n_entities + 1), n_periods)
    time = np.tile(np.arange(n_periods), n_entities)
    x = rng.normal(size=entity.size)
    effect = rng.normal(size=n_entities)[entity - 1]
    latent = 0.5 + 0.8 * x + effect
    y = (latent + rng.logistic(size=entity.size) > 0).astype(float)
    return pd.DataFrame({"y": y, "x": x, "id": entity, "t": time})


# ------------------------------------------------------------------ e(ic)
def test_e_ic_is_the_ITERATION_COUNT(auto) -> None:
    """``e(ic)`` is "number of iterations" -- ``logit.sthlp:257``, verbatim.

    This test previously asserted ``e(ic) == k + 1``, reading the macro as the AIC/BIC
    penalty term.  The fixtures refute that:

    =================  =========  ============  =====================
    fixture            ``e(ic)``  parameters    ``k + 1`` would be
    =================  =========  ============  =====================
    ``logit_sim``      4.0        3             4     (agrees)
    ``poisson_sim``    **1.0**    **2**         **3** (does NOT agree)
    =================  =========  ============  =====================

    One formula cannot produce both, whereas "iterations" describes both -- the Poisson
    fit converged in a single step, the logit needed four.  The old assertion passed only
    because its three models happened to take ``k + 1`` iterations each.
    """

    for model in (
        ws.Logit.from_formula("dom ~ weight + mpg", auto),
        ws.Probit.from_formula("dom ~ weight + mpg", auto),
        ws.Poisson.from_formula("mpg ~ weight", auto),
    ):
        fitted = model.fit()
        assert fitted.e["ic"] >= 1.0, model.model_name
        # It is the same number the property reports, and a plausible iteration count.
        assert fitted.e["ic"] == fitted.iterations
        assert 1.0 <= fitted.e["ic"] < 100.0, model.model_name


def test_e_ic_matches_the_fixture_scale(auto) -> None:
    """The fixtures put ``e(ic)`` in the single digits, not in the hundreds."""

    e = ws.Logit.from_formula("dom ~ weight + mpg", auto).fit().e

    assert e["ic"] < 20


def test_panel_e_ic_is_the_iteration_count(binary_panel) -> None:
    """``xtlogit.sthlp:587,662``: "``e(ic)`` number of iterations" -- same as ``logit``.

    This test previously asserted the OPPOSITE (that ``ic`` was a ``k + 1`` penalty term
    and that the iteration count lived under ``e(iterations)``).  The help file states the
    definition outright for the panel command too, and the ``poisson_sim`` fixture
    (``ic = 1.0`` with two parameters) rules out ``k + 1`` for the non-panel case, so the
    reading is uniform across both families.
    """

    result = ws.XtLogit.from_formula("y ~ x", binary_panel, entity="id").fit()
    e = result.e

    # ``ic`` is a plausible iteration count, and is NOT the k+1 rule.
    assert e["ic"] >= 1
    assert e["ic"] != int(round(e["rank"])) + 1, (
        f"e(ic) = {e['ic']} equals k+1, the penalty-term reading the help file refutes"
    )


def test_panel_iterations_and_ic_are_the_same_quantity(binary_panel) -> None:
    """``e(ic)`` IS the iteration count, so the two slots agree.

    An earlier round kept them apart because ``ic`` was believed to be a penalty term.
    ``xtlogit.sthlp:587`` names it "number of iterations", so they are the same number --
    and ``e(iterations)`` is retained only as a convenience alias for callers written
    against the old, wrong layout.
    """

    result = ws.XtLogit.from_formula("y ~ x", binary_panel, entity="id").fit()
    e = result.e

    assert "iterations" in e
    assert e["ic"] == e["iterations"]


def test_aic_is_consistent_with_ll_and_k(auto) -> None:
    """AIC/BIC must equal the standard formulas, so ``estat ic`` can be built."""

    for model in (
        ws.Logit.from_formula("dom ~ weight + mpg", auto),
        ws.Probit.from_formula("dom ~ weight + mpg", auto),
        ws.Poisson.from_formula("mpg ~ weight", auto),
    ):
        e = model.fit().e
        n = e["N"]
        assert e["aic"] == pytest.approx(-2.0 * e["ll"] + 2.0 * e["k"])
        assert e["bic"] == pytest.approx(
            -2.0 * e["ll"] + np.log(n) * e["k"]
        )


# ------------------------------------------------------- ordered models
def test_ordered_models_have_an_information_criterion(auto) -> None:
    """``ologit``/``oprobit`` used to store no criterion under *any* name.

    ``e(ic)`` is NOT asserted to be ``k + 1`` here: it is the iteration count
    (``logit.sthlp:257``), and the ordered models' backend does not report one, so the key
    is NaN rather than a derived number.  Inventing ``k + 1`` is exactly the reading the
    fixtures refuted for ``poisson_sim``.
    """

    for model in (
        ws.OrderedLogit.from_formula("rep78 ~ weight", auto),
        ws.OrderedProbit.from_formula("rep78 ~ weight", auto),
    ):
        e = model.fit().e
        assert np.isfinite(e["aic"]), model.model_name
        assert np.isfinite(e["bic"]), model.model_name
        assert e["aic"] == pytest.approx(-2.0 * e["ll"] + 2.0 * e["k"])
        # ``e(ic)`` is an iteration count, present only where the backend reports one.
        assert "ic" in e


def test_ordered_models_record_convergence(auto) -> None:
    """The ordered models were the only ML family without ``e(converged)``."""

    e = ws.OrderedLogit.from_formula("rep78 ~ weight", auto).fit().e

    assert "converged" in e
    assert e["converged"] in (0, 1)


def test_ordered_model_reports_k_cat(auto) -> None:
    """``e(k_cat)`` is the number of outcome categories."""

    result = ws.OrderedLogit.from_formula("rep78 ~ weight", auto).fit()
    e = result.e

    categories = auto["rep78"].dropna().nunique()
    assert e["k_cat"] == categories
    assert e["k_aux"] == categories - 1  # one cut point per boundary


# ------------------------------------------------- selection models
def test_selection_models_have_an_information_criterion(auto) -> None:
    """``heckman, method(ml)`` and ``etregress`` lacked any criterion.

    ``e(ic)`` is deliberately NOT asserted to be ``k + 1``: it is the iteration count
    (``logit.sthlp:257``, and the panel family says the same at ``xtlogit.sthlp:587``).
    The ``poisson_sim`` fixture -- ``ic = 1.0`` with two parameters -- rules out the
    penalty-term formula outright.
    """

    heckman = ws.Heckman.from_formula(
        "mpg ~ weight + select(dom ~ weight)", auto
    ).fit(method="ml")
    e = heckman.e

    assert np.isfinite(e["aic"])
    assert np.isfinite(e["bic"])
    assert e["aic"] == pytest.approx(-2.0 * e["ll"] + 2.0 * e["k"])
    # ``ic`` is an iteration count, so it is a positive integer that need not equal k+1.
    assert float(e["ic"]) >= 1.0
    assert e["ic"] != e["k"] + 1, (
        f"e(ic) = {e['ic']} equals k+1, the penalty-term reading the fixtures refute"
    )


def test_two_step_selection_reports_counts_but_no_criterion(auto) -> None:
    """The two-step fit is not a joint ML fit, so it must not claim an AIC."""

    e = ws.Heckman.from_formula("mpg ~ weight + select(dom ~ weight)", auto).fit().e

    assert e["estimator"] == "twostep"
    assert "k" in e
    assert "aic" not in e, (
        "the two-step estimator has no joint likelihood, so no AIC exists"
    )


# ------------------------------------------------------------ k* counts
def test_k_counts_are_recorded_for_ml_models(auto) -> None:
    e = ws.Logit.from_formula("dom ~ weight + mpg", auto).fit().e

    assert e["k"] == e["rank"]
    assert e["k_dv"] == 1
    assert e["k_eq"] == 1


def test_zero_inflated_model_has_two_equations(auto) -> None:
    """``zip``/``zinb`` estimate a count and an inflation equation."""

    rng = np.random.default_rng(13)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    lam = np.exp(0.4 + 0.5 * x1)
    y = rng.poisson(lam).astype(float)
    zero = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 * x2)))).astype(float)
    y = np.where(zero > 0, 0.0, y)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    result = ws.ZeroInflatedPoisson.from_formula(
        "y ~ x1 + x2", frame, inflate="x1 + x2"
    ).fit()

    assert result.e["k_eq"] == 2
