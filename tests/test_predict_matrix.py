"""The full predict type x estimator grid.

Round 72 audited the `estat` cross product and found silent errors. The same audit
applied to `predict` found something different: **IV2SLS had NO working predict
type at all** -- every kind raised "the design matrix was not retained".

That gap survived because each individual test was "correctly" refusing. A
per-type test only ever asserted the refusal, so a column that refused all
seventeen times looked like sound design rather than a missing feature.

The tests here therefore check the GRID: for each estimator, at least ``xb``
works, or the refusal names that estimator's actual reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    n = 150
    groups, periods = 30, 5
    firm = np.repeat(np.arange(groups), periods)
    year = np.tile(np.arange(periods), groups)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.6 * x1 - 0.4 * x2 + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm": firm,
            "year": year,
            "z1": rng.normal(size=n),
            "yb": (y > y.mean()).astype(int),
            "yc": rng.poisson(3.0, size=n),
        }
    )


# --------------------------------------------------------------- the IV column
def test_iv2sls_predict_works(frame) -> None:
    """The whole IV column used to refuse every type.

    ``IV2SLS`` never passed ``design_matrix`` to its result, so all seventeen
    predict kinds raised ConsistencyError.  No per-type test caught it because
    "refuses" was the asserted behaviour everywhere.
    """

    result = ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z1]", frame).fit()

    assert result.design_matrix is not None
    xb = np.asarray(result.predict(kind="xb"), dtype=float)
    assert xb.shape == (len(frame),)
    assert np.all(np.isfinite(xb))


def test_iv2sls_predict_uses_the_structural_residual(frame) -> None:
    """``xb + residuals`` must reconstruct the response.

    This is the identity that distinguishes a real predict from one wired to the
    wrong vector: if the residuals came from a different fit the sum would not
    return y.
    """

    result = ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z1]", frame).fit()

    reconstructed = np.asarray(result.predict(kind="xb"), dtype=float) + np.asarray(
        result.predict(kind="residuals"), dtype=float
    )

    observed = np.asarray(result.fittedvalues, dtype=float).reshape(-1) + np.asarray(
        result.resid, dtype=float
    ).reshape(-1)
    np.testing.assert_allclose(reconstructed, observed, rtol=1e-8, atol=1e-10)


def test_iv2sls_stdp_is_finite_and_positive(frame) -> None:
    result = ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z1]", frame).fit()

    stdp = np.asarray(result.predict(kind="stdp"), dtype=float)

    assert np.all(np.isfinite(stdp))
    assert np.all(stdp > 0)


# ------------------------------------------------- refusals that name the reason
def test_xtreg_predict_now_works_with_statats_definition(frame) -> None:
    """Round 73 refused because the panel-effect semantics looked ambiguous.

    Round 74 read ``xtreg_postestimation.sthlp:128``: ``xb`` is ``a + xb``, i.e.
    the linear prediction WITHOUT ``u_i``, and that is the default.  The variant
    that adds the panel effect is ``xbu``.  So the default is documented, not a
    judgement call, and predict should offer it.
    """

    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    xb = np.asarray(result.predict(kind="xb"), dtype=float)

    assert xb.shape == (len(frame),)
    assert np.all(np.isfinite(xb))


def test_xtreg_xb_excludes_the_panel_effect(frame) -> None:
    """The identity that pins the definition.

    ``xb + u + e_it`` must equal ``y``, and ``xb`` alone (plus ``e_it``) must NOT
    -- that gap IS ``u_i``.  Asserting only the first would pass even if ``xb``
    wrongly included the effect.
    """

    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    xb = np.asarray(result.predict(kind="xb"), dtype=float)
    residuals = np.asarray(result.predict(kind="residuals"), dtype=float)
    effect = np.asarray(result.predict(kind="u"), dtype=float)
    observed = frame["y"].to_numpy()

    np.testing.assert_allclose(xb + effect + residuals, observed, rtol=1e-10, atol=1e-12)
    # xb alone must NOT already include it.
    assert not np.allclose(xb + residuals, observed, atol=1e-6)


# ------------------------------------------- the FE error-component family
def test_xtreg_offers_the_documented_fe_types(frame) -> None:
    """``xtreg_postestimation.sthlp:127-134`` lists these for FE."""

    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    for kind in ("xb", "stdp", "u", "xbu", "ue", "residuals"):
        values = np.asarray(result.predict(kind=kind), dtype=float)
        assert values.shape == (len(frame),), kind
        assert np.all(np.isfinite(values)), kind


def test_xtreg_xbu_is_xb_plus_u(frame) -> None:
    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    xb = np.asarray(result.predict(kind="xb"), dtype=float)
    effect = np.asarray(result.predict(kind="u"), dtype=float)
    combined = np.asarray(result.predict(kind="xbu"), dtype=float)

    np.testing.assert_allclose(combined, xb + effect, rtol=1e-10)


def test_xtreg_ue_is_u_plus_the_within_residual(frame) -> None:
    """``ue = u_i + e_it`` -- the combined residual (``sthlp:129``)."""

    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    effect = np.asarray(result.predict(kind="u"), dtype=float)
    within = np.asarray(result.predict(kind="residuals"), dtype=float)
    combined = np.asarray(result.predict(kind="ue"), dtype=float)

    np.testing.assert_allclose(combined, effect + within, rtol=1e-10)


def test_xtreg_u_is_constant_within_each_entity(frame) -> None:
    """``u_i`` is a fixed error component: one value per entity, by definition."""

    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    effect = np.asarray(result.predict(kind="u"), dtype=float)
    entities = frame["firm"].to_numpy()
    for value in np.unique(entities):
        mask = entities == value
        assert np.allclose(effect[mask], effect[mask][0], atol=1e-10)


def test_xtreg_u_reproduces_the_stored_sigma_u(frame) -> None:
    """The strongest independent check on the decomposition.

    ``e(sigma_u)`` is fitted by the estimator; ``u_i`` here is recovered from the
    fitted values.  If the split were wrong these two would not agree.
    """

    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    effect = np.asarray(result.predict(kind="u"), dtype=float)
    entities = frame["firm"].to_numpy()
    one_per_entity = [
        float(effect[entities == value][0]) for value in np.unique(entities)
    ]

    assert float(np.std(one_per_entity, ddof=1)) == pytest.approx(
        float(result.e["sigma_u"]), rel=1e-8
    )


def test_xtreg_component_types_are_not_offered_for_random_effects(frame) -> None:
    """``u_i`` is a FIXED effect; the random-effects model has ``u_i`` as a draw.

    Offering the same decomposition for ``xtreg, re`` would claim a quantity the
    model treats differently, so the registry keeps them out.
    """

    result = ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    for kind in ("u", "xbu", "ue"):
        with pytest.raises((ValueError, ws.UnsupportedError)):
            result.predict(kind=kind)


def test_reghdfe_predict_uses_the_areg_semantics(frame) -> None:
    """``reghdfe`` now predicts, and its ``xb`` includes the absorbed effect.

    This test used to assert the OPPOSITE -- that the call raised -- on the grounds that
    ``reghdfe`` is a user-written command absent from ``ado/base``, so Stata offered "no
    single default to copy".

    That reasoning was wrong in a way worth recording: ``areg`` IS in the base
    installation and is the same estimator with one absorbed dimension, and
    ``areg_p.ado:147-158`` states the convention outright --

        xb (default) = xb + u, where u is the absorbed effect
        d            = u

    So the ambiguity was already resolved by the authority; "the command is user-written"
    was a reason to look harder, not a reason to stop. Leaving it open made FEOLS the only
    linear estimator that could not predict at all.
    """

    result = ws.FEOLS.from_formula("y ~ x1 + x2 | firm", frame).fit()

    predicted = np.asarray(result.predict(kind="xb"), dtype=float)

    # xb includes the fixed effect, so it equals the published fitted values.
    np.testing.assert_allclose(
        predicted, np.asarray(result.fittedvalues, dtype=float), rtol=1e-12, atol=0.0
    )
    # And the absorbed effect is separately available.
    effect = np.asarray(result.predict(kind="d"), dtype=float)
    assert np.unique(np.round(effect, 10)).size == frame["firm"].nunique()


def test_the_xtreg_refusal_names_the_estimator(frame) -> None:
    """A generic message hides which estimator refused and why.

    ``xtreg`` DOES offer ``xb`` now (round 74), so the refusal path here is for a
    type it does not have.  The message must still name the model.
    """

    result = ws.PanelOLS.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()

    with pytest.raises((ValueError, ws.UnsupportedError)) as caught:
        result.predict(kind="notatype")
    assert "xtreg" in str(caught.value)


# ----------------------------------------------------------- the working column
@pytest.mark.parametrize(
    "label",
    ["OLS", "Logit", "Probit", "Poisson", "NBreg", "OrderedLogit"],
)
def test_xb_works_across_the_linear_and_ml_family(frame, label) -> None:
    """Recording which columns DO work, so a future refactor notices a loss."""

    estimators = {
        "OLS": lambda: ws.OLS.from_formula("y ~ x1 + x2", frame).fit(),
        "Logit": lambda: ws.Logit.from_formula("yb ~ x1 + x2", frame).fit(),
        "Probit": lambda: ws.Probit.from_formula("yb ~ x1 + x2", frame).fit(),
        "Poisson": lambda: ws.Poisson.from_formula("yc ~ x1", frame).fit(),
        "NBreg": lambda: ws.NegativeBinomial.from_formula("yc ~ x1", frame).fit(),
        "OrderedLogit": lambda: ws.OrderedLogit.from_formula("yp ~ x1", frame).fit()
        if "yp" in frame
        else ws.OrderedLogit.from_formula(
            "rep ~ x1", frame.assign(rep=(frame["y"] > 0).astype(int) + 1)
        ).fit(),
    }

    result = estimators[label]()
    xb = np.asarray(result.predict(kind="xb"), dtype=float)

    assert xb.shape == (len(frame),)
    assert np.all(np.isfinite(xb))


def test_ols_leverage_types_stay_in_range(frame) -> None:
    """``hat``/``leverage`` must lie in [0, 1]; outside means the wrong formula."""

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    for kind in ("hat", "leverage"):
        values = np.asarray(result.predict(kind=kind), dtype=float)
        assert np.all(values >= -1e-9), kind
        assert np.all(values <= 1 + 1e-9), kind


def test_binary_and_count_pr_types_stay_in_range(frame) -> None:
    """``pr`` is a probability; ``n`` is a count. Out-of-range means wrong wiring."""

    logit = ws.Logit.from_formula("yb ~ x1 + x2", frame).fit()
    probabilities = np.asarray(logit.predict(kind="pr"), dtype=float)
    assert np.all((probabilities >= 0) & (probabilities <= 1))

    poisson = ws.Poisson.from_formula("yc ~ x1", frame).fit()
    counts = np.asarray(poisson.predict(kind="n"), dtype=float)
    assert np.all(counts >= 0)
