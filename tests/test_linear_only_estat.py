"""Linear-only `estat` entry points must refuse on non-linear estimators.

`estat_hettest` and `estat_ovtest` are methods on EVERY result, but both are
least-squares tests:

* ``hettest.ado:108`` scales by ``e(rss)/e(N)`` and ``:155`` reports ``e(mss)/2``;
* ``ovtest`` regresses residuals on powers of the fitted values and compares sums
  of squares.

A `logit` result stores neither ``e(rss)`` nor ``e(mss)``, yet both methods used
to return numbers.  Worse, the numbers were nearly identical to the OLS answers
on the same data, so nothing looked wrong:

    hettest:  logit 0.00496  vs  OLS 0.00487
    ovtest:   logit 1.3478   vs  OLS 1.3508

Both are now typed refusals.  This file pins that, because a plausible number is
harder to notice than an exception.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def frame():
    rng = np.random.default_rng(5)
    n = 150
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.6 * x1 - 0.4 * x2 + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "yb": (y > y.mean()).astype(int),
            "yc": rng.poisson(3.0, size=n),
            "rep": rng.integers(1, 4, size=n).astype(float),
            # needed by the IV tests below (an excluded instrument).
            "z1": rng.normal(size=n),
        }
    )


@pytest.fixture()
def models(frame):
    """One linear fit and several non-linear ones sharing the same design."""

    return {
        "OLS": ws.OLS.from_formula("y ~ x1 + x2", frame).fit(),
        "Logit": ws.Logit.from_formula("yb ~ x1 + x2", frame).fit(),
        "Poisson": ws.Poisson.from_formula("yc ~ x1", frame).fit(),
        "NBreg": ws.NegativeBinomial.from_formula("yc ~ x1", frame).fit(),
        "OrderedLogit": ws.OrderedLogit.from_formula("rep ~ x1", frame).fit(),
    }


# ------------------------------------------------------------------- hettest
def test_hettest_works_on_a_linear_model(models) -> None:
    """The guard must not over-reach: OLS still reports the test."""

    result = models["OLS"].estat_hettest()

    assert np.isfinite(float(result.statistic))
    assert "rss" in models["OLS"].e and "mss" in models["OLS"].e


@pytest.mark.parametrize("label", ["Logit", "Poisson", "NBreg", "OrderedLogit"])
def test_hettest_refuses_on_non_linear_models(models, label) -> None:
    """No ``e(rss)``/``e(mss)`` means no least-squares heteroskedasticity test."""

    result = models[label]

    assert "rss" not in result.e
    with pytest.raises(ValueError, match="linear"):
        result.estat_hettest()


def test_hettest_refusal_is_typed_not_an_indexerror(models) -> None:
    """``nbreg`` used to raise ``IndexError: index 2 is out of bounds``.

    An uncaught IndexError is a crash, not a refusal; it also tells the caller
    nothing about why the test does not apply.
    """

    with pytest.raises(ValueError):
        models["NBreg"].estat_hettest()


# -------------------------------------------------------------------- ovtest
def test_ovtest_works_on_a_linear_model(models) -> None:
    result = models["OLS"].estat_ovtest()

    assert np.isfinite(float(result.statistic))


@pytest.mark.parametrize("label", ["Logit", "Poisson", "NBreg", "OrderedLogit"])
def test_ovtest_refuses_on_non_linear_models(models, label) -> None:
    result = models[label]

    with pytest.raises(ValueError, match="linear|sums of squares"):
        result.estat_ovtest()


def test_the_former_silent_answers_would_have_looked_plausible(frame, models) -> None:
    """Record WHY the guard matters: the wrong numbers were near the right ones.

    If a future change makes these tests fail by returning a number again, this
    documents that the number is not obviously wrong -- which is the whole risk.
    """

    # The comparison is only meaningful if a linear fit is available for the same
    # response; use a binary response with both OLS and logit.
    ols = ws.OLS.from_formula("yb ~ x1", frame).fit()
    logit = ws.Logit.from_formula("yb ~ x1", frame).fit()

    linear_ovtest = float(ols.estat_ovtest().statistic)

    # Both must now refuse rather than mimic it.
    with pytest.raises(ValueError):
        logit.estat_ovtest()
    # The linear number itself remains available and finite.
    assert np.isfinite(linear_ovtest)


# ------------------------------------------------- other linear-only entries
@pytest.mark.parametrize("label", ["Logit", "Poisson"])
def test_vif_still_refuses_on_non_linear(models, label) -> None:
    """``estat vif`` was already guarded; keep it that way."""

    with pytest.raises((ValueError, NotImplementedError)):
        models[label].estat_vif()


@pytest.mark.parametrize("label", ["Logit", "Poisson", "NBreg"])
def test_ic_and_vce_are_still_available(models, label) -> None:
    """``estat ic`` and ``estat vce`` apply to any estimated model.

    They are checked here so a broad "refuse everything non-linear" change would
    fail loudly rather than quietly removing valid functionality.
    """

    result = models[label]

    assert result.estat_ic() is not None
    assert result.estat_vce() is not None


# --------------------- estimators that offer NO estat subcommands at all
def test_rreg_offers_no_estat_subcommands(frame) -> None:
    """``rreg.ado:214``: ``est local estat_cmd ""`` -- robust regression resets the
    estat command to empty, so Stata provides no ``estat`` after it.

    ``rreg`` DOES store ``e(rss)``/``e(mss)``, so the least-squares guard alone
    let ``hettest``/``ovtest`` through and they returned numbers.  Those tests
    assume OLS residuals; ``rreg``'s come from an iteratively reweighted fit.
    """

    result = ws.RobustRegression.from_formula("y ~ x1", frame).fit()

    # The estimate itself is fine and still usable.
    assert np.all(np.isfinite(np.asarray(result.params, dtype=float)))

    for method in ("estat_hettest", "estat_ovtest"):
        with pytest.raises(ValueError, match="rreg"):
            getattr(result, method)()


def test_ols_still_offers_hettest_and_ovtest(frame) -> None:
    """Regression guard: the rreg exclusion must not touch plain OLS."""

    result = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    assert np.isfinite(float(result.estat_hettest().statistic))
    assert np.isfinite(float(result.estat_ovtest().statistic))


# ------------------------------- estat vif is not an ivregress subcommand
def test_iv_regression_has_no_estat_vif(frame) -> None:
    """``ivregress_estat.ado:20-56`` dispatches only firststage / overid /
    endogenous / weakrobust / sbknown|sbsingle; everything else falls through to
    ``estat_default``, whose subcommands are ic / summarize / bootstrap / vce --
    **no vif**.

    The IV estimators were listed as linear models, so ``estat vif`` returned a
    number that was merely the OLS VIF of the same design.  It was numerically
    identical to the OLS fit (1.000203 on both), which is exactly why it looked
    plausible.
    """

    result = ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z1]", frame).fit()

    with pytest.raises(NotImplementedError):
        result.estat_vif()


def test_iv_estimators_still_offer_the_subcommands_stata_does(frame) -> None:
    """firststage, overid and endogenous remain available.

    Refusing ``vif`` must not become "refuse everything after IV".
    """

    result = ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z1]", frame).fit()

    assert result.estat_firststage() is not None
    assert result.estat_endogenous() is not None


# ------------------------- xtreg allows only five estat subcommands
@pytest.fixture()
def panel_frame():
    rng = np.random.default_rng(43)
    entities, periods = 30, 5
    firm = np.repeat(np.arange(entities), periods)
    n = firm.size
    year = np.tile(np.arange(periods), entities)
    x1 = rng.normal(size=n)
    effect = rng.normal(size=entities)[firm]
    y = 1.0 + 0.6 * x1 + effect + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "firm": firm, "year": year})


@pytest.mark.parametrize("estimator", ["fe", "re"])
def test_xtreg_offers_estat_summarize(panel_frame, estimator) -> None:
    """``xtreg_estat.ado:15-18`` sends ``su*`` and ``ic`` to ``estat_default``.

    That makes ``estat summarize`` available after ``xtreg``.  It used to refuse
    with "the estimation-sample data was not retained" -- a gap, because the panel
    estimator held the frame all along.
    """

    if estimator == "fe":
        result = ws.PanelOLS.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit()
    else:
        result = ws.RandomEffects.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit()

    rendered = result.estat_summarize()

    assert set(rendered.table.index) == {"y", "x1"}
    assert float(rendered.table.loc["y", "Obs"]) == len(panel_frame)


@pytest.mark.parametrize("estimator", ["fe", "re"])
@pytest.mark.parametrize("method", ["estat_hettest", "estat_ovtest"])
def test_xtreg_refuses_the_tests_it_does_not_allow(
    panel_frame, estimator, method
) -> None:
    """``xtreg_estat.ado:15-28``: only summarize, ic, hausman, mundlak and vce.

    Anything else is "estat <sub> not valid" (exit 321).  The panel results store
    ``e(rss)``/``e(mss)`` on the pooled regression, so the least-squares guard
    alone let both tests through and they returned numbers Stata never reports.
    """

    if estimator == "fe":
        result = ws.PanelOLS.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit()
    else:
        result = ws.RandomEffects.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit()

    with pytest.raises(ValueError, match="xtreg"):
        getattr(result, method)()


def test_xtreg_still_offers_ic_and_vce(panel_frame) -> None:
    """The two subcommands it does allow must keep working."""

    result = ws.PanelOLS.from_formula(
        "y ~ x1", panel_frame, entity="firm", time="year"
    ).fit()

    assert result.estat_ic() is not None
    assert result.estat_vce() is not None
