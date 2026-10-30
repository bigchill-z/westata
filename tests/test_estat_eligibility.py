"""Which `estat` subcommands each estimator allows -- taken from the ado source.

Rounds 80-83 showed that inferring eligibility from ``e()`` contents gets it wrong
in BOTH directions at once: the panel results store ``e(rss)``/``e(mss)``, so the
least-squares guard let ``hettest``/``ovtest`` through while ``estat summarize``
was refused for lack of retained data.

The authoritative source is the command's ``*_estat.ado`` dispatch table.  This
file records the ones that have been read, so a future change that widens or
narrows a surface has to face the Stata list.

| command | allowed | source |
|---|---|---|
| ``xtreg`` | summarize, ic, hausman, mundlak, vce | ``xtreg_estat.ado:15-28`` |
| ``logit`` | classification, gof, auc, + estat_default | ``logit_estat.ado:8-24`` |
| ``rreg`` | none (``e(estat_cmd)`` cleared) | ``rreg.ado:214`` |
| ``ivregress`` | firststage, overid, endogenous, weakrobust, sbknown/sbsingle, + estat_default | ``ivregress_estat.ado:20-56`` |
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(47)
    n = 250
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.6 * x1 - 0.4 * x2 + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": latent,
            "x1": x1,
            "x2": x2,
            "yb": (latent > np.median(latent)).astype(int),
            "z1": rng.normal(size=n),
        }
    )


@pytest.fixture(scope="module")
def panel_frame():
    rng = np.random.default_rng(53)
    entities, periods = 30, 5
    firm = np.repeat(np.arange(entities), periods)
    n = firm.size
    year = np.tile(np.arange(periods), entities)
    x1 = rng.normal(size=n)
    effect = rng.normal(size=entities)[firm]
    y = 1.0 + 0.6 * x1 + effect + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "firm": firm, "year": year})


# --------------------------------------------------------------- xtreg surface
XTREG_ALLOWED = ("estat_summarize", "estat_ic", "estat_vce")
XTREG_REFUSED = ("estat_hettest", "estat_ovtest")


@pytest.mark.parametrize("estimator", ["fe", "re"])
def test_xtreg_allows_exactly_what_the_dispatch_table_allows(
    panel_frame, estimator
) -> None:
    """``xtreg_estat.ado:15-28``: summarize/ic go to estat_default, vce too; the
    rest are "estat <sub> not valid"."""

    if estimator == "fe":
        result = ws.PanelOLS.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit()
    else:
        result = ws.RandomEffects.from_formula(
            "y ~ x1", panel_frame, entity="firm", time="year"
        ).fit()

    for method in XTREG_ALLOWED:
        assert getattr(result, method)() is not None, (estimator, method)

    for method in XTREG_REFUSED:
        with pytest.raises(ValueError):
            getattr(result, method)()


# --------------------------------------------------------------- logit surface
LOGIT_ALLOWED = ("estat_classification", "estat_ic", "estat_summarize", "estat_vce")


def test_logit_allows_its_documented_subcommands(frame) -> None:
    """``logit_estat.ado:8-24`` dispatches classification/gof/auc and falls back to
    ``estat_default`` for the rest."""

    result = ws.Logit.from_formula("yb ~ x1 + x2", frame).fit()

    for method in LOGIT_ALLOWED:
        assert getattr(result, method)() is not None, method


@pytest.mark.parametrize("method", ["estat_hettest", "estat_ovtest", "estat_vif"])
def test_logit_refuses_the_tests_it_does_not_dispatch(frame, method) -> None:
    """Only classification/gof/auc are logit-specific; hettest/ovtest/vif are not
    among them and are not estat_default subcommands either."""

    result = ws.Logit.from_formula("yb ~ x1 + x2", frame).fit()

    with pytest.raises((ValueError, NotImplementedError)):
        getattr(result, method)()


def test_logit_gof_is_now_implemented(frame) -> None:
    """``logit_estat.ado:17-19`` dispatches ``gof`` to ``lfit``.

    Round 85 recorded this as a gap and noted the refusal named the wrong reason
    ("gof is only defined after poisson").  Round 86 implemented the default
    Pearson branch -- ``lfit.ado:44-97`` collapses by covariate pattern and forms

        chi2 = sum (obs - exp)^2 / (exp (1 - exp / w)),   df = m - nvar

    where ``nvar`` counts the intercept too (``lfit.ado:480-504``).
    """

    result = ws.Logit.from_formula("yb ~ x1", frame).fit()

    rendered = result.estat_gof()

    assert hasattr(rendered, "patterns")  # LfitResult, not the Poisson GofResult
    assert np.isfinite(rendered.chi2)
    assert rendered.df == float(rendered.patterns - 2)  # one regressor + _cons


# ---------------------------------------------------------------- rreg surface
def test_rreg_allows_no_estat_subcommands(frame) -> None:
    """``rreg.ado:214`` clears ``e(estat_cmd)`` outright."""

    result = ws.RobustRegression.from_formula("y ~ x1", frame).fit()

    for method in ("estat_hettest", "estat_ovtest"):
        with pytest.raises(ValueError, match="rreg"):
            getattr(result, method)()


# ----------------------------------------------------------------- IV surface
IV_ALLOWED = ("estat_firststage", "estat_endogenous")
IV_REFUSED = ("estat_vif",)


def test_iv_allows_firststage_and_endogenous_only(frame) -> None:
    """``ivregress_estat.ado:20-56`` dispatches those two plus a few others, but
    never ``vif`` -- ``estat_default`` does not offer it."""

    result = ws.IV2SLS.from_formula("y ~ x1 + [x2 ~ z1]", frame).fit()

    for method in IV_ALLOWED:
        assert getattr(result, method)() is not None, method

    for method in IV_REFUSED:
        with pytest.raises(NotImplementedError):
            getattr(result, method)()
