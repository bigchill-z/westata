"""``e(aic)`` agrees with ``estat ic`` for the ML family, and does NOT for ``regress``.

This is a distinction that looks like an inconsistency and is not. Both sides are
deliberate:

**ML family (``logit``/``probit``/``poisson``/...)** -- ``e(aic) = -2 ll + 2 k``, which is
exactly what ``estat ic`` reports (``estimates.ado:985-1002``). The two agree, and a caller
can use either.

**``regress``** -- Stata's ``regress`` stores NO information criteria at all (its
stored-results block has no ``e(aic)``, and ``regress_auto.json`` contains neither key).
This package still exposes the two keys for callers, but on the Gaussian-deviance scale
``N log(rss/N) + 2k``, which omits the ``N(1 + log 2pi)`` constant that ``e(ll)`` carries.
So they differ by exactly that constant -- a fixed offset, not a bug in either.

Round 220 established this and round 234 stopped PRINTING the regress pair in the summary.
This file pins the numeric relationship, so a later "make them consistent" change has to
confront it.

The lesson worth keeping: **an inconsistency is not automatically a defect.** Establishing
WHICH reading is correct (here: ML yes, regress no) is the work; flattening the difference
would destroy information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 300, seed: int = 8400):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    latent = 0.4 + 0.9 * covariate
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * covariate + rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


ML_BUILDERS = {
    "logit": lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
    "probit": lambda frame: ws.Probit.from_formula("yb ~ x", frame).fit(),
    "poisson": lambda frame: ws.Poisson.from_formula("ycnt ~ x", frame).fit(),
}


# ------------------------------------------------------------- the ML family
@pytest.mark.parametrize("name", sorted(ML_BUILDERS))
def test_the_ml_family_agrees_with_estat_ic(frame, name: str) -> None:
    """``e(aic) = -2 ll + 2 k``, which is what ``estat ic`` reports."""

    fitted = ML_BUILDERS[name](frame)
    table = fitted.estat_ic()
    e = fitted.e

    assert float(e["aic"]) == pytest.approx(float(table.aic), rel=1e-10)


@pytest.mark.parametrize("name", sorted(ML_BUILDERS))
def test_the_ml_family_uses_the_textbook_formula(frame, name: str) -> None:
    e = ML_BUILDERS[name](frame).e

    assert float(e["aic"]) == pytest.approx(
        -2.0 * float(e["ll"]) + 2.0 * float(e["k"]), rel=1e-10
    )


# ------------------------------------------------------------------ regress
def test_regress_does_not_agree_with_estat_ic(frame) -> None:
    """Deliberate: its ``e(aic)`` is on the Gaussian-deviance scale."""

    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    table = fitted.estat_ic()
    e = fitted.e

    assert float(e["aic"]) != pytest.approx(float(table.aic), rel=1e-6)


def test_the_regress_gap_is_the_omitted_constant(frame) -> None:
    """The difference is exactly ``N(1 + log 2pi)`` -- a constant, not noise.

    Asserting the exact gap is what proves the difference comes from the omitted
    likelihood constant rather than from an arithmetic mistake on one side.
    """

    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    e = fitted.e
    n = float(e["N"])

    gap = float(fitted.estat_ic().aic) - float(e["aic"])

    assert gap == pytest.approx(n * (1.0 + np.log(2.0 * np.pi)), rel=1e-10)


def test_regress_estat_ic_follows_the_documented_formula(frame) -> None:
    """``estimates.ado:985-1002``: ``AIC = -2 e(ll) + 2 e(df)``."""

    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    e = fitted.e
    table = fitted.estat_ic()

    assert float(table.aic) == pytest.approx(
        -2.0 * float(e["ll"]) + 2.0 * float(table.df), rel=1e-12
    )


# ------------------------------------------------------- the two are distinct
def test_the_families_disagree(frame) -> None:
    """Guard: a single shared definition cannot satisfy both, which is the whole point."""

    ml_agrees = abs(
        float(ML_BUILDERS["logit"](frame).e["aic"])
        - float(ML_BUILDERS["logit"](frame).estat_ic().aic)
    ) < 1e-6
    ols_fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    ols_agrees = abs(float(ols_fitted.e["aic"]) - float(ols_fitted.estat_ic().aic)) < 1e-6

    assert ml_agrees is True
    assert ols_agrees is False


# ------------------------------------------------------------ nothing regressed
def test_every_estimator_still_stores_both_criteria(frame) -> None:
    estimators = dict(ML_BUILDERS)
    estimators["regress"] = lambda frame: ws.OLS.from_formula("y ~ x", frame).fit()

    for name, builder in estimators.items():
        e = builder(frame).e
        assert "aic" in e, name
        assert "bic" in e, name
        assert np.isfinite(float(e["aic"])), name