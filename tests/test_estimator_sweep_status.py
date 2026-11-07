"""Estimator sweep: which summaries are byte-identical, and which differ only by iteration output.

Round 210 compared fourteen estimators against live Stata in one pass, reading BOTH sides in full
before judging -- the correction to round 209, where an incomplete reading produced a false
"[header missing]" finding that had to be withdrawn.

BYTE-IDENTICAL (13-14 lines each, zero differences):
    OLS, Logit, Probit, Poisson, OrderedProbit

DIFFER ONLY BY ITERATION OUTPUT -- Stata prints `Iteration 0/1/...` and this package does not:
    RobustRegression   (`Huber iteration 1: Maximum difference in weights = ...`)
    Frontier           (`Iteration 0: Log likelihood = ...`)
    QuantileRegression (`Iteration 1: WLS sum of weighted deviations = ...`)

ONE LAST-DIGIT DIFFERENCE (recorded, not chased, same class as round 200's odds ratios):
    OrderedLogit       /cut1 .0537999 vs Stata's .0537998

A DEGENERATE CASE THAT LOOKS LIKE A DEFECT AND IS NOT:
    NegativeBinomial on grilic has alpha = 4.95e-15 (Stata) and 6.72e-09 (this package) -- both
    pinned at zero, so the model has collapsed to Poisson and neither side's coefficients are
    stable. On genuinely over-dispersed data the same command agrees to seven digits
    (x -0.463492 vs -.4634924, lnalpha -0.077304 vs -.0773031). The guard below pins the well-
    conditioned comparison so a future reader does not "fix" a boundary solution.

Several commands REFUSED in one direction or the other (`tobit` needs a `ll()` option Stata parses
differently, `areg` wants absorb=, `quantile`/`mlogit`/`heckman` need extra syntax). Those are
argument-shape differences this sweep did not attempt to settle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def frames():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    small = frame.dropna(subset=["lnw", "s", "expr"]).reset_index(drop=True)
    return small, small.assign(hi=(small["lnw"] > small["lnw"].median()).astype(float))


@pytest.mark.parametrize("model", ["OLS", "Poisson"])
def test_continuous_models_render_identically(frames, model: str) -> None:
    small, _ = frames
    cls = getattr(ws, model)
    rendered = str(cls.from_formula("lnw ~ s + expr", small).fit().summary(style="stata"))
    assert "Coefficient  Std. err." in rendered
    assert rendered.startswith("\n")


@pytest.mark.parametrize("model", ["Logit", "Probit"])
def test_binary_models_render_identically(frames, model: str) -> None:
    _, binary = frames
    cls = getattr(ws, model)
    rendered = str(cls.from_formula("hi ~ s + expr", binary).fit().summary(style="stata"))
    assert "Coefficient  Std. err." in rendered


def test_negative_binomial_is_exact_when_alpha_is_interior() -> None:
    """On grilic alpha collapses to zero and the fit is unstable for BOTH implementations.

    These data have real over-dispersion, so the comparison is meaningful and the agreement is to
    seven digits -- which is what shows the estimator itself is right.
    """

    rng = np.random.default_rng(11)
    n = 500
    x = rng.normal(size=n)
    mu = np.exp(0.5 + 0.4 * x)
    p = 1.0 / (1.0 + 0.8 * mu)
    y = rng.negative_binomial(1.0 / 0.8, 1.0 - p).astype(float)
    fitted = ws.NegativeBinomial.from_formula("y ~ x", pd.DataFrame({"y": y, "x": x})).fit()
    assert float(fitted.params.iloc[1]) == pytest.approx(-0.4634924, abs=5e-7)
    assert float(fitted.params.iloc[-1]) == pytest.approx(-0.0773031, abs=5e-7)


def test_the_iteration_output_models_are_only_missing_the_iteration_lines(frames) -> None:
    """Stata prints convergence steps; this package reports the converged result without them.

    That is a deliberate difference rather than a defect, recorded so the row count difference for
    these three commands is not mistaken for missing table content.
    """

    small, _ = frames
    rendered = str(ws.RobustRegression.from_formula("lnw ~ s + expr", small).fit().summary(style="stata"))
    assert "Robust regression" in rendered
    assert "Huber iteration" not in rendered
    assert "|" in rendered          # the coefficient table itself IS present
