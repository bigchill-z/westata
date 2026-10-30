"""The ``score`` option is correct for every model that offers it.

Round 217 found ``probit`` was returning the LOGIT expression ``y - p`` instead of its own
``(y - p) phi(eta) / (p (1 - p))``. The tell was a non-zero sum: with an intercept the
first-order condition is

    sum_i d ll_i / d eta_i == 0

and probit's sum was 0.25. That identity is a reliable, fixture-free acceptance test for
every single-equation score, so this file applies it across the whole surface.

``ologit`` is deliberately absent: its parameterisation here does not match
``F(cut_k - eta)`` -- the fitted probabilities from ``predict, pr`` disagree with the
hand-built ones -- so its ``scores`` stays unregistered rather than shipping a formula
that could not be verified. See the registry entry for ``predict, scores``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.predict import supported_types

#: Commands whose ``score`` is a single equation-level variable.
SINGLE_EQUATION = ("regress", "logit", "probit", "poisson")


def _frame(n: int = 500, seed: int = 4700):
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


def _fitter(name: str, response: str, frame: pd.DataFrame, formula: str):
    mapping = {
        "regress": ws.OLS,
        "logit": ws.Logit,
        "probit": ws.Probit,
        "poisson": ws.Poisson,
    }
    return mapping[name].from_formula(formula, frame).fit()


BUILDERS = {
    "regress": "y",
    "logit": "yb",
    "probit": "yb",
    "poisson": "ycnt",
}


# ------------------------------------------------------- the acceptance identity
@pytest.mark.parametrize("model_name", SINGLE_EQUATION)
def test_the_score_sums_to_zero_with_an_intercept(model_name: str) -> None:
    """THE identity: ``sum_i d ll_i / d eta_i == 0`` at the maximum-likelihood estimate.

    It is the first-order condition for the constant term, and it needs no captured Stata
    fixture. Probit failed it in round 217 because it reused the logit expression.
    """

    frame = _frame()
    fitted = _fitter(model_name, BUILDERS[model_name], frame, f"{BUILDERS[model_name]} ~ x")

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert abs(float(scores.sum())) < 1e-6


@pytest.mark.parametrize("model_name", SINGLE_EQUATION)
def test_the_identity_does_NOT_hold_without_an_intercept(model_name: str) -> None:
    """Guard: dropping the constant frees the sum, so a constant-zero return fails here.

    Without this pair, a function returning zeros would satisfy the previous test.
    """

    frame = _frame()
    fitted = _fitter(
        model_name, BUILDERS[model_name], frame, f"{BUILDERS[model_name]} ~ x - 1"
    )

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert abs(float(scores.sum())) > 1e-3


# ------------------------------------------------------------ the exact formulas
def test_the_logit_score_is_y_minus_p() -> None:
    frame = _frame()
    fitted = ws.Logit.from_formula("yb ~ x", frame).fit()
    observed = np.asarray(frame["yb"], dtype=float)
    probability = np.asarray(fitted.fittedvalues, dtype=float)

    assert np.allclose(
        np.asarray(fitted.predict(kind="score"), dtype=float),
        observed - probability,
        rtol=1e-10,
    )


def test_the_probit_score_carries_the_density_ratio() -> None:
    """``(y - p) phi(eta) / (p (1 - p))`` -- NOT ``y - p``.

    Round 217's defect: the two families were handled by one branch, so probit inherited
    logit's expression. They agree in sign only.
    """

    from scipy import stats as scipy_stats

    frame = _frame()
    fitted = ws.Probit.from_formula("yb ~ x", frame).fit()
    observed = np.asarray(frame["yb"], dtype=float)
    linear = np.asarray(fitted.predict(kind="xb"), dtype=float)
    probability = scipy_stats.norm.cdf(linear)
    density = scipy_stats.norm.pdf(linear)
    expected = (observed - probability) * density / (probability * (1 - probability))

    assert np.allclose(
        np.asarray(fitted.predict(kind="score"), dtype=float), expected, rtol=1e-10
    )


def test_the_probit_score_is_not_the_logit_one() -> None:
    """Guard: on the same data the two vectors must differ in magnitude."""

    frame = _frame()
    logit_scores = np.asarray(
        ws.Logit.from_formula("yb ~ x", frame).fit().predict(kind="score"), dtype=float
    )
    probit_scores = np.asarray(
        ws.Probit.from_formula("yb ~ x", frame).fit().predict(kind="score"), dtype=float
    )

    assert not np.allclose(logit_scores, probit_scores, rtol=1e-6)
    # Same sign pattern: both are (y - p) up to a positive factor.
    assert np.all(np.sign(logit_scores) == np.sign(probit_scores))


def test_the_poisson_score_is_y_minus_mu() -> None:
    frame = _frame()
    fitted = ws.Poisson.from_formula("ycnt ~ x", frame).fit()

    assert np.allclose(
        np.asarray(fitted.predict(kind="score"), dtype=float),
        np.asarray(fitted.resid, dtype=float),
        rtol=1e-10,
    )


def test_the_linear_score_is_the_residual() -> None:
    frame = _frame()
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()

    assert np.allclose(
        np.asarray(fitted.predict(kind="score"), dtype=float),
        np.asarray(fitted.resid, dtype=float),
        rtol=1e-12,
    )


# ------------------------------------------------------------------- the shape
@pytest.mark.parametrize("model_name", SINGLE_EQUATION)
def test_the_score_is_one_variable(model_name: str) -> None:
    """``predict.sthlp:192-195``: plural ``scores`` is for MULTI-EQUATION models."""

    frame = _frame()
    fitted = _fitter(model_name, BUILDERS[model_name], frame, f"{BUILDERS[model_name]} ~ x")

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert scores.ndim == 1
    assert scores.shape == (len(fitted.model_data),)


# ------------------------------------------------- what is deliberately absent
def test_ologit_does_not_offer_scores() -> None:
    """Its parameterisation could not be reconciled with a closed form here.

    The fitted probabilities from ``predict, pr`` disagree with the hand-built
    ``F(cut_k - eta)`` values, so the score's denominator is not the one a hand derivation
    would use. Registering the type on an unverified formula is the failure mode this
    session keeps removing, so it stays absent.
    """

    assert "scores" not in supported_types("ologit")
    assert "score" not in supported_types("ologit")


def test_nbreg_offers_neither_spelling() -> None:
    """``nbreg`` documents ``scores`` only, and it means TWO variables."""

    assert "score" not in supported_types("nbreg")
    assert "scores" not in supported_types("nbreg")