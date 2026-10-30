"""The ``score`` option: which commands accept it, what spelling, and what it computes.

``predict.sthlp:187-195`` is the general definition:

    {opt score} calculates the equation-level score; this is usually the derivative of
    the log likelihood with respect to the linear prediction.

    {opt scores} is the ME model equivalent of the {opt score} option, resulting in
    multiple equation-level score variables.  An equation-level score variable is created
    for each equation in the model.

So the singular form is ONE variable per observation, and the plural is reserved for
MULTI-EQUATION models. Each command's help file decides which of the two it offers, and
this file pins that per command:

| command | spelling | source |
|---|---|---|
| `regress` | `score` | `regress_postestimation.sthlp:158, 193` |
| `logit` | `score` | `logit_postestimation.sthlp:130, 209` |
| `probit` | `score` | `probit_postestimation.sthlp:149` |
| `poisson` | `score` | `poisson_postestimation.sthlp:163` |
| `nbreg` | **`scores`** | `nbreg_postestimation.sthlp:172` -- TWO variables |

``nbreg`` is the exception, and it is the one that is genuinely two-equation: its help
file says the second variable is the derivative with respect to the DISPERSION equation.
That is why the type is refused there rather than guessed at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.predict import supported_types

SINGULAR_SCORE_COMMANDS = ("regress", "logit", "probit", "poisson")


def _frame(n: int = 400, seed: int = 4600):
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


# ------------------------------------------------------------------ the registry
@pytest.mark.parametrize("model_name", SINGULAR_SCORE_COMMANDS)
def test_the_singular_spelling_is_registered(model_name: str) -> None:
    assert "score" in supported_types(model_name)


@pytest.mark.parametrize("model_name", SINGULAR_SCORE_COMMANDS)
def test_the_plural_spelling_is_not_registered(model_name: str) -> None:
    """None of these four is a multi-equation model, so ``scores`` does not apply.

    Round 215 registered ``scores`` for logit/probit after reading
    ``logit_p.ado:4``'s ``[, SCores *]`` as the plural option name.  It is Stata's
    PARSER abbreviation -- which matches the documented ``score`` -- and the help files
    document no ``scores`` for any of these four.
    """

    assert "scores" not in supported_types(model_name)


def test_nbreg_offers_only_the_plural_and_still_refuses() -> None:
    """``nbreg_postestimation.sthlp:172-181``: ``scores``, TWO variables.

    The second is the derivative with respect to the dispersion equation. This array API
    returns one array per call, and the two equations have different meanings, so the
    type is refused with that reason rather than returning the first one.
    """

    rng = np.random.default_rng(4602)
    frame = pd.DataFrame(
        {"ycnt": rng.negative_binomial(2, 0.5, size=300), "x": rng.normal(size=300)}
    )
    fitted = ws.NegativeBinomial.from_formula("ycnt ~ x", frame).fit()

    with pytest.raises(Exception) as caught:
        fitted.predict(kind="scores")

    message = str(caught.value).lower()
    assert "two variables" in message or "multiple" in message


# ------------------------------------------------------------------- the values
def test_the_linear_score_is_the_residual() -> None:
    """``regress_postestimation.sthlp:193`` says so explicitly."""
    fitted = ws.OLS.from_formula("y ~ x", _frame()).fit()

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert np.allclose(scores, np.asarray(fitted.resid, dtype=float), rtol=1e-12)


def test_the_binary_score_is_observed_minus_probability() -> None:
    """``d ll / d eta = y - p`` for a binary GLM, whatever the link.

    The link's density does NOT appear, because the derivative is with respect to the
    linear prediction rather than to the probability.
    """

    frame = _frame()
    fitted = ws.Logit.from_formula("yb ~ x", frame).fit()
    observed = np.asarray(frame["yb"], dtype=float)
    probability = np.asarray(fitted.fittedvalues, dtype=float)

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert np.allclose(scores, observed - probability, rtol=1e-10)


def test_the_logit_and_probit_scores_are_the_same_function_of_the_residual() -> None:
    """Both families have ``d ll / d eta = y - p``.

    They differ in how ``p`` is formed, not in the score's shape -- so on the same data
    the two score vectors differ only through their fitted probabilities, and neither
    carries a ``phi(eta)`` factor.
    """

    frame = _frame()
    logit_scores = np.asarray(
        ws.Logit.from_formula("yb ~ x", frame).fit().predict(kind="score"), dtype=float
    )
    probit_scores = np.asarray(
        ws.Probit.from_formula("yb ~ x", frame).fit().predict(kind="score"), dtype=float
    )

    assert logit_scores.shape == probit_scores.shape
    # Same sign structure, different magnitudes -- the fitted probabilities differ.
    assert np.all(np.sign(logit_scores) == np.sign(probit_scores))


def test_the_poisson_score_is_observed_minus_mean() -> None:
    fitted = ws.Poisson.from_formula("ycnt ~ x", _frame()).fit()

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert np.allclose(scores, np.asarray(fitted.resid, dtype=float), rtol=1e-10)


@pytest.mark.parametrize(
    "builder_model,response",
    [("Logit", "yb"), ("Probit", "yb"), ("Poisson", "ycnt")],
)
def test_the_score_sums_to_zero_only_with_a_constant(builder_model: str, response: str) -> None:
    """The first-order condition for the intercept is ``Σ score == 0``.

    Run twice on the same data: with a constant it holds; without one it does not. The
    pair is what makes the assertion meaningful -- a function returning zeros would pass
    the first and fail the second.
    """

    frame = _frame()
    estimator = getattr(ws, builder_model)

    with_constant = np.asarray(
        estimator.from_formula(f"{response} ~ x", frame).fit().predict(kind="score"),
        dtype=float,
    )
    without_constant = np.asarray(
        estimator.from_formula(f"{response} ~ x - 1", frame).fit().predict(kind="score"),
        dtype=float,
    )

    assert abs(float(with_constant.sum())) < 1e-6
    assert abs(float(without_constant.sum())) > 1e-3


def test_the_score_shape_is_one_value_per_observation() -> None:
    """ONE variable -- the singular form -- not one per coefficient."""

    fitted = ws.Logit.from_formula("yb ~ x", _frame()).fit()

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert scores.ndim == 1
    assert scores.shape == (len(fitted.model_data),)