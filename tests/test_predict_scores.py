"""``predict, score`` -- the derivative of the log likelihood w.r.t. the linear index.

``predict.sthlp:187-195`` is the authority:

    {opt score} calculates the equation-level score; this is usually the derivative of
    the log likelihood with respect to the linear prediction.

    {opt scores} is the ME model equivalent of the {opt score} option, resulting in
    multiple equation-level score variables.  An equation-level score variable is created
    for each equation in the model; ancillary parameters -- such as ln(sigma) and
    atanh(rho) -- make up separate equations.

So the singular form is ONE variable holding ``d ll_i / d eta_i``, and the PLURAL form is
reserved for MULTI-EQUATION models, one variable per equation.

**A correction worth recording.** Round 215 read ``logit_p.ado:4``'s
``syntax ... [, SCores *]`` as evidence that logit accepts the plural spelling, and
registered ``scores`` for logit/probit with a stacked interpretation ("one column per
coefficient").  That was wrong twice over:

* ``SCores`` is Stata's PARSER abbreviation, which matches the documented option
  ``score``; it does not rename it;
* ``scores`` means one variable per EQUATION, not one per coefficient, so the stacked
  ``(n, k)`` array was not the right shape for it either.

Neither ``logit_postestimation.sthlp`` nor ``probit_postestimation.sthlp`` documents
``scores`` at all.  Both now expose only ``score``, and the identity below pins it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.predict import supported_types


def _frame(n: int = 400, seed: int = 4300):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    return pd.DataFrame(
        {
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-(0.4 + 0.9 * covariate)))).astype(
                int
            ),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)),
            "x": covariate,
        }
    )


@pytest.fixture(scope="module")
def logit_fit():
    return ws.Logit.from_formula("yb ~ x", _frame()).fit()


@pytest.fixture(scope="module")
def poisson_fit():
    return ws.Poisson.from_formula("ycnt ~ x", _frame()).fit()


# ---------------------------------------------------------------- the registry
@pytest.mark.parametrize("model_name", ["logit", "probit", "poisson"])
def test_score_is_registered(model_name: str) -> None:
    assert "score" in supported_types(model_name)


@pytest.mark.parametrize("model_name", ["logit", "probit", "poisson"])
def test_the_plural_spelling_is_NOT_registered(model_name: str) -> None:
    """``scores`` belongs to multi-equation models (``predict.sthlp:192-195``).

    None of these three is one: a binary or count fit has a single equation, so there is
    only ever one score variable and Stata names the option ``score``.
    """

    assert "scores" not in supported_types(model_name)


# --------------------------------------------------------------------- the value
def test_the_poisson_score_is_the_residual() -> None:
    """For Poisson ``d ll / d eta = y - mu``, and ``mu = exp(xb)``.

    The response residual ``y - mu`` IS that derivative, which is why the stored
    ``resid`` is exactly what ``predict, score`` returns.
    """

    fitted = ws.Poisson.from_formula("ycnt ~ x", _frame()).fit()
    scores = np.asarray(fitted.predict(kind="score"), dtype=float)
    residuals = np.asarray(fitted.resid, dtype=float)

    assert np.allclose(scores, residuals, rtol=1e-10)


def test_the_logit_score_is_the_logit_derivative() -> None:
    """For a binary logit ``d ll / d eta = y - p``.

    A wrong link derivative (e.g. the probit density ratio) would break the MLE identity
    without changing the shape.
    """

    frame = _frame()
    fitted = ws.Logit.from_formula("yb ~ x", frame).fit()
    observed = np.asarray(frame["yb"], dtype=float)
    probability = np.asarray(fitted.fittedvalues, dtype=float)

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert np.allclose(scores, observed - probability, rtol=1e-10)


@pytest.mark.parametrize("fixture_name", ["logit_fit", "poisson_fit"])
def test_the_score_sums_to_zero_when_the_model_has_a_constant(
    fixture_name, request
) -> None:
    """``sum_i d ll_i / d eta_i == 0`` at the MLE **when an intercept is present**.

    That is the first-order condition for the constant.  Without an intercept the sum is
    free -- which the next test asserts, so this one cannot be satisfied by a constant.
    """

    fitted = request.getfixturevalue(fixture_name)

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert abs(float(scores.sum())) < 1e-6


def test_without_a_constant_the_sum_is_not_forced_to_zero() -> None:
    """Guard for the previous test: dropping the intercept frees that sum.

    A function returning a constant (or an all-zero array) would pass the intercept test
    above and fail here.
    """

    rng = np.random.default_rng(4302)
    n = 300
    covariate = rng.normal(size=n)
    frame = pd.DataFrame(
        {"ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)), "x": covariate}
    )
    fitted = ws.Poisson.from_formula("ycnt ~ x - 1", frame).fit()

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert abs(float(scores.sum())) > 1e-3


def test_the_score_has_one_value_per_observation() -> None:
    """ONE variable, not one per coefficient -- the singular form's shape."""

    fitted = ws.Logit.from_formula("yb ~ x", _frame()).fit()

    scores = np.asarray(fitted.predict(kind="score"), dtype=float)

    assert scores.ndim == 1
    assert scores.shape == (len(fitted.model_data),)