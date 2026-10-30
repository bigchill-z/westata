"""``predict, pr(n)`` for the count families.

`poisson_postestimation.sthlp:105, 147-148`:

    pr(n)    probability Pr(y = n), n a nonnegative integer (number or variable)
    pr(a,b)  probability Pr(a < y < b)

Until round 126 the type was not offered at all, and calling ``predict(kind="pr")``
reported ``poisson``'s available list without it — so a documented prediction looked
like a command that simply does not have it, rather than one needing an argument.

Distinguishing those two is the point: without ``n`` the type is INCOMPLETE, which the
error says, instead of pretending it is unavailable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


@pytest.fixture(scope="module")
def poisson_result():
    rng = np.random.default_rng(443)
    n = 300
    frame = pd.DataFrame(
        {"ycnt": rng.poisson(3.0, size=n), "x1": rng.normal(size=n)}
    )
    return ws.Poisson.from_formula("ycnt ~ x1", frame).fit()


def test_pr_is_offered_by_the_count_families() -> None:
    """The dispatcher must list ``pr`` as available for poisson."""

    from westata.postestimation.predict import SUPPORTED_TYPES

    assert "pr" in SUPPORTED_TYPES["poisson"]
    assert "pr" in SUPPORTED_TYPES["nbreg"]


def test_pr_matches_the_poisson_mass_function(poisson_result) -> None:
    """``pr(n)`` is ``Pr(y = n)`` evaluated at the fitted mean."""

    for count in (0, 1, 2, 5, 10):
        got = np.asarray(poisson_result.predict(kind="pr", n=count))
        mean = np.asarray(poisson_result.predict(kind="n"))
        expected = scipy_stats.poisson.pmf(count, mean)
        np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_the_mass_function_sums_to_one(poisson_result) -> None:
    """The definitional property of a pmf.

    A wrong formula can still produce plausible individual values; summing over the
    support cannot be satisfied accidentally.
    """

    total = np.zeros_like(np.asarray(poisson_result.predict(kind="n")))
    for count in range(60):
        total = total + np.asarray(poisson_result.predict(kind="pr", n=count))

    np.testing.assert_allclose(total, 1.0, atol=1e-10)


def test_a_missing_count_is_reported_as_incomplete_not_unsupported(
    poisson_result,
) -> None:
    """``pr`` without ``n`` is an incomplete call, not a missing capability.

    The distinction matters: "unsupported" tells a user to stop, "incomplete" tells
    them what to add.
    """

    with pytest.raises(ValueError, match="needs the count"):
        poisson_result.predict(kind="pr")


def test_pr_is_not_the_same_as_n(poisson_result) -> None:
    """``pr(n)`` is a probability; ``n`` is the expected count.

    They coincide only where the mean happens to make them equal, so this asserts a
    clear separation rather than a numeric difference.
    """

    mass = np.asarray(poisson_result.predict(kind="pr", n=3))
    mean = np.asarray(poisson_result.predict(kind="n"))

    assert np.all(mass <= 1.0)
    assert np.all(mass >= 0.0)
    assert np.all(mean > 0.0)
    assert not np.allclose(mass, mean)


def test_pr_accepts_a_nonnegative_integer() -> None:
    """``n`` is documented as "a nonnegative integer"."""

    rng = np.random.default_rng(449)
    n = 200
    frame = pd.DataFrame(
        {"ycnt": rng.poisson(1.5, size=n), "x1": rng.normal(size=n)}
    )
    result = ws.Poisson.from_formula("ycnt ~ x1", frame).fit()

    zero = np.asarray(result.predict(kind="pr", n=0))
    assert np.all(zero > 0.0)
    assert np.all(zero <= 1.0)


def test_nbreg_uses_the_negative_binomial_not_the_poisson() -> None:
    """``nbreg`` is a NEGATIVE BINOMIAL; using the Poisson pmf would be wrong.

    The first version of this implementation reached for ``poisson.pmf`` for both
    families.  That returns a plausible number and silently ignores the dispersion
    parameter -- the error is exactly the over-dispersion the model was fitted to
    capture.

    Stata: ``Var(y) = mu + alpha*mu^2`` with ``alpha = exp(lnalpha)``.  scipy's
    ``nbinom`` takes ``(size, p)`` with ``size = 1/alpha`` and
    ``p = 1/(1 + alpha*mu)``.
    """

    rng = np.random.default_rng(457)
    n = 250
    frame = pd.DataFrame(
        {"ycnt": rng.poisson(3.0, size=n), "x1": rng.normal(size=n)}
    )
    result = ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit()

    mean = np.asarray(result.predict(kind="n"))
    got = np.asarray(result.predict(kind="pr", n=2))

    names = [str(name) for name in result.param_names]
    alpha = float(np.exp(np.asarray(result.params, dtype=float)[names.index("/lnalpha")]))
    expected = scipy_stats.nbinom.pmf(2, 1.0 / alpha, 1.0 / (1.0 + alpha * mean))

    np.testing.assert_allclose(got, expected, rtol=1e-12)
    # And it must NOT be the Poisson mass: if it were, the dispersion was ignored.
    assert not np.allclose(got, scipy_stats.poisson.pmf(2, mean))


def test_nbreg_pr_sums_to_one() -> None:
    """The nbreg pmf must also integrate to 1 over the support."""

    rng = np.random.default_rng(461)
    n = 200
    frame = pd.DataFrame(
        {"ycnt": rng.poisson(2.0, size=n), "x1": rng.normal(size=n)}
    )
    result = ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit()

    total = np.zeros_like(np.asarray(result.predict(kind="n")))
    for count in range(80):
        total = total + np.asarray(result.predict(kind="pr", n=count))

    np.testing.assert_allclose(total, 1.0, atol=1e-10)


def test_poisson_pr_is_the_poisson(poisson_result) -> None:
    """The contrast case: poisson must stay the Poisson."""

    mean = np.asarray(poisson_result.predict(kind="n"))
    got = np.asarray(poisson_result.predict(kind="pr", n=4))

    np.testing.assert_allclose(got, scipy_stats.poisson.pmf(4, mean), rtol=1e-12)


# ------------------------------------------- the interval form, pr(a, b)
def test_pr_interval_excludes_both_endpoints(poisson_result) -> None:
    """``poisson_postestimation.sthlp:106`` prints ``Pr(a < y < b)`` -- STRICT.

    The help text uses the underline-less-than markup on BOTH sides, so ``pr(1, 4)``
    covers y = 2 and y = 3 but not y = 1 and not y = 4.  Including either endpoint
    would be a one-mass-point error that looks harmless.
    """

    mean = np.asarray(poisson_result.predict(kind="n"))
    got = np.asarray(poisson_result.predict(kind="pr", lower=1, upper=4))
    expected = scipy_stats.poisson.pmf(2, mean) + scipy_stats.poisson.pmf(3, mean)

    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_an_empty_interval_is_zero(poisson_result) -> None:
    """``pr(2, 3)`` contains no integer, so the probability is exactly 0.

    This is the sharpest check that the endpoints are excluded: ``cdf(b) - cdf(a)``
    would return P(y = 3) instead of 0.
    """

    got = np.asarray(poisson_result.predict(kind="pr", lower=2, upper=3))

    np.testing.assert_allclose(got, 0.0, atol=1e-15)


def test_an_open_ended_interval_uses_the_whole_tail(poisson_result) -> None:
    """A missing bound means "no bound" on that side."""

    mean = np.asarray(poisson_result.predict(kind="n"))
    above = np.asarray(poisson_result.predict(kind="pr", lower=2))
    below = np.asarray(poisson_result.predict(kind="pr", upper=2))

    np.testing.assert_allclose(above, 1.0 - scipy_stats.poisson.cdf(2, mean), rtol=1e-12)
    np.testing.assert_allclose(below, scipy_stats.poisson.cdf(1, mean), rtol=1e-12)


def test_the_interval_and_the_point_form_agree(poisson_result) -> None:
    """``pr(k)`` must equal ``pr(k-1, k+1)`` -- one mass point either way."""

    for count in (0, 1, 3, 7):
        point = np.asarray(poisson_result.predict(kind="pr", n=count))
        interval = np.asarray(
            poisson_result.predict(kind="pr", lower=count - 1, upper=count + 1)
        )
        np.testing.assert_allclose(point, interval, rtol=1e-12), count


def test_an_interval_spanning_the_support_sums_to_one(poisson_result) -> None:
    """Covering every possible count leaves probability 1."""

    got = np.asarray(poisson_result.predict(kind="pr", lower=-1, upper=1000))

    np.testing.assert_allclose(got, 1.0, atol=1e-10)


def test_nbreg_interval_uses_the_negative_binomial() -> None:
    """The interval form must respect the dispersion parameter too."""

    rng = np.random.default_rng(463)
    n = 200
    frame = pd.DataFrame(
        {"ycnt": rng.poisson(2.0, size=n), "x1": rng.normal(size=n)}
    )
    result = ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit()

    mean = np.asarray(result.predict(kind="n"))
    names = [str(name) for name in result.param_names]
    alpha = float(
        np.exp(np.asarray(result.params, dtype=float)[names.index("/lnalpha")])
    )
    distribution = scipy_stats.nbinom(1.0 / alpha, 1.0 / (1.0 + alpha * mean))
    expected = distribution.cdf(3) - distribution.cdf(1)

    got = np.asarray(result.predict(kind="pr", lower=1, upper=4))

    np.testing.assert_allclose(got, expected, rtol=1e-12)


# ------------------------------------------------- ir and score
def test_ir_is_the_incidence_rate(poisson_result) -> None:
    """``poisson_postestimation.sthlp:141-142``: ``ir`` is ``exp(xb)``.

    With no offset and no exposure -- the only case this package fits -- that is
    identically ``n``, so ``ir`` is an alias, not a second formula.
    """

    incidence = np.asarray(poisson_result.predict(kind="ir"))
    events = np.asarray(poisson_result.predict(kind="n"))

    np.testing.assert_allclose(incidence, events, rtol=1e-15)
    assert np.all(incidence > 0.0)


def test_ir_equals_exp_of_the_linear_prediction(poisson_result) -> None:
    """Spelled out, so a future refactor cannot silently redefine it."""

    incidence = np.asarray(poisson_result.predict(kind="ir"))
    linear = np.asarray(poisson_result.predict(kind="xb"))

    np.testing.assert_allclose(incidence, np.exp(linear), rtol=1e-12)


def test_score_is_the_loglik_derivative(poisson_result) -> None:
    """``poisson_postestimation.sthlp:163-164``: the score is ``d ll / d eta``.

    For a Poisson that is ``y - mu``.  Its mean over the estimation sample is zero at
    the MLE -- a property a wrong formula will not satisfy.
    """

    score = np.asarray(poisson_result.predict(kind="score"))
    mean = np.asarray(poisson_result.predict(kind="n"))
    observed = np.asarray(poisson_result.resid) + np.asarray(
        poisson_result.fittedvalues
    )

    np.testing.assert_allclose(score, observed - mean, rtol=1e-12)
    assert abs(float(score.mean())) < 1e-9


def test_score_is_not_the_plain_residual(poisson_result) -> None:
    """They coincide for a Poisson response residual, but the score is derived from
    the LIKELIHOOD -- keeping the two definitions separate is what lets ``nbreg``
    refuse instead of reusing this expression."""

    score = np.asarray(poisson_result.predict(kind="score"))
    assert np.all(np.isfinite(score))


def test_nbreg_does_not_claim_score() -> None:
    """``nbreg`` spells it ``scores`` (with the s) and it is not implemented.

    It must therefore NOT be registered: a registry entry that raises is exactly the
    "lie in the registry" the registry test guards against.  ``poisson`` keeps
    ``score``.
    """

    from westata.postestimation.predict import SUPPORTED_TYPES

    assert "score" in SUPPORTED_TYPES["poisson"]
    assert "score" not in SUPPORTED_TYPES["nbreg"]
    assert "scores" not in SUPPORTED_TYPES["nbreg"]


def test_nbreg_scores_explains_why_it_cannot_be_one_array() -> None:
    """The refusal must name the MULTI-variable shape, not just list what is available.

    ``nbreg_postestimation.sthlp:172-181``: "The first new variable will contain the
    derivative of the log likelihood with respect to the regression equation.  The
    second new variable will contain the derivative of the log likelihood with respect
    to the dispersion equation."

    A generic "not in the list" message would read as "this package lacks the feature"
    and would hide that reusing the Poisson ``y - mu`` ignores the dispersion.
    """

    rng = np.random.default_rng(491)
    n = 200
    frame = pd.DataFrame(
        {"ycnt": rng.poisson(3.0, size=n), "x1": rng.normal(size=n)}
    )
    result = ws.NegativeBinomial.from_formula("ycnt ~ x1", frame).fit()

    with pytest.raises(Exception) as caught:
        result.predict(kind="scores")

    message = str(caught.value).lower()
    assert "multiple variables" in message
    assert "dispersion" in message


def test_the_two_families_spell_the_option_differently() -> None:
    """``poisson`` documents ``score``; ``nbreg`` documents ``scores``.

    Recording the asymmetry keeps a future "unify the names" change from silently
    renaming one of them away from its own help file.
    """

    from pathlib import Path

    base = Path(r"D:\Program Files\StataNow19\ado\base")
    poisson_help = (base / "p" / "poisson_postestimation.sthlp").read_text(
        encoding="utf-8", errors="ignore"
    )
    nbreg_help = (base / "n" / "nbreg_postestimation.sthlp").read_text(
        encoding="utf-8", errors="ignore"
    )

    assert "{opt score}" in poisson_help
    assert "{opt scores}" not in poisson_help
    assert "{opt scores}" in nbreg_help
    assert "{opt score}" not in nbreg_help