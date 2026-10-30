"""The four marginal-effect forms, and why only ``dydx`` is implemented.

``margins.sthlp:265-266``:

    dydx(varlist), eyex(varlist), dyex(varlist), and eydx(varlist)
        request that margins report derivatives of the response with respect to
        varlist rather than on the response itself.  eyex(), dyex(), and eydx()
        report derivatives as elasticities.

Before this round the three elasticity forms were not accepted as arguments at all, so
the caller got ``TypeError: margins() got an unexpected keyword argument 'eyex'``. That
says nothing about whether the form exists in Stata -- it does.

The refusal now names the reason: the scaling factor is PER OBSERVATION, so it must
multiply the estimate and its gradient BEFORE averaging, and the standard error carries
the same factor. Averaging first and scaling the mean gives a different number, which is
why this is a refusal rather than a quick multiplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(809)
    n = 200
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * rng.normal(size=n) + rng.normal(size=n),
            "x1": rng.normal(size=n),
        }
    )
    return ws.OLS.from_formula("y ~ x1", frame).fit()


@pytest.mark.parametrize("form", ["eyex", "dyex", "eydx"])
def test_every_elasticity_form_produces_a_number(fitted, form) -> None:
    """All three are documented Stata options and all three are implemented."""

    rendered = fitted.margins(**{form: "x1"})

    assert np.isfinite(float(rendered.estimate))
    assert np.isfinite(float(rendered.std_err))


def test_eyex_is_the_ratio_form(fitted) -> None:
    """``eyex`` = ``d ln E[y] / d ln x`` = ``(x/E[y]) * dE[y]/dx``.

    The factor varies per observation, so the average is taken over
    ``x_i / E[y_i]`` -- not over ``x_i`` divided by the mean response, which is a
    different number.
    """

    frame = fitted.model_data
    slope = float(fitted.params["x1"])
    prediction = np.asarray(fitted.predict(), dtype=float)
    covariate = np.asarray(frame["x1"], dtype=float)

    expected = slope * float(np.mean(covariate / prediction))

    assert float(fitted.margins(eyex="x1").estimate) == pytest.approx(
        expected, rel=1e-10
    )


def test_eydx_is_the_inverse_response_form(fitted) -> None:
    """``eydx`` = ``d ln E[y] / d x`` = ``(1/E[y]) * dE[y]/dx``."""

    frame = fitted.model_data
    slope = float(fitted.params["x1"])
    prediction = np.asarray(fitted.predict(), dtype=float)

    expected = slope * float(np.mean(1.0 / prediction))

    assert float(fitted.margins(eydx="x1").estimate) == pytest.approx(
        expected, rel=1e-10
    )


def test_the_three_forms_are_distinct(fitted) -> None:
    """Guard: if the factors were interchangeable the tests above prove nothing."""

    values = [
        float(fitted.margins(**{form: "x1"}).estimate)
        for form in ("dydx", "dyex", "eyex", "eydx")
    ]

    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            assert values[left] != pytest.approx(values[right], rel=1e-6)


def test_the_ratio_forms_refuse_an_out_of_domain_response() -> None:
    """``eyex``/``eydx`` take a logarithm of ``E[y]``, so the domain matters.

    An OLS fit on a response that spans zero produces both negative and near-zero
    fitted values.  Either way the elasticity is undefined, and the call must say so
    rather than divide anyway -- ``1/E[y]`` evaluates happily for a negative number.

    My first version branched on whether an EXACT zero occurred, which is too narrow: it
    fell through to the ``else`` branch and asserted a finite result on data that
    actually had 56 negative predictions.
    """

    rng = np.random.default_rng(823)
    n = 120
    covariate = rng.normal(size=n)
    frame = pd.DataFrame(
        {"y": covariate + rng.normal(size=n), "x1": covariate}
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()
    prediction = np.asarray(result.predict(), dtype=float)

    assert np.any(prediction <= 0.0)  # guard: this fixture must be out of domain

    with pytest.raises(ValueError, match="ln|zero"):
        result.margins(eydx="x1")


def test_the_elastic_forms_work_for_a_nonlinear_model() -> None:
    """They are response-scale factors, so they apply to any model with predictions.

    For ``logit`` the response is a probability in ``(0, 1)``, so ``1/E[y]`` is always
    finite and the elasticity is well defined.
    """

    rng = np.random.default_rng(811)
    n = 200
    latent = rng.normal(size=n)
    frame = pd.DataFrame(
        {"yb": (latent > 0).astype(int), "x1": rng.normal(size=n)}
    )
    logit = ws.Logit.from_formula("yb ~ x1", frame).fit()

    for form in ("dyex", "eyex", "eydx"):
        rendered = logit.margins(**{form: "x1"})
        assert np.isfinite(float(rendered.estimate)), form


def test_two_forms_at_once_is_refused(fitted) -> None:
    """``margins`` reports one derivative form per call."""

    with pytest.raises(ValueError, match="one derivative form"):
        fitted.margins(dydx="x1", eyex="x1")


def test_two_elasticity_forms_at_once_is_refused(fitted) -> None:
    with pytest.raises(ValueError, match="one derivative form"):
        fitted.margins(eyex="x1", eydx="x1")


def test_dydx_still_works(fitted) -> None:
    """Regression guard for the new keyword parameters."""

    rendered = fitted.margins(dydx="x1")

    assert np.isfinite(float(rendered.estimate))
    assert np.isfinite(float(rendered.std_err))


def test_the_adjusted_prediction_form_is_unaffected(fitted) -> None:
    """``margins`` with no derivative at all must keep working."""

    rendered = fitted.margins(at={"x1": 0.0})

    assert np.isfinite(float(rendered.estimate))


@pytest.mark.parametrize("form", ["eyex", "eydx"])
def test_the_ratio_forms_are_not_the_rescaled_derivative(fitted, form) -> None:
    """``eyex``/``eydx`` factor through ``E[y_i]``, which VARIES per observation.

    A first version refused these on the grounds that the factor cannot be pulled out of
    the average. That was right for a constant factor and wrong here: the only step that
    must stay per-observation is the DIVISION by ``E[y_i]``, and that is cheap. The
    forms are implemented; this test records that they are NOT ``dydx`` times a scalar,
    so a future "simplification" to one multiplication would fail it.
    """

    plain = float(fitted.margins(dydx="x1").estimate)
    scaled = float(fitted.margins(**{form: "x1"}).estimate)

    assert scaled != pytest.approx(plain, rel=1e-6)


def test_dyex_is_the_derivative_times_the_mean_covariate(fitted) -> None:
    """``dyex`` = ``d E[y] / d ln(x)`` = ``x * d E[y] / d x``.

    For the linear predictor the derivative is the same at every observation, so the
    average of ``x_i * b`` is ``mean(x) * b`` -- checked against the ``dydx`` result and
    the raw covariate mean rather than against the implementation's own arithmetic.
    """

    frame = fitted.model_data
    derivative = float(fitted.margins(dydx="x1").estimate)
    semi = float(fitted.margins(dyex="x1").estimate)

    assert semi == pytest.approx(
        derivative * float(frame["x1"].mean()), rel=1e-10
    )


def test_dyex_scales_the_standard_error_by_the_same_factor(fitted) -> None:
    """The delta-method variance is ``G' V G``, so scaling ``G`` scales the SE too.

    A version that scaled only the estimate would report an interval that no longer
    matches the statistic -- structurally valid, numerically wrong.
    """

    frame = fitted.model_data
    factor = float(frame["x1"].mean())
    plain = fitted.margins(dydx="x1")
    semi = fitted.margins(dyex="x1")

    assert float(semi.std_err) == pytest.approx(
        float(plain.std_err) * abs(factor), rel=1e-10
    )


def test_dyex_works_with_an_at_adjustment(fitted) -> None:
    """``at()`` replaces the covariate values, so the factor becomes that value.

    This exposed a REAL bug: ``at()`` used to be applied AFTER the derivative form was
    chosen, so ``margins(dyex="x1", at={"x1": 2})`` returned exactly the unadjusted
    number -- the factor was still the original mean.
    """

    adjusted = float(fitted.margins(dyex="x1", at={"x1": 2.0}).estimate)
    derivative = float(fitted.margins(dydx="x1", at={"x1": 2.0}).estimate)
    unadjusted = float(fitted.margins(dyex="x1").estimate)

    # Every x is 2, so mean(x) is exactly 2.
    assert adjusted == pytest.approx(2.0 * derivative, rel=1e-10)
    assert adjusted != pytest.approx(unadjusted, rel=1e-6)


def test_dyex_works_with_over(fitted) -> None:
    """Each ``over`` row reports its own group's margin, so its own factor.

    The factor must be the mean of the covariate WITHIN the group -- using the pooled
    mean for every group would give identical semi-elasticities whenever the derivative
    is constant, which is exactly the case here.
    """

    frame = fitted.model_data
    groups = np.where(np.arange(len(frame)) % 2 == 0, "a", "b")
    rows = fitted.margins(dyex="x1", over=groups)
    derivative_rows = fitted.margins(dydx="x1", over=groups)

    assert len(rows) == 2
    by_label = {row.label: row for row in rows}
    derivative_by_label = {row.label: row for row in derivative_rows}
    values = np.asarray(frame["x1"], dtype=float)
    for label in ("a", "b"):
        group_mean = float(np.mean(values[groups == label]))
        assert float(by_label[label].estimate) == pytest.approx(
            float(derivative_by_label[label].estimate) * group_mean, rel=1e-10
        )


def test_the_group_factors_really_differ() -> None:
    """Guard for the previous test: equal group means would make it vacuous.

    The two groups are given very different covariate levels, so a version that used
    the POOLED mean for every row produces two identical numbers -- and would pass a
    weaker assertion.
    """

    rng = np.random.default_rng(827)
    x1 = np.concatenate([rng.normal(0, 1, 100), rng.normal(5, 1, 100)])
    frame = pd.DataFrame({"y": 1.0 + 0.8 * x1 + rng.normal(size=200), "x1": x1})
    result = ws.OLS.from_formula("y ~ x1", frame).fit()
    groups = np.array(["a"] * 100 + ["b"] * 100)

    rows = result.margins(dyex="x1", over=groups)
    estimates = sorted(float(row.estimate) for row in rows)

    assert estimates[0] != pytest.approx(estimates[1], rel=1e-6)


def test_an_unknown_keyword_is_still_a_type_error(fitted) -> None:
    """The recognised forms must not swallow genuinely unknown options."""

    with pytest.raises(TypeError):
        fitted.margins(not_a_form="x1")