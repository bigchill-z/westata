"""``linktest`` -- Pregibon's link test, from readable ``linktest.ado`` source.

``logit_postestimation.sthlp`` lists three postestimation commands after ``logit``:
``estat classification``, ``estat gof`` and ``linktest``. The first two were implemented;
``linktest`` was not.

The procedure (``linktest.ado:106-134``):

    _predict `hat', xb                 // the fitted model's linear index
    gen `hat2' = `hat'^2
    `nxtcmd' `lhs' _hat _hatsq ...     // REFIT the same estimator with both added
    ret scalar t  = _b[_hatsq]/_se[_hatsq]
    ret scalar df = e(df_r)

A significant ``_hatsq`` means the linear index is mis-specified -- the FUNCTIONAL FORM is
wrong, which is a different diagnosis from a poor fit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


def _linear(*, curvature: float = 0.0, n: int = 400, seed: int = 3600):
    """A linear model, optionally with a quadratic term the linear fit cannot capture."""

    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    response = (
        1.0
        + 0.8 * covariate
        + curvature * covariate ** 2
        + rng.normal(size=n) * 0.5
    )
    return pd.DataFrame({"y": response, "x": covariate})


@pytest.fixture(scope="module")
def well_specified():
    return ws.OLS.from_formula("y ~ x", _linear(curvature=0.0)).fit()


@pytest.fixture(scope="module")
def misspecified():
    return ws.OLS.from_formula("y ~ x", _linear(curvature=1.5, seed=3602)).fit()


# ---------------------------------------------------------------- the procedure
def test_the_hatsq_coefficient_is_from_a_refit(well_specified) -> None:
    """``linktest.ado:110-129``: the model is REFIT with ``_hat`` and ``_hatsq`` added.

    Rebuilding that refit by hand must reproduce the reported coefficient, which is what
    distinguishes a real link test from a number that merely looks plausible.
    """

    frame = pd.DataFrame(well_specified.model_data).reset_index(drop=True)
    linear = np.asarray(well_specified.predict(kind="xb"), dtype=float)
    augmented = frame.assign(_hat=linear, _hatsq=linear ** 2)
    refit = ws.OLS.from_formula("y ~ _hat + _hatsq", augmented).fit()
    expected = float(np.asarray(refit.params, dtype=float)[
        list(refit.param_names).index("_hatsq")
    ])

    assert well_specified.linktest().hatsq_coefficient == pytest.approx(
        expected, rel=1e-10
    )


def test_the_t_statistic_is_coefficient_over_standard_error(well_specified) -> None:
    frame = pd.DataFrame(well_specified.model_data).reset_index(drop=True)
    linear = np.asarray(well_specified.predict(kind="xb"), dtype=float)
    augmented = frame.assign(_hat=linear, _hatsq=linear ** 2)
    refit = ws.OLS.from_formula("y ~ _hat + _hatsq", augmented).fit()
    position = list(refit.param_names).index("_hatsq")
    expected = float(np.asarray(refit.params, dtype=float)[position]) / float(
        np.asarray(refit.bse, dtype=float)[position]
    )

    assert well_specified.linktest().t_statistic == pytest.approx(expected, rel=1e-10)


def test_the_degrees_of_freedom_come_from_the_refit(well_specified) -> None:
    """``linktest.ado:132`` reads ``e(df_r)`` AFTER the refit, so it is n - 3, not n - 2."""

    reported = well_specified.linktest()

    assert reported.df == pytest.approx(len(well_specified.model_data) - 3.0)


def test_the_p_value_is_two_sided(well_specified) -> None:
    reported = well_specified.linktest()

    assert reported.pvalue == pytest.approx(
        float(2.0 * scipy_stats.t.sf(abs(reported.t_statistic), reported.df)),
        rel=1e-10,
    )


# ---------------------------------------------------------------- the diagnosis
def test_a_well_specified_model_is_not_rejected(well_specified) -> None:
    """A correctly linear fit must not flag a link error."""

    assert well_specified.linktest().pvalue > 0.05


def test_a_quadratic_misspecification_is_detected(misspecified) -> None:
    """A model fitted without a real quadratic term must flag one.

    This is the whole point: ``linktest`` diagnoses the FUNCTIONAL FORM, and data built
    with curvature cannot be described by a straight line.
    """

    assert misspecified.linktest().pvalue < 0.001


def test_the_two_panels_differ(well_specified, misspecified) -> None:
    """Guard: if both behaved alike the direction tests are vacuous."""

    quiet = well_specified.linktest().pvalue
    loud = misspecified.linktest().pvalue

    assert loud < 0.001 < 0.05 < quiet


def test_the_hat_coefficient_is_near_one_when_the_link_is_right(well_specified) -> None:
    """``_hat`` should recover the identity map when the linear index is correct.

    A link test's ``_hat`` coefficient near 1 with ``_hatsq`` near 0 is the signature of a
    well-specified linear predictor; the fitted model IS a linear function of x, so the
    refit reproduces it.
    """

    reported = well_specified.linktest()

    assert 0.5 < reported.hat_coefficient < 1.5


# ------------------------------------------------------------------ other models
@pytest.mark.parametrize("builder", ["logit", "probit", "poisson"])
def test_the_test_runs_for_the_binary_and_count_families(builder: str) -> None:
    """``linktest`` is documented after ``logit``, and applies to the same GLM family."""

    rng = np.random.default_rng(3604)
    n = 300
    covariate = rng.normal(size=n)
    latent = 0.5 + 0.8 * covariate
    frame = pd.DataFrame(
        {
            "yb": (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-latent))).astype(int),
            "ycnt": rng.poisson(np.exp(0.3 + 0.4 * covariate)),
            "x": covariate,
        }
    )
    models = {
        "logit": lambda: ws.Logit.from_formula("yb ~ x", frame).fit(),
        "probit": lambda: ws.Probit.from_formula("yb ~ x", frame).fit(),
        "poisson": lambda: ws.Poisson.from_formula("ycnt ~ x", frame).fit(),
    }

    reported = models[builder]().linktest()

    assert np.isfinite(reported.t_statistic)
    assert 0.0 <= reported.pvalue <= 1.0


# ------------------------------------------------------------------- refusals
@pytest.mark.parametrize("name", ["zip", "zinb", "sureg", "mvreg", "eivreg"])
def test_the_commands_stat_lists_are_refused(name: str) -> None:
    """``linktest.ado:52-66`` names them explicitly.

    Their linear index is not a single ``xb``, so ``_hat``/``_hatsq`` are undefined.
    """

    rng = np.random.default_rng(3606)
    n = 120
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "ycnt": rng.poisson(2.0, size=n),
        }
    )

    if name in ("zip", "zinb"):
        result = ws.ZeroInflatedPoisson.from_formula("ycnt ~ x", frame).fit()
    elif name == "eivreg":
        result = ws.Eivreg.from_formula(
            "y ~ x", frame, reliab={"x": 0.9}
        ).fit()
    elif name == "sureg":
        # ``sureg`` takes a SEQUENCE of equations, not one combined formula -- passing a
        # string raised "equations must be a sequence of formulas".  The multi-equation
        # fixtures are built directly instead of skipping, because a skip here would
        # leave the refusal untested (the lesson from round 189).
        multi = frame.assign(y2=rng.normal(size=n), x2=rng.normal(size=n))
        result = ws.Sureg.from_formula(["y ~ x", "y2 ~ x2"], multi).fit()
    elif name == "mvreg":
        # ``mvreg`` uses Stata's ``outcomes = regressors`` spelling, not ``~``:
        # "mvreg: the formula must separate the outcomes from the regressors with '='".
        multi = frame.assign(y2=rng.normal(size=n), x2=rng.normal(size=n))
        result = ws.Mvreg.from_formula("y y2 = x x2", multi).fit()
    else:  # pragma: no cover - the parametrisation is fixed
        pytest.fail(f"no builder for {name}")

    with pytest.raises(NotImplementedError):
        result.linktest()


def test_an_iv_fit_is_refused(well_specified) -> None:
    """``linktest.ado:63`` refuses anything starting with ``iv``."""

    rng = np.random.default_rng(3608)
    n = 200
    instrument = rng.normal(size=n)
    endogenous = 0.7 * instrument + rng.normal(size=n) * 0.3
    exog = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.8 * exog + 0.5 * endogenous + rng.normal(size=n),
            "x": exog,
            "e": endogenous,
            "z": instrument,
        }
    )
    iv = ws.IV2SLS.from_formula("y ~ x + [e ~ z]", frame).fit()

    with pytest.raises(NotImplementedError):
        iv.linktest()


# ---------------------------------------------------------------- the r() names
def test_to_dict_reports_t_and_df(well_specified) -> None:
    """``linktest.ado:131-132`` stores exactly ``r(t)`` and ``r(df)``.

    The p-value is this package's convenience and is named ``p`` rather than ``r(p)``
    precisely because Stata does not return it.
    """

    table = well_specified.linktest().to_dict()

    assert set(table) == {"t", "df", "p"}


def test_the_rendered_text_states_the_hypothesis(well_specified) -> None:
    rendered = str(well_specified.linktest())

    assert "link test" in rendered.lower()
    assert "_hatsq" in rendered
    assert "P>|t|" in rendered