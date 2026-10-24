"""Conformance tests for ``predict`` and ``margins``.

Every marginal effect is checked against the closed-form derivative for that
model, never against the implementation itself:

* logit:   dE[y]/dx_j = mean(p_i (1-p_i)) * b_j
* probit:  dE[y]/dx_j = mean(phi(xb_i)) * b_j
* poisson: dE[y]/dx_j = mean(mu_i) * b_j
* OLS:     the marginal effect *is* the coefficient, with the coefficient's
           own standard error
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.errors import UnsupportedError


def _binary(n: int = 500, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    return pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})


def _count(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.6 * x1)).astype(float)
    return pd.DataFrame({"y": y, "x1": x1})


def _linear(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# ------------------------------------------------------------------ predict


def test_ols_predict_xb_equals_fittedvalues():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    np.testing.assert_allclose(res.predict(kind="xb"), res.fittedvalues, rtol=1e-12)


def test_logit_predict_pr_is_a_probability():
    res = ws.Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    probabilities = res.predict(kind="pr")
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    # And it must equal the logistic transform of xb.
    np.testing.assert_allclose(
        probabilities, 1.0 / (1.0 + np.exp(-res.predict(kind="xb"))), rtol=1e-12
    )


def test_poisson_predict_n_is_positive_mean():
    res = ws.Poisson.from_formula("y ~ x1", _count()).fit()
    means = res.predict(kind="n")
    assert np.all(means > 0)
    np.testing.assert_allclose(
        means, np.exp(res.predict(kind="xb")), rtol=1e-12
    )


def test_unsupported_predict_kind_raises():
    res = ws.OLS.from_formula("y ~ x1", _linear()).fit()
    with pytest.raises(UnsupportedError, match="predict"):
        res.predict(kind="pr")


# ------------------------------------------------------- marginal effects


def test_logit_average_marginal_effect_matches_analytic_formula():
    df = _binary()
    res = ws.Logit.from_formula("y ~ x1 + x2", df).fit()

    n = len(df)
    X = np.column_stack([np.ones(n), df["x1"], df["x2"]])
    beta = res.params.to_numpy()
    probability = 1.0 / (1.0 + np.exp(-(X @ beta)))

    for position, name in [(1, "x1"), (2, "x2")]:
        analytic = float(np.mean(probability * (1 - probability) * beta[position]))
        outcome = res.margins(dydx=name)
        assert outcome.estimate == pytest.approx(analytic, rel=1e-8), name


def test_probit_average_marginal_effect_matches_analytic_formula():
    rng = np.random.default_rng(11)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.0 * x1 - 0.6 * x2 + rng.normal(size=n)
    df = pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1, "x2": x2})
    res = ws.Probit.from_formula("y ~ x1 + x2", df).fit()

    X = np.column_stack([np.ones(n), x1, x2])
    beta = res.params.to_numpy()
    density = stats.norm.pdf(X @ beta)

    for position, name in [(1, "x1"), (2, "x2")]:
        analytic = float(np.mean(density * beta[position]))
        outcome = res.margins(dydx=name)
        assert outcome.estimate == pytest.approx(analytic, rel=1e-8), name


def test_poisson_average_marginal_effect_matches_analytic_formula():
    df = _count()
    res = ws.Poisson.from_formula("y ~ x1", df).fit()

    n = len(df)
    X = np.column_stack([np.ones(n), df["x1"]])
    beta = res.params.to_numpy()
    mean = np.exp(X @ beta)
    analytic = float(np.mean(mean * beta[1]))

    assert res.margins(dydx="x1").estimate == pytest.approx(analytic, rel=1e-8)


def test_ols_marginal_effect_is_the_coefficient_with_its_own_se():
    res = ws.OLS.from_formula("y ~ x1 + x2", _linear()).fit()
    for name in ("x1", "x2"):
        outcome = res.margins(dydx=name)
        assert outcome.estimate == pytest.approx(res.params[name], rel=1e-8)
        # Delta-method SE of a linear model equals the coefficient SE.
        assert outcome.std_err == pytest.approx(res.bse[name], rel=1e-5)


def test_margins_at_fixes_covariates():
    res = ws.Logit.from_formula("y ~ x1 + x2", _binary()).fit()

    # At a fixed x2, dE[y]/dx1 = mean over the sample of p(1-p)*b1 with x2 held.
    df = _binary()
    X = np.column_stack([np.ones(len(df)), df["x1"], np.full(len(df), 1.0)])
    beta = res.params.to_numpy()
    probability = 1.0 / (1.0 + np.exp(-(X @ beta)))
    analytic = float(np.mean(probability * (1 - probability) * beta[1]))

    outcome = res.margins(dydx="x1", at={"x2": 1.0})
    assert outcome.estimate == pytest.approx(analytic, rel=1e-8)


def test_margins_over_reports_one_row_per_group():
    df = _binary()
    df["group"] = np.arange(len(df)) % 3
    res = ws.Logit.from_formula("y ~ x1 + x2", df).fit()

    rows = res.margins(dydx="x1", over="group")
    assert len(rows) == 3
    assert len({row.estimate for row in rows}) == 3


def test_predictive_margin_matches_mean_prediction():
    res = ws.Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    outcome = res.margins()
    assert outcome.estimate == pytest.approx(
        float(np.mean(res.predict(kind="pr"))), rel=1e-8
    )


def test_margins_unknown_variable_raises():
    res = ws.OLS.from_formula("y ~ x1", _linear()).fit()
    with pytest.raises(KeyError, match="not in"):
        res.margins(dydx="not_a_variable")


def test_margins_without_design_matrix_raises():
    res = ws.OLS.from_formula("y ~ x1", _linear()).fit()
    res.design_matrix = None
    with pytest.raises(UnsupportedError, match="design matrix"):
        res.margins(dydx="x1")


def test_margins_requires_standard_errors_from_model_vce():
    # A robust VCE must propagate into the marginal-effect standard error.
    df = _linear()
    plain = ws.OLS.from_formula("y ~ x1 + x2", df).fit()
    robust = ws.OLS.from_formula("y ~ x1 + x2", df).fit(cov_type="HC1")
    assert plain.margins(dydx="x1").std_err != pytest.approx(
        robust.margins(dydx="x1").std_err, rel=1e-3
    )


# ------------------------------------ Stata-captured regression (P0-7, M1)
#
# Everything below is pinned to numbers captured from a live StataNow 19.5 (MP)
# session on 2026-10-24, with the 1978 automobile data that ships as
# ``tests/data/auto.dta`` (``sysuse auto, clear``; N = 74):
#
#     . logit foreign weight mpg
#     . margins, dydx(weight)
#       r(table)[1,1] = -0.000456944823386641   (b)
#       r(table)[2,1] =  5.70605514650599e-05   (se)
#       r(Jacobian)   = [-1.108580469274756e-01, -6.649265095237061e-04,
#                        -4.796996154758345e-05]
#     . margins, dydx(weight) at(mpg=20)
#       b = -0.000351977277986498   se = 1.77996692282805e-06
#     . probit foreign weight mpg ; margins, dydx(weight)
#       b = -0.000464912943838621   se = 5.64992247182475e-05
#     . regress mpg weight i.foreign
#     . margins, over(foreign)          se = [0.472474113575479, 0.726387480366952]
#     . margins, dydx(foreign)          b/se (non-base row) = -1.650029105842638 /
#                                                            1.07599411819307
#
# The old implementation differentiated the covariate with a fixed h = 1e-6 and
# missed these SEs by 7.1e-3 (full sample) and 5.4e-2 (at(mpg=20)); the repair
# differentiates analytically and lands on Stata to <=3e-10.

AUTO = Path(__file__).parent / "data" / "auto.dta"


def _auto() -> pd.DataFrame:
    return pd.read_stata(AUTO, convert_categoricals=False)


def test_logit_ame_standard_error_matches_stata():
    """``margins, dydx(weight)`` after ``logit foreign weight mpg``."""

    frame = _auto()
    outcome = ws.Logit.from_formula("foreign ~ weight + mpg", frame).fit().margins(
        dydx="weight"
    )

    assert outcome.estimate == pytest.approx(-0.000456944823386641, rel=1e-6)
    assert outcome.std_err == pytest.approx(5.70605514650599e-05, rel=1e-6)


def test_logit_ame_at_fixed_covariate_matches_stata():
    """``margins, dydx(weight) at(mpg=20)``: the old SE was off by 5.4e-2."""

    frame = _auto()
    outcome = ws.Logit.from_formula("foreign ~ weight + mpg", frame).fit().margins(
        dydx="weight", at={"mpg": 20}
    )

    assert outcome.estimate == pytest.approx(-0.000351977277986498, rel=1e-6)
    assert outcome.std_err == pytest.approx(1.77996692282805e-06, rel=1e-6)


def test_probit_ame_standard_error_matches_stata():
    frame = _auto()
    outcome = ws.Probit.from_formula("foreign ~ weight + mpg", frame).fit().margins(
        dydx="weight"
    )

    assert outcome.estimate == pytest.approx(-0.000464912943838621, rel=1e-6)
    assert outcome.std_err == pytest.approx(5.64992247182475e-05, rel=1e-6)


def test_logit_ame_gradient_matches_stata_jacobian():
    """The delta-method gradient *is* Stata's ``r(Jacobian)`` (no differencing).

    Captured gradient: ``[-1.108580469274756e-01, -6.649265095237061e-04,
    -4.796996154758345e-05]`` in Stata's ``e(b)`` order ``[weight, mpg, _cons]``.
    """

    from westata.postestimation.margins import _statistic_and_gradient

    frame = _auto()
    result = ws.Logit.from_formula("foreign ~ weight + mpg", frame).fit()
    exog = np.asarray(result.design_matrix[0], dtype=float)
    names = list(result.design_matrix[1])

    _, gradient = _statistic_and_gradient(
        result, exog, "pr", derivative_with_respect_to=names.index("weight")
    )
    stata_order = [
        gradient[names.index("weight")],
        gradient[names.index("mpg")],
        gradient[names.index("Intercept")],
    ]
    expected = [-1.108580469274756e-01, -6.649265095237061e-04, -4.796996154758345e-05]
    np.testing.assert_allclose(stata_order, expected, rtol=1e-9)


def test_regress_predictive_margins_over_groups_match_stata():
    """``margins, over(foreign)``: an exact linear combination of ``e(V)``."""

    frame = _auto()
    rows = ws.OLS.from_formula("mpg ~ weight + C(foreign)", frame).fit().margins(
        over="foreign"
    )

    expected = [0.472474113575479, 0.726387480366952]
    assert [row.estimate for row in rows] == pytest.approx(
        [19.826923076925, 24.7727272727292], rel=1e-9
    )
    assert [row.std_err for row in rows] == pytest.approx(expected, rel=1e-9)


def test_regress_factor_discrete_change_matches_stata():
    """``margins, dydx(foreign)`` equals the coefficient, so its SE is ``bse``.

    Stata reports ``-1.650029105842638`` with ``se = 1.07599411819307``, the
    coefficient's own standard error; the old numeric gradient managed only
    1.8e-9 because it differenced a function that is exactly linear.
    """

    frame = _auto()
    result = ws.OLS.from_formula("mpg ~ weight + C(foreign)", frame).fit()
    row = result.margins(dydx="C(foreign)[T.1]", factor_vars=["C(foreign)[T.1]"])
    row = row[0] if isinstance(row, list) else row

    assert row.estimate == pytest.approx(-1.650029105842638, rel=1e-12)
    assert row.std_err == pytest.approx(1.07599411819307, rel=1e-12)


def test_poisson_ame_standard_error_matches_stata():
    """``poisson rep78 mpg weight`` + ``margins, dydx(weight)`` (N = 69)."""

    frame = _auto().dropna(subset=["rep78"])
    outcome = ws.Poisson.from_formula("rep78 ~ mpg + weight", frame).fit().margins(
        dydx="weight"
    )

    assert outcome.estimate == pytest.approx(-0.000298305029001251, rel=1e-6)
    assert outcome.std_err == pytest.approx(0.000471476771714765, rel=1e-6)


def test_tobit_margins_default_is_the_linear_prediction_not_e_ab():
    """M7 (new finding): tobit's default ``margins`` differentiates ``xb``.

    Captured on StataNow 19.5 (2026-10-24), ``tobit mpg weight foreign, ll(17)``:

        . margins, dydx(weight)                      b = -0.00779786618189768
        . margins, dydx(weight) predict(xb)          b = -0.00779786618189768
        . margins, dydx(weight) predict(e(17,.))     b = -0.00477824532202177
        . margins, dydx(weight) predict(ystar(17,.)) b = -0.00548991091113322
        . margins                                    b = 20.6918844403444
        . margins, predict(e(17,.))                  b = 22.7510610608599

    and ``e(marginsok)`` reads ``default XB Pr(passthru) E(passthru)
    YStar(passthru)``, i.e. ``xb`` is flagged ``default``.  The old
    ``DEFAULT_PREDICT["tobit"] = "e"`` therefore made ``margins`` succeed while
    reporting ``e(17,.)`` -- a 39%-different quantity -- which is the silent
    wrong-answer failure mode this project forbids.
    """

    frame = _auto()
    result = ws.Tobit.from_formula("mpg ~ weight + foreign", frame, lower=17).fit()

    default = result.margins(dydx="weight")
    # The default equals the coefficient itself (xb is linear), to Stata's digits.
    assert default.estimate == pytest.approx(-0.00779786618189768, rel=1e-9)
    assert default.std_err == pytest.approx(0.000870168871185609, rel=1e-9)
    assert default.prediction == "xb"

    # The censored expectation is still reachable, and is a *different* number.
    conditional = result.margins(dydx="weight", predict_kind="e")
    assert conditional.estimate == pytest.approx(-0.00477824532202177, rel=1e-6)
    assert conditional.estimate != pytest.approx(default.estimate, rel=1e-3)

    # The predictive margin likewise follows xb: mean(xb) = 20.6918844403444.
    margin = result.margins()
    assert margin.estimate == pytest.approx(20.6918844403444, rel=1e-9)
    assert margin.std_err == pytest.approx(0.492180562049591, rel=1e-9)
    assert result.margins(predict_kind="e").estimate == pytest.approx(
        22.7510610608599, rel=1e-9
    )


def test_ordered_logit_ame_matches_stata():
    """``ologit rep78 weight mpg`` + ``margins, dydx(weight)`` (N = 69).

    Captured: ``b = 9.40690482679723e-06, se = 1.52602157141091e-05`` for the
    default ``predict(pr)`` (category 1), and
    ``b = 2.82484885639941e-05, se = 4.15550150811616e-05`` for
    ``predict(pr outcome(2))``.  The AME is ~1e-5, so a numeric gradient had no
    significant digits left at all.
    """

    frame = _auto().dropna(subset=["rep78"])
    result = ws.OrderedLogit.from_formula("rep78 ~ weight + mpg", frame).fit()

    default = result.margins(dydx="weight")
    assert default.estimate == pytest.approx(9.40690482679723e-06, rel=1e-6)
    assert default.std_err == pytest.approx(1.52602157141091e-05, rel=1e-6)

    second = result.margins(dydx="weight", outcome=2)
    assert second.estimate == pytest.approx(2.82484885639941e-05, rel=1e-6)
    assert second.std_err == pytest.approx(4.15550150811616e-05, rel=1e-6)


def test_ordered_logit_pr_matches_stata_category_probability():
    """``ologit`` ``predict, pr`` is P(y = 1) by default (Stata: 0.0e+00)."""

    frame = _auto().dropna(subset=["rep78"])
    result = ws.OrderedLogit.from_formula("rep78 ~ weight + mpg", frame).fit()
    probability = np.asarray(result.predict(kind="pr"), dtype=float)
    # P(y=1) = Lambda(cut1 - eta), recomputed from the fitted parameters.
    names = list(result.param_names)
    params = np.asarray(result.params, dtype=float)
    design = np.asarray(result.design_matrix[0], dtype=float)
    eta = design @ params[[names.index(name) for name in design_names(result)]]
    cut1 = params[names.index("/cut1")]
    expected = 1.0 / (1.0 + np.exp(-(cut1 - eta)))
    np.testing.assert_allclose(probability, expected, rtol=1e-12)


def design_names(result) -> list[str]:
    """Slope names of an ordered result, in design-column order."""

    return [
        name for name in result.param_names if not str(name).startswith("/")
    ]
