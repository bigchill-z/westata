"""Tests for ``nlcom`` / ``testnl`` (Stata ``[R] nlcom``, ``[R] testnl``).

Every expected value is derived independently from the delta-method formula

    Var(g(b)) = G V G'

with ``G = dg/db`` written out by hand; the implementation's own gradient is
never used to build an expected value.  The expression language is also attacked
with unsafe inputs, because the parser is the only place where user text could
reach code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.postestimation.lincom import LincomResult
from westata.postestimation.nonlinear import nlcom, testnl
from westata.postestimation.tests import WaldTest


def _frame(n: int = 200, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + rng.normal(scale=0.6, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


@pytest.fixture(scope="module")
def ols():
    return ws.OLS.from_formula("y ~ x1 + x2 + x3", _frame()).fit()


@pytest.fixture(scope="module")
def logit():
    rng = np.random.default_rng(3)
    n = 400
    x1 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 + rng.logistic(size=n)
    frame = pd.DataFrame({"y": (latent > 0).astype(float), "x1": x1})
    return ws.Logit.from_formula("y ~ x1", frame).fit()


# --------------------------------------------------- identity with lincom


def test_single_coefficient_matches_lincom(ols):
    nonlinear = nlcom(ols, "_b[x1]")
    linear = ols.lincom("x1")

    assert isinstance(nonlinear, LincomResult)
    assert nonlinear.estimate == pytest.approx(linear.estimate, rel=1e-12)
    assert nonlinear.std_err == pytest.approx(linear.std_err, rel=1e-9)
    assert nonlinear.statistic == pytest.approx(linear.statistic, rel=1e-8)
    # The estimate, standard error and statistic agree exactly ...
    # ... but the p-value and interval do NOT: nlcom uses the normal reference
    # while lincom follows the model's t. Stata behaves the same way.
    assert nonlinear.distribution == "z"
    assert linear.distribution == "t"


def test_single_coefficient_matches_lincom_on_a_z_model(logit):
    nonlinear = nlcom(logit, "_b[x1]")
    linear = logit.lincom("x1")

    assert nonlinear.estimate == pytest.approx(linear.estimate, rel=1e-12)
    assert nonlinear.std_err == pytest.approx(linear.std_err, rel=1e-9)
    assert nonlinear.statistic == pytest.approx(linear.statistic, rel=1e-8)
    assert nonlinear.pvalue == pytest.approx(linear.pvalue, rel=1e-8)
    assert nonlinear.distribution == "z"


def test_linear_combination_matches_lincom(ols):
    nonlinear = nlcom(ols, "_b[x1] + _b[x2]")
    linear = ols.lincom("x1 + x2")

    assert nonlinear.estimate == pytest.approx(linear.estimate, rel=1e-12)
    assert nonlinear.std_err == pytest.approx(linear.std_err, rel=1e-8)


def test_scaled_combination_matches_lincom(ols):
    nonlinear = nlcom(ols, "2*_b[x1] - 3*_b[x2]")
    linear = ols.lincom("2*x1 - 3*x2")
    assert nonlinear.estimate == pytest.approx(linear.estimate, rel=1e-12)
    assert nonlinear.std_err == pytest.approx(linear.std_err, rel=1e-8)


def test_intercept_is_addressable(ols):
    outcome = nlcom(ols, "_b[Intercept]")
    assert outcome.estimate == pytest.approx(float(ols.params["Intercept"]), rel=1e-12)


# -------------------------------------------------- analytic delta method


def test_exp_matches_analytic_delta_method(ols):
    b = float(ols.params["x1"])
    se = float(ols.bse["x1"])

    outcome = nlcom(ols, "exp(_b[x1])")
    assert outcome.estimate == pytest.approx(np.exp(b), rel=1e-12)
    # d/db exp(b) = exp(b), so the delta-method SE is exp(b) * se.
    assert outcome.std_err == pytest.approx(np.exp(b) * se, rel=1e-7)


def test_ratio_matches_hand_computed_delta_method(ols):
    names = list(ols.param_names)
    cov = np.asarray(ols.cov_params(), dtype=float)
    b1 = float(ols.params["x1"])
    b2 = float(ols.params["x2"])
    i, j = names.index("x1"), names.index("x2")

    # g = b1 / b2  ->  dg/db1 = 1/b2, dg/db2 = -b1/b2^2
    gradient = np.zeros(len(names))
    gradient[i] = 1.0 / b2
    gradient[j] = -b1 / b2**2
    expected_se = float(np.sqrt(gradient @ cov @ gradient))

    outcome = nlcom(ols, "_b[x1] / _b[x2]")
    assert outcome.estimate == pytest.approx(b1 / b2, rel=1e-12)
    assert outcome.std_err == pytest.approx(expected_se, rel=1e-7)


def test_sqrt_and_log_are_supported(ols):
    logged = nlcom(ols, "log(exp(_b[x1]))")
    assert logged.estimate == pytest.approx(float(ols.params["x1"]), rel=1e-9)
    assert logged.std_err == pytest.approx(float(ols.bse["x1"]), rel=1e-7)


def test_sqrt_is_supported(ols):
    squared = nlcom(ols, "sqrt(_b[x1]**2)")
    assert squared.estimate == pytest.approx(abs(float(ols.params["x1"])), rel=1e-12)


def test_ci_uses_the_normal_reference_evEN_after_regress(ols):
    """Stata's nlcom reports a normal interval even after ``regress``.

    The committed fixture is decisive: ``nlcom exp(_b[x1])`` prints
    [6.652973, 7.943221], which is ``est -/+ 1.959964*se``. A t(196) interval
    would be [6.648965, 7.947229]. So nlcom does NOT follow the model's
    distribution the way lincom does.
    """

    outcome = nlcom(ols, "exp(_b[x1])")
    critical = stats.norm.ppf(0.975)
    assert outcome.ci_lower == pytest.approx(
        outcome.estimate - critical * outcome.std_err, rel=1e-10
    )
    assert outcome.ci_upper == pytest.approx(
        outcome.estimate + critical * outcome.std_err, rel=1e-10
    )
    assert outcome.distribution == "z"
    assert outcome.df is None


def test_level_widens_the_interval(ols):
    narrow = nlcom(ols, "exp(_b[x1])", level=0.90)
    wide = nlcom(ols, "exp(_b[x1])", level=0.99)
    assert (wide.ci_upper - wide.ci_lower) > (narrow.ci_upper - narrow.ci_lower)


def test_eform_exponentiates_estimate_and_interval(ols):
    plain = nlcom(ols, "_b[x1]")
    transformed = nlcom(ols, "_b[x1]", eform=True)

    assert transformed.estimate == pytest.approx(np.exp(plain.estimate), rel=1e-12)
    assert transformed.ci_lower == pytest.approx(np.exp(plain.ci_lower), rel=1e-12)
    assert transformed.ci_upper == pytest.approx(np.exp(plain.ci_upper), rel=1e-12)
    # Stata keeps the standard error and statistic on the untransformed scale.
    assert transformed.std_err == pytest.approx(plain.std_err, rel=1e-12)
    assert transformed.statistic == pytest.approx(plain.statistic, rel=1e-12)
    assert transformed.eform is True


def test_expression_is_attached_to_the_result(ols):
    outcome = nlcom(ols, "exp(_b[x1])")
    assert outcome.expression == "exp(_b[x1])"
    assert "Coef." in str(outcome)


# ------------------------------------------------------------- testnl


def test_testnl_single_coefficient_equals_zero(ols):
    outcome = testnl(ols, "_b[x1] = 0")

    b = float(ols.params["x1"])
    se = float(ols.bse["x1"])
    assert isinstance(outcome, WaldTest)
    assert outcome.statistic == pytest.approx((b / se) ** 2, rel=1e-8)
    assert outcome.df == 1
    assert outcome.distribution == "chi2"
    assert outcome.df_denom is None
    assert outcome.pvalue == pytest.approx(stats.chi2.sf((b / se) ** 2, 1), rel=1e-8)


def test_testnl_nonlinear_matches_analytic(ols):
    b = float(ols.params["x1"])
    se = float(ols.bse["x1"])

    outcome = testnl(ols, "exp(_b[x1]) = 2")
    expected = (np.exp(b) - 2.0) ** 2 / (np.exp(b) ** 2 * se**2)
    assert outcome.statistic == pytest.approx(expected, rel=1e-7)


def test_testnl_equality_of_two_coefficients_matches_linear_test(ols):
    outcome = testnl(ols, "_b[x1] = _b[x2]")
    linear = ols.test("x1 = x2")
    # The linear test reports F = W/q with q = 1, so statistic * df is the chi2 W.
    assert outcome.statistic == pytest.approx(linear.statistic * linear.df, rel=1e-7)
    assert outcome.df == 1


def test_testnl_joint_comma_separated(ols):
    outcome = testnl(ols, "_b[x1] = 0, _b[x2] = 0")
    assert outcome.df == 2
    assert outcome.distribution == "chi2"

    names = list(ols.param_names)
    cov = np.asarray(ols.cov_params(), dtype=float)
    b = np.asarray(ols.params, dtype=float)
    i, j = names.index("x1"), names.index("x2")
    R = np.zeros((2, len(names)))
    R[0, i] = 1.0
    R[1, j] = 1.0
    middle = R @ cov @ R.T
    expected = float(b @ R.T @ np.linalg.solve(middle, R @ b))
    assert outcome.statistic == pytest.approx(expected, rel=1e-7)
    assert outcome.pvalue == pytest.approx(stats.chi2.sf(expected, 2), rel=1e-7)


def test_testnl_joint_space_separated(ols):
    spaced = testnl(ols, "_b[x1] = 0 _b[x2] = 0")
    comma = testnl(ols, "_b[x1] = 0, _b[x2] = 0")
    assert spaced.df == 2
    assert spaced.statistic == pytest.approx(comma.statistic, rel=1e-10)


def test_testnl_parenthesised_constraints(ols):
    grouped = testnl(ols, "(_b[x1] = 0) (_b[x2] = 0)")
    comma = testnl(ols, "_b[x1] = 0, _b[x2] = 0")
    assert grouped.df == 2
    assert grouped.statistic == pytest.approx(comma.statistic, rel=1e-10)


def test_testnl_without_equals_tests_zero(ols):
    bare = testnl(ols, "_b[x1]")
    explicit = testnl(ols, "_b[x1] = 0")
    assert bare.df == 1
    assert bare.statistic == pytest.approx(explicit.statistic, rel=1e-10)


def test_testnl_works_on_a_z_model(logit):
    outcome = testnl(logit, "exp(_b[x1]) = 1")
    b = float(logit.params["x1"])
    se = float(logit.bse["x1"])
    expected = (np.exp(b) - 1.0) ** 2 / (np.exp(b) ** 2 * se**2)
    assert outcome.statistic == pytest.approx(expected, rel=1e-7)
    assert outcome.df == 1


def test_testnl_constraint_is_rendered(ols):
    """Stata numbers each constraint, e.g. "( 1) _b[x1] = 0"."""

    outcome = testnl(ols, "_b[x1] = 0, _b[x2] = 0")
    assert "_b[x1]" in outcome.constraint
    assert "_b[x2]" in outcome.constraint
    # Stata right-aligns the index in a two-character field.
    assert "( 1)" in outcome.constraint
    assert "( 2)" in outcome.constraint
    assert "chi2(" in str(outcome)


# --------------------------------------------------------------- safety


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os')",
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "_b[x1].__class__",
        "lambda: 1",
        "[1, 2, 3]",
        "{'a': 1}",
        "(1, 2)",
        "_b[x1] if _b[x2] else 0",
        "_b[x1] > 0",
        "_b[x1] % 2",
        "x1",
        "max(_b[x1], _b[x2])",
        "ln(_b[x1])",
        "exp(_b[x1], 2)",
        "exp(value=_b[x1])",
        "_b[x1] + 'text'",
    ],
)
def test_unsafe_expressions_are_rejected(ols, expression):
    with pytest.raises(ValueError):
        nlcom(ols, expression)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os') = 0",
        "open('/etc/passwd') = 0",
        "_b[x1].__class__ = 0",
        "[1, 2, 3] = 0",
    ],
)
def test_testnl_unsafe_expressions_are_rejected(ols, expression):
    with pytest.raises(ValueError):
        testnl(ols, expression)


def test_unknown_coefficient_raises_keyerror(ols):
    with pytest.raises(KeyError, match="not found"):
        nlcom(ols, "_b[nonexistent]")
    with pytest.raises(KeyError, match="not found"):
        testnl(ols, "_b[nonexistent] = 0")


def test_equals_sign_is_rejected_by_nlcom(ols):
    with pytest.raises(ValueError, match="testnl"):
        nlcom(ols, "_b[x1] = 1")


def test_division_by_zero_is_reported(ols):
    with pytest.raises(ZeroDivisionError):
        nlcom(ols, "_b[x1] / 0")


def test_empty_expression_is_rejected(ols):
    with pytest.raises(ValueError):
        nlcom(ols, "   ")
    with pytest.raises(ValueError):
        testnl(ols, "")


def test_non_string_expression_is_rejected(ols):
    with pytest.raises(TypeError):
        nlcom(ols, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        testnl(ols, 1)  # type: ignore[arg-type]


def test_invalid_level_is_rejected(ols):
    with pytest.raises(ValueError):
        nlcom(ols, "_b[x1]", level=1.5)


# ------------------------------------------------- factor / odd names


def test_quoted_factor_coefficient_names_are_addressable():
    rng = np.random.default_rng(11)
    n = 150
    g = rng.choice(["a", "b", "c"], size=n)
    y = 1.0 + np.where(g == "b", 0.8, 0.0) + np.where(g == "c", -0.4, 0.0)
    y = y + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"y": y, "g": pd.Categorical(g)})
    result = ws.OLS.from_formula("y ~ C(g)", frame).fit()

    name = next(value for value in result.param_names if value.startswith("C(g)"))
    b = float(result.params[name])
    se = float(result.bse[name])

    # A coefficient name such as ``C(g)[T.b]`` is not valid Python inside
    # ``_b[...]``; Stata's bracket spelling cannot be parsed, so the quoted
    # spelling is the supported form.
    quoted = nlcom(result, f'exp(_b["{name}"])')
    assert quoted.estimate == pytest.approx(np.exp(b), rel=1e-12)
    assert quoted.std_err == pytest.approx(np.exp(b) * se, rel=1e-7)

    with pytest.raises(ValueError):
        nlcom(result, f"exp(_b[{name}])")


def test_power_and_unary_minus(ols):
    outcome = nlcom(ols, "-(_b[x1]**2)")
    b = float(ols.params["x1"])
    cov = np.asarray(ols.cov_params(), dtype=float)
    names = list(ols.param_names)
    gradient = np.zeros(len(names))
    gradient[names.index("x1")] = -2.0 * b
    expected_se = float(np.sqrt(gradient @ cov @ gradient))
    assert outcome.estimate == pytest.approx(-(b**2), rel=1e-12)
    assert outcome.std_err == pytest.approx(expected_se, rel=1e-7)
