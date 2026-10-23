"""Tests for the nonlinear least-squares (``Nl``) and stochastic frontier
(``Frontier``) estimators.

Reference values are the captured StataNow 19.5 output in
``tests/conformance/stata_fixtures/p10_outputs.txt``:

=========================  ==================================================
fixture section            Stata values
=========================  ==================================================
``nl_0`` (lines 85-109)
                           ``/b0`` 4.943526 (se .0381662),
                           ``/b1`` .7912568 (se .0080793);
                           Residual SS 6.9274269, Model SS 1010.4869,
                           Total 1017.4143, R-squared .9932,
                           Adj R-squared .9931, Root MSE .1524677,
                           Res. dev. -279.1251
``frontier_0`` (lines 142-170)
                           x1 .7813064, x2 .517943, _cons 1.047813;
                           /lnsig2v -3.395558, /lnsig2u -1.705448;
                           sigma_v .1830897, sigma_u .4262523,
                           sigma2 .2152128, lambda 2.328106;
                           Log likelihood -96.054179, Wald chi2(2) 4462.60,
                           LR test of sigma_u=0: chibar2(01) = 23.60
=========================  ==================================================

The samples reproduce Stata's data-generating process exactly (same seeds and
draw order as ``tests/conformance/test_stata_p10.py``), so these are the
fixture's own numbers rather than "close enough to some DGP".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.errors import ConsistencyError, ConvergenceError, UnsupportedError
from westata.estimators.nonlinear import (
    Frontier,
    Nl,
    compile_expression,
    frontier_loglike,
)

# --------------------------------------------------------------- data frames


def nl_frame(n: int = 300, seed: int = 103) -> pd.DataFrame:
    """The ``nl_0`` fixture sample."""

    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.1, 4.0, size=n))
    return pd.DataFrame(
        {"y": 5.0 * np.exp(-0.8 * x) + rng.normal(scale=0.15, size=n), "x": x}
    )


def frontier_frame(n: int = 400, seed: int = 113) -> pd.DataFrame:
    """The ``frontier_0`` fixture sample (half-normal inefficiency)."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    inefficiency = np.abs(rng.normal(scale=0.4, size=n))
    return pd.DataFrame(
        {
            "y": 1.0 + 0.8 * x1 + 0.5 * x2 - inefficiency + rng.normal(scale=0.2, size=n),
            "x1": x1,
            "x2": x2,
        }
    )


# Stata reference values, copied from p10_outputs.txt.
STATA_NL_B0 = 4.943526
STATA_NL_B1 = 0.7912568
STATA_NL_SE_B0 = 0.0381662
STATA_NL_SE_B1 = 0.0080793
STATA_NL_RSS = 6.9274269
STATA_NL_MSS = 1010.4869
STATA_NL_TOTAL = 1017.4143
STATA_NL_R2 = 0.9932
STATA_NL_R2_A = 0.9931
STATA_NL_RMSE = 0.1524677
STATA_NL_DEV = -279.1251

STATA_FR_X1 = 0.7813064
STATA_FR_X2 = 0.517943
STATA_FR_CONS = 1.047813
STATA_FR_LNSIG2V = -3.395558
STATA_FR_LNSIG2U = -1.705448
STATA_FR_SE_X1 = 0.0138266
STATA_FR_SE_X2 = 0.0150432
STATA_FR_SE_CONS = 0.0272013
STATA_FR_SE_LNSIG2V = 0.2098701
STATA_FR_SE_LNSIG2U = 0.154228
STATA_FR_SIGMA_V = 0.1830897
STATA_FR_SIGMA_U = 0.4262523
STATA_FR_SIGMA2 = 0.2152128
STATA_FR_LAMBDA = 2.328106
STATA_FR_LL = -96.054179
STATA_FR_WALD = 4462.60
STATA_FR_LR = 23.60


# --------------------------------------------------------------------- nl


def test_nl_recovers_stata_estimates():
    """``nl (y = {b0} * exp(-{b1} * x))``: Stata /b0 4.943526, /b1 .7912568."""

    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame()).fit()

    assert float(result.params["/b0"]) == pytest.approx(STATA_NL_B0, rel=1e-3)
    assert float(result.params["/b1"]) == pytest.approx(STATA_NL_B1, rel=1e-3)


def test_nl_standard_errors_match_stata():
    """Stata se .0381662 and .0080793 from s^2 (J'J)^-1, s^2 = RSS/(n-k)."""

    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame()).fit()

    assert float(result.bse["/b0"]) == pytest.approx(STATA_NL_SE_B0, rel=1e-3)
    assert float(result.bse["/b1"]) == pytest.approx(STATA_NL_SE_B1, rel=1e-3)


def test_nl_parameter_names_carry_the_leading_slash():
    """Stata prints the parameters of ``nl`` as ``/b0``, ``/b1``."""

    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame()).fit()

    assert result.param_names == ["/b0", "/b1"]
    assert "/b0" in result.params.index
    # The slash is a display prefix, not a doubled or decorated one.
    assert all(name.count("/") == 1 and name.index("/") == 0 for name in result.param_names)


def test_nl_reports_stata_summary_of_fit_scalars():
    """Residual SS 6.9274269, Model SS 1010.4869, R2 .9932, RMSE .1524677."""

    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame()).fit()

    assert result.e["rss"] == pytest.approx(STATA_NL_RSS, rel=1e-6)
    assert result.e["mss"] == pytest.approx(STATA_NL_MSS, rel=1e-5)
    assert result.e["mss"] + result.e["rss"] == pytest.approx(STATA_NL_TOTAL, rel=1e-7)
    assert result.e["r2"] == pytest.approx(STATA_NL_R2, rel=1e-4)
    assert result.e["r2_a"] == pytest.approx(STATA_NL_R2_A, rel=1e-4)
    assert result.e["rmse"] == pytest.approx(STATA_NL_RMSE, rel=1e-6)
    # "Res. dev." is -2 lnL at the ML scale sigma^2 = RSS/n, not RSS/(n-k).
    assert result.e["deviance"] == pytest.approx(STATA_NL_DEV, rel=1e-6)
    assert result.e["N"] == 300
    assert result.e["df_m"] == 2
    assert result.e["df_r"] == 298


def test_nl_total_sum_of_squares_is_uncentred_without_a_constant_term():
    """The fixture's Total is ``sum(y^2)``: Model + Residual = 1017.4143.

    ``{b0}*exp(...)`` is not an additive constant, so Stata does *not* centre the
    total; centring would give 809.8 and R-squared .9914 instead of the printed
    .9932.  This is the difference between reproducing the fixture and merely
    fitting the same model.
    """

    frame = nl_frame()
    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", frame).fit()

    y = frame["y"].to_numpy()
    assert result.e["tss"] == pytest.approx(float(np.sum(y**2)), rel=1e-12)
    assert result.e["tss"] == pytest.approx(STATA_NL_TOTAL, rel=1e-7)
    assert result.e["mss"] + result.e["rss"] == pytest.approx(result.e["tss"], rel=1e-12)
    assert result.e["has_constant"] == 0
    # The centred total is visibly different, which is why this test exists.
    assert abs(result.e["tss"] - float(np.sum((y - y.mean()) ** 2))) > 200.0


def test_nl_rejects_a_dangerous_expression():
    with pytest.raises(ValueError):
        Nl.from_formula("y = {b0} * __import__('os').system('echo x')", nl_frame())


def test_nl_never_executes_the_expression(tmp_path):
    """Validation must precede evaluation: the file must survive."""

    victim = tmp_path / "must_survive.txt"
    victim.write_text("still here", encoding="utf-8")
    expression = f"y = {{b0}} * __import__('os').remove({str(victim)!r})"

    with pytest.raises(ValueError):
        Nl.from_formula(expression, nl_frame())

    assert victim.exists()


def test_nl_rejects_an_unknown_function():
    with pytest.raises(ValueError, match="frobnicate"):
        Nl.from_formula("y = {b0} * frobnicate(x)", nl_frame())


@pytest.mark.parametrize(
    "expression",
    [
        "y = {b0} * (x).real",  # attribute access
        "y = {b0} * x.__class__",  # dunder attribute
        "y = {b0} * (lambda t: t)(x)",  # lambda
        "y = {b0} * [i for i in x][0]",  # list comprehension
        "y = {b0} * {i: i for i in x}",  # dict comprehension
        "y = {b0} * (x if x > 0 else 0)",  # conditional expression
        "y = {b0} * x @ x",  # matrix multiply
        "y = {b0} * x[0]",  # subscript
        "y = {b0} * exp(x, 2)",  # wrong arity
        "y = {b0} * exp(value=x)",  # keyword argument
        "y = {b0} * open('f')",  # unknown name called
    ],
)
def test_nl_rejects_constructs_outside_the_whitelist(expression):
    with pytest.raises(ValueError):
        Nl.from_formula(expression, nl_frame())


def test_nl_rejects_an_unknown_data_name():
    with pytest.raises(ValueError, match="nope"):
        Nl.from_formula("y = {b0} + {b1} * nope", nl_frame())


def test_nl_accepts_stata_function_names():
    """``ln`` is Stata's log and is accepted alongside ``log``."""

    frame = pd.DataFrame({"y": [-1.0, -2.0], "x": [1.0, 2.0]})
    model = Nl.from_formula("y = {b0} * ln(x)", frame)
    assert model._parameter_names == ["b0"]


def test_nl_rejects_a_formula_without_an_equals_sign():
    with pytest.raises(ValueError, match="="):
        Nl.from_formula("y ~ {b0} * exp(-{b1} * x)", nl_frame())


def test_nl_rejects_a_formula_without_parameters():
    with pytest.raises(ConsistencyError, match="no parameters"):
        Nl.from_formula("y = x", nl_frame())


def test_nl_rejects_a_non_dataframe():
    with pytest.raises(TypeError):
        Nl.from_formula("y = {b0} * x", {"y": [1.0], "x": [1.0]})


def test_nl_rejects_an_unknown_dependent_variable():
    with pytest.raises(ConsistencyError, match="not found"):
        Nl.from_formula("q = {b0} * x", nl_frame())


def test_nl_rejects_an_unparsable_expression():
    with pytest.raises(ValueError):
        Nl.from_formula("y = {b0} * (x + )", nl_frame())


def test_nl_on_a_linear_model_reproduces_ols():
    rng = np.random.default_rng(11)
    n = 200
    x = rng.normal(size=n)
    frame = pd.DataFrame({"y": 1.5 + 2.5 * x + rng.normal(size=n), "x": x})

    nonlinear = Nl.from_formula("y = {b0} + {b1} * x", frame).fit()
    plain = ws.OLS.from_formula("y ~ x", frame).fit()

    assert float(nonlinear.params["/b0"]) == pytest.approx(
        float(plain.params["Intercept"]), abs=1e-6
    )
    assert float(nonlinear.params["/b1"]) == pytest.approx(
        float(plain.params["x"]), abs=1e-6
    )
    # An additive parameter *is* a constant term, so Stata centres the total and
    # the whole fit block has to agree with regress.  ``ws.OLS`` does not record
    # e(rss), so rss is rebuilt from its residual vector.
    assert nonlinear.e["has_constant"] == 1
    ols_rss = float(np.sum(np.asarray(plain.resid) ** 2))
    y = frame["y"].to_numpy()
    ols_tss = float(np.sum((y - y.mean()) ** 2))
    assert nonlinear.e["rss"] == pytest.approx(ols_rss, rel=1e-10)
    assert nonlinear.e["tss"] == pytest.approx(ols_tss, rel=1e-10)
    assert nonlinear.e["mss"] == pytest.approx(ols_tss - ols_rss, rel=1e-10)
    assert nonlinear.e["r2"] == pytest.approx(float(plain.e["r2"]), rel=1e-10)
    assert nonlinear.e["rmse"] == pytest.approx(float(plain.e["rmse"]), rel=1e-10)
    # df_m drops by one exactly because a constant is present.
    assert nonlinear.e["df_m"] == 1


def test_nl_reports_a_t_reference_distribution():
    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame()).fit()

    assert result.dist == "t"
    assert result.model_name == "nl"
    assert result.nobs == 300
    assert result.e["df_r"] == result.df_resid
    assert np.asarray(result.resid).shape == (300,)


def test_nl_accepts_explicit_starting_values():
    """``start=`` (Stata's initial values) must not change the optimum."""

    default = Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame()).fit()
    supplied = Nl.from_formula(
        "y = {b0} * exp(-{b1} * x)", nl_frame(), start={"b0": 1.0, "b1": 0.2}
    ).fit()

    assert float(supplied.params["/b0"]) == pytest.approx(
        float(default.params["/b0"]), rel=1e-6
    )
    assert float(supplied.params["/b1"]) == pytest.approx(
        float(default.params["/b1"]), rel=1e-6
    )


def test_nl_accepts_stata_inline_initial_values():
    """Stata writes initial values inside the braces: ``{b0=1}``."""

    result = Nl.from_formula("y = {b0=1.0} * exp(-{b1=0.2} * x)", nl_frame()).fit()

    assert float(result.params["/b0"]) == pytest.approx(STATA_NL_B0, rel=1e-4)
    assert float(result.params["/b1"]) == pytest.approx(STATA_NL_B1, rel=1e-4)
    assert result.param_names == ["/b0", "/b1"]


def test_nl_rejects_a_start_value_for_an_unknown_parameter():
    with pytest.raises(ValueError, match="unknown parameters"):
        Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame(), start={"b2": 1.0})


def test_nl_strict_raises_when_the_iteration_limit_is_hit():
    model = Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame())
    with pytest.raises(ConvergenceError):
        model.fit(maxiter=1, strict=True)


def test_nl_rejects_an_unsupported_cov_type():
    with pytest.raises(ValueError, match="cov_type"):
        Nl.from_formula("y = {b0} * x", nl_frame()).fit(cov_type="robust")


def test_nl_reports_stata_style_summary():
    text = (
        Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame())
        .fit()
        .summary(style="stata")
    )

    assert "Number of obs" in text
    assert "R-squared" in text
    assert "Root MSE" in text
    assert "P>|t|" in text
    assert "/b0" in text
    assert "/b1" in text
    # Stata's printed values must be visible in the rendering.
    assert "4.943526" in text
    assert ".7912568" in text


def test_nl_exposes_the_standard_e_keys():
    result = Nl.from_formula("y = {b0} * exp(-{b1} * x)", nl_frame()).fit()

    for key in [
        "N",
        "df_m",
        "df_r",
        "rss",
        "mss",
        "tss",
        "r2",
        "r2_a",
        "rmse",
        "ll",
        "deviance",
        "sigma",
        "converged",
        "cmd",
    ]:
        assert key in result.e, key
    assert result.e["converged"] == 1
    assert result.e["sigma"] == pytest.approx(result.e["rmse"], rel=1e-12)
    assert result.e["ll"] == pytest.approx(-0.5 * result.e["deviance"], rel=1e-12)


def test_compile_expression_rejects_a_bare_unknown_name():
    """Every unknown symbol is rejected at validation, as a ValueError.

    The check is static, so it fires when the expression is built rather than
    when it is evaluated: an unknown name cannot reach the evaluation path and
    turn into a ``KeyError`` later.  That is the property this pins down.
    """

    with pytest.raises(ValueError, match="neither a parameter nor a column"):
        compile_expression("a + b", ["a"], ["c"])


def test_compile_expression_rejects_an_unknown_name_at_evaluation():
    with pytest.raises(ValueError):
        compile_expression("a + b", ["a"], ["c"])(
            {"a": 1.0}, pd.DataFrame({"x": [1.0, 2.0]})
        )


def test_compile_expression_evaluates_a_permitted_expression():
    function = compile_expression("a * exp(-b * x)", ["a", "b"], ["x"])
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    values = function({"a": 2.0, "b": 0.5}, frame)

    np.testing.assert_allclose(values, 2.0 * np.exp(-0.5 * np.array([1.0, 2.0])))
    # A parameter-only expression broadcasts to the sample size.
    constant = compile_expression("a + 1", ["a"], ["x"])({"a": 3.0}, frame)
    np.testing.assert_allclose(constant, [4.0, 4.0])


# ----------------------------------------------------------------- frontier


def test_frontier_coefficients_match_stata():
    """Stata: x1 .7813064, x2 .517943, _cons 1.047813 (n=400, seed 113)."""

    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    assert float(result.params["x1"]) == pytest.approx(STATA_FR_X1, rel=2e-2)
    assert float(result.params["x2"]) == pytest.approx(STATA_FR_X2, rel=2e-2)
    assert float(result.params["Intercept"]) == pytest.approx(STATA_FR_CONS, rel=2e-2)


def test_frontier_coefficients_match_stata_closely():
    """Measured agreement is far tighter than the 2% contract; record it."""

    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    assert float(result.params["x1"]) == pytest.approx(STATA_FR_X1, rel=1e-6)
    assert float(result.params["x2"]) == pytest.approx(STATA_FR_X2, rel=1e-6)
    assert float(result.params["Intercept"]) == pytest.approx(STATA_FR_CONS, rel=1e-6)


def test_frontier_reports_finite_positive_variance_components():
    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    for key in ("sigma_u", "sigma_v", "lambda", "sigma2"):
        assert key in result.e, key
        value = float(result.e[key])
        assert np.isfinite(value), key
        assert value > 0, key

    sigma_u = float(result.e["sigma_u"])
    sigma_v = float(result.e["sigma_v"])
    assert float(result.e["sigma2"]) == pytest.approx(sigma_u**2 + sigma_v**2, rel=1e-12)
    assert float(result.e["lambda"]) == pytest.approx(sigma_u / sigma_v, rel=1e-12)
    assert float(result.e["gamma"]) == pytest.approx(
        sigma_u**2 / float(result.e["sigma2"]), rel=1e-12
    )


def test_frontier_variance_components_match_stata():
    """Stata: sigma_v .1830897, sigma_u .4262523, sigma2 .2152128, lambda 2.328106."""

    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    assert float(result.e["sigma_v"]) == pytest.approx(STATA_FR_SIGMA_V, rel=1e-4)
    assert float(result.e["sigma_u"]) == pytest.approx(STATA_FR_SIGMA_U, rel=1e-4)
    assert float(result.e["sigma2"]) == pytest.approx(STATA_FR_SIGMA2, rel=1e-4)
    assert float(result.e["lambda"]) == pytest.approx(STATA_FR_LAMBDA, rel=1e-4)


def test_frontier_log_likelihood_and_wald_test_match_stata():
    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    assert float(result.e["ll"]) == pytest.approx(STATA_FR_LL, rel=1e-6)
    assert float(result.e["chi2"]) == pytest.approx(STATA_FR_WALD, rel=1e-3)
    assert result.e["chi2type"] == "Wald"
    assert float(result.e["p"]) < 1e-10


def test_frontier_standard_errors_match_stata():
    """The observed information reproduces Stata's se to 6 significant digits."""

    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    assert float(result.bse["x1"]) == pytest.approx(STATA_FR_SE_X1, rel=1e-3)
    assert float(result.bse["x2"]) == pytest.approx(STATA_FR_SE_X2, rel=1e-3)
    assert float(result.bse["Intercept"]) == pytest.approx(STATA_FR_SE_CONS, rel=1e-3)
    # Stata prints the two variance parameters in the same table (fixture 162-163).
    assert float(result.params["/lnsig2v"]) == pytest.approx(STATA_FR_LNSIG2V, rel=1e-4)
    assert float(result.params["/lnsig2u"]) == pytest.approx(STATA_FR_LNSIG2U, rel=1e-4)
    assert float(result.bse["/lnsig2v"]) == pytest.approx(STATA_FR_SE_LNSIG2V, rel=1e-3)
    assert float(result.bse["/lnsig2u"]) == pytest.approx(STATA_FR_SE_LNSIG2U, rel=1e-3)


def test_frontier_lr_test_of_no_inefficiency_matches_stata():
    """Stata: "LR test of sigma_u=0: chibar2(01) = 23.60"."""

    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    assert float(result.e["lr_chi2"]) == pytest.approx(STATA_FR_LR, rel=1e-3)
    # A boundary test: the p-value halves the chi2(1) tail.
    assert float(result.e["p_lr"]) == pytest.approx(
        0.5 * float(stats.chi2.sf(result.e["lr_chi2"], 1.0)), rel=1e-12
    )


def test_frontier_rejects_an_unsupported_distribution():
    with pytest.raises(UnsupportedError, match="truncated"):
        Frontier.from_formula("y ~ x1 + x2", frontier_frame(), distribution="truncated")


@pytest.mark.parametrize("distribution", ["truncated", "exponential", "half"])
def test_frontier_rejects_every_other_distribution(distribution):
    with pytest.raises(UnsupportedError):
        Frontier.from_formula("y ~ x1 + x2", frontier_frame(), distribution=distribution)


def test_frontier_accepts_explicit_halfnormal_spelling():
    result = Frontier.from_formula(
        "y ~ x1 + x2", frontier_frame(), distribution="half-normal"
    ).fit()
    assert result.e["distribution"] == "halfnormal"


def test_frontier_reports_a_z_reference_distribution():
    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    assert result.dist == "z"
    assert result.model_name == "frontier"
    assert result.nobs == 400
    assert result.e["N"] == 400
    assert result.e["df_m"] == 2
    assert np.asarray(result.resid).shape == (400,)
    assert result.e["converged"] == 1


def test_frontier_reports_stata_style_summary():
    text = (
        Frontier.from_formula("y ~ x1 + x2", frontier_frame())
        .fit()
        .summary(style="stata")
    )

    assert "Number of obs" in text
    assert "Wald chi2(2)" in text
    assert "P>|z|" in text
    assert "/lnsig2v" in text
    assert "/lnsig2u" in text
    assert ".7813064" in text


def test_frontier_exposes_the_standard_e_keys():
    result = Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit()

    for key in [
        "N",
        "df_m",
        "df_r",
        "ll",
        "ll_0",
        "chi2",
        "p",
        "chi2type",
        "sigma_u",
        "sigma_v",
        "sigma2",
        "sigma2_u",
        "sigma2_v",
        "lambda",
        "gamma",
        "lr_chi2",
        "distribution",
        "converged",
        "cmd",
    ]:
        assert key in result.e, key


def test_frontier_likelihood_uses_logcdf_in_the_far_tail():
    """Guarding ``Phi`` underflow is a property of the likelihood, so test it there.

    At ``z = -40`` the correct ``ln Phi(z)`` is about ``-804``;
    ``ln(stats.norm.cdf(-40))`` is ``-inf`` and would silently drop the
    observation from the sum.  Calling the likelihood directly isolates that from
    the optimiser, whose own outcome on such a sample is a boundary solution
    (``sigma_v -> 0``) rather than anything about the tail.
    """

    exog = np.column_stack([np.ones(3), [0.0, 1.0, -1.0]])
    y = np.array([100.0, 101.0, 99.0])
    # Layout is [beta0, beta1, lnsig2v, lnsig2u]:
    # sigma_v^2 = 1, sigma_u^2 = e^-4, sigma = sqrt(1 + e^-4), lambda = e^-2.
    theta = np.array([100.0, 1.0, 0.0, -4.0])

    llf = frontier_loglike(theta, y, exog)
    assert np.isfinite(llf)

    # Compare against an independent, naive implementation for this mild case.
    sigma = np.sqrt(np.exp(0.0) + np.exp(-4.0))
    lam = np.exp(0.5 * (-4.0 - 0.0))
    z = (y - exog @ theta[:2]) / sigma
    expected = float(
        np.sum(
            np.log(2.0)
            - np.log(sigma)
            - 0.5 * z**2
            - 0.5 * np.log(2 * np.pi)
            + np.log(stats.norm.cdf(-z * lam))
        )
    )
    assert llf == pytest.approx(expected, rel=1e-10)

    # Now push the outcome far into the right tail of e (a large positive
    # residual), where -z*lambda is very negative: the naive ln(Phi(.)) form
    # underflows to -inf while the logcdf form stays finite.
    far = np.array([500.0, 500.0, 500.0])
    z_far = (far - exog @ theta[:2]) / sigma
    with np.errstate(divide="ignore"):
        naive = np.log(stats.norm.cdf(-z_far * lam))
    assert not np.all(np.isfinite(naive))
    assert np.isfinite(frontier_loglike(theta, far, exog))


def test_frontier_gaussian_limit_when_inefficiency_is_absent():
    """With no true inefficiency the frontier must collapse toward OLS.

    The likelihood is flat in sigma_u at the boundary, so only coefficient
    agreement with OLS is asserted (Stata behaves the same way).
    """

    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    frame = pd.DataFrame({"y": 1.0 + 2.0 * x + rng.normal(scale=0.5, size=n), "x": x})

    frontier = Frontier.from_formula("y ~ x", frame).fit()
    plain = ws.OLS.from_formula("y ~ x", frame).fit()

    assert float(frontier.params["x"]) == pytest.approx(
        float(plain.params["x"]), abs=0.05
    )
    assert float(frontier.e["sigma_u"]) < 0.5


def test_frontier_works_with_a_single_regressor():
    rng = np.random.default_rng(17)
    n = 300
    x = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": 0.5
            + 1.2 * x
            - np.abs(rng.normal(scale=0.3, size=n))
            + rng.normal(scale=0.2, size=n),
            "x": x,
        }
    )
    result = Frontier.from_formula("y ~ x", frame).fit()

    assert np.all(np.isfinite(result.bse.to_numpy()))
    assert float(result.params["x"]) == pytest.approx(1.2, abs=0.1)
    assert float(result.e["lambda"]) > 0


def test_frontier_rejects_an_unsupported_cov_type():
    with pytest.raises(ValueError, match="cov_type"):
        Frontier.from_formula("y ~ x1 + x2", frontier_frame()).fit(cov_type="cluster")


def test_frontier_rejects_a_formula_without_a_tilde():
    with pytest.raises(ValueError, match="~"):
        Frontier.from_formula("y = x1", frontier_frame())


def test_frontier_strict_raises_when_the_iteration_limit_is_hit():
    model = Frontier.from_formula("y ~ x1 + x2", frontier_frame())
    with pytest.raises(ConvergenceError):
        model.fit(maxiter=1, strict=True)


def test_nl_accepts_subset():
    frame = nl_frame()
    full = Nl.from_formula("y = {b0} * exp(-{b1} * x)", frame).fit()
    part = Nl.from_formula(
        "y = {b0} * exp(-{b1} * x)", frame, subset=frame["x"] > 0.5
    ).fit()

    assert part.nobs < full.nobs
    # Dropping the tail changes the estimate slightly; the parameter names and the
    # sample bookkeeping are what subset must get right.
    assert part.param_names == ["/b0", "/b1"]
    assert part.nobs == int((frame["x"] > 0.5).sum())
    assert float(part.params["/b1"]) == pytest.approx(
        float(full.params["/b1"]), abs=0.05
    )
