"""Conformance tests for ``test`` and ``lincom``.

Every assertion is checked against an independent hand computation from
``[R] test`` / ``[R] lincom`` (W = (Rb-r)'(RVR')^-1(Rb-r)), never against the
implementation under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from westata.estimators.hdfe import FEOLS
from westata.estimators.iv import IV2SLS
from westata.estimators.linear import OLS


def _data(n: int = 80, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + rng.normal(scale=0.6, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


@pytest.fixture(scope="module")
def ols_result():
    return OLS.from_formula("y ~ x1 + x2 + x3", _data()).fit()


def _manual_wald(params, cov, R, r):
    diff = R @ params - r
    W = float(diff @ np.linalg.solve(R @ cov @ R.T, diff))
    return W


def test_single_coefficient_zero_test_matches_manual_and_tstat(ols_result):
    names = ols_result.param_names
    R = np.zeros((1, len(names)))
    R[0, names.index("x1")] = 1.0

    outcome = ols_result.test("x1")

    W = _manual_wald(
        ols_result.params.to_numpy(), ols_result.cov_params(), R, np.zeros(1)
    )
    q = 1
    df_resid = ols_result.df_resid
    assert outcome.statistic == pytest.approx(W / q, rel=1e-12)
    assert outcome.distribution == "F"
    assert outcome.df_denom == pytest.approx(df_resid)

    # For q=1 the F test is exactly the square of the t statistic.
    t = ols_result.tvalues["x1"]
    assert outcome.statistic == pytest.approx(t**2, rel=1e-10)
    assert outcome.pvalue == pytest.approx(ols_result.pvalues["x1"], rel=1e-10)


def test_joint_test_of_two_coefficients(ols_result):
    names = ols_result.param_names
    R = np.zeros((2, len(names)))
    R[0, names.index("x1")] = 1.0
    R[1, names.index("x2")] = 1.0

    outcome = ols_result.test("x1 x2")

    W = _manual_wald(
        ols_result.params.to_numpy(), ols_result.cov_params(), R, np.zeros(2)
    )
    assert outcome.df == 2
    assert outcome.statistic == pytest.approx(W / 2, rel=1e-12)
    assert outcome.pvalue == pytest.approx(
        stats.f.sf(W / 2, 2, ols_result.df_resid), rel=1e-12
    )


def test_equality_hypothesis_x1_equals_x2(ols_result):
    names = ols_result.param_names
    R = np.zeros((1, len(names)))
    R[0, names.index("x1")] = 1.0
    R[0, names.index("x2")] = -1.0

    outcome = ols_result.test("x1 = x2")

    W = _manual_wald(
        ols_result.params.to_numpy(), ols_result.cov_params(), R, np.zeros(1)
    )
    assert outcome.statistic == pytest.approx(W, rel=1e-12)


def test_hypothesis_against_nonzero_constant(ols_result):
    names = ols_result.param_names
    R = np.zeros((1, len(names)))
    R[0, names.index("x1")] = 1.0

    outcome = ols_result.test("x1 = 2")

    W = _manual_wald(
        ols_result.params.to_numpy(), ols_result.cov_params(), R, np.array([2.0])
    )
    assert outcome.statistic == pytest.approx(W, rel=1e-12)


def test_equality_chain_tests_pairwise_against_first(ols_result):
    names = ols_result.param_names
    R = np.zeros((2, len(names)))
    R[0, names.index("x1")] = 1.0
    R[0, names.index("x2")] = -1.0
    R[1, names.index("x1")] = 1.0
    R[1, names.index("x3")] = -1.0

    outcome = ols_result.test("x1 = x2 = x3")

    W = _manual_wald(
        ols_result.params.to_numpy(), ols_result.cov_params(), R, np.zeros(2)
    )
    assert outcome.df == 2
    assert outcome.statistic == pytest.approx(W / 2, rel=1e-12)


def test_wildcard_selects_matching_coefficients(ols_result):
    outcome = ols_result.test("x*")
    assert outcome.df == 3


def test_unknown_coefficient_raises(ols_result):
    with pytest.raises(KeyError, match="not found"):
        ols_result.test("nonexistent")


def test_iv_result_uses_chi2(  ):
    rng = np.random.default_rng(9)
    n = 120
    z = rng.normal(size=n)
    x = 0.8 * z + rng.normal(size=n)
    y = 1.0 + 2.0 * x + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "z": z})
    res = IV2SLS.from_formula("y ~ [x ~ z]", df).fit()

    outcome = res.test("x")
    assert outcome.distribution == "chi2"
    assert outcome.df_denom is None
    assert outcome.pvalue == pytest.approx(
        stats.chi2.sf(outcome.statistic, 1), rel=1e-12
    )


def test_coef_option_returns_constrained_estimates(ols_result):
    names = ols_result.param_names
    out = ols_result.test("x1 = 0", coef=True)
    constrained = out["params"]

    # The constrained estimate must satisfy the constraint exactly.
    assert constrained["x1"] == pytest.approx(0.0, abs=1e-12)
    # And the other coefficients must move only through the correction term.
    assert not np.allclose(
        constrained.drop("x1").to_numpy(),
        ols_result.params.drop("x1").to_numpy(),
        rtol=1e-6,
    )


def test_statsmodels_wald_test_scalar_true_is_f(ols_result):
    names = ols_result.param_names
    R = np.zeros((1, len(names)))
    R[0, names.index("x1")] = 1.0

    out = ols_result.wald_test(R)
    assert out.distribution == "F"
    assert out.statistic == pytest.approx(ols_result.tvalues["x1"] ** 2, rel=1e-10)


def test_statsmodels_wald_test_scalar_false_is_chi2(ols_result):
    names = ols_result.param_names
    R = np.zeros((1, len(names)))
    R[0, names.index("x1")] = 1.0

    out = ols_result.wald_test(R, scalar=False)
    assert out.distribution == "chi2"
    W = _manual_wald(
        ols_result.params.to_numpy(), ols_result.cov_params(), R, np.zeros(1)
    )
    assert out.statistic == pytest.approx(W, rel=1e-12)


# ---------------------------------------------------------------------- lincom


def test_lincom_single_coefficient_matches_t_stat(ols_result):
    out = ols_result.lincom("x1")
    assert out.estimate == pytest.approx(ols_result.params["x1"], rel=1e-12)
    assert out.std_err == pytest.approx(ols_result.bse["x1"], rel=1e-12)
    assert out.statistic == pytest.approx(ols_result.tvalues["x1"], rel=1e-12)
    assert out.pvalue == pytest.approx(ols_result.pvalues["x1"], rel=1e-12)


def test_lincom_sum_of_coefficients(ols_result):
    names = ols_result.param_names
    weights = np.zeros(len(names))
    weights[names.index("x1")] = 1.0
    weights[names.index("x2")] = 1.0

    out = ols_result.lincom("x1 + x2")

    expected = float(weights @ ols_result.params.to_numpy())
    expected_se = float(
        np.sqrt(weights @ ols_result.cov_params() @ weights)
    )
    assert out.estimate == pytest.approx(expected, rel=1e-12)
    assert out.std_err == pytest.approx(expected_se, rel=1e-12)


def test_lincom_scaled_difference(ols_result):
    names = ols_result.param_names
    weights = np.zeros(len(names))
    weights[names.index("x1")] = 2.0
    weights[names.index("x2")] = -3.0

    out = ols_result.lincom("2*x1 - 3*x2")

    expected = float(weights @ ols_result.params.to_numpy())
    assert out.estimate == pytest.approx(expected, rel=1e-12)


def test_lincom_confidence_interval_uses_t_distribution(ols_result):
    out = ols_result.lincom("x1")
    critical = stats.t.ppf(0.975, ols_result.df_resid)
    assert out.ci_lower == pytest.approx(
        out.estimate - critical * out.std_err, rel=1e-12
    )
    assert out.ci_upper == pytest.approx(
        out.estimate + critical * out.std_err, rel=1e-12
    )


def test_lincom_eform_exponentiates(ols_result):
    plain = ols_result.lincom("x1")
    transformed = ols_result.lincom("x1", eform=True)
    assert transformed.estimate == pytest.approx(np.exp(plain.estimate), rel=1e-12)
    assert transformed.ci_lower == pytest.approx(np.exp(plain.ci_lower), rel=1e-12)


def test_lincom_rejects_equals_sign(ols_result):
    with pytest.raises(ValueError, match="equals"):
        ols_result.lincom("x1 = 0")


def test_lincom_custom_level(ols_result):
    out = ols_result.lincom("x1", level=0.90)
    critical = stats.t.ppf(0.95, ols_result.df_resid)
    assert out.ci_lower == pytest.approx(
        out.estimate - critical * out.std_err, rel=1e-12
    )


def test_hdfe_result_supports_test_and_lincom():
    rng = np.random.default_rng(21)
    n_entities, n_periods = 20, 5
    entity = np.repeat(np.arange(n_entities), n_periods)
    x1 = rng.normal(size=n_entities * n_periods)
    x2 = rng.normal(size=n_entities * n_periods)
    y = 1.0 + 2.0 * x1 - x2 + rng.normal(size=n_entities)[entity] + rng.normal(
        scale=0.3, size=n_entities * n_periods
    )
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity})

    res = FEOLS.from_formula("y ~ x1 + x2 | firm", df).fit()
    outcome = res.test("x1 x2")
    assert outcome.df == 2
    assert np.isfinite(outcome.pvalue)
    assert np.isfinite(res.lincom("x1 + x2").estimate)
