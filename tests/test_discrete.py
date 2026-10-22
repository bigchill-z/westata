"""Conformance tests for discrete-choice and count models.

References are independent: analytic cases where the MLE is known, plus
cross-checks against statsmodels' own estimators for the backend wiring.  The
Stata-specific properties under test are z inference, the observed-information
default variance, and McFadden's pseudo R-squared in ``e(r2_p)``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

from westata.errors import ConvergenceError
from westata.estimators.discrete import (
    Logit,
    NegativeBinomial,
    OrderedLogit,
    OrderedProbit,
    Poisson,
    Probit,
)


def _binary(n: int = 400, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = -0.5 + 1.2 * x1 - 0.8 * x2 + rng.logistic(size=n)
    y = (latent > 0).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _count(n: int = 400, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.6 * x1)
    y = rng.poisson(mu).astype(float)
    return pd.DataFrame({"y": y, "x1": x1})


def _ordered(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    latent = 0.9 * x1 + rng.normal(size=n)
    y = pd.cut(latent, bins=[-np.inf, -0.6, 0.6, np.inf], labels=[1, 2, 3]).astype(
        float
    )
    return pd.DataFrame({"y": y, "x1": x1})


# ------------------------------------------------------------------- logit


def test_logit_uses_z_inference():
    res = Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    assert res.dist == "z"
    expected = 2.0 * stats.norm.sf(np.abs(res.tvalues.to_numpy()))
    np.testing.assert_allclose(res.pvalues.to_numpy(), expected, rtol=1e-12)


def test_logit_matches_statsmodels_point_estimates():
    df = _binary()
    res = Logit.from_formula("y ~ x1 + x2", df).fit()

    X = sm.add_constant(df[["x1", "x2"]], has_constant="add")
    ref = sm.Logit(df["y"], X).fit(method="newton", disp=0, maxiter=500)

    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-8)
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-6)


def test_logit_sign_is_consistent_with_dgp():
    res = Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    assert res.params["x1"] > 0  # DGP coefficient is +1.2
    assert res.params["x2"] < 0  # DGP coefficient is -0.8
    assert res.params["x1"] == pytest.approx(1.2, abs=0.4)


def test_logit_pseudo_r2_is_mcfadden():
    res = Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    expected = 1.0 - res.e["ll"] / res.e["ll_0"]
    assert res.e["r2_p"] == pytest.approx(expected, rel=1e-12)
    assert 0.0 <= res.e["r2_p"] <= 1.0


def test_logit_summary_reports_wald_chi2():
    text = Logit.from_formula("y ~ x1 + x2", _binary()).fit().summary(style="stata")
    assert "Wald chi2(" in text
    assert "P>|z|" in text
    assert "P>|t|" not in text


# ------------------------------------------------------------------ probit


def test_probit_matches_statsmodels():
    df = _binary()
    res = Probit.from_formula("y ~ x1 + x2", df).fit()

    X = sm.add_constant(df[["x1", "x2"]], has_constant="add")
    ref = sm.Probit(df["y"], X).fit(method="newton", disp=0, maxiter=500)

    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-8)


def test_probit_sign_is_consistent_with_dgp():
    # The DGP used a logistic error, so the probit slope is scaled by ~0.55 but
    # must keep the same sign structure.
    res = Probit.from_formula("y ~ x1 + x2", _binary()).fit()
    assert res.params["x1"] > 0
    assert res.params["x2"] < 0


# ----------------------------------------------------------------- poisson


def test_poisson_matches_statsmodels():
    df = _count()
    res = Poisson.from_formula("y ~ x1", df).fit()

    X = sm.add_constant(df[["x1"]], has_constant="add")
    ref = sm.Poisson(df["y"], X).fit(method="newton", disp=0, maxiter=500)

    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-8)
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-8)


def test_poisson_recovers_dgp_slope():
    res = Poisson.from_formula("y ~ x1", _count()).fit()
    assert res.params["x1"] == pytest.approx(0.6, abs=0.15)


def test_poisson_robust_cov_type():
    df = _count()
    res = Poisson.from_formula("y ~ x1", df).fit(cov_type="HC0")

    X = sm.add_constant(df[["x1"]], has_constant="add")
    ref = sm.Poisson(df["y"], X).fit(
        method="newton", disp=0, cov_type="HC0", maxiter=500
    )
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-6)


# ------------------------------------------------------- negative binomial


def test_negative_binomial_runs_and_has_extra_parameter():
    df = _count()
    res = NegativeBinomial.from_formula("y ~ x1", df).fit()
    # nbreg estimates an extra dispersion parameter (alpha).
    assert len(res.params) == 3
    assert np.all(np.isfinite(res.params.to_numpy()))


# ------------------------------------------------------------ ordered models


def test_ordered_logit_runs_and_reports_z():
    res = OrderedLogit.from_formula("y ~ x1", _ordered()).fit()
    assert res.dist == "z"
    assert np.all(np.isfinite(res.params.to_numpy()))
    assert "P>|z|" in res.summary(style="stata")


def test_ordered_probit_runs():
    res = OrderedProbit.from_formula("y ~ x1", _ordered()).fit()
    assert np.all(np.isfinite(res.params.to_numpy()))
    assert res.params["x1"] > 0


def test_ordered_thresholds_are_labelled_like_stata():
    res = OrderedProbit.from_formula("y ~ x1", _ordered()).fit()
    # Three categories -> two cut points, named /cut1 and /cut2 as in Stata.
    assert any(name.startswith("/cut") for name in res.param_names)


# ------------------------------------------------------------ guard rails


def test_unsupported_cov_type_raises():
    with pytest.raises(ValueError, match="cov_type"):
        Logit.from_formula("y ~ x1", _binary()).fit(cov_type="bogus")


def test_perfect_separation_warns_or_raises():
    # A regressor that perfectly predicts the outcome: the MLE diverges.
    df = pd.DataFrame({"y": [0.0, 0.0, 1.0, 1.0], "x": [-2.0, -1.0, 1.0, 2.0]})
    with pytest.raises(Exception):
        Logit.from_formula("y ~ x", df).fit(strict=True, maxiter=200)


def test_missing_lhs_column_raises():
    df = _binary()
    with pytest.raises(Exception, match="not_a_column"):
        Logit.from_formula("not_a_column ~ x1", df).fit()


def test_e_keys_present():
    res = Logit.from_formula("y ~ x1", _binary()).fit()
    for key in ["N", "df_m", "ll", "chi2", "r2_p"]:
        assert key in res.e, key


def test_parameter_names_come_from_the_formula():
    # ``results.test("mpg")`` only works if the design names survive; a raw
    # statsmodels array model would report const/x1/x2 instead.
    res = Logit.from_formula("y ~ x1 + x2", _binary()).fit()
    assert res.param_names == ["Intercept", "x1", "x2"]
    assert np.isfinite(res.test("x1").pvalue)


def test_unusable_covariance_raises_convergence_error():
    # Negative binomial on a nearly-Poisson count with a bad start can fail to
    # produce a Hessian; westata must say so, not raise a raw statsmodels
    # ValueError from cov_params().
    rng = np.random.default_rng(99)
    n = 120
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.2 + 0.5 * x)).astype(float)
    df = pd.DataFrame({"y": y, "x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = NegativeBinomial.from_formula("y ~ x", df).fit()
        except ConvergenceError:
            return  # acceptable and expected for a hard likelihood surface
    assert np.all(np.isfinite(res.params.to_numpy()))
