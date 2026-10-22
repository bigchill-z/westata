"""Conformance tests for the self-implemented Tobit estimator.

The likelihood is specified from ``[R] tobit`` (p. 4):

    lnL = sum_uncensored [-ln(sigma) - 0.5 ln(2pi) - 0.5 ((y-xb)/sigma)^2]
        + sum_left  ln Phi((a-xb)/sigma)
        + sum_right ln(1 - Phi((b-xb)/sigma))

Tests check the likelihood itself, recovery of a known DGP, the censoring
bookkeeping Stata prints (Uncensored / Left-censored / Right-censored), and the
Stata-style reporting (LR chi2, Pseudo R2, /sigma).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, stats

from westata.errors import ConsistencyError, ConvergenceError
from westata.estimators.tobit import Tobit


def _left_censored(n: int = 1500, seed: int = 11):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.65 * x1 + 0.3 * x2 + rng.normal(scale=0.35, size=n)
    y = np.maximum(latent, 2.0)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _two_sided(n: int = 1500, seed: int = 13):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    latent = 0.5 + 0.8 * x1 + rng.normal(scale=0.5, size=n)
    y = np.clip(latent, -0.4, 1.6)
    return pd.DataFrame({"y": y, "x1": x1})


# --------------------------------------------------------------- likelihood


def test_loglike_matches_independent_formula():
    df = _left_censored(n=300)
    model = Tobit.from_formula("y ~ x1 + x2", df, lower=2.0)

    beta = np.array([0.9, 0.5, 0.2])
    sigma = 0.4
    theta = np.concatenate([beta, [np.log(sigma)]])

    # Independent implementation of the [R] tobit likelihood.
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df["x1"], df["x2"]])
    xb = X @ beta
    left = y <= 2.0
    unc = ~left
    expected = 0.0
    expected += float(
        np.sum(
            -np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * ((y[unc] - xb[unc]) / sigma) ** 2
        )
    )
    expected += float(np.sum(stats.norm.logcdf((2.0 - xb[left]) / sigma)))

    assert model._loglike(theta) == pytest.approx(expected, rel=1e-12)


def test_right_censoring_contributes_survival_probability():
    df = _two_sided(n=400)
    model = Tobit.from_formula("y ~ x1", df, lower=-0.4, upper=1.6)

    beta = np.array([0.4, 0.7])
    sigma = 0.6
    theta = np.concatenate([beta, [np.log(sigma)]])

    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df["x1"]])
    xb = X @ beta
    left = y <= -0.4
    right = y >= 1.6
    unc = ~(left | right)

    expected = float(
        np.sum(
            -np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * ((y[unc] - xb[unc]) / sigma) ** 2
        )
    )
    expected += float(np.sum(stats.norm.logcdf((-0.4 - xb[left]) / sigma)))
    expected += float(np.sum(stats.norm.logsf((1.6 - xb[right]) / sigma)))

    assert model._loglike(theta) == pytest.approx(expected, rel=1e-12)


# ----------------------------------------------------------------- recovery


def test_recovers_dgp_parameters():
    df = _left_censored()
    res = Tobit.from_formula("y ~ x1 + x2", df, lower=2.0).fit()

    assert res.params["x1"] == pytest.approx(0.65, abs=0.08)
    assert res.params["x2"] == pytest.approx(0.30, abs=0.08)
    assert res.params["Intercept"] == pytest.approx(1.00, abs=0.15)
    assert res.e["sigma"] == pytest.approx(0.35, abs=0.06)


def test_censoring_counts_match_the_data():
    df = _left_censored()
    res = Tobit.from_formula("y ~ x1 + x2", df, lower=2.0).fit()

    n_left = int((df["y"] <= 2.0).sum())
    assert res.e["n_left"] == n_left
    assert res.e["n_uncensored"] == len(df) - n_left
    assert res.e["n_left"] + res.e["n_uncensored"] == len(df)


def test_two_sided_censoring_runs():
    df = _two_sided()
    res = Tobit.from_formula("y ~ x1", df, lower=-0.4, upper=1.6).fit()

    assert res.e["n_left"] > 0
    assert res.e["n_right"] > 0
    assert res.params["x1"] == pytest.approx(0.8, abs=0.15)


def test_standard_errors_are_positive_and_finite():
    df = _left_censored()
    res = Tobit.from_formula("y ~ x1 + x2", df, lower=2.0).fit()
    assert np.all(np.isfinite(res.bse.to_numpy()))
    assert np.all(res.bse.to_numpy() > 0)


def test_tobit_estimate_differs_from_naive_ols():
    # The whole point of Tobit: censoring biases OLS towards zero, so the Tobit
    # slope on x1 must exceed the OLS slope on the same censored data.
    import westata as ws

    df = _left_censored()
    ols = ws.OLS.from_formula("y ~ x1 + x2", df).fit()
    tobit = Tobit.from_formula("y ~ x1 + x2", df, lower=2.0).fit()
    assert tobit.params["x1"] > ols.params["x1"]


# ---------------------------------------------------------------- reporting


def test_summary_uses_lr_chi2_and_pseudo_r2():
    df = _left_censored()
    text = Tobit.from_formula("y ~ x1 + x2", df, lower=2.0).fit().summary(
        style="stata"
    )
    assert "LR chi2(" in text
    assert "Pseudo R2" in text
    assert "Log likelihood" in text
    assert "P>|t|" in text


def test_sigma_reported_like_stata():
    df = _left_censored()
    res = Tobit.from_formula("y ~ x1 + x2", df, lower=2.0).fit()
    assert "/sigma" in res.param_names
    assert res.e["var_e"] == pytest.approx(res.e["sigma"] ** 2, rel=1e-12)


def test_e_keys_present():
    df = _left_censored()
    res = Tobit.from_formula("y ~ x1 + x2", df, lower=2.0).fit()
    for key in ["N", "ll", "chi2", "r2_p", "sigma", "n_left", "n_uncensored"]:
        assert key in res.e, key


# -------------------------------------------------------------- guard rails


def test_no_censoring_limit_raises():
    with pytest.raises(ConsistencyError, match="censoring limit"):
        Tobit.from_formula("y ~ x1", _left_censored(), lower=None, upper=None)


def test_inverted_limits_raise():
    with pytest.raises(ConsistencyError, match="below upper"):
        Tobit.from_formula("y ~ x1", _left_censored(), lower=5.0, upper=1.0)


def test_unsupported_cov_type_raises():
    with pytest.raises(ValueError, match="cov_type"):
        Tobit.from_formula("y ~ x1", _left_censored(), lower=2.0).fit(
            cov_type="cluster"
        )


def test_maximum_likelihood_beats_arbitrary_point():
    df = _left_censored(n=400)
    model = Tobit.from_formula("y ~ x1 + x2", df, lower=2.0)
    res = model.fit()

    k = 3
    theta_hat = np.concatenate(
        [res.params.to_numpy()[:k], [np.log(res.e["sigma"])]]
    )
    # Perturbing away from the optimum must lower the likelihood.
    assert model._loglike(theta_hat) > model._loglike(theta_hat + 0.1)
    assert model._loglike(theta_hat) > model._loglike(theta_hat - 0.1)
