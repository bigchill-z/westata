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


# ---------------------------------------------- Stata conformance (T1-T3)
#
# Captured live from StataNow 19.5 MP on 2026-10-24, DGP n=500 seed=2024:
# x1,x2 ~ N(0,1), ystar = 1 + 0.8 x1 - 0.5 x2 + N(0,1), y = max(ystar, 0).
# `tobit y x1 x2, ll(0)` reported
#     x1 .84959386  x2 -.52643973  _cons .92658230  var(e.y) 1.11808396
# with e(V) diagonal (0.05221914988^2, 0.04964909387^2, 0.05122533893^2,
# 2.7179562390530054^2), e(df_m)=2, e(df_r)=498.
#
# The three defects these pin down:
#   T1  se(/sigma) was se(ln sigma): the covariance was never transformed from
#       the log-scale parameterisation, so sigma's z was overstated by 1/sigma
#       (0.0406037 vs Stata's 0.0406037... after the delta method).
#   T2  e(df_r) was N - k = 496; Stata's is N - df_m = 498.
#   T3  the coefficient SEs came from a finite-difference Hessian at step 1e-5.

TOBIT_DGP_N = 500


def _tobit_conformance_frame():
    rng = np.random.default_rng(2024)
    n = TOBIT_DGP_N
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    ystar = 1.0 + 0.8 * x1 - 0.5 * x2 + rng.normal(size=n)
    frame = pd.DataFrame({"x1": x1, "x2": x2})
    frame["y"] = np.clip(ystar, 0.0, None)
    frame["yu"] = np.minimum(ystar, 2.5)
    frame["yb"] = np.clip(ystar, 0.0, 2.0)
    return frame


STATA_TOBIT = {
    "ll": {
        "params": {"x1": 0.8495938564718171, "x2": -0.5264397336099021,
                   "Intercept": 0.9265823047044895},
        "se": {"x1": 0.0522191498799564, "x2": 0.04964909386660454,
               "Intercept": 0.05122533893099292, "/sigma": 0.04060374414635429},
        "df_m": 2.0, "df_r": 498.0, "sigma": 1.0573942015731823,
    },
    "ul": {
        "params": {"x1": 0.8138658226255706, "x2": -0.5045438544766526,
                   "Intercept": 0.9470464876280105},
        "se": {"x1": 0.04806497488068231, "x2": 0.0458215513579502,
               "Intercept": 0.04683163300182566, "/sigma": 0.035419636598977754},
        "df_m": 2.0, "df_r": 498.0, "sigma": 1.0165140661636944,
    },
    "both": {
        "params": {"x1": 0.8463986604607947, "x2": -0.5410777522746777,
                   "Intercept": 0.9288309077571877},
        "se": {"x1": 0.061966984308833975, "x2": 0.05655768946328548,
               "Intercept": 0.05309843934038632, "/sigma": 0.05383672836926836},
        "df_m": 2.0, "df_r": 498.0, "sigma": 1.071207063656481,
    },
}

TOBIT_VARIANTS = {
    "ll": ("y", 0.0, None),
    "ul": ("yu", None, 2.5),
    "both": ("yb", 0.0, 2.0),
}


@pytest.mark.parametrize("variant", sorted(STATA_TOBIT))
def test_sigma_standard_error_uses_the_sigma_jacobian(variant):
    """T1: ``se(/sigma) = se(ln sigma) * sigma_hat``, all three variants.

    Reporting ``se(ln sigma)`` in the sigma row is not a rounding difference: the
    ratio is exactly ``1/sigma_hat``, so every test on the scale parameter had a
    z statistic overstated by that factor.
    """

    lhs, lower, upper = TOBIT_VARIANTS[variant]
    reference = STATA_TOBIT[variant]
    res = Tobit.from_formula(
        f"{lhs} ~ x1 + x2", _tobit_conformance_frame(), lower=lower, upper=upper
    ).fit()

    assert float(res.bse["/sigma"]) == pytest.approx(
        reference["se"]["/sigma"], rel=1e-6
    )


@pytest.mark.parametrize("variant", sorted(STATA_TOBIT))
def test_residual_dof_is_n_minus_df_m(variant):
    """T2: ``e(df_r) = N - e(df_m)``, and ``df_m = rank - 1 - hascons``.

    Stata's ``tobit.ado`` sets ``df_m = e(rank) - 1 - hascons`` and
    ``df_r = e(N) - e(df_m)``: neither the scale parameter nor (with a constant)
    the intercept consumes a degree of freedom. westata used ``N - k``, i.e. 496
    where Stata reports 498, and every t statistic used the wrong denominator.
    """

    lhs, lower, upper = TOBIT_VARIANTS[variant]
    reference = STATA_TOBIT[variant]
    res = Tobit.from_formula(
        f"{lhs} ~ x1 + x2", _tobit_conformance_frame(), lower=lower, upper=upper
    ).fit()

    assert res.e["df_m"] == reference["df_m"]
    assert res.e["df_r"] == reference["df_r"]
    assert res.df_resid == reference["df_r"]
    assert res.e["df_r"] + res.e["df_m"] == TOBIT_DGP_N


@pytest.mark.parametrize("variant", sorted(STATA_TOBIT))
def test_coefficient_standard_errors_match_stata(variant):
    """T3: observed information, not a coarse finite-difference Hessian.

    The old central-difference Hessian at ``step=1e-5`` left every coefficient
    standard error 8.9e-5 away from Stata's.  The analytic Hessian lands at 3e-8
    (limited by Stata's own stopping tolerance, which the score at its
    estimates shows: |score| ~ 1e-6 there).
    """

    lhs, lower, upper = TOBIT_VARIANTS[variant]
    reference = STATA_TOBIT[variant]
    res = Tobit.from_formula(
        f"{lhs} ~ x1 + x2", _tobit_conformance_frame(), lower=lower, upper=upper
    ).fit()

    for name, expected in reference["se"].items():
        if name == "/sigma":
            continue
        assert float(res.bse[name]) == pytest.approx(expected, rel=1e-6), name
    for name, expected in reference["params"].items():
        assert float(res.params[name]) == pytest.approx(expected, rel=1e-6), name
    assert float(res.e["sigma"]) == pytest.approx(reference["sigma"], rel=1e-6)


def test_stdp_uses_the_coefficient_block_of_the_covariance():
    """T3's acceptance test: ``stdp = sqrt(x' V x)`` from the analytic V."""

    frame = _tobit_conformance_frame()
    res = Tobit.from_formula("y ~ x1 + x2", frame, lower=0.0).fit()

    cov = np.asarray(res.cov_params(), dtype=float)
    X = np.column_stack([np.ones(len(frame)), frame["x1"], frame["x2"]])
    stdp = np.sqrt(np.einsum("ij,jk,ik->i", X, cov[:-1, :-1], X))

    # Captured from Stata's `predict double _stdp, stdp` on the same data.
    assert stdp[0] == pytest.approx(0.08376789256819156, rel=1e-6)
    assert stdp[250] == pytest.approx(0.06800766700228333, rel=1e-6)
    assert float(np.max(stdp)) == pytest.approx(0.19301463582789766, rel=1e-6)


def test_analytic_score_and_hessian_agree_with_finite_differences():
    """The two analytic derivatives are checked against central differences.

    This is the guard that keeps the closed forms honest: if either is edited
    into a wrong sign or a missing term, the fitted standard errors drift away
    from Stata's without any test noticing.
    """

    frame = _left_censored(n=200, seed=11)
    model = Tobit.from_formula("y ~ x1 + x2", frame, lower=2.0)
    theta = np.array([0.9, 0.5, 0.2, np.log(0.4)])
    k = 3

    step = 1e-6
    numeric = np.zeros(theta.size)
    for i in range(theta.size):
        e = np.zeros(theta.size)
        e[i] = step
        numeric[i] = (
            model._loglike(theta + e) - model._loglike(theta - e)
        ) / (2 * step)
    np.testing.assert_allclose(model._score(theta), numeric, rtol=1e-6, atol=1e-6)

    analytic = model._information(theta, k)
    numeric_hessian = np.empty_like(analytic)
    for j in range(theta.size):
        e = np.zeros(theta.size)
        e[j] = step
        numeric_hessian[:, j] = (
            model._score(theta + e) - model._score(theta - e)
        ) / (2 * step)
    np.testing.assert_allclose(analytic, numeric_hessian, rtol=1e-5, atol=1e-5)
