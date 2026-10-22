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
