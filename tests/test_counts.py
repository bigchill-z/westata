"""Tests for the zero-inflated and truncated count models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import gammaln

import westata as ws
from westata.errors import ConsistencyError
from westata.estimators.counts import (
    TruncatedNegativeBinomial,
    TruncatedPoisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)


def _poisson_frame(n: int = 800, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.4 + 0.6 * x1)
    return pd.DataFrame({"y": rng.poisson(mu).astype(float), "x1": x1})


def _excess_zero_frame(n: int = 1200, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.4 + 0.6 * x1)
    structural = rng.random(size=n) < 0.25
    counts = np.where(structural, 0, rng.poisson(mu)).astype(float)
    return pd.DataFrame({"y": counts, "x1": x1})


def _overdispersed_frame(n: int = 1200, seed: int = 47) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.4 + 0.6 * x1)
    lam = rng.gamma(shape=1.0 / 0.5, scale=mu * 0.5)
    return pd.DataFrame({"y": rng.poisson(lam).astype(float), "x1": x1})


# ------------------------------------------------- zero-inflated reduction


def test_zip_without_inflation_reduces_to_poisson():
    frame = _poisson_frame()
    zipped = ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate=None).fit()
    plain = ws.Poisson.from_formula("y ~ x1", frame).fit()

    np.testing.assert_allclose(
        zipped.params[["Intercept", "x1"]].to_numpy(),
        plain.params[["Intercept", "x1"]].to_numpy(),
        rtol=1e-5,
    )


def test_zip_loglikelihood_matches_the_formula():
    frame = _excess_zero_frame(n=400)
    result = ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate="x1").fit()

    params = result.params
    beta = np.array([float(params["Intercept"]), float(params["x1"])])
    gamma = np.array(
        [float(params["inflate:Intercept"]), float(params["inflate:x1"])]
    )
    X = np.column_stack([np.ones(len(frame)), frame["x1"].to_numpy()])
    mu = np.exp(X @ beta)
    pi = 1.0 / (1.0 + np.exp(-(X @ gamma)))
    y = frame["y"].to_numpy()

    poisson_pmf = np.exp(y * np.log(mu) - mu - gammaln(y + 1))
    likelihood = np.where(
        y == 0,
        pi + (1 - pi) * poisson_pmf,
        (1 - pi) * poisson_pmf,
    )
    expected = float(np.sum(np.log(np.maximum(likelihood, 1e-300))))
    assert float(result.e["ll"]) == pytest.approx(expected, rel=1e-8)


def test_zip_recovers_an_inflation_effect():
    frame = _excess_zero_frame()
    result = ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate="x1").fit()
    # The DGP has a constant inflation probability, so the slope in the inflate
    # equation should be near zero and the intercept negative.
    assert abs(float(result.params["inflate:x1"])) < 0.6
    assert float(result.params["inflate:Intercept"]) < 0.0


def test_zinb_runs_on_overdispersed_data():
    frame = _overdispersed_frame()
    result = ZeroInflatedNegativeBinomial.from_formula(
        "y ~ x1", frame, inflate="_cons"
    ).fit()
    assert np.all(np.isfinite(result.params.to_numpy()))
    assert "/lnalpha" in result.param_names


# ------------------------------------------------------- truncated counts


def test_ztp_loglikelihood_matches_the_formula():
    frame = _poisson_frame().query("y > 0").reset_index(drop=True)
    result = TruncatedPoisson.from_formula("y ~ x1", frame, lower=1).fit()

    beta = np.array([float(result.params["Intercept"]), float(result.params["x1"])])
    X = np.column_stack([np.ones(len(frame)), frame["x1"].to_numpy()])
    mu = np.exp(X @ beta)
    y = frame["y"].to_numpy()

    log_pmf = y * np.log(mu) - mu - gammaln(y + 1)
    normaliser = np.log(1.0 - stats.poisson.cdf(0, mu))
    expected = float(np.sum(log_pmf - normaliser))
    assert float(result.e["ll"]) == pytest.approx(expected, rel=1e-8)


def test_ztp_rejects_data_below_the_truncation_point():
    frame = _excess_zero_frame()  # contains zeros
    with pytest.raises(ConsistencyError, match="below the truncation"):
        TruncatedPoisson.from_formula("y ~ x1", frame, lower=1)


def test_ztp_rejects_an_inverted_range():
    frame = _poisson_frame()
    with pytest.raises(ConsistencyError, match="must not"):
        TruncatedPoisson.from_formula("y ~ x1", frame, lower=5, upper=2)


def test_tnbreg_runs_on_truncated_overdispersed_data():
    frame = _overdispersed_frame().query("y > 0").reset_index(drop=True)
    result = TruncatedNegativeBinomial.from_formula("y ~ x1", frame, lower=1).fit()
    assert np.all(np.isfinite(result.params.to_numpy()))
    assert "/lnalpha" in result.param_names
    assert float(result.e["alpha"]) > 0


# --------------------------------------------------------------- plumbing


@pytest.mark.parametrize(
    "model_name, frame, kwargs",
    [
        ("ZeroInflatedPoisson", "excess", {"inflate": "x1"}),
        ("TruncatedPoisson", "positive", {"lower": 1}),
    ],
)
def test_required_e_keys_and_summary(model_name, frame, kwargs):
    data = _excess_zero_frame() if frame == "excess" else _poisson_frame().query("y > 0").reset_index(drop=True)
    model = globals()[model_name].from_formula("y ~ x1", data, **kwargs)
    result = model.fit()

    for key in ["N", "df_m", "ll", "chi2", "r2_p", "rank"]:
        assert key in result.e, key
    assert "Coef." in result.summary(style="stata")
