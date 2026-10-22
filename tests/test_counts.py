"""Conformance tests for zero-inflated and truncated count models.

Two independent kinds of evidence are used, because either one alone is easy to
fool:

1. **Hand-written likelihoods.** Each test re-derives the density from the
   textbook definition (``sum(log(pi*1{y=0} + (1-pi)*poisson_pmf))`` and
   ``sum(log f(k) - log(1 - f(0)))``) and compares against the estimator's own
   log-likelihood to 1e-10.  This catches a wrong parameterisation even when no
   Stata output is available.
2. **Captured StataNow 19.5 output** in
   ``tests/conformance/stata_fixtures/p6p7_outputs.txt``, for the point
   estimates, the log-likelihood and the model-level statistics Stata prints.

The single strongest structural check is the degenerate case: with
``inflate=None`` the zero-inflated model *is* the plain Poisson, so its
coefficients must equal ``westata.Poisson``'s to 1e-6.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, gammaln

from westata.errors import ConsistencyError, ConvergenceError
from westata.estimators.counts import (
    TruncatedNegativeBinomial,
    TruncatedPoisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from westata.estimators.discrete import NegativeBinomial, Poisson

# ------------------------------------------------------------------- fixtures


def count_frame(n: int = 1200, seed: int = 43) -> pd.DataFrame:
    """Stata fixture DGP: Poisson counts with 25% structural zeros."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.4 + 0.6 * x1)
    structural_zero = rng.random(size=n) < 0.25
    counts = np.where(structural_zero, 0, rng.poisson(mu)).astype(float)
    return pd.DataFrame({"y": counts, "x1": x1})


def overdispersed_frame(n: int = 1200, seed: int = 47) -> pd.DataFrame:
    """Stata fixture DGP: NB2 counts with alpha = 0.5 (overdispersed)."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.4 + 0.6 * x1)
    lam = rng.gamma(shape=1.0 / 0.5, scale=mu * 0.5)
    return pd.DataFrame({"y": rng.poisson(lam).astype(float), "x1": x1})


def inflated_frame(n: int = 4000, seed: int = 61) -> pd.DataFrame:
    """Known inflation effect: pi = logistic(-0.4 + 1.1 * z)."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    z = rng.normal(size=n)
    mu = np.exp(0.3 + 0.5 * x1)
    pi = expit(-0.4 + 1.1 * z)
    structural = rng.random(size=n) < pi
    counts = np.where(structural, 0, rng.poisson(mu)).astype(float)
    return pd.DataFrame({"y": counts, "x1": x1, "z": z})


def small_frame(n: int = 80, seed: int = 71) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    z = rng.normal(size=n)
    mu = np.exp(0.2 + 0.4 * x1)
    pi = expit(-0.3 + 0.7 * z)
    counts = np.where(rng.random(size=n) < pi, 0, rng.poisson(mu)).astype(float)
    return pd.DataFrame({"y": counts, "x1": x1, "z": z})


def positive(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.query("y > 0").reset_index(drop=True)


# ------------------------------------------------- hand-written densities (NB2)


def poisson_logpmf(y: float, mu: float) -> float:
    return float(y * np.log(mu) - mu - gammaln(y + 1.0))


def nb_logpmf(y, mu, alpha: float) -> float:
    """NB2 with ``Var = mu + alpha*mu^2`` (Stata's ``alpha``)."""

    size = 1.0 / alpha
    p = size / (size + mu)
    return float(stats.nbinom.logpmf(y, size, p))


def zip_loglike(y, X, Z, theta) -> float:
    """``sum log(pi*1{y=0} + (1-pi)*poisson_pmf)``, evaluated per observation."""

    k = X.shape[1]
    mu = np.exp(X @ theta[:k])
    pi = expit(Z @ theta[k:])
    total = 0.0
    for yi, mui, pii in zip(y, mu, pi, strict=True):
        f = poisson_logpmf(yi, mui)
        if yi == 0:
            total += float(np.log(pii + (1.0 - pii) * np.exp(f)))
        else:
            total += float(np.log1p(-pii)) + f
    return total


def zinb_loglike(y, X, Z, theta) -> float:
    k = X.shape[1]
    mu = np.exp(X @ theta[:k])
    alpha = float(np.exp(theta[-1]))
    pi = expit(Z @ theta[k:-1])
    total = 0.0
    for yi, mui, pii in zip(y, mu, pi, strict=True):
        f = nb_logpmf(yi, mui, alpha)
        if yi == 0:
            total += float(np.log(pii + (1.0 - pii) * np.exp(f)))
        else:
            total += float(np.log1p(-pii)) + f
    return total


def zero_truncated_loglike(y, X, theta, alpha: float | None = None) -> float:
    """``sum [log f(y) - log(1 - f(0))]`` for Poisson or NB2 ``f``."""

    k = X.shape[1] if alpha is None else X.shape[1]
    mu = np.exp(X @ theta[:k])
    total = 0.0
    for yi, mui in zip(y, mu, strict=True):
        if alpha is None:
            f = float(stats.poisson.pmf(yi, mui))
            f0 = float(stats.poisson.pmf(0.0, mui))
        else:
            size = 1.0 / alpha
            p = size / (size + mui)
            f = float(stats.nbinom.pmf(yi, size, p))
            f0 = float(stats.nbinom.pmf(0.0, size, p))
        total += float(np.log(f) - np.log1p(-f0))
    return total


def truncated_loglike(y, X, theta, lower: int, upper: int | None) -> float:
    """Normalise by ``F(upper) - F(lower - 1)`` over the integer support."""

    mu = np.exp(X @ theta)
    total = 0.0
    for yi, mui in zip(y, mu, strict=True):
        f = float(stats.poisson.pmf(yi, mui))
        if upper is None:
            denominator = 1.0 - float(stats.poisson.cdf(lower - 1, mui))
        else:
            denominator = float(stats.poisson.cdf(upper, mui)) - float(
                stats.poisson.cdf(lower - 1, mui)
            )
        total += float(np.log(f) - np.log(denominator))
    return total


# ------------------------------------- zero inflation: the degenerate case


def test_zip_without_inflate_reproduces_poisson_coefficients():
    """Primary regression: no inflation equation -> exactly the plain Poisson."""

    frame = count_frame(n=600)
    zipped = ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate=None).fit()
    plain = Poisson.from_formula("y ~ x1", frame).fit()

    np.testing.assert_allclose(
        zipped.params.to_numpy(), plain.params.to_numpy(), atol=1e-6
    )
    np.testing.assert_allclose(zipped.bse.to_numpy(), plain.bse.to_numpy(), atol=1e-5)
    assert zipped.e["ll"] == pytest.approx(plain.e["ll"], abs=1e-8)
    assert zipped.e["ll_0"] == pytest.approx(plain.e["ll_0"], abs=1e-8)


def test_zinb_without_inflate_reproduces_nbreg_coefficients():
    frame = overdispersed_frame(n=800)
    zipped = ZeroInflatedNegativeBinomial.from_formula("y ~ x1", frame).fit()
    plain = NegativeBinomial.from_formula("y ~ x1", frame).fit()

    np.testing.assert_allclose(
        zipped.params.to_numpy(), plain.params.to_numpy(), atol=1e-4
    )


# ------------------------------------------------ zero inflation: likelihood


def test_zip_loglike_matches_hand_written_formula():
    frame = small_frame()
    model = ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate="z")
    y, X, Z = frame["y"].to_numpy(), model._X, model._inflate_exog

    theta = model.start_params()
    assert model._loglike(theta) == pytest.approx(
        zip_loglike(y, X, Z, theta), abs=1e-10
    )

    # A second, non-trivial point exercises both branches of the mixture.
    theta2 = theta + np.array([0.3, -0.2, 0.5, -0.4])
    assert model._loglike(theta2) == pytest.approx(
        zip_loglike(y, X, Z, theta2), abs=1e-10
    )


def test_zip_loglike_at_the_estimate_matches_hand_written_formula():
    """The reported ``e(ll)``, not just an arbitrary point, must be the formula."""

    frame = count_frame(n=400)
    result = ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate="x1").fit()
    model = ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate="x1")

    X = np.column_stack([np.ones(len(frame)), frame["x1"].to_numpy()])
    expected = zip_loglike(
        frame["y"].to_numpy(), X, X, result.params.to_numpy()
    )
    assert result.e["ll"] == pytest.approx(expected, abs=1e-8)


def test_zinb_loglike_matches_hand_written_formula():
    frame = small_frame()
    model = ZeroInflatedNegativeBinomial.from_formula("y ~ x1", frame, inflate="z")
    y, X, Z = frame["y"].to_numpy(), model._X, model._inflate_exog

    theta = model.start_params()
    assert model._loglike(theta) == pytest.approx(
        zinb_loglike(y, X, Z, theta), abs=1e-10
    )


# ------------------------------------------ zero inflation: effect recovery


def test_zip_recovers_the_inflation_effect():
    """A DGP with a known, x-dependent zero process must be recovered."""

    frame = inflated_frame()
    result = ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate="z").fit()

    params = result.params
    # Main equation: mu = exp(0.3 + 0.5 x1).
    assert params["Intercept"] == pytest.approx(0.3, abs=0.08)
    assert params["x1"] == pytest.approx(0.5, abs=0.08)
    # Inflation equation: pi = logistic(-0.4 + 1.1 z).
    assert params["inflate:Intercept"] == pytest.approx(-0.4, abs=0.12)
    assert params["inflate:z"] == pytest.approx(1.1, abs=0.12)


def test_zip_inflate_terms_are_labelled_like_stata():
    result = ZeroInflatedPoisson.from_formula(
        "y ~ x1", inflated_frame(n=600), inflate="z"
    ).fit()
    assert result.param_names == [
        "Intercept",
        "x1",
        "inflate:Intercept",
        "inflate:z",
    ]


def test_zip_inflation_improves_the_likelihood_over_plain_poisson():
    frame = inflated_frame(n=1500)
    plain = ZeroInflatedPoisson.from_formula("y ~ x1", frame).fit()
    inflated = ZeroInflatedPoisson.from_formula("y ~ x1", frame, inflate="z").fit()
    assert inflated.e["ll"] > plain.e["ll"]


def test_zip_matches_the_stata_fixture():
    result = ZeroInflatedPoisson.from_formula(
        "y ~ x1", count_frame(), inflate="x1"
    ).fit()

    # StataNow 19.5: zip y x1, inflate(x1)
    #   main  x1 .559456, _cons .4039659
    #   inflate x1 .0033665, _cons -1.016455
    #   log likelihood -1700.705, LR chi2(1) 248.04
    assert result.params["x1"] == pytest.approx(0.559456, rel=1e-4)
    assert result.params["Intercept"] == pytest.approx(0.4039659, rel=1e-4)
    assert result.params["inflate:x1"] == pytest.approx(0.0033665, abs=1e-3)
    assert result.params["inflate:Intercept"] == pytest.approx(-1.016455, rel=1e-4)
    assert result.e["ll"] == pytest.approx(-1700.705, rel=1e-6)
    assert result.e["ll_0"] == pytest.approx(-1824.7258, rel=1e-5)
    assert result.e["chi2"] == pytest.approx(248.04, rel=1e-3)
    # Stata prints LR chi2(1): the inflation equation does not add model df.
    assert result.e["df_m"] == 1
    assert result.e["rank"] == 4


def test_zinb_matches_the_stata_fixture():
    result = ZeroInflatedNegativeBinomial.from_formula(
        "y ~ x1", overdispersed_frame(), inflate="_cons"
    ).fit()

    # StataNow 19.5: zinb y x1, inflate(_cons)
    #   main x1 .5829196, _cons .4350183, /lnalpha -.6779927
    #   log likelihood -2028.728, LR chi2(1) 293.89
    assert result.params["x1"] == pytest.approx(0.5829196, rel=1e-4)
    assert result.params["Intercept"] == pytest.approx(0.4350183, rel=1e-4)
    assert result.params["/lnalpha"] == pytest.approx(-0.6779927, rel=1e-3)
    assert result.e["ll"] == pytest.approx(-2028.728, rel=1e-5)
    assert result.e["ll_0"] == pytest.approx(-2175.6742, rel=1e-5)
    assert result.e["chi2"] == pytest.approx(293.89, rel=1e-3)
    assert result.e["df_m"] == 1


def test_zinb_runs_on_overdispersed_data_with_excess_zeros():
    rng = np.random.default_rng(83)
    n = 1500
    x1 = rng.normal(size=n)
    mu = np.exp(0.4 + 0.5 * x1)
    lam = rng.gamma(shape=1.0 / 0.5, scale=mu * 0.5)
    counts = rng.poisson(lam).astype(float)
    counts[rng.random(size=n) < 0.2] = 0.0
    frame = pd.DataFrame({"y": counts, "x1": x1})

    result = ZeroInflatedNegativeBinomial.from_formula(
        "y ~ x1", frame, inflate="_cons"
    ).fit()
    assert np.all(np.isfinite(result.params.to_numpy()))
    assert "/lnalpha" in result.param_names
    assert result.params["x1"] == pytest.approx(0.5, abs=0.15)


# -------------------------------------------------------------- zero truncated


def test_truncated_poisson_loglike_matches_hand_written_zero_truncated_formula():
    frame = positive(count_frame(n=150))
    model = TruncatedPoisson.from_formula("y ~ x1", frame, lower=1)

    theta = np.array([0.4, 0.6])
    expected = zero_truncated_loglike(frame["y"].to_numpy(), model._X, theta)
    assert model._loglike(theta) == pytest.approx(expected, abs=1e-10)


def test_truncated_poisson_reported_ll_matches_hand_written_formula():
    frame = positive(count_frame())
    result = TruncatedPoisson.from_formula("y ~ x1", frame, lower=1).fit()

    X = np.column_stack([np.ones(len(frame)), frame["x1"].to_numpy()])
    expected = zero_truncated_loglike(
        frame["y"].to_numpy(), X, result.params.to_numpy()
    )
    assert result.e["ll"] == pytest.approx(expected, abs=1e-8)


def test_truncated_poisson_matches_the_stata_ztp_fixture():
    frame = positive(count_frame())
    result = TruncatedPoisson.from_formula("y ~ x1", frame, lower=1).fit()

    # StataNow 19.5: ztp y x1
    #   x1 .5415945, _cons .4189287, log likelihood -916.60416
    #   LR chi2(1) 270.42, Pseudo R2 .1286
    assert result.model_name == "ztp"
    assert result.params["x1"] == pytest.approx(0.5415945, rel=1e-5)
    assert result.params["Intercept"] == pytest.approx(0.4189287, rel=1e-5)
    assert result.e["ll"] == pytest.approx(-916.60416, rel=1e-6)
    assert result.e["chi2"] == pytest.approx(270.42, rel=1e-4)
    assert result.e["r2_p"] == pytest.approx(0.1286, abs=5e-4)
    # Standard errors are the observed information, as Stata reports them.
    assert result.bse["x1"] == pytest.approx(0.0350626, rel=1e-3)
    assert result.bse["Intercept"] == pytest.approx(0.04102, rel=1e-3)


def test_truncated_poisson_general_upper_limit():
    frame = count_frame(n=250)
    frame = frame.loc[(frame["y"] >= 1) & (frame["y"] <= 4)].reset_index(drop=True)
    model = TruncatedPoisson.from_formula("y ~ x1", frame, lower=1, upper=4)

    theta = np.array([0.4, 0.6])
    expected = truncated_loglike(frame["y"].to_numpy(), model._X, theta, 1, 4)
    assert model._loglike(theta) == pytest.approx(expected, abs=1e-10)

    result = model.fit()
    assert np.all(np.isfinite(result.params.to_numpy()))
    # A general limit is Stata's tpoisson, not ztp.
    assert result.model_name == "tpoisson"
    assert np.allclose(result.fittedvalues, result.fittedvalues)


def test_truncated_poisson_general_lower_limit_is_tpoisson():
    frame = count_frame(n=300)
    frame = frame.loc[frame["y"] >= 2].reset_index(drop=True)
    result = TruncatedPoisson.from_formula("y ~ x1", frame, lower=2).fit()
    assert result.model_name == "tpoisson"
    assert result.e["truncation_point"] == 2.0


def test_truncated_poisson_rejects_data_below_the_lower_limit():
    frame = count_frame(n=200)
    assert frame["y"].min() == 0.0  # the data violate the truncation
    with pytest.raises(ConsistencyError, match="truncation"):
        TruncatedPoisson.from_formula("y ~ x1", frame, lower=1)


def test_truncated_poisson_rejects_data_above_the_upper_limit():
    frame = positive(count_frame(n=200))
    assert frame["y"].max() > 3
    with pytest.raises(ConsistencyError, match="truncation"):
        TruncatedPoisson.from_formula("y ~ x1", frame, lower=1, upper=3)


def test_truncated_models_reject_inverted_limits():
    frame = positive(count_frame(n=100))
    with pytest.raises(ConsistencyError, match="must not exceed"):
        TruncatedPoisson.from_formula("y ~ x1", frame, lower=4, upper=2)


def test_truncated_models_reject_non_integer_outcomes():
    frame = positive(count_frame(n=100)).copy()
    frame.loc[0, "y"] = 1.5
    with pytest.raises(ConsistencyError, match="integer"):
        TruncatedPoisson.from_formula("y ~ x1", frame, lower=1)


def test_truncated_negative_binomial_matches_the_stata_tnbreg_fixture():
    result = TruncatedNegativeBinomial.from_formula(
        "y ~ x1", positive(overdispersed_frame()), lower=1
    ).fit()

    # StataNow 19.5: tnbreg y x1, ll(0)
    #   x1 .6035713, _cons .3967922, /lnalpha -.5884602
    #   log likelihood -1321.6972, LR chi2(1) 171.80, Pseudo R2 .0610
    assert result.model_name == "tnbreg"
    assert result.params["x1"] == pytest.approx(0.6035713, rel=1e-4)
    assert result.params["Intercept"] == pytest.approx(0.3967922, rel=1e-3)
    assert result.params["/lnalpha"] == pytest.approx(-0.5884602, rel=2e-3)
    assert result.e["ll"] == pytest.approx(-1321.6972, rel=1e-6)
    assert result.e["chi2"] == pytest.approx(171.80, rel=1e-3)
    assert result.e["r2_p"] == pytest.approx(0.0610, abs=5e-4)
    assert result.e["alpha"] == pytest.approx(np.exp(-0.5884602), rel=2e-3)


def test_truncated_negative_binomial_loglike_matches_hand_written_formula():
    frame = positive(overdispersed_frame(n=200))
    model = TruncatedNegativeBinomial.from_formula("y ~ x1", frame, lower=1)

    theta = np.array([0.4, 0.5, np.log(0.5)])
    expected = zero_truncated_loglike(
        frame["y"].to_numpy(), model._X, theta, alpha=0.5
    )
    assert model._loglike(theta) == pytest.approx(expected, abs=1e-10)


def test_truncated_negative_binomial_produces_finite_parameters():
    frame = positive(overdispersed_frame())
    result = TruncatedNegativeBinomial.from_formula("y ~ x1", frame, lower=1).fit()
    assert np.all(np.isfinite(result.params.to_numpy()))
    assert np.all(np.isfinite(result.bse.to_numpy()))
    assert np.all(result.bse.to_numpy() > 0)
    assert result.e["alpha"] > 0


# ---------------------------------------------------------------- reporting


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(
            lambda frame: ZeroInflatedPoisson.from_formula("y ~ x1", frame),
            id="zip-no-inflate",
        ),
        pytest.param(
            lambda frame: ZeroInflatedPoisson.from_formula(
                "y ~ x1", frame, inflate="z"
            ),
            id="zip-inflate",
        ),
        pytest.param(
            lambda frame: ZeroInflatedNegativeBinomial.from_formula(
                "y ~ x1", frame, inflate="z"
            ),
            id="zinb",
        ),
    ],
)
def test_e_keys_present_for_zero_inflated_models(builder):
    result = builder(inflated_frame(n=800)).fit()
    for key in ("N", "df_m", "ll", "chi2", "p", "r2_p", "rank"):
        assert key in result.e, key
    assert result.dist == "z"
    assert result.e["N"] == result.nobs
    assert result.e["rank"] == len(result.params)
    assert np.isfinite(result.e["ll"])
    assert np.isfinite(result.e["chi2"])
    assert 0.0 <= result.e["p"] <= 1.0


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(
            lambda frame: TruncatedPoisson.from_formula("y ~ x1", frame, lower=1),
            id="ztp",
        ),
        pytest.param(
            lambda frame: TruncatedNegativeBinomial.from_formula(
                "y ~ x1", frame, lower=1
            ),
            id="tnbreg",
        ),
    ],
)
def test_e_keys_present_for_truncated_models(builder):
    result = builder(positive(count_frame(n=800))).fit()
    for key in ("N", "df_m", "ll", "chi2", "p", "r2_p", "rank"):
        assert key in result.e, key
    assert result.dist == "z"
    assert result.e["N"] == result.nobs
    assert result.e["rank"] == len(result.params)


def test_stata_summary_renders_for_every_model():
    inflated = inflated_frame(n=800)
    truncated = positive(count_frame(n=800))

    results = [
        ZeroInflatedPoisson.from_formula("y ~ x1", inflated).fit(),
        ZeroInflatedPoisson.from_formula("y ~ x1", inflated, inflate="z").fit(),
        ZeroInflatedNegativeBinomial.from_formula("y ~ x1", inflated).fit(),
        TruncatedPoisson.from_formula("y ~ x1", truncated, lower=1).fit(),
        TruncatedNegativeBinomial.from_formula("y ~ x1", truncated, lower=1).fit(),
    ]
    for result in results:
        text = result.summary(style="stata")
        assert "Number of obs" in text
        assert "P>|z|" in text
        assert "P>|t|" not in text
        assert "LR chi2(" in text
        assert "Pseudo R2" in text
        assert "Log likelihood" in text
        assert "Coef." in text


def test_pseudo_r2_is_mcfadden_against_the_constant_only_model():
    result = TruncatedPoisson.from_formula(
        "y ~ x1", positive(count_frame()), lower=1
    ).fit()
    expected = 1.0 - result.e["ll"] / result.e["ll_0"]
    assert result.e["r2_p"] == pytest.approx(expected, rel=1e-12)


def test_fitted_values_are_positive_and_finite():
    result = ZeroInflatedPoisson.from_formula(
        "y ~ x1", count_frame(n=400), inflate="x1"
    ).fit()
    assert np.all(np.isfinite(result.fittedvalues))
    assert np.all(result.fittedvalues > 0)


# -------------------------------------------------------------- guard rails


def test_non_convergence_raises_with_strict():
    model = ZeroInflatedPoisson.from_formula(
        "y ~ x1", inflated_frame(n=400), inflate="z"
    )
    with pytest.raises(ConvergenceError):
        model.fit(strict=True, maxiter=1)


def test_non_convergence_without_strict_only_warns():
    model = TruncatedNegativeBinomial.from_formula(
        "y ~ x1", positive(overdispersed_frame(n=300)), lower=1
    )
    with pytest.warns(RuntimeWarning):
        model.fit(maxiter=1)


def test_unsupported_cov_type_raises():
    model = ZeroInflatedPoisson.from_formula("y ~ x1", count_frame(n=200))
    with pytest.raises(ValueError, match="cov_type"):
        model.fit(cov_type="bogus")


def test_inflate_must_name_existing_columns():
    with pytest.raises(Exception, match="available columns|not_in_frame"):
        ZeroInflatedPoisson.from_formula(
            "y ~ x1", count_frame(n=200), inflate="not_in_frame"
        )


def test_missing_lhs_column_raises():
    with pytest.raises(Exception, match="not_a_column"):
        TruncatedPoisson.from_formula(
            "not_a_column ~ x1", positive(count_frame(n=200)), lower=1
        )
