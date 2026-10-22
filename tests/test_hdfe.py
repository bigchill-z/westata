"""Conformance tests for :mod:`westata.estimators.hdfe` (Stata ``reghdfe``/``areg``).

Two independent references are used:

* explicit dummy-variable OLS via statsmodels (the definition of a fixed-effects
  estimator: they are just dummies),
* iterative two-way demeaning + OLS on the demeaned data (Frisch-Waugh-Lovell,
  the algebra ``reghdfe``/``PyFixest`` actually use).

The covariance tests cross-validate against :func:`westata.vce.compute_cov`.  For
HDFE the ``(N-1)/(N-K)`` factor must count the absorbed fixed-effect parameters,
so the westata CR1 matrix -- built on the *demeaned* design, whose column count
only counts reported regressors -- is rescaled to the same ``K`` before
comparing.  That rescaling is the substantive difference between a demeaned OLS
and ``reghdfe``, so it is asserted explicitly rather than hidden.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from westata.data.weights import normalize_weights
from westata.errors import WeightError
from westata.estimators.hdfe import FEOLS
from westata.estimators.linear import OLS
from westata.vce.dispatch import compute_cov


# --------------------------------------------------------------------------- data
def _panel(n_firms: int = 10, n_years: int = 6, seed: int = 0) -> pd.DataFrame:
    """Balanced two-way panel with firm and year effects, 5 clusters of 2 firms."""

    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(n_years), n_firms)
    n = firm.size
    alpha = rng.normal(scale=2.0, size=n_firms)[firm]
    gamma = rng.normal(size=n_years)[year]
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.5 * x1 - 0.6 * x2 + alpha + gamma + rng.normal(scale=0.4, size=n)
    return pd.DataFrame(
        {
            "y": y,
            "x1": x1,
            "x2": x2,
            "firm": firm,
            "year": year,
            "cluster": firm // 2,
        }
    )


def _dummy_design(df: pd.DataFrame, fe: list[str], regressors: list[str]):
    """``[const, regressors..., dummies(fe, drop_first)]`` for statsmodels."""

    dummies = pd.get_dummies(df[fe].astype("category"), drop_first=True).astype(float)
    X = np.column_stack(
        [np.ones(len(df)), df[regressors].to_numpy(dtype=float), dummies.to_numpy()]
    )
    return X, dummies.shape[1]


def _demean(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Remove one-way group means (the ``areg``/one-way within transform)."""

    series = pd.Series(np.asarray(values, dtype=float))
    means = series.groupby(pd.Series(group)).transform("mean").to_numpy()
    return np.asarray(values, dtype=float) - means


def _demean_two_way(values: np.ndarray, fe: list[str], df: pd.DataFrame,
                    tol: float = 1e-15, maxiter: int = 10_000) -> np.ndarray:
    """Iterative two-way demeaning (alternating projections), self-contained.

    Not reusing the estimator's internals keeps this a real reference: it uses
    only pandas groupby means plus the FWL theorem.
    """

    out = np.array(values, dtype=float, copy=True)
    groups = [df[name].to_numpy() for name in fe]
    for _ in range(maxiter):
        previous = out.copy()
        for group in groups:
            out = _demean(out, group)
        if np.max(np.abs(out - previous)) < tol:
            return out
    raise AssertionError("two-way demeaning did not converge in the reference")


# ------------------------------------------------------- 1. one fixed effect
def test_single_fixed_effect_matches_dummy_variable_ols():
    df = _panel()
    X, _ = _dummy_design(df, ["firm"], ["x1", "x2"])
    ref = sm.OLS(df["y"], X).fit()

    res = FEOLS.from_formula("y ~ x1 + x2 | firm", df).fit()

    assert list(res.params.index) == ["x1", "x2"]
    np.testing.assert_allclose(
        res.params.to_numpy(), ref.params[1:3], rtol=0, atol=1e-10
    )
    # Dummy OLS has N - (2 slopes + 9 dummies + const) = N - 12 residual df.
    assert res.df_resid == ref.df_resid


# ------------------------------------------------- 2. two-way fixed effects
def test_two_way_fixed_effects_match_two_way_demeaning():
    df = _panel()
    y_tilde = _demean_two_way(df["y"].to_numpy(), ["firm", "year"], df)
    x_tilde = np.column_stack(
        [
            _demean_two_way(df["x1"].to_numpy(), ["firm", "year"], df),
            _demean_two_way(df["x2"].to_numpy(), ["firm", "year"], df),
        ]
    )

    beta, *_ = np.linalg.lstsq(x_tilde, y_tilde, rcond=None)
    res = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit()

    np.testing.assert_allclose(res.params.to_numpy(), beta, rtol=0, atol=1e-10)
    assert res.df_resid == len(df) - (2 + 10 + 6 - 1)


# ------------------------------------------------ 3. no fixed effects -> OLS
def test_no_fixed_effects_matches_plain_ols():
    df = _panel()
    ref = OLS.from_formula("y ~ x1 + x2", df).fit()

    res = FEOLS.from_formula("y ~ x1 + x2", df).fit()

    assert list(res.params.index) == ["Intercept", "x1", "x2"]
    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=0, atol=1e-12)
    assert res.df_resid == ref.df_resid == len(df) - 3
    assert res.e["df_m"] == 2


# ------------------------------------------------------- 4. cluster-robust
def test_cluster_inference_uses_g_minus_one_and_finite_standard_errors():
    df = _panel()
    n_clusters = df["cluster"].nunique()

    res = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit(
        cov_type="cluster", cov_kwds={"cluster": "cluster"}
    )

    assert res.df_resid == n_clusters - 1
    assert res.e["N_clust"] == n_clusters
    # Residual df stays N - K (Stata's e(df_r)); the two must differ here.
    assert res.e["df_r"] == len(df) - res.e["rank"]
    assert res.e["df_r"] != res.df_resid

    bse = res.bse.to_numpy()
    assert np.all(np.isfinite(bse)) and np.all(bse > 0)
    assert np.all(np.isfinite(res.pvalues.to_numpy()))


def test_cluster_covariance_is_westata_cr1_scaled_to_total_parameter_count():
    df = _panel()
    res = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit(
        cov_type="cluster", cov_kwds={"cluster": "cluster"}
    )

    # The demeaned design has only the 2 reported columns, but reghdfe's
    # (N-1)/(N-K) correction counts the 15 absorbed parameters as well.
    y_tilde = _demean_two_way(df["y"].to_numpy(), ["firm", "year"], df)
    x_tilde = np.column_stack(
        [
            _demean_two_way(df["x1"].to_numpy(), ["firm", "year"], df),
            _demean_two_way(df["x2"].to_numpy(), ["firm", "year"], df),
        ]
    )
    beta, *_ = np.linalg.lstsq(x_tilde, y_tilde, rcond=None)
    resid = y_tilde - x_tilde @ beta

    raw, _ = compute_cov(
        x_tilde, resid, cov_type="cluster", cov_kwds={"groups": df["cluster"]}
    )
    n, n_columns, k_total = len(df), x_tilde.shape[1], int(res.e["rank"])
    expected = raw * (n - n_columns) / (n - k_total)

    np.testing.assert_allclose(res.cov, expected, rtol=0, atol=1e-12)


def test_nonrobust_covariance_matches_westata_iid_with_absorbed_df():
    df = _panel()
    res = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit()

    y_tilde = _demean_two_way(df["y"].to_numpy(), ["firm", "year"], df)
    x_tilde = np.column_stack(
        [
            _demean_two_way(df["x1"].to_numpy(), ["firm", "year"], df),
            _demean_two_way(df["x2"].to_numpy(), ["firm", "year"], df),
        ]
    )
    beta, *_ = np.linalg.lstsq(x_tilde, y_tilde, rcond=None)
    resid = y_tilde - x_tilde @ beta

    expected, df_t = compute_cov(
        x_tilde, resid, cov_type="nonrobust", df_resid=len(df) - int(res.e["rank"])
    )
    assert df_t == res.df_resid
    np.testing.assert_allclose(res.cov, expected, rtol=0, atol=1e-12)


def test_hc1_scaling_uses_total_parameter_count():
    df = _panel()
    res = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit(cov_type="HC1")

    n, k_total = len(df), int(res.e["rank"])
    # HC1 = N/(N-K) * HC0, with K counting absorbed fixed effects.
    assert res.cov_type == "HC1"
    assert res.df_resid == n - k_total


# ---------------------------------------------------------------- 5. e() keys
def test_e_contains_all_required_scalars():
    df = _panel()
    res = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit()

    for key in ("N", "df_m", "df_r", "r2", "r2_a", "rmse", "F", "ll", "rank"):
        assert key in res.e, key
        assert np.isfinite(float(res.e[key])), key

    assert res.e["N"] == len(df)
    assert res.e["df_m"] == 2
    assert res.e["rank"] == 2 + 10 + 6 - 1
    assert res.e["df_r"] == len(df) - res.e["rank"]
    assert res.e["df_r"] == res.df_resid
    # rmse is the residual standard deviation on N-K (not sqrt(rss/N)).
    resid = res.resid
    assert res.e["rmse"] == pytest.approx(np.sqrt(resid @ resid / res.e["df_r"]), rel=1e-12)
    assert res.e["r2"] == pytest.approx(res.rsquared)
    assert res.e["r2_a"] == pytest.approx(res.rsquared_adj)
    # Adjusted R^2 uses the same df as the variance estimator.
    assert res.e["r2_a"] == pytest.approx(
        1 - (1 - res.e["r2"]) * (len(df) - 1) / res.e["df_r"], rel=1e-12
    )


def test_goodness_of_fit_scalars_do_not_depend_on_the_covariance():
    # r2/rmse/df_r/r2_a are sample statistics: clustering changes the t/F
    # reference df (G - 1) but must not rescale rmse or the adjusted R^2.
    # The cluster variable must not be nested in a fixed effect, otherwise the
    # absorbed parameters stop counting towards K (tested separately below).
    df = _panel()
    df["cluster"] = np.random.default_rng(7).integers(0, 8, size=len(df))
    plain = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit()
    clustered = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit(
        cov_type="cluster", cov_kwds={"cluster": "cluster"}
    )

    assert clustered.df_resid != plain.df_resid
    for key in ("r2", "r2_a", "rmse", "df_r", "N", "rank"):
        assert clustered.e[key] == pytest.approx(plain.e[key], rel=1e-12), key
    # rmse is a residual standard deviation on N - K, never sqrt(rss / N).
    resid = plain.resid
    assert plain.e["rmse"] > np.sqrt(resid @ resid / plain.nobs)


def test_fixed_effect_nested_in_the_cluster_is_not_counted_in_the_df():
    # reghdfe's default `dofadjustments(nested)`: a fixed effect fully nested in
    # the cluster variable contributes no parameters to the (N-1)/(N-K) factor,
    # because the cluster absorbs it.  PyFixest's k_fixef="nonnested" does the
    # same, so e(rank) and e(df_r) shrink here while the slopes stay identical.
    df = _panel()  # cluster == firm // 2 is nested within the firm effect
    plain = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit()
    nested = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit(
        cov_type="cluster", cov_kwds={"cluster": "cluster"}
    )

    assert nested.e["rank"] < plain.e["rank"]
    assert nested.e["df_r"] == len(df) - nested.e["rank"]
    np.testing.assert_allclose(nested.params.to_numpy(), plain.params.to_numpy(),
                               rtol=0, atol=1e-10)


def test_f_statistic_is_the_wald_test_of_the_slopes():
    df = _panel()
    res = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit()

    b = res.params.to_numpy()
    V = res.cov
    expected = float(b @ np.linalg.solve(V, b) / 2)
    assert res.e["F"] == pytest.approx(expected, rel=1e-10)
    assert res.fvalue == pytest.approx(expected, rel=1e-10)


# ------------------------------------------------------------- 6. summary
def test_stata_summary_renders_coef_and_model_name():
    df = _panel()
    text = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit().summary(style="stata")

    for token in ("Coef.", "Std. err.", "P>|t|", "Number of obs", "reghdfe"):
        assert token in text


# ------------------------------------------------- 7. absorbed regressors
def test_absorbed_regressor_is_reported_and_warned():
    df = _panel()
    # Constant within firm: fully absorbed by the firm fixed effect.
    df["firm_mean"] = df.groupby("firm")["x1"].transform("mean")

    with pytest.warns(RuntimeWarning, match="firm_mean"):
        res = FEOLS.from_formula("y ~ x1 + firm_mean | firm", df).fit()

    assert res.absorbed == ["firm_mean"]
    assert "firm_mean" not in res.params.index


def test_collinear_duplicate_regressor_is_reported():
    df = _panel()
    df["x1_copy"] = df["x1"]

    with pytest.warns(RuntimeWarning, match="x1_copy"):
        res = FEOLS.from_formula("y ~ x1 + x1_copy | firm", df).fit()

    assert res.absorbed == ["x1_copy"]


def test_no_absorbed_columns_is_an_empty_list():
    df = _panel()
    assert FEOLS.from_formula("y ~ x1 | firm", df).fit().absorbed == []


# ------------------------------------------------------- 8. error handling
@pytest.mark.parametrize("cov_type", ["HAC", "twoway", "robust", "bogus"])
def test_unknown_cov_type_raises_value_error(cov_type):
    df = _panel()
    with pytest.raises(ValueError, match="cov_type"):
        FEOLS.from_formula("y ~ x1 | firm", df).fit(cov_type=cov_type)


@pytest.mark.parametrize("cov_type", ["HC2", "HC3"])
def test_hc2_hc3_are_rejected_rather_than_silently_approximated(cov_type):
    # PyFixest refuses these for fixed-effect models: the leverage of the full
    # dummy design is required and not available.  reghdfe does not offer them
    # either, so westata must say so instead of returning an approximation.
    df = _panel()
    with pytest.raises(ValueError, match=cov_type):
        FEOLS.from_formula("y ~ x1 | firm", df).fit(cov_type=cov_type)


def test_cluster_requires_cluster_labels():
    df = _panel()
    with pytest.raises(ValueError, match="cluster"):
        FEOLS.from_formula("y ~ x1 | firm", df).fit(cov_type="cluster")


def test_cov_type_is_case_insensitive():
    df = _panel()
    lower = FEOLS.from_formula("y ~ x1 + x2 | firm", df).fit(cov_type="cluster",
                                                             cov_kwds={"cluster": "cluster"})
    upper = FEOLS.from_formula("y ~ x1 + x2 | firm", df).fit(cov_type="CLUSTER",
                                                             cov_kwds={"cluster": "cluster"})
    np.testing.assert_allclose(lower.cov, upper.cov, rtol=0, atol=0)


# --------------------------------------------------------------- 9. weights
def test_weights_require_an_explicit_weight_type():
    df = _panel()
    with pytest.raises(WeightError):
        FEOLS.from_formula("y ~ x1 | firm", df, weights=np.ones(len(df)))


def test_aweight_matches_statsmodels_wls_with_dummies():
    df = _panel()
    rng = np.random.default_rng(99)
    df["w"] = rng.uniform(0.5, 2.0, size=len(df))

    X, _ = _dummy_design(df, ["firm"], ["x1", "x2"])
    ref = sm.WLS(df["y"], X, weights=df["w"]).fit()

    res = FEOLS.from_formula(
        "y ~ x1 + x2 | firm", df, weights=df["w"], weight_type="aweight"
    ).fit()

    np.testing.assert_allclose(res.params.to_numpy(), ref.params[1:3], rtol=0, atol=1e-10)
    # aweights keep the observation count as the effective sample size.
    assert res.e["N"] == len(df)


def test_pweight_defaults_to_robust_inference():
    df = _panel()
    df["w"] = np.random.default_rng(5).uniform(0.5, 2.0, size=len(df))

    res = FEOLS.from_formula(
        "y ~ x1 + x2 | firm", df, weights=df["w"], weight_type="pweight"
    ).fit()
    assert res.cov_type == "HC1"


# ----------------------------------------------------- 10. misc invariants
def test_residuals_and_fitted_values_reconstruct_the_dependent_variable():
    df = _panel()
    res = FEOLS.from_formula("y ~ x1 + x2 | firm + year", df).fit()
    assert res.nobs == len(df)
    np.testing.assert_allclose(
        res.fittedvalues + res.resid, df["y"].to_numpy(), rtol=0, atol=1e-8
    )


def test_dist_is_t_and_model_name_is_reghdfe():
    df = _panel()
    res = FEOLS.from_formula("y ~ x1 | firm", df).fit()
    assert res.dist == "t"
    assert res.model_name == "reghdfe"
