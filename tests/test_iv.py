"""IV conformance tests against linearmodels (same algebra as Stata ``ivregress``).

The reference implementation is linearmodels' ``IV2SLS``/``IVLIML``/``IVGMM``,
which solves the same estimating equations Stata's ``ivregress`` does.  Point
estimates and covariances are therefore comparable to machine precision; the
Stata-specific layer tested here is the degrees-of-freedom convention, the
``e()`` scalars and the ``cov_type`` mapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from linearmodels.iv import IV2SLS as LMIV2SLS
from linearmodels.iv import IVGMM as LMIVGMM
from linearmodels.iv import IVLIML as LMIVLIML

from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.iv import IV2SLS, IVGMM, IVLIML
from westata.protocols import SSC

JUST = "y ~ 1 + x2 + [x ~ z1]"
OVER = "y ~ 1 + x2 + [x ~ z1 + z2]"


def _just_identified(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """One endogenous regressor, one instrument (exactly identified)."""

    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    v = rng.normal(size=n)
    u = 0.6 * v + rng.normal(size=n)
    x = 0.8 * z1 + v
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x - 1.0 * x2 + u
    return pd.DataFrame({"y": y, "x": x, "x2": x2, "z1": z1})


def _over_identified(n: int = 300, seed: int = 1) -> pd.DataFrame:
    """One endogenous regressor, two instruments (over-identified)."""

    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    v = rng.normal(size=n)
    u = 0.6 * v + rng.normal(size=n)
    x = 0.8 * z1 + 0.3 * z2 + v
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x - 1.0 * x2 + u
    return pd.DataFrame({"y": y, "x": x, "x2": x2, "z1": z1, "z2": z2})


def _reference(cls, formula: str, df: pd.DataFrame, **fit_kw):
    return cls.from_formula(formula, df).fit(**fit_kw)


# --------------------------------------------------------------------------
# Point estimates
# --------------------------------------------------------------------------
def test_2sls_exactly_identified_matches_linearmodels():
    df = _just_identified()
    res = IV2SLS.from_formula(JUST, df).fit()
    ref = _reference(LMIV2SLS, JUST, df, cov_type="unadjusted", debiased=True)
    np.testing.assert_allclose(
        res.params.to_numpy(), np.asarray(ref.params), rtol=1e-12, atol=1e-12
    )
    assert res.param_names == list(ref.params.index)


def test_2sls_over_identified_matches_linearmodels():
    df = _over_identified()
    res = IV2SLS.from_formula(OVER, df).fit()
    ref = _reference(LMIV2SLS, OVER, df, cov_type="unadjusted", debiased=True)
    np.testing.assert_allclose(
        res.params.to_numpy(), np.asarray(ref.params), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize(
    ("westata_cls", "lm_cls"),
    [(IVLIML, LMLIML := LMIVLIML), (IVGMM, LMGMM := LMIVGMM)],
)
def test_liml_and_gmm_match_linearmodels(westata_cls, lm_cls):
    df = _over_identified()
    res = westata_cls.from_formula(OVER, df).fit()
    ref = _reference(lm_cls, OVER, df, cov_type="unadjusted", debiased=True)
    np.testing.assert_allclose(
        res.params.to_numpy(), np.asarray(ref.params), rtol=1e-10, atol=1e-10
    )


def test_model_without_exogenous_regressors():
    df = _just_identified(seed=4)
    formula = "y ~ 0 + [x ~ z1]"
    res = IV2SLS.from_formula(formula, df).fit()
    ref = _reference(LMIV2SLS, formula, df, cov_type="unadjusted", debiased=False)
    np.testing.assert_allclose(res.params.to_numpy(), np.asarray(ref.params), rtol=1e-12)
    assert res.e["df_m"] == 1


# --------------------------------------------------------------------------
# Covariance mapping
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("westata_cls", "lm_cls"),
    [(IV2SLS, LMIV2SLS), (IVLIML, LMIVLIML), (IVGMM, LMIVGMM)],
)
def test_hc1_matches_linearmodels_robust_debiased(westata_cls, lm_cls):
    df = _over_identified()
    res = westata_cls.from_formula(OVER, df).fit(cov_type="HC1")
    ref = _reference(lm_cls, OVER, df, cov_type="robust", debiased=True)
    np.testing.assert_allclose(
        res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-10, atol=1e-12
    )
    assert res.cov_type == "HC1"


def test_hc0_matches_linearmodels_robust_undebiased():
    df = _just_identified()
    res = IV2SLS.from_formula(JUST, df).fit(cov_type="HC0")
    ref = _reference(LMIV2SLS, JUST, df, cov_type="robust", debiased=False)
    np.testing.assert_allclose(res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-12)


def test_hc1_is_hc0_scaled_by_n_over_n_minus_k():
    df = _just_identified()
    n, k = len(df), 3
    hc0 = IV2SLS.from_formula(JUST, df).fit(cov_type="HC0")
    hc1 = IV2SLS.from_formula(JUST, df).fit(cov_type="HC1")
    np.testing.assert_allclose(
        hc1.bse.to_numpy(),
        hc0.bse.to_numpy() * np.sqrt(n / (n - k)),
        rtol=1e-12,
    )


@pytest.mark.parametrize("cov_type", ["HC2", "HC3"])
@pytest.mark.parametrize("cls", [IV2SLS, IVLIML, IVGMM])
def test_leverage_hc_rejected_for_all_iv(cls, cov_type):
    # Stata's ivregress has no HC2/HC3, and linearmodels has no leverage
    # corrected IV sandwich; westata refuses rather than mislabelling an HC0
    # number as HC2/HC3.
    df = _just_identified()
    with pytest.raises(UnsupportedError):
        cls.from_formula(JUST, df).fit(cov_type=cov_type)


def test_cluster_matches_linearmodels_clustered():
    df = _over_identified(n=240)
    df["g"] = np.repeat(np.arange(24), 10)
    res = IV2SLS.from_formula(OVER, df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["g"]}
    )
    ref = _reference(
        LMIV2SLS, OVER, df, cov_type="clustered", clusters=df["g"], debiased=True
    )
    np.testing.assert_allclose(res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-12)
    # CR1 inference uses G - 1 degrees of freedom, e(df_r) stays at N - K.
    assert res.df_resid == df["g"].nunique() - 1
    assert res.e["df_r"] == len(df) - 3


def test_cluster_for_liml_matches_linearmodels():
    df = _over_identified(n=240)
    df["g"] = np.repeat(np.arange(24), 10)
    res = IVLIML.from_formula(OVER, df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["g"]}
    )
    ref = _reference(
        LMIVLIML, OVER, df, cov_type="clustered", clusters=df["g"], debiased=True
    )
    np.testing.assert_allclose(res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-10)


def test_cluster_requires_clusters_kwds():
    df = _just_identified()
    with pytest.raises(ValueError, match="clusters"):
        IV2SLS.from_formula(JUST, df).fit(cov_type="cluster")


def test_ssc_adj_off_matches_linearmodels_undebiased():
    df = _over_identified()
    for westata_cls, lm_cls in [(IV2SLS, LMIV2SLS), (IVLIML, LMIVLIML), (IVGMM, LMIVGMM)]:
        res = westata_cls.from_formula(OVER, df).fit(
            cov_type="HC1", ssc=SSC(adj=False)
        )
        ref = _reference(lm_cls, OVER, df, cov_type="robust", debiased=False)
        np.testing.assert_allclose(
            res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-12
        )


def test_cluster_adj_off_drops_g_over_g_minus_1():
    df = _over_identified(n=240)
    df["g"] = np.repeat(np.arange(24), 10)
    g = df["g"]
    adjusted = IVLIML.from_formula(OVER, df).fit(
        cov_type="cluster", cov_kwds={"clusters": g}
    )
    raw = IVLIML.from_formula(OVER, df).fit(
        cov_type="cluster", cov_kwds={"clusters": g}, ssc=SSC(adj=True, cluster_adj=False)
    )
    n_groups = g.nunique()
    np.testing.assert_allclose(
        (raw.bse.to_numpy() / adjusted.bse.to_numpy()) ** 2,
        np.full(len(raw.bse), (n_groups - 1) / n_groups),
        rtol=1e-12,
    )


def test_clusters_align_on_index_after_listwise_deletion():
    df = _over_identified(n=240)
    df["g"] = np.repeat(np.arange(24), 10)
    df.loc[[0, 17, 240 - 1], "z1"] = np.nan  # linearmodels drops these rows
    res = IV2SLS.from_formula(OVER, df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["g"]}
    )
    surviving = df.dropna(subset=["y", "x", "x2", "z1", "z2"])
    ref = _reference(
        LMIV2SLS,
        OVER,
        df,
        cov_type="clustered",
        clusters=surviving["g"],
        debiased=True,
    )
    assert res.nobs == len(surviving)
    np.testing.assert_allclose(res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-12)


def test_hc_cov_kwds_are_rejected_when_unused():
    df = _just_identified()
    with pytest.raises(ValueError, match="cov_kwds"):
        IV2SLS.from_formula(JUST, df).fit(cov_type="HC1", cov_kwds={"clusters": [0, 1] * 100})


# --------------------------------------------------------------------------
# Stata degrees of freedom and e() scalars
# --------------------------------------------------------------------------
def test_df_resid_excludes_excluded_instruments():
    df = _over_identified(n=300)  # K = intercept + x2 + x = 3
    res = IV2SLS.from_formula(OVER, df).fit()
    assert res.nobs == 300
    assert res.df_resid == 297
    assert res.e["df_r"] == 297
    assert res.e["df_m"] == 2
    assert res.e["rank"] == 3
    # Adding a second instrument must not consume a degree of freedom.
    just = IV2SLS.from_formula("y ~ 1 + x2 + [x ~ z1]", df).fit()
    assert just.df_resid == 297


def test_e_scalars_are_populated():
    df = _over_identified(n=300)
    res = IV2SLS.from_formula(OVER, df).fit()
    required = ["N", "df_m", "df_r", "r2", "r2_a", "rmse", "F", "ll", "rank"]
    for key in required:
        assert key in res.e, key
        assert np.isfinite(res.e[key]), key
    ref = _reference(LMIV2SLS, OVER, df, cov_type="unadjusted", debiased=True)
    assert res.e["r2"] == pytest.approx(float(ref.rsquared), rel=1e-12)
    assert res.e["r2_a"] == pytest.approx(float(ref.rsquared_adj), rel=1e-12)
    rss = float(np.sum(np.asarray(res.resid) ** 2))
    assert res.e["rmse"] == pytest.approx(np.sqrt(rss / 297), rel=1e-12)
    assert res.e["F"] == pytest.approx(float(ref.f_statistic.stat), rel=1e-10)
    # First-stage F is exposed separately from the overall Wald F.
    assert res.e["F_first"] > 10


def test_e_scalars_for_liml_and_gmm():
    df = _over_identified()
    for cls in (IVLIML, IVGMM):
        res = cls.from_formula(OVER, df).fit()
        for key in ["N", "df_m", "df_r", "r2", "r2_a", "rmse", "F", "ll", "rank"]:
            assert key in res.e, (cls.__name__, key)
            assert np.isfinite(res.e[key]), (cls.__name__, key)
        assert res.e["df_r"] == len(df) - 3


def test_summary_stata_renders():
    df = _over_identified()
    text = IV2SLS.from_formula(OVER, df).fit().summary(style="stata")
    # Stata's ivregress table reports z / P>|z|, not t (see [R] ivregress,
    # Example 1: "Wald chi2(2) = 90.76" with a z column).
    for token in ["Coef.", "Std. err.", "P>|z|", "Number of obs", "R-squared"]:
        assert token in text
    assert "P>|t|" not in text


def test_iv_uses_normal_distribution_like_stata():
    df = _over_identified()
    res = IV2SLS.from_formula(OVER, df).fit()
    assert res.dist == "z"
    from scipy import stats

    expected = 2.0 * stats.norm.sf(np.abs(res.tvalues.to_numpy()))
    np.testing.assert_allclose(res.pvalues.to_numpy(), expected, rtol=1e-12)


# --------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cov_type", ["HAC", "twoway", "Weiss", ""])
def test_unknown_cov_type_raises_value_error(cov_type):
    df = _just_identified()
    with pytest.raises(ValueError, match="Unknown cov_type"):
        IV2SLS.from_formula(JUST, df).fit(cov_type=cov_type)


def test_nonrobust_alias_accepted():
    df = _just_identified()
    default = IV2SLS.from_formula(JUST, df).fit()
    named = IV2SLS.from_formula(JUST, df).fit(cov_type="nonrobust")
    np.testing.assert_allclose(named.bse.to_numpy(), default.bse.to_numpy(), rtol=1e-15)
    np.testing.assert_allclose(named.cov, default.cov, rtol=1e-15)


def test_weights_are_out_of_scope():
    df = _just_identified()
    with pytest.raises(UnsupportedError):
        IV2SLS.from_formula(JUST, df, weights=df["x2"])
    with pytest.raises(UnsupportedError):
        IV2SLS(
            df["y"].to_numpy(),
            np.column_stack([np.ones(len(df)), df["x2"]]),
            df["x"].to_numpy(),
            df[["z1"]].to_numpy(),
            weights=df["x2"].to_numpy(),
        )


def test_weak_instrument_warns_but_fits():
    rng = np.random.default_rng(7)
    n = 200
    z1 = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.01 * z1 + v
    y = 1.0 + 2.0 * x + 0.6 * v + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "z1": z1})
    with pytest.warns(UserWarning, match="weak"):
        res = IV2SLS.from_formula("y ~ 1 + [x ~ z1]", df).fit()
    assert np.all(np.isfinite(res.params.to_numpy()))


def test_array_constructor_matches_formula():
    df = _over_identified()
    X_exog = np.column_stack([np.ones(len(df)), df["x2"]])
    model = IV2SLS(
        df["y"].to_numpy(),
        X_exog,
        df["x"].to_numpy(),
        df[["z1", "z2"]].to_numpy(),
        param_names=["Intercept", "x2", "x"],
    )
    res = model.fit()
    from_formula = IV2SLS.from_formula(OVER, df).fit()
    np.testing.assert_allclose(res.params.to_numpy(), from_formula.params.to_numpy(), rtol=1e-12)
    assert res.param_names == ["Intercept", "x2", "x"]


def test_constructor_validates_shapes():
    df = _just_identified(n=50)
    with pytest.raises(ValueError):
        IV2SLS(df["y"].to_numpy()[:49], df["x2"].to_numpy(), df["x"].to_numpy(), df["z1"].to_numpy())
    with pytest.raises(ValueError):
        IV2SLS(
            df["y"].to_numpy(),
            df["x2"].to_numpy(),
            df["x"].to_numpy(),
            df["z1"].to_numpy(),
            param_names=["a", "b", "c"],
        )


def test_constructor_rejects_under_identification():
    df = _just_identified(n=50)
    with pytest.raises(ValueError, match="instruments"):
        IV2SLS(
            df["y"].to_numpy(),
            np.ones(len(df)),
            df["x"].to_numpy(),
            np.empty((len(df), 0)),
        )


def test_rank_deficient_instruments_raise_consistency_error():
    df = _just_identified()
    df["z1_copy"] = df["z1"]
    with pytest.raises(ConsistencyError):
        IV2SLS.from_formula("y ~ 1 + x2 + [x ~ z1 + z1_copy]", df).fit()


def test_under_identified_raises_value_error():
    df = _just_identified()
    with pytest.raises(ValueError, match="instruments"):
        # One excluded instrument for two endogenous regressors: not identified.
        IV2SLS.from_formula("y ~ 1 + [x + x2 ~ z1]", df).fit()


def test_categorical_dummies_and_string_clusters():
    rng = np.random.default_rng(11)
    n = 150
    z1 = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.8 * z1 + v
    g = rng.integers(0, 4, size=n)
    y = 1.0 + 2.0 * x + 0.5 * (g == 1) + 0.6 * v + rng.normal(size=n)
    df = pd.DataFrame(
        {"y": y, "x": x, "z1": z1, "g": g, "grp": [f"c{i % 6}" for i in range(n)]}
    )
    res = IV2SLS.from_formula("y ~ 1 + C(g) + [x ~ z1]", df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["grp"]}
    )
    ref = _reference(
        LMIV2SLS,
        "y ~ 1 + C(g) + [x ~ z1]",
        df,
        cov_type="clustered",
        clusters=df["grp"],
        debiased=True,
    )
    np.testing.assert_allclose(res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-12)
    # Constant + 3 dummies = 4 exogenous columns; df_m counts the 3 dummies.
    assert res.e["df_m"] == 4
    assert res.e["df_r"] == n - 5


def test_twoway_clustering_is_rejected():
    df = _just_identified()
    two_way = np.column_stack([np.arange(len(df)) % 10, np.arange(len(df)) % 7])
    with pytest.raises(UnsupportedError, match="two-way"):
        IV2SLS.from_formula(JUST, df).fit(
            cov_type="cluster", cov_kwds={"clusters": two_way}
        )


def test_constant_detected_in_column_span_without_explicit_intercept():
    # A full dummy set without an explicit intercept still spans the constant,
    # so Stata (and linearmodels) centre TSS and count df_m = K.
    rng = np.random.default_rng(2)
    n = 180
    z1 = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.8 * z1 + v
    g = rng.integers(0, 3, size=n)
    y = 1.0 + 2.0 * x + 0.4 * g + 0.5 * v + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "x": x, "z1": z1, "g": g})
    formula = "y ~ 0 + C(g) + [x ~ z1]"
    res = IV2SLS.from_formula(formula, df).fit()
    ref = _reference(LMIV2SLS, formula, df, cov_type="unadjusted", debiased=True)
    assert res.e["r2"] == pytest.approx(float(ref.rsquared), rel=1e-12)
    assert res.e["r2_a"] == pytest.approx(float(ref.rsquared_adj), rel=1e-12)
    assert res.e["df_m"] == 4
    assert res.e["F"] == pytest.approx(float(ref.f_statistic.stat), rel=1e-10)


