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
        LMIV2SLS, OVER, df, cov_type="clustered", clusters=df["g"], debiased=False
    )
    np.testing.assert_allclose(res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-10)
    # IV inference is normal-based, so the reference df is G-1 for clustered
    # inference; e(df_r) stays at N because Stata stores missing there.
    assert res.df_resid == df["g"].nunique() - 1
    assert res.e["df_r"] == len(df)


def test_cluster_for_liml_matches_linearmodels():
    """Stata's ``ivregress liml, vce(cluster)`` defaults to CR0, not CR1.

    Live capture (StataNow 19.5 MP, 2026-10-24): on the 140x14 panel
    ``ivregress liml y x1 (e1 = z1 z2), vce(cluster gid)`` produces SEs that
    match ``debiased=False`` to 1.1e-14; adding ``small`` (CR1) moves them by
    4.53e-2 = sqrt(G/(G-1) * (N-1)/(N-K)).  An earlier version of this test
    pinned ``debiased=True``, i.e. it asserted the CR1 variant Stata only
    reports with ``small``.
    """
    df = _over_identified(n=240)
    df["g"] = np.repeat(np.arange(24), 10)
    res = IVLIML.from_formula(OVER, df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["g"]}
    )
    ref = _reference(
        LMIVLIML, OVER, df, cov_type="clustered", clusters=df["g"], debiased=False
    )
    np.testing.assert_allclose(res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-10)
    # ssc=SSC(adj=True) is Stata's `small` option: CR1.
    with_small = IVLIML.from_formula(OVER, df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["g"]}, ssc=SSC(adj=True)
    )
    ref_small = _reference(
        LMIVLIML, OVER, df, cov_type="clustered", clusters=df["g"], debiased=True
    )
    np.testing.assert_allclose(
        with_small.bse.to_numpy(), np.asarray(ref_small.std_errors), rtol=1e-10
    )
    n, k = len(df), 3
    n_groups = df["g"].nunique()
    cr1_over_cr0 = np.sqrt(
        n_groups / (n_groups - 1.0) * (n - 1) / (n - k)
    )
    np.testing.assert_allclose(
        with_small.bse.to_numpy() / res.bse.to_numpy(),
        np.full(k, cr1_over_cr0),
        rtol=1e-10,
    )


def test_cluster_requires_clusters_kwds():
    df = _just_identified()
    with pytest.raises(ValueError, match="clusters"):
        IV2SLS.from_formula(JUST, df).fit(cov_type="cluster")


def test_hc1_keeps_its_n_minus_k_factor_even_when_ssc_adj_is_off():
    """HC1 and the nonrobust IV denominator are separate conventions.

    Stata's ivregress uses s^2 = ESS/N by default (no ``small``), but HC1 is
    still ``N/(N-K) * HC0``.  ``ssc.adj=False`` must therefore not flatten HC1
    onto HC0 -- an earlier implementation did exactly that.
    """
    df = _over_identified()
    for cls in (IV2SLS, IVLIML, IVGMM):
        res = cls.from_formula(OVER, df).fit(cov_type="HC1", ssc=SSC(adj=False))
        hc0 = cls.from_formula(OVER, df).fit(cov_type="HC0")
        ratio = (res.bse.to_numpy() / hc0.bse.to_numpy())
        n, k = len(df), 3
        np.testing.assert_allclose(ratio, np.sqrt(n / (n - k)), rtol=1e-10)


def test_cluster_adj_off_drops_g_over_g_minus_1():
    """``ssc=SSC(adj=False)`` is Stata's default CR0; ``adj=True`` is ``small``.

    Stata's ``ivregress ..., vce(cluster g)`` applies neither ``G/(G-1)`` nor
    ``(N-1)/(N-K)`` unless ``small`` is given (live capture 2026-10-24), so the
    ``small`` variant is exactly the default times CR1/CR0.
    """
    df = _over_identified(n=240)
    df["g"] = np.repeat(np.arange(24), 10)
    g = df["g"]
    default = IVLIML.from_formula(OVER, df).fit(
        cov_type="cluster", cov_kwds={"clusters": g}
    )
    small = IVLIML.from_formula(OVER, df).fit(
        cov_type="cluster", cov_kwds={"clusters": g}, ssc=SSC(adj=True)
    )
    n_groups = g.nunique()
    n, k = len(df), 3
    expected_ratio = np.sqrt(
        n_groups / (n_groups - 1.0) * (n - 1) / (n - k)
    )
    np.testing.assert_allclose(
        (small.bse.to_numpy() / default.bse.to_numpy()),
        np.full(len(small.bse), expected_ratio),
        rtol=1e-10,
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
        debiased=False,
    )
    assert res.nobs == len(surviving)
    # debiased=False: Stata's ivregress VCE has no ``small`` factor by default.
    np.testing.assert_allclose(res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-10)


def test_hc_cov_kwds_are_rejected_when_unused():
    df = _just_identified()
    with pytest.raises(ValueError, match="cov_kwds"):
        IV2SLS.from_formula(JUST, df).fit(cov_type="HC1", cov_kwds={"clusters": [0, 1] * 100})


# --------------------------------------------------------------------------
# L1: vce(robust) is HC0 and vce(robust) small is HC1
# --------------------------------------------------------------------------
#: Live capture, StataNow 19.5 (MP), 2026-10-24, on the fixed panel produced by
#: the retest's ``linear_iv`` generator with seed 20261024 (140 obs, 14
#: clusters, instruments z1 z2), reordered to westata's [Intercept, x1, e1]::
#:
#:     ivregress 2sls y x1 (e1 = z1 z2), vce(robust)
#:         -> se = [.05607703848079, .05246009123669, .07300778706249]
#:     ivregress 2sls y x1 (e1 = z1 z2), vce(robust) small
#:         -> se = [.05668769574323, .05303136134241, .07380281362936]
STATA_L1_SE_ROBUST = np.array([0.05607703848079, 0.05246009123669, 0.07300778706249])
STATA_L1_SE_ROBUST_SMALL = np.array([0.05668769574323, 0.05303136134241, 0.07380281362936])


def _stata_l1_panel() -> pd.DataFrame:
    """The retest's 140x14 IV panel (same seed as retest/domains/linear_iv)."""

    rng = np.random.default_rng(20261024)
    n, g = 140, 14
    gid = np.repeat(np.arange(g), n // g)
    x1 = rng.normal(size=n)
    z = rng.normal(size=(n, 2))
    e1 = 0.8 * z[:, 0] + rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.4 * e1 + rng.normal(0.0, 0.7, n)
    return pd.DataFrame(
        {"y": y, "x1": x1, "e1": e1, "z1": z[:, 0], "z2": z[:, 1], "gid": gid}
    )


def test_iv_vce_robust_is_hc0_and_small_is_hc1():
    """Stata: ``vce(robust)`` = HC0, ``vce(robust) small`` = HC1 (opposite of OLS).

    An earlier version of ``iv.py`` documented HC1 as the equivalent of
    ``vce(robust)``, which made every "robust" IV standard error
    ``sqrt(N/(N-K))`` too large (1.09e-2 relative on this data).
    """

    df = _stata_l1_panel()
    formula = "y ~ x1 + [e1 ~ z1 + z2]"
    hc0 = IV2SLS.from_formula(formula, df).fit(cov_type="HC0")
    hc1 = IV2SLS.from_formula(formula, df).fit(cov_type="HC1")
    np.testing.assert_allclose(hc0.bse.to_numpy(), STATA_L1_SE_ROBUST, rtol=1e-6)
    np.testing.assert_allclose(hc1.bse.to_numpy(), STATA_L1_SE_ROBUST_SMALL, rtol=1e-6)
    # The two really do differ, by exactly sqrt(N/(N-K)).
    n, k = len(df), 3
    np.testing.assert_allclose(
        hc1.bse.to_numpy() / hc0.bse.to_numpy(),
        np.full(k, np.sqrt(n / (n - k))),
        rtol=1e-12,
    )
    # And ``vce(robust) small`` / ``vce(robust)`` in Stata is that same ratio.
    np.testing.assert_allclose(
        STATA_L1_SE_ROBUST_SMALL / STATA_L1_SE_ROBUST,
        np.full(k, np.sqrt(n / (n - k))),
        rtol=1e-6,
    )


# --------------------------------------------------------------------------
# L2/L3: Stata's clustered defaults, and GMM's clustered weight matrix
# --------------------------------------------------------------------------
#: Live capture, StataNow 19.5 (2026-10-24), same 140x14 panel, in
#: [Intercept, x1, e1] order::
#:
#:     ivregress liml y x1 (e1 = z1 z2), vce(cluster gid)
#:         -> se = [.04174899098950, .05239940413109, .06219927341224]
#:     ivregress gmm  y x1 (e1 = z1 z2), vce(cluster gid)
#:         -> b  = [.94148181673416, 2.06205332268846, -.49647420664484]
#:            se = [.03989222231688, .04950581528835, .05994273495944]
#:            J  = 1.1097180436518683
#:     ivregress gmm  y x1 (e1 = z1 z2)        (heteroskedastic W)
#:         -> b  = [.95149366676723, 2.05002044968350, -.48210120281584]
#:            J  = 0.8351609988436965
STATA_L2_LIML_CLUSTER_SE = np.array(
    [0.04174899098950, 0.05239940413109, 0.06219927341224]
)
STATA_L3_GMM_CLUSTER_B = np.array([0.94148181673416, 2.06205332268846, -0.49647420664484])
STATA_L3_GMM_CLUSTER_SE = np.array([0.03989222231688, 0.04950581528835, 0.05994273495944])
STATA_L3_GMM_DEFAULT_B = np.array([0.95149366676723, 2.05002044968350, -0.48210120281584])
STATA_L3_GMM_CLUSTER_J = 1.1097180436518683
STATA_L3_GMM_DEFAULT_J = 0.8351609988436965


def test_liml_cluster_defaults_to_cr0_like_stata():
    df = _stata_l1_panel()
    formula = "y ~ x1 + [e1 ~ z1 + z2]"
    res = IVLIML.from_formula(formula, df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["gid"]}
    )
    np.testing.assert_allclose(res.bse.to_numpy(), STATA_L2_LIML_CLUSTER_SE, rtol=1e-6)


def test_gmm_cluster_clusters_the_weight_matrix():
    """``ivregress gmm, vce(cluster g)`` clusters ``W`` too, moving the estimates.

    The pre-fix westata always used the heteroskedastic ``W``, so the
    coefficients were 3.58e-2 away from Stata's before any standard error was
    even considered.  This test pins Stata's coefficient vector.
    """

    df = _stata_l1_panel()
    formula = "y ~ x1 + [e1 ~ z1 + z2]"
    clustered = IVGMM.from_formula(formula, df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["gid"]}
    )
    np.testing.assert_allclose(
        clustered.params.to_numpy(), STATA_L3_GMM_CLUSTER_B, rtol=1e-6
    )
    np.testing.assert_allclose(clustered.bse.to_numpy(), STATA_L3_GMM_CLUSTER_SE, rtol=1e-6)
    assert clustered.e["J"] == pytest.approx(STATA_L3_GMM_CLUSTER_J, rel=1e-10)
    # The heteroskedastic-W estimator must NOT reproduce Stata's clustered fit --
    # otherwise the assertion above would pass for the wrong reason.
    hetero = IVGMM.from_formula(formula, df).fit()
    assert not np.allclose(hetero.params.to_numpy(), STATA_L3_GMM_CLUSTER_B, rtol=1e-4)
    # ... while it is still what Stata reports for the non-clustered fit.
    np.testing.assert_allclose(hetero.params.to_numpy(), STATA_L3_GMM_DEFAULT_B, rtol=1e-6)
    assert hetero.e["J"] == pytest.approx(STATA_L3_GMM_DEFAULT_J, rel=1e-10)


def test_gmm_cluster_reproduces_the_retest_panel():
    """The exact panel ``retest/domains/linear_iv/run.py`` item 12d used.

    That run measured a 3.583e-2 coefficient gap for ``ivregress gmm yiv x1
    (e1 = z1 z2), vce(cluster gid)``; the heteroskedastic-W estimator (the
    pre-fix behaviour, exercised here through the non-clustered ``fit``) still
    reproduces that gap, and the clustered-W fit now matches Stata.
    """

    rng = np.random.default_rng(20261024)
    n, g = 140, 14
    gid = np.repeat(np.arange(g), n // g)
    tt = np.tile(np.arange(10), g)
    cont = rng.choice([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], size=n)
    fe_g = rng.normal(0.0, 0.5, g)[gid]
    fe_t = rng.normal(0.0, 0.3, 10)[tt]
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + fe_g + fe_t + rng.normal(0.0, 0.7, n)
    w = rng.integers(1, 5, size=n).astype(float)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    e1 = 0.8 * z1 + 0.3 * z2 + rng.normal(size=n) + 0.5 * rng.normal(size=n)
    yiv = 1.0 + 2.0 * x1 - 0.4 * e1 + rng.normal(0.0, 0.7, n)
    df = pd.DataFrame(
        {"y": y, "yiv": yiv, "x1": x1, "x2": x2, "gid": gid, "t": tt,
         "cont": cont, "w": w, "z1": z1, "z2": z2, "e1": e1}
    )
    formula = "yiv ~ x1 + [e1 ~ z1 + z2]"
    # Stata (live capture, 2026-10-24): ivregress gmm yiv x1 (e1 = z1 z2),
    # vce(cluster gid) -> b = [1.0126539516130684, 2.0935721060105577,
    # -.48536211184222755] in [_cons, x1, e1] order.
    stata_b = np.array([1.0126539516130684, 2.0935721060105577, -0.48536211184222755])
    clustered = IVGMM.from_formula(formula, df).fit(
        cov_type="cluster", cov_kwds={"clusters": df["gid"]}
    )
    np.testing.assert_allclose(clustered.params.to_numpy(), stata_b, rtol=1e-6)
    # The pre-fix estimator (heteroskedastic W) misses by ~3.6e-2, which is the
    # gap the retest reported; this keeps the test honest about what changed.
    hetero = IVGMM.from_formula(formula, df).fit()
    gap = np.max(np.abs(hetero.params.to_numpy() - stata_b) / np.abs(stata_b))
    assert 1e-2 < gap < 6e-2, gap




# --------------------------------------------------------------------------
# Stata degrees of freedom and e() scalars
# --------------------------------------------------------------------------
def test_df_r_uses_n_because_stata_leaves_it_missing():
    """Stata's ivregress stores ``e(df_r) = .`` (missing).

    Verified against StataNow 19.5: ``ereturn list`` after
    ``ivregress 2sls y x_exog (x_endog = z1 z2)`` reports ``e(df_r) = .`` and
    ``e(rmse) = sqrt(e(rss)/e(N))``.  IV inference is normal-based, so there is
    no residual-degrees-of-freedom concept; recording N-K would be a number
    Stata never reports.
    """
    df = _over_identified(n=300)  # K = intercept + x2 + x = 3
    res = IV2SLS.from_formula(OVER, df).fit()
    assert res.nobs == 300
    assert res.e["df_r"] == 300
    assert res.df_resid == 300
    assert res.e["df_m"] == 2
    assert res.e["rank"] == 3
    # Adding a second instrument must not consume a degree of freedom.
    just = IV2SLS.from_formula("y ~ 1 + x2 + [x ~ z1]", df).fit()
    assert just.e["df_r"] == 300


def test_e_scalars_are_populated():
    df = _over_identified(n=300)
    res = IV2SLS.from_formula(OVER, df).fit()
    required = ["N", "df_m", "df_r", "r2", "r2_a", "rmse", "F", "ll", "rank"]
    for key in required:
        assert key in res.e, key
        assert np.isfinite(res.e[key]), key
    ref = _reference(LMIV2SLS, OVER, df, cov_type="unadjusted", debiased=True)
    assert res.e["r2"] == pytest.approx(float(ref.rsquared), rel=1e-12)
    # R2_a keeps the N-K denominator even though the VCE does not.
    assert res.e["r2_a"] == pytest.approx(float(ref.rsquared_adj), rel=1e-12)
    rss = float(np.sum(np.asarray(res.resid) ** 2))
    # Stata: e(rmse) = sqrt(rss/N) for ivregress.
    assert res.e["rmse"] == pytest.approx(np.sqrt(rss / len(df)), rel=1e-12)
    # Stata's ivregress does NOT store e(F): ``ereturn list`` after
    # ``ivregress 2sls`` shows e(chi2) only (verified on StataNow 19.5). westata
    # exposes e(F) = e(chi2)/e(df_m) as a convenience, so assert that relation
    # rather than a hard-coded number from a different sample.
    assert np.isfinite(res.e["chi2"]) and res.e["chi2"] > 0
    assert res.e["F"] == pytest.approx(res.e["chi2"] / res.e["df_m"], rel=1e-12)
    # First-stage F is exposed separately from the overall Wald F.
    assert res.e["F_first"] > 10


def test_e_scalars_for_liml_and_gmm():
    df = _over_identified()
    for cls in (IVLIML, IVGMM):
        res = cls.from_formula(OVER, df).fit()
        for key in ["N", "df_m", "df_r", "r2", "r2_a", "rmse", "F", "ll", "rank"]:
            assert key in res.e, (cls.__name__, key)
            assert np.isfinite(res.e[key]), (cls.__name__, key)
        # ivregress leaves e(df_r) missing; N is the comparable quantity.
        assert res.e["df_r"] == len(df)


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
        debiased=False,
    )
    np.testing.assert_allclose(res.bse.to_numpy(), np.asarray(ref.std_errors), rtol=1e-10)
    # Constant + 3 dummies = 4 exogenous columns; df_m counts the 3 dummies.
    assert res.e["df_m"] == 4
    # ivregress stores e(df_r) = . ; westata records N as the comparable count.
    assert res.e["df_r"] == n


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
    ref = _reference(LMIV2SLS, formula, df, cov_type="unadjusted", debiased=False)
    assert res.e["r2"] == pytest.approx(float(ref.rsquared), rel=1e-12)
    assert res.e["r2_a"] == pytest.approx(float(ref.rsquared_adj), rel=1e-12)
    assert res.e["df_m"] == 4
    # ivregress stores chi2, not F.
    assert res.e["chi2"] > 0


