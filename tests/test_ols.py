"""OLS conformance tests against statsmodels (same algebra as Stata ``regress``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from westata.errors import ConsistencyError, WeightError
from westata.estimators.linear import OLS


def _data(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    g = rng.integers(0, 3, size=n)
    y = 1.0 + 2.0 * x1 - 1.5 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})


def _statsmodels_fit(df: pd.DataFrame, cov_type: str = "nonrobust", **kw):
    X = sm.add_constant(df[["x1", "x2"]], has_constant="add")
    model = sm.OLS(df["y"], X)
    return model.fit(cov_type=cov_type, **kw)


def test_point_estimates_match_statsmodels():
    df = _data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit()
    ref = _statsmodels_fit(df)
    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-12)


def test_hc1_matches_statsmodels():
    df = _data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(cov_type="HC1")
    ref = _statsmodels_fit(df, cov_type="HC1")
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-12)


def test_cluster_matches_statsmodels_cr1():
    df = _data()
    groups = df["g"]
    res = OLS.from_formula("y ~ x1 + x2", df).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )
    ref = _statsmodels_fit(
        df, cov_type="cluster", cov_kwds={"groups": groups}, use_correction=True
    )
    np.testing.assert_allclose(res.bse.to_numpy(), ref.bse.to_numpy(), rtol=1e-12)
    assert res.df_resid == groups.nunique() - 1


def test_e_scalars_are_populated():
    df = _data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit()
    assert res.e["N"] == len(df)
    assert res.e["rank"] == 3
    assert res.e["df_m"] == 2
    assert res.e["df_r"] == len(df) - 3
    assert res.e["r2"] == pytest.approx(res.rsquared)


def test_stata_summary_labels():
    df = _data()
    text = OLS.from_formula("y ~ x1 + x2", df).fit().summary(style="stata")
    for token in ["Number of obs", "Coef.", "Std. err.", "P>|t|", "conf. interval"]:
        assert token in text


def test_rank_deficient_raises_consistency_error():
    df = _data()
    df["x1_copy"] = df["x1"]
    with pytest.raises(ConsistencyError):
        OLS.from_formula("y ~ x1 + x1_copy", df).fit()


def test_weights_require_weight_type():
    df = _data()
    # Stata never guesses a weight type; P1 implements the four types and
    # rejects a bare weight vector so a silent default cannot change the VCE.
    with pytest.raises(WeightError):
        OLS.from_formula("y ~ x1", df, weights=df["x1"])


# ==========================================================================
# Stata e() conformance for the four VCEs Stata distinguishes.
#
# Every constant below is a live capture from StataNow 19.5 (MP) on
# 2026-10-24, on ``sysuse auto`` (n = 74) with the derived variables
#
#     gen byte gid = mod(_n, 12)      // 12 clusters
#     gen int  w   = mod(_n, 4) + 1   // 1..4, sum(w) = 185
#
# The retest found three separate defects here where the *number* was right but
# the *definition* was wrong:
#
#   * ``e(F)`` was the classic R^2-form F under every VCE (P1-15);
#   * ``e(df_r)`` was ``N-K`` under clustering where Stata stores ``G-1`` (P1-16);
#   * ``e(rss)``, ``e(mss)``, ``e(sum_w)`` and ``e(N_clust)`` were missing.
# ==========================================================================
AUTO = Path(__file__).resolve().parent / "data" / "auto.dta"


@pytest.fixture(scope="module")
def auto() -> pd.DataFrame:
    if not AUTO.exists():  # pragma: no cover - the fixture file is in-repo
        pytest.skip("auto.dta not downloaded")
    df = pd.read_stata(AUTO)[["mpg", "weight", "foreign"]].copy()
    df["foreign"] = (df["foreign"] == "Foreign").astype(float)
    n = len(df)
    # Stata's _n is 1-based: mod(_n, 12) for row i (0-based) is (i+1) % 12.
    df["gid"] = np.array([(i + 1) % 12 for i in range(n)], dtype=float)
    df["w"] = np.array([((i + 1) % 4) + 1 for i in range(n)], dtype=float)
    return df


_AUTO_FORMULA = "mpg ~ weight + foreign"

#: Stata e(F) per VCE (``ereturn list``, live capture 2026-10-24).
STATA_E_F = {
    "nonrobust": 69.74846262000308,
    "robust": 73.80592552526059,
    "cluster": 81.951329431322,
    "pweight": 59.710228990293984,
    "aweight": 67.34659209101721,
    "fweight": 172.6349262051427,
}


def test_e_f_nonrobust_matches_stata(auto):
    res = OLS.from_formula(_AUTO_FORMULA, auto).fit()
    assert res.e["F"] == pytest.approx(STATA_E_F["nonrobust"], rel=1e-9)


def test_e_f_robust_is_the_robust_wald_not_the_r2_form(auto):
    """``regress mpg weight foreign, vce(robust)`` -> Stata e(F) = 73.80592552526.

    The pre-fix value was the nonrobust 69.74846262000 (R^2-form F), 5.5e-2
    away; under clustering the same mistake was 37.5% off.
    """

    res = OLS.from_formula(_AUTO_FORMULA, auto).fit(cov_type="HC1")
    assert res.e["F"] == pytest.approx(STATA_E_F["robust"], rel=1e-9)
    assert res.e["F"] != pytest.approx(STATA_E_F["nonrobust"], rel=1e-6)


def test_e_f_cluster_is_the_cr1_wald(auto):
    res = OLS.from_formula(_AUTO_FORMULA, auto).fit(
        cov_type="cluster", cov_kwds={"groups": auto["gid"]}
    )
    assert res.e["F"] == pytest.approx(STATA_E_F["cluster"], rel=1e-9)


@pytest.mark.parametrize("weight_type", ["pweight", "aweight", "fweight"])
def test_e_f_under_weights(auto, weight_type):
    res = OLS.from_formula(
        _AUTO_FORMULA, auto, weights=auto["w"], weight_type=weight_type
    ).fit()
    assert res.e["F"] == pytest.approx(STATA_E_F[weight_type], rel=1e-9)


def test_e_f_is_always_the_wald_of_the_reported_matrix(auto):
    """One identity behind all four cases: e(F) = b_s' V_ss^-1 b_s / df_m."""

    for fit_kw in (
        {},
        {"cov_type": "HC1"},
        {"cov_type": "cluster", "cov_kwds": {"groups": auto["gid"]}},
    ):
        res = OLS.from_formula(_AUTO_FORMULA, auto).fit(**fit_kw)
        names = list(res.param_names)
        slopes = [i for i, nm in enumerate(names) if nm != "Intercept"]
        b = np.asarray(res.params, dtype=float)[slopes]
        V = res.cov[np.ix_(slopes, slopes)]
        expected = float(b @ np.linalg.solve(V, b) / len(slopes))
        assert res.e["F"] == pytest.approx(expected, rel=1e-12)


def test_e_f_is_zero_for_the_intercept_only_model(auto):
    """Stata: ``regress mpg`` reports e(F) = 0, not missing."""

    res = OLS.from_formula("mpg ~ 1", auto).fit()
    assert res.e["F"] == 0.0


def test_e_df_r_is_g_minus_1_under_clustering(auto):
    """Stata: ``e(df_r) = 11`` (= G-1) where ``N-K = 71``; ``e(N_clust) = 12``."""

    res = OLS.from_formula(_AUTO_FORMULA, auto).fit(
        cov_type="cluster", cov_kwds={"groups": auto["gid"]}
    )
    assert res.e["df_r"] == 11
    assert res.e["N_clust"] == 12
    assert res.df_resid == 11
    assert res.e["df_r"] != len(auto) - 3
    # e(rmse) keeps the N - K denominator even though e(df_r) does not.
    assert res.e["rmse"] == pytest.approx(
        np.sqrt(res.e["rss"] / (len(auto) - 3)), rel=1e-12
    )


def test_e_df_r_stays_n_minus_k_without_clustering(auto):
    res = OLS.from_formula(_AUTO_FORMULA, auto).fit()
    assert res.e["df_r"] == len(auto) - 3
    assert "N_clust" not in res.e


#: Stata's (e(rss), e(mss), e(sum_w)) per case (live capture 2026-10-24).
STATA_SUMS = {
    "nonrobust": (824.1717612920727, 1619.287698167387, 74.0),
    "pweight": (951.827263110844, 1805.6992233756428, 185.0),
    "aweight": (951.827263110844, 1805.6992233756428, 185.0),
    "fweight": (2379.56815777711, 4514.248058439108, 185.0),
    "noconstant": (5124.219675059274, 30883.780324940726, 74.0),
}


def test_e_rss_and_e_mss_match_stata(auto):
    res = OLS.from_formula(_AUTO_FORMULA, auto).fit()
    rss, mss, sum_w = STATA_SUMS["nonrobust"]
    assert res.e["rss"] == pytest.approx(rss, rel=1e-12)
    assert res.e["mss"] == pytest.approx(mss, rel=1e-12)
    assert res.e["sum_w"] == sum_w
    assert 1.0 - res.e["rss"] / (res.e["rss"] + res.e["mss"]) == pytest.approx(
        res.e["r2"], rel=1e-12
    )


@pytest.mark.parametrize("weight_type", ["pweight", "aweight", "fweight"])
def test_e_sum_w_is_the_raw_weight_sum(auto, weight_type):
    """``e(sum_w) = sum(w)`` = 185 even where estimation renormalises the weights.

    ``aweight``/``pweight`` weights are rescaled to sum to N for estimation, so
    the stored scalar has to come from the *un-normalised* vector.  Using the
    normalized one would report 74 and quietly disagree with Stata.
    """

    res = OLS.from_formula(
        _AUTO_FORMULA, auto, weights=auto["w"], weight_type=weight_type
    ).fit()
    rss, mss, sum_w = STATA_SUMS[weight_type]
    assert res.e["sum_w"] == sum_w
    assert res.e["sum_w"] == pytest.approx(float(auto["w"].sum()))
    assert res.e["rss"] == pytest.approx(rss, rel=1e-10)
    assert res.e["mss"] == pytest.approx(mss, rel=1e-10)


def test_e_rss_and_mss_with_noconstant(auto):
    """Without a constant Stata does not centre TSS, so mss is not the demeaned SS."""

    res = OLS.from_formula("mpg ~ 0 + weight + foreign", auto).fit()
    rss, mss, _ = STATA_SUMS["noconstant"]
    assert res.e["rss"] == pytest.approx(rss, rel=1e-12)
    assert res.e["mss"] == pytest.approx(mss, rel=1e-12)
    # Stata's r2 = 1 - rss/sum(y^2) (uncentred) ...
    assert res.e["r2"] == pytest.approx(1.0 - rss / (rss + mss), rel=1e-12)
    # ... and r2_a = 1 - (1-r2)*N/(N-K), i.e. with c = 0.
    assert res.e["r2_a"] == pytest.approx(0.8537391937770055, rel=1e-12)


def test_e_sum_w_equals_n_for_unweighted_fits(auto):
    res = OLS.from_formula(_AUTO_FORMULA, auto).fit()
    assert res.e["sum_w"] == float(len(auto))


def test_e_rss_matches_the_residual_sum_of_squares(auto):
    res = OLS.from_formula(_AUTO_FORMULA, auto).fit()
    assert res.e["rss"] == pytest.approx(
        float(np.sum(np.asarray(res.resid) ** 2)), rel=1e-12
    )


def test_e_ll_0_is_the_intercept_only_likelihood(auto):
    """Stata's ``e(ll_0)`` = the Gaussian log-likelihood of ``regress y``.

    Live capture (2026-10-24, auto.dta): nonrobust / ``vce(robust)`` /
    ``vce(cluster gid)`` all report -234.39433764823468, ``[pweight=w]``
    -238.86834919298698 and ``[fweight=w]`` -597.1708729824675; ``noconstant``
    stores no ``e(ll_0)`` at all.
    """

    res = OLS.from_formula(_AUTO_FORMULA, auto).fit()
    assert res.e["ll_0"] == pytest.approx(-234.39433764823468, rel=1e-12)
    # It does not move with the VCE.
    for fit_kw in (
        {"cov_type": "HC1"},
        {"cov_type": "cluster", "cov_kwds": {"groups": auto["gid"]}},
    ):
        assert OLS.from_formula(_AUTO_FORMULA, auto).fit(**fit_kw).e[
            "ll_0"
        ] == pytest.approx(-234.39433764823468, rel=1e-12)
    weighted = OLS.from_formula(
        _AUTO_FORMULA, auto, weights=auto["w"], weight_type="pweight"
    ).fit()
    assert weighted.e["ll_0"] == pytest.approx(-238.86834919298698, rel=1e-12)
    fweighted = OLS.from_formula(
        _AUTO_FORMULA, auto, weights=auto["w"], weight_type="fweight"
    ).fit()
    assert fweighted.e["ll_0"] == pytest.approx(-597.1708729824675, rel=1e-12)
    # No constant -> Stata reports no ll_0, and neither do we.
    assert "ll_0" not in OLS.from_formula("mpg ~ 0 + weight + foreign", auto).fit().e
