"""Documented behaviour of the cluster-robust degrees of freedom.

Stata ``[R] regress`` -> "Degrees of freedom":

    For one-way cluster-robust variance estimators, p-values and confidence
    intervals are derived from a t distribution with G - 1 degrees of freedom,
    where G is the total number of clusters.

Stata also stores that same number in ``e(df_r)``: live capture on StataNow
19.5 (MP), 2026-10-24, ``sysuse auto`` + ``gen byte gid = mod(_n, 12)``
(G = 12 clusters, N = 74, K = 3)::

    regress mpg weight foreign, vce(cluster gid)
    ereturn list  ->  e(N) = 74   e(df_r) = 11   e(N_clust) = 12
                      e(rmse) = 3.4070592856515844 (= sqrt(rss/(N-K)))

so ``e(df_r) = G - 1 = 11``, *not* ``N - K = 71``.  An earlier version of this
file asserted ``e(df_r) = N - K``, which was wrong: Stata never reports that
number under clustering (``e(rmse)`` is the quantity that keeps ``N - K``).

westata therefore keeps the two concepts aligned but explicit:

``results.df_resid``
    The reference-distribution degrees of freedom -- ``G - 1`` under clustering,
    ``N - K`` otherwise.  Used for t/p/CI and for the displayed ``F(df_m, .)``.
``results.e["df_r"]``
    Stata's stored scalar, which is the same ``G - 1`` under clustering and
    ``N - K`` otherwise.

These tests pin the split so a future refactor cannot silently collapse it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.estimators.linear import OLS


def _cluster_data(n_clusters: int = 6, per_cluster: int = 8, seed: int = 3):
    rng = np.random.default_rng(seed)
    n = n_clusters * per_cluster
    g = np.repeat(np.arange(n_clusters), per_cluster)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Cluster-correlated errors, so the clustered VCE is not degenerate.
    shock = rng.normal(size=n_clusters)[g]
    y = 1.0 + 2.0 * x1 - x2 + shock + rng.normal(scale=0.3, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "g": g})


def test_cluster_stores_g_minus_1_in_e_df_r_like_stata():
    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["g"]}
    )

    n_groups = df["g"].nunique()
    n = len(df)
    k = 3  # Intercept, x1, x2

    # Stata: e(df_r) = e(N_clust) - 1 under vce(cluster id).
    assert res.e["df_r"] == n_groups - 1
    assert res.e["N_clust"] == n_groups
    # results.df_resid carries the same reference-distribution df.
    assert res.df_resid == n_groups - 1
    # ... and it really is not N - K, otherwise this test proves nothing.
    assert res.e["df_r"] != n - k
    # e(rmse) keeps the N - K denominator; only the inference df changes.
    rss = float(np.sum(np.asarray(res.resid) ** 2))
    assert res.e["rmse"] == pytest.approx(np.sqrt(rss / (n - k)), rel=1e-12)


def test_cluster_pvalues_use_g_minus_1_not_n_minus_k():
    from scipy import stats

    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["g"]}
    )

    t = res.tvalues.to_numpy()
    expected = 2.0 * stats.t.sf(np.abs(t), res.df_resid)
    np.testing.assert_allclose(res.pvalues.to_numpy(), expected, rtol=1e-12)

    wrong = 2.0 * stats.t.sf(np.abs(t), len(df) - 3)
    # Sanity check that the two reference distributions actually differ here,
    # so the assertion above is not vacuous.
    assert not np.allclose(res.pvalues.to_numpy(), wrong, rtol=1e-6)


def test_cluster_summary_reports_g_minus_1_in_the_f_test():
    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["g"]}
    )
    text = res.summary(style="stata")
    n_groups = res.df_resid
    assert f"F(2, {int(n_groups)})" in text


def test_cluster_e_f_is_the_wald_statistic_of_that_vce():
    """Stata's e(F) under clustering is the CR1 Wald/k, not the R^2-form F.

    Live capture (StataNow 19.5, 2026-10-24, auto.dta)::

        regress mpg weight foreign, vce(cluster gid)   ->  e(F) = 81.951329431322
        regress mpg weight foreign, vce(robust)        ->  e(F) = 73.805925525261
        regress mpg weight foreign                     ->  e(F) = 69.748462620003
    """

    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["g"]}
    )
    names = list(res.param_names)
    slopes = [i for i, nm in enumerate(names) if nm != "Intercept"]
    b = np.asarray(res.params, dtype=float)[slopes]
    V = res.cov[np.ix_(slopes, slopes)]
    expected = float(b @ np.linalg.solve(V, b) / len(slopes))
    assert res.e["F"] == pytest.approx(expected, rel=1e-12)

    plain = OLS.from_formula("y ~ x1 + x2", df).fit()
    assert res.e["F"] != pytest.approx(plain.e["F"], rel=1e-6)


def test_nonrobust_keeps_df_resid_equal_to_e_df_r():
    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit()
    assert res.df_resid == res.e["df_r"] == len(df) - 3
    assert "N_clust" not in res.e


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_heteroskedastic_variants_use_n_minus_k(cov_type):
    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(cov_type=cov_type)
    assert res.df_resid == len(df) - 3
    assert res.e["df_r"] == len(df) - 3
