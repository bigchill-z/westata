"""Documented behaviour of the cluster-robust degrees of freedom.

Stata ``[R] regress`` -> "Degrees of freedom" (p. 26 of the reference manual):

    For one-way cluster-robust variance estimators, p-values and confidence
    intervals are derived from a t distribution with G - 1 degrees of freedom,
    where G is the total number of clusters.

That sentence governs *inference* (t tests, confidence intervals and the
model F test).  It does not redefine ``e(df_r)``, which Stata documents as
"residual degrees of freedom" (``e(N) - rank(e(V))``).

westata therefore keeps the two concepts separate and explicit:

``results.df_resid``
    The reference-distribution degrees of freedom -- ``G - 1`` under clustering,
    ``N - K`` otherwise.  Used for t/p/CI and for the displayed ``F(df_m, .)``.
``results.e["df_r"]``
    The residual degrees of freedom ``N - K``, matching Stata's scalar.

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


def test_cluster_uses_g_minus_1_for_inference_and_n_minus_k_for_e_df_r():
    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["g"]}
    )

    n_groups = df["g"].nunique()
    n = len(df)
    k = 3  # Intercept, x1, x2

    # Inference uses G - 1 (Stata [R] regress, "Degrees of freedom").
    assert res.df_resid == n_groups - 1
    # e(df_r) stays the residual degrees of freedom N - K.
    assert res.e["df_r"] == n - k

    # The two must not accidentally coincide in this test, otherwise it proves
    # nothing about which quantity is used where.
    assert res.df_resid != res.e["df_r"]


def test_cluster_pvalues_use_g_minus_1_not_n_minus_k():
    from scipy import stats

    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(
        cov_type="cluster", cov_kwds={"groups": df["g"]}
    )

    t = res.tvalues.to_numpy()
    expected = 2.0 * stats.t.sf(np.abs(t), res.df_resid)
    np.testing.assert_allclose(res.pvalues.to_numpy(), expected, rtol=1e-12)

    wrong = 2.0 * stats.t.sf(np.abs(t), res.e["df_r"])
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


def test_nonrobust_keeps_df_resid_equal_to_e_df_r():
    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit()
    assert res.df_resid == res.e["df_r"] == len(df) - 3


@pytest.mark.parametrize("cov_type", ["HC0", "HC1", "HC2", "HC3"])
def test_heteroskedastic_variants_use_n_minus_k(cov_type):
    df = _cluster_data()
    res = OLS.from_formula("y ~ x1 + x2", df).fit(cov_type=cov_type)
    assert res.df_resid == len(df) - 3
    assert res.e["df_r"] == len(df) - 3
