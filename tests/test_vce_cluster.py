"""Tests for one-way cluster-robust VCE (CR1, Stata ``vce(cluster id)``).

CR1 from ``[R] regress`` (Methods and formulas):

    V = G/(G-1) * (N-1)/(N-K) * (X'X)^-1 [sum_g X_g' u_g u_g' X_g] (X'X)^-1

with ``df_t = G - 1``.  statsmodels' ``cov_type="cluster"`` with
``use_correction=True`` implements exactly this, so it is used as an independent
oracle next to the hand formula.  The hand formula remains the source of truth:
if statsmodels ever drifts, the hand-formula test stays and the statsmodels test
is what gets reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from westata.protocols import SSC
from westata.vce.dispatch import compute_cov


def _toy(n: int = 48, seed: int = 3):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([0.5, 1.0, -1.5])
    y = X @ beta + rng.normal(size=n)
    res = sm.OLS(y, X).fit()
    groups = rng.integers(0, 7, size=n)
    return X, y, groups, res


def _raw_sandwich(X, resid, groups):
    """CR0 sandwich with f = 1, hand-written (loop over groups, not rows)."""
    k = X.shape[1]
    bread = np.linalg.inv(X.T @ X)
    xu = X * resid[:, None]
    meat = np.zeros((k, k))
    for gval in np.unique(groups):
        score = xu[groups == gval].sum(axis=0)
        meat += np.outer(score, score)
    return bread @ meat @ bread


def test_cr1_matches_hand_computed_formula():
    X, _, groups, res = _toy()
    n, k = X.shape
    G = len(np.unique(groups))

    cov, df_t = compute_cov(
        X, res.resid, cov_type="cluster", cov_kwds={"groups": groups}
    )

    expected = (
        G / (G - 1.0) * (n - 1.0) / (n - k) * _raw_sandwich(X, res.resid, groups)
    )
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=0.0)
    assert df_t == G - 1


def test_cr1_matches_statsmodels_cluster_with_correction():
    X, y, groups, _ = _toy()
    # use_correction lives in cov_kwds in statsmodels; True is CR1 (= Stata).
    expected = (
        sm.OLS(y, X)
        .fit(
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": True},
        )
        .cov_params()
    )

    cov, df_t = compute_cov(
        X,
        sm.OLS(y, X).fit().resid,
        cov_type="cluster",
        cov_kwds={"groups": groups},
    )

    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=1e-14)
    assert df_t == len(np.unique(groups)) - 1


def test_raw_sandwich_matches_statsmodels_without_correction():
    X, y, groups, _ = _toy()
    expected = (
        sm.OLS(y, X)
        .fit(
            cov_type="cluster",
            cov_kwds={"groups": groups, "use_correction": False},
        )
        .cov_params()
    )

    cov, _ = compute_cov(
        X,
        sm.OLS(y, X).fit().resid,
        cov_type="cluster",
        cov_kwds={"groups": groups},
        ssc=SSC(adj=False),
    )

    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=1e-14)


def test_cluster_adj_false_keeps_only_df_correction():
    X, _, groups, res = _toy()
    n, k = X.shape
    sandwich = _raw_sandwich(X, res.resid, groups)

    cov, _ = compute_cov(
        X,
        res.resid,
        cov_type="cluster",
        cov_kwds={"groups": groups},
        ssc=SSC(cluster_adj=False),
    )

    np.testing.assert_allclose(
        cov, (n - 1.0) / (n - k) * sandwich, rtol=1e-12, atol=0.0
    )


def test_cluster_df_t_is_g_minus_one():
    X, _, groups, res = _toy()
    _, df_t = compute_cov(
        X, res.resid, cov_type="cluster", cov_kwds={"groups": groups}
    )
    assert df_t == len(np.unique(groups)) - 1


def test_singleton_clusters_reduce_cr1_to_hc1():
    """With G = N the CR1 factor N/(N-K) is exactly the HC1 factor."""

    X, _, _, res = _toy()
    n, k = X.shape
    cov_cluster_cov, df_cluster = compute_cov(
        X, res.resid, cov_type="cluster", cov_kwds={"groups": np.arange(n)}
    )
    cov_hc1, _ = compute_cov(X, res.resid, cov_type="HC1")

    np.testing.assert_allclose(cov_cluster_cov, cov_hc1, rtol=1e-12, atol=0.0)
    assert df_cluster == n - 1


def test_group_labels_are_irrelevant():
    X, _, groups, res = _toy()
    labels = np.array([f"industry-{g}" for g in groups])

    cov_int, df_int = compute_cov(
        X, res.resid, cov_type="cluster", cov_kwds={"groups": groups}
    )
    cov_str, df_str = compute_cov(
        X, res.resid, cov_type="cluster", cov_kwds={"groups": labels}
    )

    np.testing.assert_allclose(cov_str, cov_int, rtol=1e-12, atol=1e-15)
    assert df_str == df_int


def test_groups_accepts_pandas_series():
    X, _, groups, res = _toy()
    series = pd.Series(groups, index=np.arange(X.shape[0])[::-1])

    cov_series, df_series = compute_cov(
        X, res.resid, cov_type="cluster", cov_kwds={"groups": series}
    )
    cov_array, df_array = compute_cov(
        X, res.resid, cov_type="cluster", cov_kwds={"groups": groups}
    )

    np.testing.assert_allclose(cov_series, cov_array, rtol=0.0, atol=0.0)
    assert df_series == df_array


def test_categorical_group_labels_match_unencoded_labels():
    """``df["industry"]`` is often categorical; unused levels must not add groups."""

    X, _, groups, res = _toy()
    categorical = pd.Series(
        pd.Categorical(groups, categories=[6, 5, 4, 3, 2, 1, 0, 10, 11])
    )

    cov_cat, df_cat = compute_cov(
        X, res.resid, cov_type="cluster", cov_kwds={"groups": categorical}
    )
    cov_raw, df_raw = compute_cov(
        X, res.resid, cov_type="cluster", cov_kwds={"groups": groups}
    )

    np.testing.assert_allclose(cov_cat, cov_raw, rtol=1e-12, atol=1e-15)
    assert df_cat == df_raw == len(np.unique(groups)) - 1


def test_missing_groups_key_raises():
    X, _, _, res = _toy()
    with pytest.raises(ValueError, match="groups"):
        compute_cov(X, res.resid, cov_type="cluster")
    with pytest.raises(ValueError, match="groups"):
        compute_cov(X, res.resid, cov_type="cluster", cov_kwds={})


def test_groups_length_mismatch_raises():
    X, _, groups, res = _toy()
    with pytest.raises(ValueError, match="length"):
        compute_cov(
            X, res.resid, cov_type="cluster", cov_kwds={"groups": groups[:-1]}
        )


def test_two_dimensional_groups_raise():
    X, _, groups, res = _toy()
    two_way = np.column_stack([groups, groups[::-1]])
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_cov(X, res.resid, cov_type="cluster", cov_kwds={"groups": two_way})


def test_single_group_raises():
    X, _, _, res = _toy()
    with pytest.raises(ValueError, match="at least 2"):
        compute_cov(
            X,
            res.resid,
            cov_type="cluster",
            cov_kwds={"groups": np.zeros(X.shape[0], dtype=int)},
        )


def test_missing_group_labels_raise():
    X, _, groups, res = _toy()
    with_nan = groups.astype(float)
    with_nan[0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        compute_cov(
            X, res.resid, cov_type="cluster", cov_kwds={"groups": with_nan}
        )
