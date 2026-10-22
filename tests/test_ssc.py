"""Tests for westata's Stata small-sample corrections (``vce.ssc``).

The correction factors are the quantities that most often separate a
"Stata-identical" standard error from a merely plausible one, so each factor is
pinned against the formula in ``[R] regress`` (Methods and formulas):

* HC1 robust: ``N / (N - K)``
* cluster CR1: ``G / (G - 1) * (N - 1) / (N - K)``

Defaults come from :class:`westata.protocols.SSC`
(``adj=True, cluster_adj=True, df="stata"``), i.e. Stata's defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

from westata.protocols import SSC
from westata.vce.dispatch import compute_cov
from westata.vce.ssc import cluster_factor, hc1_factor, resolve_ssc


def _toy(n: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    resid = rng.normal(size=n)
    return X, resid, np.linalg.inv(X.T @ X)


def _cluster_sandwich(X, resid, groups):
    """Raw CR0 sandwich (f = 1), written out independently of the library."""
    k = X.shape[1]
    xu = X * resid[:, None]
    meat = np.zeros((k, k))
    for gval in np.unique(groups):
        score = xu[groups == gval].sum(axis=0)
        meat += np.outer(score, score)
    bread = np.linalg.inv(X.T @ X)
    return bread @ meat @ bread


# ---------------------------------------------------------------------------
# defaults
# ---------------------------------------------------------------------------


def test_resolve_ssc_none_uses_stata_defaults():
    resolved = resolve_ssc(None)
    assert isinstance(resolved, SSC)
    assert resolved.adj is True
    assert resolved.cluster_adj is True
    assert resolved.df == "stata"


def test_resolve_ssc_passes_through_custom_object():
    custom = SSC(adj=False, cluster_adj=False, df="none")
    assert resolve_ssc(custom) is custom


def test_hc1_factor_is_n_over_n_minus_k():
    n, k = 30, 3
    assert hc1_factor(n, k, None) == pytest.approx(n / (n - k))
    assert hc1_factor(n, k, SSC()) == pytest.approx(n / (n - k))


def test_hc1_factor_toggles_with_adj():
    assert hc1_factor(30, 3, SSC(adj=False)) == 1.0


def test_cluster_factor_default_is_stata_cr1():
    n, k, groups = 30, 3, 6
    expected = (n - 1.0) / (n - k) * groups / (groups - 1.0)
    assert cluster_factor(n, k, groups, None) == pytest.approx(expected)


def test_cluster_factor_adj_false_removes_all_correction():
    assert cluster_factor(30, 3, 6, SSC(adj=False)) == 1.0


def test_cluster_factor_cluster_adj_false_keeps_only_df_correction():
    n, k, groups = 30, 3, 6
    assert cluster_factor(n, k, groups, SSC(cluster_adj=False)) == pytest.approx(
        (n - 1.0) / (n - k)
    )
    assert cluster_factor(n, k, groups, SSC(adj=False, cluster_adj=False)) == 1.0


# ---------------------------------------------------------------------------
# the flags actually change computed covariances (end-to-end through dispatch)
# ---------------------------------------------------------------------------


def test_cluster_adj_toggle_changes_cov():
    X, resid, _ = _toy()
    n, k = X.shape
    groups = np.repeat(np.arange(6), n // 6)
    G = len(np.unique(groups))
    sandwich = _cluster_sandwich(X, resid, groups)

    cov_default, _ = compute_cov(
        X, resid, cov_type="cluster", cov_kwds={"groups": groups}
    )
    cov_no_cluster_adj, _ = compute_cov(
        X,
        resid,
        cov_type="cluster",
        cov_kwds={"groups": groups},
        ssc=SSC(cluster_adj=False),
    )
    cov_no_adj, _ = compute_cov(
        X,
        resid,
        cov_type="cluster",
        cov_kwds={"groups": groups},
        ssc=SSC(adj=False),
    )

    np.testing.assert_allclose(
        cov_default,
        (n - 1.0) / (n - k) * G / (G - 1.0) * sandwich,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        cov_no_cluster_adj,
        (n - 1.0) / (n - k) * sandwich,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(cov_no_adj, sandwich, rtol=1e-12, atol=0.0)
    assert not np.allclose(cov_default, cov_no_adj)


def test_adj_toggle_changes_hc1_cov():
    X, resid, _ = _toy()
    n, k = X.shape

    cov_adj, _ = compute_cov(X, resid, cov_type="HC1")
    cov_no_adj, _ = compute_cov(X, resid, cov_type="HC1", ssc=SSC(adj=False))

    np.testing.assert_allclose(cov_adj, n / (n - k) * cov_no_adj, rtol=1e-12, atol=0.0)


def test_ssc_none_equals_default_ssc():
    X, resid, _ = _toy()
    groups = np.repeat(np.arange(5), X.shape[0] // 5)
    cov_none, df_none = compute_cov(
        X, resid, cov_type="cluster", cov_kwds={"groups": groups}
    )
    cov_default, df_default = compute_cov(
        X, resid, cov_type="cluster", cov_kwds={"groups": groups}, ssc=SSC()
    )
    np.testing.assert_allclose(cov_none, cov_default, rtol=0.0, atol=0.0)
    assert df_none == df_default
