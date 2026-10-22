"""Tests for heteroskedasticity-robust VCEs (HC0-HC3).

Formulas follow ``[R] regress`` (Methods and formulas) / White (1980):

* ``HC0 = bread @ (X' diag(u_i^2) X) @ bread``
* ``HC1 = N / (N - K) * HC0``  (Stata's ``regress, robust``)
* ``HC2`` uses ``u_i^2 / (1 - h_i)``
* ``HC3`` uses ``u_i^2 / (1 - h_i)^2`` with ``h_i = x_i' (X'X)^-1 x_i``

Every hand-computed expectation is written out with plain numpy so the test is
an independent oracle, not a re-run of the implementation.  statsmodels is used
as a second, independent implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm

from westata.protocols import SSC
from westata.vce.dispatch import compute_cov
from westata.vce.robust import bread, cov_hc, leverage


def _toy(n: int = 40, seed: int = 1):
    """Return (X, resid, bread) for an independent hand calculation."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    resid = rng.normal(size=n)
    return X, resid, np.linalg.inv(X.T @ X)


def _ols_data(n: int = 50, seed: int = 2):
    """Return (X, y, statsmodels OLS results) with non-trivial residuals."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([1.0, -2.0, 0.5])
    y = X @ beta + rng.normal(size=n)
    return X, y, sm.OLS(y, X).fit()


def test_bread_is_inverse_xtx():
    X, resid, _ = _toy()
    np.testing.assert_allclose(bread(X), np.linalg.inv(X.T @ X), rtol=1e-14, atol=0.0)


def test_leverage_matches_diagonal_of_hat_matrix():
    X, _, bread_inv = _toy()
    expected = np.diag(X @ bread_inv @ X.T)
    np.testing.assert_allclose(leverage(X, bread_inv), expected, rtol=1e-12, atol=0.0)


def test_hc0_matches_manual_formula():
    X, resid, bread_inv = _toy()
    n, k = X.shape

    cov, df_t = compute_cov(X, resid, cov_type="HC0")

    meat = (X.T * resid**2) @ X
    expected = bread_inv @ meat @ bread_inv
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=0.0)
    assert df_t == n - k


def test_hc1_matches_manual_formula():
    X, resid, bread_inv = _toy()
    n, k = X.shape

    cov, df_t = compute_cov(X, resid, cov_type="HC1")

    meat = (X.T * resid**2) @ X
    expected = (n / (n - k)) * bread_inv @ meat @ bread_inv
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=0.0)
    assert df_t == n - k


def test_hc2_matches_manual_formula():
    X, resid, bread_inv = _toy()
    n, k = X.shape

    cov, df_t = compute_cov(X, resid, cov_type="HC2")

    h = np.diag(X @ bread_inv @ X.T)
    meat = (X.T * (resid**2 / (1.0 - h))) @ X
    expected = bread_inv @ meat @ bread_inv
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=0.0)
    assert df_t == n - k


def test_hc3_matches_manual_formula():
    X, resid, bread_inv = _toy()
    n, k = X.shape

    cov, df_t = compute_cov(X, resid, cov_type="HC3")

    h = np.diag(X @ bread_inv @ X.T)
    meat = (X.T * (resid**2 / (1.0 - h) ** 2)) @ X
    expected = bread_inv @ meat @ bread_inv
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=0.0)
    assert df_t == n - k


def test_hc1_equals_hc0_times_n_over_n_minus_k():
    X, resid, _ = _toy()
    n, k = X.shape
    hc0, _ = compute_cov(X, resid, cov_type="HC0")
    hc1, _ = compute_cov(X, resid, cov_type="HC1")
    np.testing.assert_allclose(hc1, (n / (n - k)) * hc0, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("variant", ["HC0", "HC1", "HC2", "HC3"])
def test_robust_df_t_is_n_minus_k(variant):
    X, resid, _ = _toy()
    n, k = X.shape
    _, df_t = compute_cov(X, resid, cov_type=variant)
    assert df_t == n - k


@pytest.mark.parametrize("variant", ["HC0", "HC1", "HC2", "HC3"])
def test_robust_matches_statsmodels(variant):
    X, y, res = _ols_data()
    expected = sm.OLS(y, X).fit(cov_type=variant).cov_params()

    cov, df_t = compute_cov(X, res.resid, cov_type=variant)

    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=1e-14)
    assert df_t == res.df_resid


def test_cov_hc_accepts_explicit_bread_and_ssc():
    X, resid, bread_inv = _toy()
    n, k = X.shape

    direct = cov_hc(X, resid, variant="HC1", bread=bread_inv, ssc=SSC(adj=False))
    expected = bread_inv @ ((X.T * resid**2) @ X) @ bread_inv
    np.testing.assert_allclose(direct, expected, rtol=1e-12, atol=0.0)


def test_cov_hc_rejects_unknown_variant():
    X, resid, _ = _toy()
    with pytest.raises(ValueError, match="variant"):
        cov_hc(X, resid, variant="HC9")
