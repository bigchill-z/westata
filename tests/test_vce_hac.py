"""Tests for the Newey-West HAC covariance (``westata.vce.hac``).

Transcribed from ``newey.ado:133-183``:

    for j = 0..lag:
        cross_j = sum_t g_t g_{t-j}'      with g_t = x_t * e_t
        tt_j    = (cross_j + cross_j') * (1 - j/(1+lag))
        j == 0:  xtx = tt_0 * 0.5         -> equals cross_0 (it is symmetric)
        j >  0:  xtx = xtx + tt_j

The ``0.5`` on the j = 0 term only undoes the duplicated transpose, so the
accumulated matrix is

    xtx = cross_0 + sum_{j>=1} w_j (cross_j + cross_j'),   w_j = 1 - j/(1+lag)

``newey.sthlp:113-114`` states that ``lag(0)`` gives the same output as
``vce(robust)``.  In Stata ``vce(robust)`` carries the ``N/(N-K)`` factor, so
``lag(0)`` matches **HC1**, not HC0 -- a distinction worth a dedicated test,
because comparing against HC0 shows a clean ratio of exactly ``N/(N-K)`` and
looks like a kernel bug rather than a scaling convention.
"""

from __future__ import annotations

import numpy as np
import pytest

from westata.protocols import SSC
from westata.vce.dispatch import SUPPORTED_COV_TYPES, compute_cov
from westata.vce.hac import bartlett_weights, cov_hac, hac_meat


@pytest.fixture()
def sample():
    rng = np.random.default_rng(11)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    resid = rng.normal(size=n)
    return X, resid


# ------------------------------------------------------------------ kernels
def test_bartlett_weights_follow_the_ado() -> None:
    for lags in (0, 1, 2, 3, 7):
        expected = [1.0 - j / (1.0 + lags) for j in range(lags + 1)]
        assert np.allclose(bartlett_weights(lags), expected), lags


def test_bartlett_weights_reject_a_negative_lag() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        bartlett_weights(-1)


def test_bartlett_weights_start_at_one_and_end_above_zero() -> None:
    """lap 0 has weight 1; the last lag has weight 1/(1+lag) > 0."""

    weights = bartlett_weights(4)

    assert weights[0] == 1.0
    assert weights[-1] == pytest.approx(1.0 / 5.0)
    assert np.all(np.diff(weights) < 0)


# --------------------------------------------------------------------- meat
def test_meat_matches_a_direct_transcription(sample) -> None:
    X, resid = sample
    scores = X * resid[:, None]
    n = scores.shape[0]

    for lags in (0, 1, 2, 4):
        expected = scores.T @ scores
        for j in range(1, lags + 1):
            cross = scores[j:].T @ scores[: n - j]
            weight = 1.0 - j / (1.0 + lags)
            expected = expected + weight * (cross + cross.T)
        assert np.allclose(hac_meat(X, resid, lags), expected), lags


def test_meat_at_lag_zero_is_the_hc0_meat(sample) -> None:
    X, resid = sample
    scores = X * resid[:, None]

    assert np.allclose(hac_meat(X, resid, 0), scores.T @ scores)


def test_meat_is_symmetric(sample) -> None:
    X, resid = sample

    meat = hac_meat(X, resid, 3)

    assert np.allclose(meat, meat.T)


def test_larger_lag_changes_the_meat(sample) -> None:
    X, resid = sample

    assert not np.allclose(hac_meat(X, resid, 0), hac_meat(X, resid, 2))


def test_meat_rejects_a_negative_lag(sample) -> None:
    X, resid = sample

    with pytest.raises(ValueError, match="nonnegative"):
        hac_meat(X, resid, -1)


def test_meat_rejects_misaligned_input(sample) -> None:
    X, resid = sample

    with pytest.raises(ValueError, match="rows"):
        hac_meat(X, resid[:-1], 1)


# ------------------------------------------------------------- dispatch level
def test_hac_is_a_supported_cov_type() -> None:
    assert "HAC" in SUPPORTED_COV_TYPES


def test_newey_alias_is_accepted(sample) -> None:
    X, resid = sample

    canonical, _ = compute_cov(X, resid, cov_type="HAC", cov_kwds={"maxlags": 2})
    alias, _ = compute_cov(X, resid, cov_type="newey", cov_kwds={"lags": 2})

    assert np.allclose(canonical, alias)


def test_maxlags_is_required(sample) -> None:
    """``newey.sthlp:55``: lag() is required, with no default."""

    X, resid = sample

    with pytest.raises(ValueError, match="maxlags"):
        compute_cov(X, resid, cov_type="HAC")


def test_negative_maxlags_is_refused(sample) -> None:
    """``newey.ado:24-27`` rejects a negative lag."""

    X, resid = sample

    with pytest.raises(ValueError, match="nonnegative"):
        compute_cov(X, resid, cov_type="HAC", cov_kwds={"maxlags": -1})


def test_lag_zero_equals_hc1(sample) -> None:
    """``newey.sthlp:113-114``: lag(0) reproduces ``vce(robust)``.

    Stata's ``vce(robust)`` includes the ``N/(N-K)`` factor, so the match is
    against HC1.  Against HC0 the ratio would be exactly ``N/(N-K)``.
    """

    X, resid = sample
    n, k = X.shape

    hac, _ = compute_cov(X, resid, cov_type="HAC", cov_kwds={"maxlags": 0})
    hc1, _ = compute_cov(X, resid, cov_type="HC1")

    assert np.allclose(hac, hc1, rtol=1e-12)
    # And explicitly NOT HC0, by the documented factor.
    hc0, _ = compute_cov(X, resid, cov_type="HC0")
    assert np.allclose(hac / hc0, n / (n - k), rtol=1e-12)


def test_larger_lag_increases_variance_under_positive_autocorrelation() -> None:
    """Direction check: with positively autocorrelated scores HAC grows.

    This has to be constructed carefully.  Adding lags does **not** monotonically
    increase a HAC variance -- the cross terms carry the sign of the
    autocorrelation, so with negative autocorrelation the variance *shrinks*.
    An earlier version of this test simply compared lag 0, 2, 5 on an AR(1)
    residual and failed, because the realised sample autocovariances were not all
    positive.  Here the regressor is deliberately serially correlated in the same
    direction as the residual, so every added lag contributes positively.
    """

    rng = np.random.default_rng(5)
    n = 400
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.8 * x[t - 1] + rng.normal()
    resid = np.zeros(n)
    for t in range(1, n):
        resid[t] = 0.8 * resid[t - 1] + rng.normal()
    X = np.column_stack([np.ones(n), x])
    scores = X * resid[:, None]

    # Every lag-1..5 cross-covariance of the scores must be positive for the
    # monotone claim to be meaningful.
    for j in range(1, 6):
        cross = scores[j:].T @ scores[: n - j]
        assert cross[1, 1] > 0, j

    variance = []
    for lags in (0, 2, 5):
        cov, _ = compute_cov(X, resid, cov_type="HAC", cov_kwds={"maxlags": lags})
        variance.append(float(cov[1, 1]))

    assert variance[0] < variance[1] < variance[2], variance


def test_larger_lag_shrinks_variance_under_negative_autocorrelation() -> None:
    """The mirror case, which fixes the direction convention.

    Without this test the implementation could invert the kernel weight and the
    positive-autocorrelation check would still pass.
    """

    rng = np.random.default_rng(7)
    n = 400
    x = rng.normal(size=n)
    resid = np.zeros(n)
    for t in range(1, n):
        # Alternating sign -> strongly negative lag-1 autocorrelation.
        resid[t] = -0.7 * resid[t - 1] + rng.normal()
    X = np.column_stack([np.ones(n), x])
    scores = X * resid[:, None]

    cross = scores[1:].T @ scores[: n - 1]
    assert cross[1, 1] < 0

    variance = []
    for lags in (0, 2):
        cov, _ = compute_cov(X, resid, cov_type="HAC", cov_kwds={"maxlags": lags})
        variance.append(float(cov[1, 1]))

    assert variance[1] < variance[0], variance


def test_df_is_n_minus_k(sample) -> None:
    X, resid = sample
    n, k = X.shape

    _, df = compute_cov(X, resid, cov_type="HAC", cov_kwds={"maxlags": 2})

    assert df == float(n - k)


def test_ssc_adj_off_drops_the_factor(sample) -> None:
    X, resid = sample
    n, k = X.shape

    adjusted, _ = compute_cov(X, resid, cov_type="HAC", cov_kwds={"maxlags": 1})
    plain, _ = compute_cov(
        X, resid, cov_type="HAC", cov_kwds={"maxlags": 1}, ssc=SSC(adj=False)
    )

    assert np.allclose(adjusted / plain, n / (n - k), rtol=1e-12)


def test_weighted_hac_uses_the_weighted_space(sample) -> None:
    """The dispatcher hands ``hac_meat`` X~ = sqrt(w)X and e~ = sqrt(w)e."""

    X, resid = sample
    w = np.linspace(0.5, 1.5, X.shape[0])

    cov, _ = compute_cov(
        X, resid, cov_type="HAC", cov_kwds={"maxlags": 1}, weights=w
    )

    Xw = X * np.sqrt(w)[:, None]
    rw = resid * np.sqrt(w)
    bread = np.linalg.inv(Xw.T @ Xw)
    expected = cov_hac(
        Xw, rw, lags=1, bread=bread, nobs=float(X.shape[0]),
        df_resid=float(X.shape[0] - X.shape[1]),
    )

    assert np.allclose(cov, expected, rtol=1e-12)
