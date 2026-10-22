"""Tests for the VCE dispatcher (``cov_type``/``cov_kwds`` -> covariance).

``compute_cov`` is the single entry point every estimator uses; these tests pin
its public contract:

* ``nonrobust``: ``sigma2 * (X'X)^-1`` with ``df_t = N - K``
* robust HC0-HC3: ``df_t = N - K``
* cluster: ``df_t = G - 1``
* unknown ``cov_type`` fails loudly and lists the supported values
"""

from __future__ import annotations

import numpy as np
import pytest

from westata.protocols import SSC
from westata.vce import compute_cov as package_compute_cov
from westata.vce.dispatch import SUPPORTED_COV_TYPES, compute_cov


def _toy(n: int = 30, seed: int = 4):
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    resid = rng.normal(size=n)
    return X, resid, np.linalg.inv(X.T @ X)


def test_default_cov_type_is_nonrobust():
    X, resid, bread = _toy()
    n, k = X.shape

    cov, df_t = compute_cov(X, resid)

    expected = (resid @ resid / (n - k)) * bread
    np.testing.assert_allclose(cov, expected, rtol=1e-12, atol=0.0)
    assert df_t == n - k


def test_nonrobust_explicit_matches_default():
    X, resid, _ = _toy()
    np.testing.assert_allclose(
        compute_cov(X, resid, cov_type="nonrobust")[0],
        compute_cov(X, resid)[0],
        rtol=0.0,
        atol=0.0,
    )


def test_nonrobust_honours_supplied_df_resid():
    X, resid, bread = _toy()
    n, k = X.shape
    df_resid = float(n - k - 1)

    cov, df_t = compute_cov(X, resid, df_resid=df_resid)

    np.testing.assert_allclose(
        cov, (resid @ resid / df_resid) * bread, rtol=1e-12, atol=0.0
    )
    assert df_t == df_resid


def test_nonrobust_df_resid_does_not_affect_robust_types():
    X, resid, _ = _toy()
    n, k = X.shape
    cov_a, df_a = compute_cov(X, resid, cov_type="HC1")
    cov_b, df_b = compute_cov(X, resid, cov_type="HC1", df_resid=float(n - k - 2))
    np.testing.assert_allclose(cov_b, cov_a, rtol=0.0, atol=0.0)
    assert df_b == df_a == n - k


def test_supported_cov_types_tuple():
    assert SUPPORTED_COV_TYPES == ("nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster")


@pytest.mark.parametrize("bad", ["bogus", "HAC", "twoway", "robust", ""])
def test_unknown_cov_type_raises_value_error(bad):
    X, resid, _ = _toy()
    with pytest.raises(ValueError) as excinfo:
        compute_cov(X, resid, cov_type=bad)

    message = str(excinfo.value)
    assert "cov_type" in message
    for name in SUPPORTED_COV_TYPES:
        assert name in message


@pytest.mark.parametrize(
    "alias,canonical", [("hc0", "HC0"), ("hc1", "HC1"), ("hc2", "HC2"), ("hc3", "HC3")]
)
def test_cov_type_is_case_insensitive(alias, canonical):
    X, resid, _ = _toy()
    np.testing.assert_allclose(
        compute_cov(X, resid, cov_type=alias)[0],
        compute_cov(X, resid, cov_type=canonical)[0],
        rtol=0.0,
        atol=0.0,
    )


def test_cluster_dispatches_and_returns_g_minus_one_df():
    X, resid, _ = _toy(n=30)
    groups = np.repeat(np.arange(6), 5)

    cov, df_t = compute_cov(
        X, resid, cov_type="cluster", cov_kwds={"groups": groups}
    )

    assert cov.shape == (X.shape[1], X.shape[1])
    assert np.isfinite(cov).all()
    np.testing.assert_allclose(cov, cov.T, rtol=1e-12, atol=1e-15)
    assert df_t == 5


def test_cluster_requires_cov_kwds_mapping_with_groups():
    X, resid, _ = _toy()
    with pytest.raises(TypeError, match="cov_kwds"):
        compute_cov(X, resid, cov_type="cluster", cov_kwds=[("groups", np.zeros(30))])
    with pytest.raises(ValueError, match="groups"):
        compute_cov(X, resid, cov_type="cluster", cov_kwds={"time": np.zeros(30)})


def test_nobs_argument_must_match_data():
    X, resid, _ = _toy()
    n = X.shape[0]
    cov_a, _ = compute_cov(X, resid, nobs=n)
    cov_b, _ = compute_cov(X, resid)
    np.testing.assert_allclose(cov_a, cov_b, rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="nobs"):
        compute_cov(X, resid, nobs=n + 1)


@pytest.mark.parametrize(
    "X,resid,match",
    [
        (np.ones(30), np.ones(30), "X"),
        (np.ones((30, 2)), np.ones(29), "resid"),
        (np.ones((2, 3)), np.ones(2), "columns"),
    ],
)
def test_shape_validation(X, resid, match):
    with pytest.raises(ValueError, match=match):
        compute_cov(X, resid)


def test_returns_ndarray_and_float():
    X, resid, _ = _toy()
    cov, df_t = compute_cov(X, resid, cov_type="HC0")
    assert isinstance(cov, np.ndarray)
    assert cov.dtype == np.float64
    assert isinstance(df_t, float)


def test_package_reexports():
    from westata.protocols import SSC as ProtocolSSC

    import westata.vce as vce

    assert package_compute_cov is compute_cov
    assert vce.SSC is ProtocolSSC
