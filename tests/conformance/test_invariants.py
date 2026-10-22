"""Tests for the OLS invariant oracle and numeric tolerance helper.

Invariants (design spec section 11.2 ``InvariantOracle``) are the cheapest
evidence level: they need no published Stata output, only identities any
correct OLS fit must satisfy.  A violation here means the estimator itself is
wrong, regardless of how close it happens to land to a fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from westata.conformance.oracles.invariants import check_ols_invariants
from westata.conformance.tolerances import (
    LINEAR_RTOL,
    MLE_RTOL,
    MLE_SE_RTOL,
    assert_close,
)


def _fit(
    *, has_intercept: bool = True, seed: int = 0, n: int = 40
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return an exactly-solved OLS fit ``(X, params, resid)``.

    The data-generating process always contains a constant: when the design
    omits the intercept, ``sum(resid)`` stays away from zero while
    ``X.T @ resid`` is still zero.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 3.0 + 2.0 * x + rng.normal(scale=0.5, size=n)
    columns = [np.ones(n), x] if has_intercept else [x]
    X = np.column_stack(columns)
    params, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ params
    return X, params, resid


# ---------------------------------------------------------------------------
# check_ols_invariants
# ---------------------------------------------------------------------------


def test_invariants_pass_on_consistent_ols_fit() -> None:
    X, params, resid = _fit()

    assert check_ols_invariants(X, resid, params, has_intercept=True) is None


def test_invariants_pass_on_consistent_fit_without_intercept() -> None:
    X, params, resid = _fit(has_intercept=False)

    assert check_ols_invariants(X, resid, params, has_intercept=False) is None


def test_invariants_fail_on_inconsistent_residuals() -> None:
    X, params, resid = _fit()
    # Perturb along a regressor: the normal equations gain an O(1e-3) entry
    # while the intercept condition still holds up to rounding.
    perturbed = resid + 1e-3 * X[:, 1]

    with pytest.raises(AssertionError, match=r"X\.T @ resid"):
        check_ols_invariants(X, perturbed, params, has_intercept=True)


def test_intercept_condition_fails_when_intercept_is_missing() -> None:
    X, params, resid = _fit(has_intercept=False)
    assert abs(float(np.sum(resid))) > 1e-6  # sanity: the data violates it

    with pytest.raises(AssertionError, match="intercept"):
        check_ols_invariants(X, resid, params, has_intercept=True)


def test_invariants_use_scale_aware_atol() -> None:
    # Large-magnitude regressors amplify floating-point cancellation in
    # ``X.T @ resid``; the default atol must still accept an exact solve.
    rng = np.random.default_rng(7)
    n = 74
    wage_like = 1000.0 + 3000.0 * rng.random(n)
    X = np.column_stack([np.ones(n), wage_like])
    y = 40.0 - 0.006 * wage_like + rng.normal(scale=2.0, size=n)
    params, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ params

    assert check_ols_invariants(X, resid, params, has_intercept=True) is None


def test_invariants_reject_mismatched_params() -> None:
    X, params, resid = _fit()

    with pytest.raises(ValueError, match="params"):
        check_ols_invariants(X, resid, params[:-1], has_intercept=True)


def test_invariants_reject_mismatched_residuals() -> None:
    X, params, resid = _fit()

    with pytest.raises(ValueError, match="resid"):
        check_ols_invariants(X, resid[:-1], params, has_intercept=True)


def test_invariants_reject_non_positive_atol() -> None:
    X, params, resid = _fit()

    with pytest.raises(ValueError, match="atol"):
        check_ols_invariants(X, resid, params, has_intercept=True, atol=0.0)


# ---------------------------------------------------------------------------
# Tolerance constants and assert_close
# ---------------------------------------------------------------------------


def test_tolerance_constants_match_spec() -> None:
    assert (LINEAR_RTOL, MLE_RTOL, MLE_SE_RTOL) == pytest.approx((1e-10, 1e-5, 1e-4))


def test_assert_close_passes_within_tolerance() -> None:
    assert assert_close(1.0 + 5e-12, 1.0, LINEAR_RTOL, label="beta[weight]") is None


def test_assert_close_supports_arrays() -> None:
    actual = np.array([1.0, -2.0, 3.5])
    expected = actual * (1.0 + 5e-12)

    assert assert_close(actual, expected, LINEAR_RTOL) is None


def test_assert_close_raises_outside_tolerance() -> None:
    with pytest.raises(AssertionError) as excinfo:
        assert_close(1.0 + 1e-8, 1.0, LINEAR_RTOL, label="beta[weight]")

    message = str(excinfo.value)
    assert "beta[weight]" in message
    assert "actual" in message
    assert "expected" in message
    assert "relative error" in message


def test_assert_close_reports_largest_array_deviation() -> None:
    with pytest.raises(AssertionError, match="max relative error"):
        assert_close(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-6]), LINEAR_RTOL)


@pytest.mark.parametrize("rtol", [0.0, -1e-10, float("inf"), float("nan")])
def test_assert_close_rejects_invalid_rtol(rtol: float) -> None:
    with pytest.raises(ValueError, match="rtol"):
        assert_close(1.0, 1.0, rtol)
