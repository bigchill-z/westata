"""Invariant oracles: identities every correct OLS fit must satisfy.

Design spec: ``docs/superpowers/specs/2026-10-22-westata-design.md`` section
11.2 (``InvariantOracle``).  Least squares is defined by the normal equations,
so any estimator (our own, statsmodels, Stata) must produce residuals
orthogonal to the design matrix, and — when a constant is included — residuals
that sum to zero.  These checks need no Stata license and no fixture.
"""

from __future__ import annotations

import math

import numpy as np


def _as_float_array(value: object, name: str) -> np.ndarray:
    """Coerce ``value`` to a float array, raising ``TypeError`` on garbage."""

    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be numeric, got {type(value).__name__}: {value!r}"
        ) from exc


def check_ols_invariants(
    X: np.ndarray,
    resid: np.ndarray,
    params: np.ndarray,
    *,
    has_intercept: bool,
    atol: float = 1e-10,
) -> None:
    """Assert the least-squares normal equations for a fitted model.

    Checks
    ------
    1. ``X.T @ resid ~= 0`` (residuals orthogonal to every regressor).
    2. When ``has_intercept`` is true, ``sum(resid) ~= 0``.

    Floating-point orthogonality is judged against the problem scale:
    ``max|X.T @ resid|`` is compared with ``atol * ||X||_inf * ||resid||_inf *
    nobs``.  A raw absolute threshold would reject exact solves on
    large-magnitude regressors (e.g. wages measured in thousands), because
    cancellation alone leaves a residual gradient of order
    ``eps * ||X|| * ||resid|| * nobs``.

    Parameters
    ----------
    X:
        ``nobs x nparams`` design matrix.
    resid:
        ``nobs`` residual vector of the fitted model.
    params:
        Fitted coefficient vector, shape ``(nparams,)``.  It is not used in the
        identity itself, but requiring it keeps callers from pairing residuals
        with the wrong design matrix.
    has_intercept:
        Whether the model includes a constant term; only then is
        ``sum(resid) = 0`` expected.
    atol:
        Scale-relative tolerance, positive and finite.

    Raises
    ------
    ValueError
        On malformed inputs (wrong shapes, non-finite data, bad ``atol``).
    AssertionError
        When the normal equations or the intercept condition do not hold.
    """

    X_arr = _as_float_array(X, "X")
    resid_arr = _as_float_array(resid, "resid")
    params_arr = _as_float_array(params, "params")

    if X_arr.ndim != 2 or X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
        raise ValueError(f"X must be a non-empty 2-D array, got shape {X_arr.shape}")
    nobs, nparams = X_arr.shape
    if params_arr.shape != (nparams,):
        raise ValueError(
            f"params must have shape ({nparams},) to match X, got {params_arr.shape}"
        )
    if resid_arr.shape != (nobs,):
        raise ValueError(
            f"resid must have shape ({nobs},) to match X, got {resid_arr.shape}"
        )
    if not math.isfinite(atol) or atol <= 0:
        raise ValueError(f"atol must be positive and finite, got {atol!r}")
    if not (np.all(np.isfinite(X_arr)) and np.all(np.isfinite(resid_arr))):
        raise ValueError("X and resid must contain only finite values")

    # 1) Normal equations: X.T @ resid == 0.
    gradient = X_arr.T @ resid_arr
    gradient_scale = max(
        1.0,
        float(np.max(np.abs(X_arr))) * float(np.max(np.abs(resid_arr))) * nobs,
    )
    gradient_threshold = atol * gradient_scale
    max_gradient = float(np.max(np.abs(gradient)))
    if max_gradient > gradient_threshold:
        raise AssertionError(
            "OLS normal equations violated: max |X.T @ resid| = "
            f"{max_gradient:.6e} exceeds scale-adjusted threshold "
            f"{gradient_threshold:.6e} "
            f"(atol={atol:.3g}, scale={gradient_scale:.6e})"
        )

    # 2) Intercept condition: sum(resid) == 0.
    if has_intercept:
        total_resid = abs(float(np.sum(resid_arr)))
        sum_scale = max(1.0, nobs * float(np.max(np.abs(resid_arr))))
        sum_threshold = atol * sum_scale
        if total_resid > sum_threshold:
            raise AssertionError(
                "OLS intercept invariant violated: |sum(resid)| = "
                f"{total_resid:.6e} exceeds scale-adjusted threshold "
                f"{sum_threshold:.6e} "
                f"(atol={atol:.3g}, scale={sum_scale:.6e})"
            )


__all__ = ["check_ols_invariants"]
