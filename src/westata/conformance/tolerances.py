"""Numeric tolerances for Stata-conformance evidence.

Design spec: ``docs/superpowers/specs/2026-10-22-westata-design.md`` section
11.4.  Linear/IV/panel estimates promise *relative* agreement of ``1e-10`` for
both point estimates and standard errors; MLE models get looser bounds because
different optimizers reach the same optimum only to within their convergence
tolerance (section 10.5).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

#: Relative tolerance for linear models (OLS/IV/panel): point estimates and SEs.
LINEAR_RTOL: float = 1e-10
#: Relative tolerance for MLE point estimates (logit, probit, ...).
MLE_RTOL: float = 1e-5
#: Relative tolerance for MLE standard errors.
MLE_SE_RTOL: float = 1e-4


def _as_float_array(value: Any, name: str) -> np.ndarray:
    """Coerce ``value`` to a float array, raising ``TypeError`` on garbage."""

    try:
        return np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be numeric, got {type(value).__name__}: {value!r}"
        ) from exc


def assert_close(actual: Any, expected: Any, rtol: float, *, label: str = "") -> None:
    """Assert that ``actual`` matches ``expected`` within relative ``rtol``.

    This is ``numpy.testing.assert_allclose`` with ``atol=0`` so the comparison
    stays purely relative (the evidence table in the design spec is stated in
    relative terms), plus a failure message that names the quantity being
    checked and reports the largest absolute/relative deviation.

    Parameters
    ----------
    actual, expected:
        Scalars or array-likes; broadcastable arrays are compared elementwise.
    rtol:
        Relative tolerance, must be positive and finite.
    label:
        Human-readable description of the compared quantity, e.g.
        ``"beta[weight]"``; included in the failure message.

    Raises
    ------
    AssertionError
        When any element differs by more than ``rtol`` (relative), or when the
        shapes are incompatible.
    ValueError
        When ``rtol`` is not positive and finite.
    """

    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError(f"rtol must be positive and finite, got {rtol!r}")

    actual_arr = _as_float_array(actual, "actual")
    expected_arr = _as_float_array(expected, "expected")
    prefix = f"{label}: " if label else ""

    try:
        np.testing.assert_allclose(actual_arr, expected_arr, rtol=rtol, atol=0.0)
    except AssertionError as exc:
        if actual_arr.shape != expected_arr.shape:
            detail = (
                "shape mismatch: "
                f"actual {actual_arr.shape} vs expected {expected_arr.shape}"
            )
        else:
            absolute = np.abs(actual_arr - expected_arr)
            scale = np.maximum(np.abs(expected_arr), np.finfo(float).tiny)
            relative = absolute / scale
            detail = (
                f"max abs difference={float(np.max(absolute)):.6e}, "
                f"max relative error={float(np.max(relative)):.6e}, rtol={rtol:.3g}\n"
                f"actual={np.array2string(actual_arr, threshold=12)}\n"
                f"expected={np.array2string(expected_arr, threshold=12)}"
            )
        raise AssertionError(f"{prefix}values are not close: {detail}") from exc


__all__ = ["LINEAR_RTOL", "MLE_RTOL", "MLE_SE_RTOL", "assert_close"]
