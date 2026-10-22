"""Weighted-data semantics following Stata ``[U] 20.24 Weighted estimation``.

Stata distinguishes four weight types, and they change the estimator in
different ways:

``fweight``
    Frequency weights: integer replication counts.  ``sum(w)`` observations are
    effectively present, so ``df_resid`` and ``N`` use ``sum(w)``.
``aweight``
    Analytic/precision weights.  Normalized to sum to ``N`` and then used like
    fweights; ``df_resid`` uses ``N``.
``pweight``
    Sampling weights.  The point estimate is the aweighted one, but the
    variance must use the linearized (robust) sandwich estimator -- Stata's
    ``regress y x [pw=w]`` is identical to adding ``vce(robust)``.
``iweight``
    Importance weights.  Used like fweights without normalization.

Reference: ``[U] 20.24`` in the Stata User's Guide (see ``docs/reference``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from westata.errors import WeightError

__all__ = ["WeightSpec", "normalize_weights", "WEIGHT_TYPES"]

WEIGHT_TYPES = ("fweight", "aweight", "pweight", "iweight")


@dataclass(frozen=True)
class WeightSpec:
    """Normalized weights plus the metadata the estimators need.

    Attributes
    ----------
    weights:
        Weights to use in the weighted least-squares problem, already
        normalized according to the Stata rules for ``weight_type``.
    weight_type:
        One of ``"fweight"``, ``"aweight"``, ``"pweight"``, ``"iweight"``.
    effective_n:
        Sample size to use in degrees-of-freedom calculations.
    implies_robust:
        True for ``pweight``: Stata's ``[pw=w]`` implies ``vce(robust)``.
    n_obs:
        Number of rows in the data (not the effective sample size).
    """

    weights: np.ndarray
    weight_type: str
    effective_n: float
    implies_robust: bool
    n_obs: int


def normalize_weights(weights: Any, weight_type: str) -> WeightSpec:
    """Normalize ``weights`` according to Stata's rules for ``weight_type``.

    Parameters
    ----------
    weights:
        Array-like of non-negative weights, one per observation.
    weight_type:
        ``"fweight"``, ``"aweight"``, ``"pweight"`` or ``"iweight"``.

    Raises
    ------
    WeightError
        If the weight type is unknown, the array is empty, has non-positive
        total weight, or contains negative or missing values.
    """

    if not isinstance(weight_type, str):
        raise WeightError(
            "weight_type must be a string such as 'aweight'; got "
            f"{type(weight_type).__name__}"
        )
    kind = weight_type.strip().lower()
    if kind not in WEIGHT_TYPES:
        raise WeightError(
            f"unknown weight_type {weight_type!r}; supported types are: "
            + ", ".join(WEIGHT_TYPES)
        )

    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size == 0:
        raise WeightError("weights array is empty")
    if np.isnan(w).any():
        raise WeightError("weights contain missing values")
    if (w < 0).any():
        raise WeightError("weights must be non-negative")
    if not np.isfinite(w).all():
        raise WeightError("weights must be finite")

    total = float(w.sum())
    if total <= 0:
        raise WeightError("weights sum to zero; at least one weight must be positive")

    n_obs = int(w.size)

    if kind == "fweight":
        return WeightSpec(
            weights=w,
            weight_type=kind,
            effective_n=total,
            implies_robust=False,
            n_obs=n_obs,
        )

    if kind == "iweight":
        return WeightSpec(
            weights=w,
            weight_type=kind,
            effective_n=total,
            implies_robust=False,
            n_obs=n_obs,
        )

    # aweight and pweight are both normalized so that the weights sum to N;
    # they differ only in how the variance is computed.
    normalized = w * (n_obs / total)
    return WeightSpec(
        weights=normalized,
        weight_type=kind,
        effective_n=float(n_obs),
        implies_robust=(kind == "pweight"),
        n_obs=n_obs,
    )
