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
    #: Rows removed because their weight was negative. Stata does this for aweight/pweight and
    #: keeps no placeholder, so the caller must drop the SAME rows from the design matrix.
    dropped: int = 0
    #: Boolean mask over the ORIGINAL rows, True where the observation survives. ``None`` when no
    #: row was dropped.
    kept: np.ndarray | None = None


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
    # NEGATIVE WEIGHTS ARE LEGAL FOR aweight AND pweight. Stata drops an observation whose weight
    # is negative rather than refusing the command, and `[aw=1/lntc]` on nerlove.dta is exactly
    # that case: lntc spans -2.501 to 4.9375, so 21 of 145 observations get a negative weight and
    # Stata reports e(N) = 124. `fweight` is different -- it needs NONINTEGER counts to be positive
    # integers and raises r(401) ("may not use noninteger frequency weights"), which this branch
    # still enforces.
    #
    # The array KEEPS ITS ORIGINAL LENGTH here, with negative entries zeroed, because the caller
    # has already built a design matrix with that many rows and checks the two for alignment. The
    # `kept` mask is what the estimator uses to drop the rows.
    dropped = 0
    kept: np.ndarray | None = None
    if (w < 0).any():
        if kind not in ("aweight", "pweight"):
            raise WeightError("weights must be non-negative")
        kept = w >= 0.0
        dropped = int((~kept).sum())
        if not kept.any():
            raise WeightError(
                "every weight is negative, so no observation is left to fit"
            )
        w = np.where(kept, w, 0.0)
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
            dropped=dropped,
            kept=kept,
        )

    if kind == "iweight":
        return WeightSpec(
            weights=w,
            weight_type=kind,
            effective_n=total,
            implies_robust=False,
            n_obs=n_obs,
            dropped=dropped,
            kept=kept,
        )

    # aweight and pweight are both normalized so that the weights sum to N; they differ only in
    # how the variance is computed.
    #
    # THE SURVIVING COUNT IS WHAT THE DEGREES OF FREEDOM USE. Stata reports e(N) = 124 for
    # `[aw=1/lntc]`, and its F statistic is F(4, 119) -- 124 minus the five parameters. Counting
    # all 145 rows gave F(4, 140), which changes the p-value. `surviving` therefore excludes the
    # rows whose weight was negative, while `n_obs` still records how many rows the caller
    # supplied so the alignment check keeps working.
    surviving = int(kept.sum()) if kept is not None else int(n_obs)
    normalized = w * (surviving / total)
    return WeightSpec(
        weights=normalized,
        weight_type=kind,
        effective_n=float(surviving),
        implies_robust=(kind == "pweight"),
        n_obs=n_obs,
        dropped=dropped,
        kept=kept,
    )
