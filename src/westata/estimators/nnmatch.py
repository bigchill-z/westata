"""``teffects nnmatch``: nearest-neighbour matching on Mahalanobis distance.

The report's C15-03 recorded this as "NNMATCH has no caliper parameter", implying the
estimator existed and only one option was missing. It did not exist at all -- there was no
nnmatch implementation -- so the option could not have been added in isolation.

Reference from a live ``teffects nnmatch (re78 <8 covariates>) (treat), atet`` on nsw_dw.dta:

    nneighbor(1) : ATET = 2453.0763
    nneighbor(4) : ATET = 2060.4859

THE TIE RULE, which is what makes the first number reproducible. Stata matches on the
Mahalanobis distance ||x_i - x_j||^2 with the inverse of the sample covariate covariance
(``metric = mahalanobis`` per ``e(metric)``), and when several controls sit at the SAME minimum
distance it uses ALL of them and averages:

    n(1), first tied control only   : 2277.0212
    n(1), all tied controls         : 2453.0763   <- Stata

47 of the 185 treated units have ties here, so the difference is not a rounding detail. This is
the same rule found in ``teffects_psmatch`` (round 52); the two estimators differ in their
distance metric, not in how they break ties.

NOTE ON A REPORTED NUMBER: the gap report lists "ATET 2277.021" against Stata's 2639.865 for
``teffects psmatch``. 2277.0212 is exactly the value produced by Mahalanobis matching WITH the
first-tie-only rule -- i.e. an nnmatch computation, not a broken psmatch. The two cases in the
report had been cross-wired.

``caliper()`` is implemented as an absolute distance cap; matches beyond it are dropped and a
treated unit with no control inside the caliper raises rather than silently borrowing a distant
one, which is what Stata reports as "no matches".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from westata.errors import ConsistencyError

__all__ = ["NnmatchResult", "teffects_nnmatch"]


@dataclass(frozen=True)
class NnmatchResult:
    """``teffects nnmatch``: ATET from Mahalanobis nearest-neighbour matching."""

    ate: float
    nobs: int
    treated: int
    controls: int
    nneighbor: int
    matches_min: int
    matches_max: int
    metric: str = "mahalanobis"
    caliper: float = float("nan")
    se: float = float("nan")
    z: float = float("nan")
    pvalue: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")

    @property
    def atet(self) -> float:
        """Alias, since the default estimand for matching is the ATET."""

        return self.ate


def teffects_nnmatch(
    response: Any,
    treatment: Any,
    covariates: Any,
    *,
    nneighbor: int = 1,
    caliper: float | None = None,
    tie_tolerance: float = 1e-10,
) -> NnmatchResult:
    """ATET by Mahalanobis-distance nearest-neighbour matching.

    ``caliper`` caps the distance a match may have; ``None`` means no cap (Stata's default).
    ``tie_tolerance`` is the absolute distance within which two controls count as equally near.
    """

    y = np.asarray(response, dtype=float).reshape(-1)
    d = np.asarray(treatment, dtype=float).reshape(-1)
    x = np.asarray(covariates, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not (y.size == d.size == x.shape[0]):
        raise ConsistencyError(
            f"teffects nnmatch: shapes disagree -- outcome {y.size}, treatment {d.size}, "
            f"covariates {x.shape[0]}"
        )
    keep = np.isfinite(y) & np.isfinite(d) & np.all(np.isfinite(x), axis=1)
    y, d, x = y[keep], d[keep], x[keep]
    levels = np.unique(d)
    if levels.size != 2 or not np.array_equal(levels, np.array([0.0, 1.0])):
        raise ConsistencyError(
            f"teffects nnmatch: treatment must be coded 0/1; found {levels.tolist()}"
        )
    nneighbor = int(nneighbor)
    if nneighbor < 1:
        raise ValueError(f"teffects nnmatch: nneighbor must be >= 1; got {nneighbor}")
    if caliper is not None:
        caliper = float(caliper)
        if not np.isfinite(caliper) or caliper <= 0.0:
            raise ValueError(
                f"teffects nnmatch: caliper must be a positive finite distance; got {caliper}"
            )

    controls = np.flatnonzero(d == 0.0)
    treated = np.flatnonzero(d == 1.0)
    if not controls.size or not treated.size:
        raise ConsistencyError(
            "teffects nnmatch: both treatment levels need at least one observation"
        )
    if controls.size < x.shape[1] + 1:
        raise ConsistencyError(
            f"teffects nnmatch: {controls.size} controls cannot estimate a "
            f"{x.shape[1]}-dimensional covariance; supply more control observations"
        )

    # Stata's default metric is the INVERSE sample covariate covariance (``e(metric)`` reads
    # "mahalanobis"). Computed over the full sample, matching the reference fit.
    covariance = np.cov(x.T)
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError as error:  # pragma: no cover - defensive
        raise ConsistencyError(
            "teffects nnmatch: the covariate covariance is singular; drop collinear "
            "covariates before matching"
        ) from error

    differences: list[float] = []
    match_counts: list[int] = []
    for index in treated:
        delta = x[controls] - x[index]
        distance = np.einsum("ij,jk,ik->i", delta, inverse, delta)
        if caliper is not None:
            distance = np.where(distance <= caliper, distance, np.inf)
            if not np.isfinite(distance).any():
                raise ConsistencyError(
                    f"teffects nnmatch: treated observation {index} has no control within "
                    f"caliper({caliper}); Stata reports this as 'no matches'"
                )
        order = np.argsort(distance)
        kth = min(nneighbor, order.size)
        threshold = distance[order[kth - 1]]
        if not np.isfinite(threshold):
            raise ConsistencyError(
                f"teffects nnmatch: treated observation {index} has fewer than "
                f"{nneighbor} controls inside caliper({caliper})"
            )
        # Every control at the k-th smallest distance (within tolerance) is a match -- the
        # tie rule that turns 2277.0212 into Stata's 2453.0763.
        selected = controls[distance <= threshold + tie_tolerance]
        differences.append(float(y[index] - np.mean(y[selected])))
        match_counts.append(int(selected.size))

    return NnmatchResult(
        ate=float(np.mean(differences)),
        nobs=int(y.size),
        treated=int(treated.size),
        controls=int(controls.size),
        nneighbor=nneighbor,
        matches_min=int(min(match_counts)),
        matches_max=int(max(match_counts)),
        caliper=float("nan") if caliper is None else caliper,
    )
