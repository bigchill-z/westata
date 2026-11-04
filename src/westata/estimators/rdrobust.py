"""Sharp regression discontinuity: Stata's ``rdrobust`` (point estimate and bandwidths).

``rdrobust`` is an SSC package with no readable ``ado`` source here, but it RUNS in the local
StataNow, so its numbers are available as a reference. That makes this a verifiable port rather
than a from-scratch design.

Reference on a synthetic sharp RD sample (n = 400, true jump 8.0)::

    Number of obs      224      176        Kernel        = Triangular
    Eff. Number of obs  96       69        VCE method    = NN
    Order est. (p)       1        1
    Order bias (q)       2        2
    BW est. (h)      0.479    0.479
    BW bias (b)      0.794    0.794
    rho (h/b)        0.602    0.602

    RD Effect   7.6896      z 20.8565    P>|z| 0.000   [6.86877, 8.2963]

and from ``e()``: ``tau_cl = 7.6896241``, ``tau_bc = 7.5812002``, ``h_r = .47857568``,
``b_l = b_r = .79436028``, ``se_tau_cl = .31709743``.

**What is reproduced exactly**: the point estimate and the effective sample counts. Local
linear regression with a triangular kernel, fitted separately on each side of the cutoff:

    tau = mu_+(0) - mu_-(0)

gives 7.6896241 against Stata's 7.6896241, with 96 left and 69 right observations inside the
bandwidth.

**What is NOT reproduced**: the standard error. The sandwich form, and its equivalence-kernel
rewrite, both give 0.32163224 against Stata's 0.31709743 -- a 1.4% gap that is not a scaling
factor (Stata/ours = 0.98581, not sqrt(n/(n-1)) or any simple adjustment). ``rdrobust`` uses a
specific variance estimator whose exact weighting this port does not yet match, so
``se``/``z``/``p``/the confidence interval are reported as unavailable rather than filled in
with a number that is nearly right.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from westata.errors import ConsistencyError, UnsupportedError

__all__ = ["RdRobustResult", "rdrobust"]


@dataclass(frozen=True)
class RdRobustResult:
    """``rdrobust``: sharp RD estimate at a cutoff.

    ``se``/``z``/``pvalue``/``ci`` are ``nan`` when the variance estimator could not be
    reproduced; ``variance_matches_stata`` records which case applies.
    """

    tau: float
    cutoff: float
    bandwidth: float
    bias_bandwidth: float
    nobs: int
    nobs_left: int
    nobs_right: int
    nobs_effective_left: int
    nobs_effective_right: int
    order: int
    bias_order: int
    kernel: str
    se: float = float("nan")
    z: float = float("nan")
    pvalue: float = float("nan")
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    variance_matches_stata: bool = False

    def __str__(self) -> str:
        lines = [
            "Sharp RD estimate",
            "",
            f"{'Number of obs':<22}{self.nobs_left:>10}{self.nobs_right:>10}"
            f"        Kernel       = {self.kernel}",
            f"{'Eff. Number of obs':<22}{self.nobs_effective_left:>10}"
            f"{self.nobs_effective_right:>10}",
            f"{'Order est. (p)':<22}{self.order:>10}{self.order:>10}",
            f"{'Order bias (q)':<22}{self.bias_order:>10}{self.bias_order:>10}",
            f"{'BW est. (h)':<22}{self.bandwidth:>10.3f}{self.bandwidth:>10.3f}",
            f"{'BW bias (b)':<22}{self.bias_bandwidth:>10.3f}{self.bias_bandwidth:>10.3f}",
            "",
            "-" * 72,
            f"{'':<20}|{'Point':>14}|{'Robust Inference':>28}",
            f"{'':<20}|{'Estimate':>14}|{'z-stat':>10}{'P>|z|':>9}{'[95% CI]':>18}",
            "-" * 72,
        ]
        if np.isfinite(self.se):
            interval = f"[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
            lines.append(
                f"{'RD Effect':<20}|{self.tau:>14.4f}|{self.z:>10.4f}"
                f"{self.pvalue:>9.3f}{interval:>24}"
            )
        else:
            lines.append(
                f"{'RD Effect':<20}|{self.tau:>14.4f}|"
                f"{'(variance not reproduced)':>38}"
            )
        lines.append("-" * 72)
        return "\n".join(lines)


def _triangular(distance: np.ndarray, bandwidth: float) -> np.ndarray:
    """Triangular kernel on ``|distance / bandwidth|``, scaled to vanish at the edge."""

    return np.maximum(0.0, 1.0 - np.abs(distance / bandwidth))


def _local_polynomial(
    distance: np.ndarray,
    response: np.ndarray,
    bandwidth: float,
    order: int,
) -> tuple[float, int]:
    """Weighted local polynomial fit; return the intercept and the observations used."""

    weights = _triangular(distance, bandwidth)
    inside = weights > 0.0
    if int(inside.sum()) <= order:
        raise ConsistencyError(
            f"rdrobust: only {int(inside.sum())} observations inside the bandwidth, "
            f"which cannot support an order-{order} fit"
        )
    x = distance[inside]
    y = response[inside]
    w = weights[inside]
    design = np.column_stack([x**power for power in range(order + 1)])
    weighted = design * w[:, None]
    try:
        coefficients = np.linalg.solve(design.T @ weighted, weighted.T @ y)
    except np.linalg.LinAlgError:
        coefficients, *_ = np.linalg.lstsq(
            design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None
        )
    return float(coefficients[0]), int(inside.sum())


def rdrobust(
    response: Any,
    running: Any,
    *,
    cutoff: float = 0.0,
    bandwidth: float | None = None,
    bias_bandwidth: float | None = None,
    order: int = 1,
    bias_order: int = 2,
    kernel: str = "triangular",
) -> RdRobustResult:
    """Sharp RD estimate at ``cutoff``.

    Bandwidth selection is not implemented: pass ``bandwidth`` explicitly, or the default
    ``None`` raises rather than silently choosing one. Stata's MSE-optimal selector is a
    separate, considerably larger piece of work.
    """

    y = np.asarray(response, dtype=float).reshape(-1)
    x = np.asarray(running, dtype=float).reshape(-1)
    if y.size != x.size:
        raise ConsistencyError(
            f"rdrobust: {y.size} outcomes against {x.size} running-variable values"
        )
    keep = np.isfinite(y) & np.isfinite(x)
    y, x = y[keep], x[keep]
    if y.size < 10:
        raise ValueError(f"rdrobust: at least 10 usable observations are required; got {y.size}")
    if str(kernel).lower() not in {"triangular", "uniform"}:
        raise UnsupportedError(
            f"rdrobust: only the triangular kernel is implemented; got {kernel!r}"
        )
    if bandwidth is None:
        raise UnsupportedError(
            "rdrobust: bandwidth selection is not implemented; pass bandwidth= explicitly. "
            "The MSE-optimal selector is a separate piece of work, and guessing a value "
            "would produce a point estimate that looks like a result."
        )
    bandwidth = float(bandwidth)
    if bandwidth <= 0.0:
        raise ValueError(f"rdrobust: bandwidth must be positive; got {bandwidth}")
    if bias_bandwidth is None:
        bias_bandwidth = float("nan")
    else:
        bias_bandwidth = float(bias_bandwidth)

    distance = x - float(cutoff)
    right = distance >= 0.0
    left = ~right

    mu_right, used_right = _local_polynomial(
        distance[right], y[right], bandwidth, order
    )
    mu_left, used_left = _local_polynomial(
        -distance[left], y[left], bandwidth, order
    )

    return RdRobustResult(
        tau=float(mu_right - mu_left),
        cutoff=float(cutoff),
        bandwidth=bandwidth,
        bias_bandwidth=bias_bandwidth,
        nobs=int(y.size),
        nobs_left=int(left.sum()),
        nobs_right=int(right.sum()),
        nobs_effective_left=int(used_left),
        nobs_effective_right=int(used_right),
        order=int(order),
        bias_order=int(bias_order),
        kernel="Triangular",
    )
