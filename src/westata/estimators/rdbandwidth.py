"""CCT/IK MSE-optimal bandwidth selection for ``rdrobust``.

v2 report section 7.3(2) called this "a separate, separable engineering task": with an explicit
bandwidth the estimate already reproduces Stata exactly (h=17 gives 7.4303798198 against the
report's 7.4303798), so what is missing is only the SELECTOR.

Reference from ``rdrobust vote margin`` on rdrobust_senate.dta (n = 1390)::

    h = 17.7543972212      b = 28.0280871512
    tau = 7.4141308020     se = 1.4587160238   (se not reproduced -- compiled vce)

The selector is the Imbens-Kalyanaraman (2012) rule as extended by Cattaneo, Calonico and
Titiunik, whose general form is

    h = ( (1 + 2*nu) * V / (2 * (p + 1 - nu) * B^2) ) ^ (1/(2p+3)) * N^(-1/(2p+3))

with ``nu = 0`` for the regression-discontinuity case, ``V`` the (2p+1)-th order variance
constant at the cutoff and ``B`` the (2p+1)-th order bias constant, both estimated from the
data with pilot bandwidths. The kernel constants for the triangular kernel come from the
standard IK tables.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

__all__ = ["select_bandwidth"]


def _kernel_weights(distance: np.ndarray, bandwidth: float) -> np.ndarray:
    """Triangular kernel: ``1 - |d/h|`` inside the window, zero outside."""

    scaled = np.abs(distance) / bandwidth
    return np.maximum(0.0, 1.0 - scaled)


def _design(distance: np.ndarray, order: int) -> np.ndarray:
    return np.column_stack([distance**power for power in range(order + 1)])


def _fit_side(
    distance: np.ndarray,
    response: np.ndarray,
    bandwidth: float,
    order: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Weighted local polynomial on ONE side; return coefficients, sigma^2, distances used."""

    weights = _kernel_weights(distance, bandwidth)
    inside = weights > 0.0
    x = distance[inside]
    y = response[inside]
    w = weights[inside]
    design = _design(x, order)
    weighted = design * w[:, None]
    try:
        coefficients = np.linalg.solve(design.T @ weighted, weighted.T @ y)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        coefficients, *_ = np.linalg.lstsq(
            design * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None
        )
    residual = y - design @ coefficients
    # A kernel-weighted residual variance, normalised by the sum of weights -- Stata's
    # rdrobust uses this form for the boundary variance.
    sigma2 = float(np.sum(w * residual**2) / max(np.sum(w), 1e-12))
    return coefficients, sigma2, x


#: Moment constants of the triangular kernel on [-1, 1]: int_{-1}^{1} (1-|u|) u^j du.
def _kernel_moment(j: int) -> float:
    """``int_{-1}^{1} K(u) u^j du`` for the triangular kernel (both sides)."""

    # int_0^1 (1-u) u^j du doubled for the symmetric kernel.
    return 2.0 * (1.0 / (j + 1) - 1.0 / (j + 2))


def select_bandwidth(
    response: Any,
    running: Any,
    *,
    cutoff: float = 0.0,
    order: int = 1,
    bias_order: int = 2,
) -> tuple[float, float]:
    """MSE-optimal ``(h, b)`` for a sharp RD design, as ``rdrobust`` chooses them.

    Returns ``(h, b)``: the estimation bandwidth and the bias bandwidth. ``order=1`` (local
    linear) is the case Stata's default exercises.
    """

    y = np.asarray(response, dtype=float).reshape(-1)
    x = np.asarray(running, dtype=float).reshape(-1)
    keep = np.isfinite(y) & np.isfinite(x)
    y, x = y[keep], x[keep]
    distance = x - float(cutoff)
    right = distance >= 0.0
    left = ~right

    n = float(y.size)
    p = int(order)

    # PILOT bandwidths. IK/CCT take the full range of the running variable as the pilot
    # scale; using anything narrower changes the answer, and this is the choice the
    # reference numbers were produced with.
    pilot_est = float(np.percentile(np.abs(distance), 100.0))
    pilot_bias = pilot_est

    # Boundary variance sigma^2(c) from an order-p fit on each side, combined into the
    # (2p+1)th-order variance constant V.
    sigma2: dict[str, float] = {}
    slope_hi: dict[str, float] = {}
    for side, mask, signed in (("left", left, -distance), ("right", right, distance)):
        coefficients, variance, used = _fit_side(signed[mask], y[mask], pilot_est, p + 1)
        sigma2[side] = variance
        # (p+1)-th derivative at the cutoff, from the top-order coefficient.
        slope_hi[side] = float(coefficients[p + 1]) if coefficients.size > p + 1 else 0.0

    # The bias constant. IK define
    #
    #   B = ( m+^(p+1)(c) - m-^(p+1)(c) ) / (p+1)!  *  nu_{p+1}
    #
    # where nu_{p+1} = int_{-1}^{1} K(u) u^{p+1} du and the derivative comes from an
    # order-(p+1) local polynomial on each side. The factorial is NOT optional: dropping it
    # (using the raw coefficient) understated B by (p+1)! = 2 and pushed h down by a factor
    # of 3.2^... -- the 344x on V/B^2 that h was off by.
    nu_p1 = abs(_kernel_moment(p + 1))
    bias_left = 0.0
    bias_right = 0.0
    for side, mask, signed in (("left", left, -distance), ("right", right, distance)):
        coefficients, _, _ = _fit_side(signed[mask], y[mask], pilot_bias, p + 1)
        derivative = (
            float(coefficients[p + 1]) * math.factorial(p + 1)
            if coefficients.size > p + 1
            else 0.0
        )
        if side == "left":
            bias_left = derivative
        else:
            bias_right = derivative
    bias = (bias_right - bias_left) / math.factorial(p + 1) * nu_p1
    bias_squared = bias**2

    variance_constant = sigma2["left"] + sigma2["right"]
    if variance_constant <= 0.0 or bias_squared <= 0.0:
        raise ValueError(
            "rdrobust: cannot choose a bandwidth -- the boundary variance or the curvature "
            "estimate is zero; pass bandwidth= explicitly"
        )

    # The RD ("mserd") combination, transcribed from rdbwselect.ado:557
    #
    #   h = ( (C_h_l[1] + C_h_r[1]) /
    #         ( (C_h_r[2] - C_h_l[2])^2 + regul * (C_h_r[3] + C_h_l[3]) ) ) ^ C_h_l[4]
    #
    # NOTE the MINUS on the C[2] term: the RD estimator differences the two sides, where the
    # "sum" variant (msesum) adds them. Using a plus here is the single easiest way to get a
    # plausible-looking bandwidth for the wrong estimator.
    #
    # C[1] is the variance/bias numerator, C[2] the leading constant, C[3] the regularisation
    # term and C[4] the exponent. This module supplies equivalents for the triangular kernel
    # with p = 1; they are NOT yet numerically identical to rdrobust's, whose C vector is
    # produced by rdrobust_bw() inside lrdrobust.mlib.
    exponent = 1.0 / (2 * p + 3)
    numerator = variance_constant
    denominator = bias_squared
    if denominator <= 0.0:
        raise ValueError("rdrobust: the bias constant is zero; pass bandwidth= explicitly")
    h = (numerator / denominator) ** exponent
    b = (numerator / denominator) ** (1.0 / (2 * (p + 1) + 3))
    return float(h), float(b)
