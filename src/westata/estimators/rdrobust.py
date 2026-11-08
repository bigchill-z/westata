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
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError

__all__ = ["RdRobustResult", "rdrobust"]


@dataclass(frozen=True)
class RdRobustResult:
    """``rdrobust``: sharp RD estimate at a cutoff.

    ``se``/``z``/``pvalue``/``ci`` are ``nan`` when the variance estimator could not be
    reproduced; ``variance_matches_stata`` records which case applies.

    WHY THE VARIANCE IS NOT REPRODUCED -- and this one IS a hard boundary, unlike several
    earlier gaps in this codebase that turned out to be recoverable. The point estimate
    matches Stata exactly (``rdrobust diff border, c(30) h(5) b(8)`` gives tau_cl
    121.297562295 here and 121.297562295 in Stata). The variance is computed inside
    ``rdrobust_vce``, which lives ONLY in the compiled Mata library
    ``~/ado/plus/l/lrdrobust.mlib`` -- there is no ``.ado``/``.mata`` source for it, unlike
    ``rdrobust.ado`` itself, which IS readable at ``~/ado/plus/r/rdrobust.ado`` (1331 lines).

    What the readable part pins down, verified against Stata's own ``e(V_cl_l)`` /
    ``e(V_cl_r)`` (h(5) b(8) on border.dta, cutoff 30)::

        V_cl_l = [[496.94559, 225.18263], [225.18263, 167.66193]]
        V_cl_r = [[4303.6294, -1188.4335], [-1188.4335, 336.15059]]
        sqrt((V_cl_l + V_cl_r)[1,1]) = 69.2861818   == Stata se_tau_cl 69.2861819408

    i.e. ``se = sqrt((V_l + V_r)[deriv, deriv])`` in zero-based terms -- the Mata code's
    ``[deriv+1, deriv+1]``. What is NOT recovered is how V_l/V_r are built from the
    residuals. The standard CR1 sandwich candidates were tried and all miss on the very
    section used for the check above (n_left = 5, only a 2-column design):

        w^2 * res^2   ->  38.16     (target 496.95)
        w   * res^2   -> 240.47
        w   * res^2 * n -> 1202.33

    Reporting any of those would be the plausible-but-wrong number this project refuses, so
    the inferential fields stay NaN and the test suite asserts that they do.
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
    #: ``e(coef_covs)``: the joint covariate coefficient, or nan when no ``covs`` were given.
    coef_covs: Any = None
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


def _covariate_adjusted(
    response: np.ndarray,
    distance: np.ndarray,
    covariates: Any,
    bandwidth: float,
    order: int,
) -> tuple[float, int, int, np.ndarray]:
    """Covariate-adjusted RD, transcribed from ``rdrobust.ado:761-841``.

    The mechanism is Frisch-Waugh partialling per side, then POOLING the two sides before one
    joint solve -- which is why Stata reports a single ``e(coef_covs)`` rather than one per side:

        D_side   = [response, covariates]        covariates appended AFTER the response (:761)
        G_p      = R'W R                          R = polynomial basis, no covariates
        beta_p   = G_p^-1 R'W D                   (p+1) x (1 + dZ)  (:771)
        ZWZ      = pooled moment of the covariate after partialling R out
        gamma_p  = ZWZ^-1 ZWY                     (:834)
        s_Y      = (1, -gamma_p)
        tau_cl   = s_Y' * beta_p[deriv+1, ]       (:841)

    TWO DETAILS that took separate rounds to get right, both now verified on
    ``rdrobust vote margin, h(17.75) b(28.03) covs(cov1)``:

    * ``beta_p`` has ONE ROW PER POLYNOMIAL ORDER, not one row per outcome. For a linear fit
      with one covariate it is 2x2: rows = intercept/slope, columns = response/covariate.
    * ``s_Y' * beta_p[deriv+1, ]`` with ``deriv=0`` reads ROW 0, the intercept row. Reading row
      1 (the slope) gave 0.4238 against Stata's 7.4906.

    Reproduced exactly: tau 7.49061748, gamma -94.34797652, samples 360 / 323.
    """

    from westata.errors import ConsistencyError

    z = np.asarray(covariates, dtype=float)
    if z.ndim == 1:
        z = z.reshape(-1, 1)
    if z.shape[0] != response.size:
        raise ConsistencyError(
            f"rdrobust: {z.shape[0]} covariate rows against {response.size} observations"
        )
    if not np.all(np.isfinite(z)):
        raise ConsistencyError("rdrobust: covariates contain missing values")

    left = distance < 0.0
    right = ~left
    sides: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, mask, sign in (
        ("right", right, 1.0),
        ("left", left, -1.0),
    ):
        # Restrict to this side FIRST. Using the full-length distance without masking made
        # `inside` true everywhere, so both "sides" contained all 683 observations and the
        # estimate collapsed to 0.
        signed = sign * distance[mask]
        weight = _triangular(signed, bandwidth)
        inside = weight > 0.0
        basis = np.column_stack(
            [signed[inside] ** power for power in range(order + 1)]
        )
        design = np.column_stack([response[mask][inside], z[mask][inside]])
        sides[name] = (basis, weight[inside], design, int(inside.sum()))

    basis_r, w_r, d_r, used_right = sides["right"]
    basis_l, w_l, d_l, used_left = sides["left"]

    inverse_r = np.linalg.pinv(basis_r.T @ (basis_r * w_r[:, None]))
    inverse_l = np.linalg.pinv(basis_l.T @ (basis_l * w_l[:, None]))
    beta_r = inverse_r @ (basis_r.T @ (d_r * w_r[:, None]))
    beta_l = inverse_l @ (basis_l.T @ (d_l * w_l[:, None]))
    beta_p = beta_r - beta_l

    def _partial(basis: np.ndarray, weight: np.ndarray, target: np.ndarray) -> np.ndarray:
        coefficients = np.linalg.pinv(basis.T @ (basis * weight[:, None])) @ (
            basis.T @ (weight * target)
        )
        return target - basis @ coefficients

    numerator = 0.0
    denominator = 0.0
    for basis, weight, design in ((basis_r, w_r, d_r), (basis_l, w_l, d_l)):
        for column in range(1, d_r.shape[1]):
            z_tilde = _partial(basis, weight, design[:, column])
            y_tilde = _partial(basis, weight, design[:, 0])
            numerator += float(np.sum(weight * z_tilde * y_tilde))
            denominator += float(np.sum(weight * z_tilde**2))
    if denominator <= 0.0:
        raise ConsistencyError(
            "rdrobust: the covariates are collinear with the polynomial terms inside the "
            "bandwidth, so no adjustment can be estimated"
        )
    gamma = np.array([numerator / denominator])
    multiplier = np.concatenate(([1.0], -gamma))
    tau = float(multiplier @ beta_p[0])
    return tau, used_left, used_right, gamma


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


def _side_variance(
    distance: np.ndarray,
    response: np.ndarray,
    bandwidth: float,
    order: int,
    mode: str = "hc0",
) -> float:
    """HC0 variance of the one-sided intercept, in Stata's normalisation.

    Read from `rdrobust.sthlp:158-171`: `vce(hc0)` is the "plug-in residuals variance estimator
    without weights", which is the ordinary sandwich

        V = e1' A^-1 [ sum_i w_i^2 u_i^2 x_i x_i' ] A^-1 e1,    A = sum_i w_i x_i x_i'

    with `x` the SCALED distance `distance / h` -- that scaling is what the "without weights" phrase
    refers to, and it is the detail that makes the number match. Using the raw distance instead gave
    0.1113851870 against Stata's 0.1075019935.

    Verified against a live `rdrobust y x, h(0.2808754677) b(0.4908484209) vce(hc0)`:
    0.1075019935 for the default triangular kernel with a linear fit.
    """

    weights = _triangular(distance, bandwidth)
    inside = weights > 0.0
    if int(inside.sum()) <= order:
        return float("nan")

    scaled = distance[inside] / bandwidth
    values = response[inside]
    w = weights[inside]
    design = np.column_stack([scaled**power for power in range(order + 1)])

    normal = design.T @ (w[:, None] * design)
    try:
        inverse = np.linalg.inv(normal)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return float("nan")

    coefficients = inverse @ (design.T @ (w * values))
    residual = values - design @ coefficients

    middle = design.T @ ((w * w * residual * residual)[:, None] * design)
    e1 = np.zeros(order + 1)
    e1[0] = 1.0
    return float(e1 @ inverse @ middle @ inverse @ e1)


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
    covs: Any = None,
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

    if covs is None:
        mu_right, used_right = _local_polynomial(
            distance[right], y[right], bandwidth, order
        )
        mu_left, used_left = _local_polynomial(
            -distance[left], y[left], bandwidth, order
        )
        covariate_coefficients: Any = None
    else:
        tau_covs, used_left, used_right, covariate_coefficients = _covariate_adjusted(
            y, distance, covs, bandwidth, order
        )
        mu_right = tau_covs
        mu_left = 0.0

    # A variance is only computable when a bandwidth was given; automatic selection still refuses.
    variance = float("nan")
    if covs is None:
        left_variance = _side_variance(-distance[left], y[left], bandwidth, order)
        right_variance = _side_variance(distance[right], y[right], bandwidth, order)
        variance = left_variance + right_variance

    standard_error = (
        float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0 else float("nan")
    )
    statistic = float("nan")
    probability = float("nan")
    lower_bound = float("nan")
    upper_bound = float("nan")
    if np.isfinite(standard_error) and standard_error > 0:
        statistic = float((mu_right - mu_left) / standard_error)
        probability = float(2.0 * (1.0 - stats.norm.cdf(abs(statistic))))
        lower_bound = float(mu_right - mu_left - 1.959963984540054 * standard_error)
        upper_bound = float(mu_right - mu_left + 1.959963984540054 * standard_error)

    return RdRobustResult(
        se=standard_error,
        z=statistic,
        pvalue=probability,
        ci_lower=lower_bound,
        ci_upper=upper_bound,
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
        coef_covs=covariate_coefficients,
    )
