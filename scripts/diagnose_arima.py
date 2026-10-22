"""Diagnose the ARIMA gap against Stata: conditional vs exact likelihood."""

from __future__ import annotations

import sys

from pathlib import Path

import numpy as np
from scipy import optimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

STATA_DIR = r"D:\Program Files\StataNow19"


def ar1_sim(n: int = 500, phi: float = 0.7, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + shocks[t]
    return y


def neg_conditional(theta: np.ndarray, y: np.ndarray) -> float:
    """Conditional log-likelihood: drop y_1, integrate from y_2 onward."""

    mu, phi, sigma = theta
    e = y[1:] - mu - phi * (y[:-1] - mu)
    n = e.size
    return float(
        0.5 * n * (np.log(2 * np.pi) + 2 * np.log(sigma)) + 0.5 * (e @ e) / sigma**2
    )


def neg_exact(theta: np.ndarray, y: np.ndarray) -> float:
    """Exact likelihood: y_1 marginal + conditional terms."""

    mu, phi, sigma = theta
    e = y[1:] - mu - phi * (y[:-1] - mu)
    n = e.size
    # Marginal variance of a stationary AR(1).
    var0 = sigma**2 / (1.0 - phi**2)
    first = (y[0] - mu) ** 2 / var0 + np.log(var0)
    return float(
        0.5 * (n + 1) * np.log(2 * np.pi)
        + 0.5 * n * 2 * np.log(sigma)
        + 0.5 * first
        + 0.5 * (e @ e) / sigma**2
    )


def fit(objective, start, y):
    result = optimize.minimize(
        objective,
        start,
        args=(y,),
        method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-12, "fatol": 1e-12},
    )
    result = optimize.minimize(
        objective,
        result.x,
        args=(y,),
        method="BFGS",
        options={"gtol": 1e-12},
    )
    return result.x


def main() -> int:
    y = ar1_sim()
    print("=" * 70)
    print("Stata")
    print("=" * 70)
    with StataOracle(stata_dir=STATA_DIR, edition="mp") as oracle:
        import pandas as pd

        oracle.send_data(pd.DataFrame({"y": y, "t": np.arange(1, len(y) + 1)}))
        oracle.run("tsset t")
        oracle.run("arima y, arima(1,0,0) nolog")
        capture = oracle.capture()
    names = list(capture.names)
    b = np.asarray(capture.b, dtype=float)
    print(f"  e(b) = {dict(zip(names, b))}")
    print(f"  sigma = {b[2]:.10f}")

    print()
    print("=" * 70)
    print("Conditional MLE (drop y_1)")
    print("=" * 70)
    theta_c = fit(neg_conditional, np.array([y.mean(), 0.5, y.std()]), y)
    print(f"  mu = {theta_c[0]:.10f}  phi = {theta_c[1]:.10f}  sigma = {theta_c[2]:.10f}")

    print()
    print("=" * 70)
    print("Exact MLE (stationary y_1 marginal)")
    print("=" * 70)
    theta_e = fit(neg_exact, np.array([y.mean(), 0.5, y.std()]), y)
    print(f"  mu = {theta_e[0]:.10f}  phi = {theta_e[1]:.10f}  sigma = {theta_e[2]:.10f}")

    print()
    print("=" * 70)
    print("Comparison with Stata (phi and const)")
    print("=" * 70)
    print(f"  Stata const = {b[0]:.10f}  phi = {b[1]:.10f}")
    for label, theta in [("conditional", theta_c), ("exact", theta_e)]:
        d_phi = abs(theta[1] - b[1]) / abs(b[1])
        d_c = abs(theta[0] - b[0]) / abs(b[0])
        print(f"  {label:12s} const rel={d_c:.3e}  phi rel={d_phi:.3e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
