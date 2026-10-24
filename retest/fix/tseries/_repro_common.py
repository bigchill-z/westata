"""Shared data + plumbing for the tseries repair repros.

The data-generating processes are the SAME ones used by the domain retest
(``retest/domains/tseries/run.py``: ``SEED = 20261024``, ``N = 300``,
``N_GARCH = 600``, GARCH seed 777) so every number here can be cross-checked
against ``retest/domains/tseries/NOTES.md``.  Nothing is read from
``tests/conformance`` or from the committed fixtures.

Run a repro with::

    cd D:\\myCodeProject\\statapy
    .venv\\Scripts\\python.exe retest/fix/tseries/repro_v1v2_var.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "retest" / "lib"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SEED = 20261024
N = 300
N_GARCH = 600

_rng = np.random.default_rng(SEED)


def ar1(phi: float = 0.6, n: int = N) -> np.ndarray:
    e = _rng.normal(size=n + 100)
    y = np.zeros(n + 100)
    for t in range(1, n + 100):
        y[t] = phi * y[t - 1] + e[t]
    return y[100:]


def ma1_random_walk(theta: float = 0.5, n: int = N) -> np.ndarray:
    e = _rng.normal(size=n + 1)
    return np.cumsum(e[1:] + theta * e[:-1])


def arma11_integrated(phi: float = 0.5, theta: float = 0.3, n: int = N) -> np.ndarray:
    e = _rng.normal(size=n + 150)
    d = np.zeros(n + 150)
    for t in range(1, n + 150):
        d[t] = phi * d[t - 1] + e[t] + theta * e[t - 1]
    return np.cumsum(d[150:])


def seasonal_series(n: int = N) -> np.ndarray:
    e = _rng.normal(size=n + 60)
    y = np.zeros(n + 60)
    for t in range(5, n + 60):
        y[t] = 0.5 * y[t - 1] + 0.4 * y[t - 4] - 0.2 * y[t - 5] + e[t]
    return y[60:60 + n]


def var2_process(n: int = N) -> np.ndarray:
    e = _rng.normal(size=(n + 100, 2))
    y = np.zeros((n + 100, 2))
    for t in range(1, n + 100):
        y[t, 0] = 0.5 * y[t - 1, 0] + 0.2 * y[t - 1, 1] + e[t, 0]
        y[t, 1] = -0.3 * y[t - 1, 0] + 0.4 * y[t - 1, 1] + e[t, 1]
    return y[100:]


def cointegrated_pair(n: int = N, phi: float = 0.4, sd: float = 0.05,
                      seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    r = np.random.default_rng(seed)
    e = r.normal(size=(n, 2))
    x = np.cumsum(e[:, 0])
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = phi * u[t - 1] + sd * e[t, 1]
    return 2.0 + 1.5 * x + u, x


def garch_process(n: int = N_GARCH, seed: int = 777) -> np.ndarray:
    r = np.random.default_rng(seed)
    omega, a, b = 0.05, 0.10, 0.85
    h = omega / (1 - a - b)
    y = np.zeros(n)
    for t in range(n):
        h = omega + a * (y[t - 1] ** 2 if t else h) + b * h
        y[t] = np.sqrt(h) * r.normal()
    return y


# Built exactly as in the domain runner so the Stata evidence lines up with
# retest/domains/tseries/NOTES.md.
Y_AR = ar1()
Y_RW = ma1_random_walk()
Y_111 = arma11_integrated()
Y_S = seasonal_series()
Y_VAR = var2_process()
YC1, YC2 = cointegrated_pair(seed=1)
# Borderline cointegration pair: trace(r=1) lands between Stata's 5% critical
# value (3.76) and statsmodels' Osterwald-Lenum one (3.8415).
YB1, YB2 = cointegrated_pair(seed=248)
Y_GARCH = garch_process()


def vol_frame() -> pd.DataFrame:
    return pd.DataFrame({"t": np.arange(1, N_GARCH + 1), "yg": Y_GARCH})


def rel(a, b) -> float:
    """max |a-b| / max(|b|, 1e-12) elementwise (b = Stata reference)."""

    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def fmt_row(label: str, before, after, stata) -> str:
    return (f"{label:<44s} before={before!s:<26s} after={after!s:<26s} "
            f"stata={stata!s}")


def start_stata():
    from stata_probe import Stata

    return Stata()
