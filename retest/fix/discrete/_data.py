"""Shared DGP and Stata/westata alignment helpers for the discrete-group probes.

``make_data`` reproduces ``retest/domains/discrete/run.py``'s generator *exactly*,
including the order the random draws are consumed, so the numbers printed here
can be compared with the domain report's 70-point table (seed 20261024, n=600).

``align_stata`` exists because the ordering of ``e(b)``/``e(V)`` differs between
the two sides and getting it backwards is silent: westata puts the constant
first, Stata puts it last, and ``zip``/``zinb`` repeat ``x1 x2 _cons`` for the
inflation equation, so name matching is unsafe too.  ``perm[i]`` is the **Stata**
index of westata parameter ``i``, and the matrix is reindexed on Stata's side --
reindexing westata's matrix instead would apply the inverse permutation and
produce ~1e1 "relative differences" that are pure bookkeeping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_data(seed: int = 20261024, n: int = 600) -> pd.DataFrame:
    """The domain report's DGP, draw for draw."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    grpf = rng.integers(0, 3, size=n)          # factor variable, 3 levels
    g = rng.integers(0, 10, size=n)            # cluster variable, 10 groups
    eta_b = 0.4 + 0.7 * x1 - 0.5 * x2
    yb = rng.binomial(1, 1 / (1 + np.exp(-eta_b)), size=n)
    mu = np.exp(0.3 + 0.5 * x1 - 0.3 * x2)
    y_c = rng.poisson(mu)
    size = 1.0 / 0.5                           # true alpha = 0.5
    y_nb = rng.negative_binomial(size, size / (size + mu), size=n)
    pz = 1 / (1 + np.exp(-(1.2 * x1 - 0.8 * x2 - 0.3)))
    zero_draw = rng.random(n) < pz
    y_zi = np.where(zero_draw, 0, rng.poisson(mu))
    y_zinb = np.where(zero_draw, 0, y_nb)
    ycat = np.digitize(eta_b + rng.logistic(size=n), [-1.0, 0.5, 2.0]) + 1
    w = rng.integers(1, 4, size=n).astype(float)
    return pd.DataFrame(
        {"yb": yb, "y_c": y_c, "y_nb": y_nb, "y_zi": y_zi, "y_zinb": y_zinb,
         "ycat": ycat, "x1": x1, "x2": x2, "grpf": grpf, "g": g, "w": w}
    )


def align_stata(perm, b, V) -> tuple[np.ndarray, np.ndarray]:
    """Reorder Stata's ``e(b)``/``e(V)`` into westata's parameter order.

    ``perm[i]`` is the Stata index of westata parameter ``i``; the returned
    array/matrix line up element-for-element with ``result.params`` and
    ``result.cov_params()``.
    """

    index = np.asarray(perm, dtype=int)
    beta = np.asarray(b, dtype=float).reshape(-1)[index]
    matrix = np.asarray(V, dtype=float)
    return beta, matrix[np.ix_(index, index)]


def rel(actual, expected, floor: float = 1e-12) -> float:
    """BRIEF's relative difference, with the report's 1e-12 denominator floor."""

    a = np.asarray(actual, dtype=float).reshape(-1)
    e = np.asarray(expected, dtype=float).reshape(-1)
    if a.size != e.size:
        return float("inf")
    return float(np.max(np.abs(a - e) / np.maximum(np.abs(e), floor)))


def tier(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value <= 1e-9:
        return "A"
    if value <= 1e-6:
        return "B"
    if value <= 1e-3:
        return "C"
    return "E"
