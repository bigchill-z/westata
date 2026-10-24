"""Shared helpers for the linear-iv repro/verification scripts.

Everything here is a *measurement* helper: it starts a real StataNow 19.5
session through ``retest/lib/stata_probe.py`` and compares against westata's
live (editable-install) results.  No fixtures, no registry.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for _p in (REPO / "retest" / "lib", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from stata_probe import Stata, rel_diff  # noqa: E402,F401

warnings.simplefilter("ignore")

AUTO = REPO / "tests" / "data" / "auto.dta"


def auto_frame() -> pd.DataFrame:
    """``auto.dta`` with the numeric columns used by ``regress``.

    ``foreign`` is converted to a 0/1 indicator so Stata and westata see the
    identical matrix; ``gid`` (12 clusters) and ``w`` (1..4) mirror the Stata
    ``mod(_n, 12)`` / ``mod(_n, 4) + 1`` expressions used in the captures.
    """

    df = pd.read_stata(AUTO)[["mpg", "weight", "foreign"]].copy()
    df["foreign"] = (df["foreign"] == "Foreign").astype(float)
    n = len(df)
    df["gid"] = np.array([(i + 1) % 12 for i in range(n)], dtype=float)
    df["w"] = np.array([((i + 1) % 4) + 1 for i in range(n)], dtype=float)
    return df


def iv_panel(seed: int = 20261024, n: int = 140, g: int = 14, n_instr: int = 2) -> pd.DataFrame:
    """Reproducible IV panel: fixed seed, ``g`` balanced clusters, ``n_instr`` instruments.

    Same generator as ``retest/domains/linear_iv/run.py`` for the 140x14x2 case,
    so the numbers in the retest report remain directly comparable.
    """

    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(g), n // g)
    x1 = rng.normal(size=n)
    z = rng.normal(size=(n, n_instr))
    e1 = 0.8 * z[:, 0] + rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.4 * e1 + rng.normal(0.0, 0.7, n)
    out = {"y": y, "x1": x1, "e1": e1, "gid": gid}
    for j in range(n_instr):
        out[f"z{j + 1}"] = z[:, j]
    return pd.DataFrame(out)


def retest_iv_panel() -> pd.DataFrame:
    """The *exact* panel ``retest/domains/linear_iv/run.py`` used for item 12d.

    Copied from that script's ``make_panel`` (seed 20261024, 140 obs, 14
    clusters) so the L3 "before" number can be checked against the 3.58e-2 the
    retest report quotes.  It carries extra columns (``y``, ``x2``, ``t``,
    ``cont``, ``w``) because the same frame feeds the non-IV items there.
    """

    rng = np.random.default_rng(20261024)
    n, g = 140, 14
    gid = np.repeat(np.arange(g), n // g)
    tt = np.tile(np.arange(10), g)
    cont = rng.choice([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5], size=n)
    fe_g = rng.normal(0.0, 0.5, g)[gid]
    fe_t = rng.normal(0.0, 0.3, 10)[tt]
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + fe_g + fe_t + rng.normal(0.0, 0.7, n)
    w = rng.integers(1, 5, size=n).astype(float)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    e1 = 0.8 * z1 + 0.3 * z2 + rng.normal(size=n) + 0.5 * rng.normal(size=n)
    yiv = 1.0 + 2.0 * x1 - 0.4 * e1 + rng.normal(0.0, 0.7, n)
    return pd.DataFrame(
        {
            "y": y,
            "yiv": yiv,
            "x1": x1,
            "x2": x2,
            "gid": gid,
            "t": tt,
            "cont": cont,
            "w": w,
            "z1": z1,
            "z2": z2,
            "e1": e1,
        }
    )


def rel(actual, expected) -> float:
    return rel_diff(actual, expected)


def fmt(value) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if value != value:
        return "nan"
    return f"{value:.6g}"


def stata_bv(cap, order):
    """Return ``(b, se)`` of a :class:`Capture` reordered to ``order``.

    Stata's ``e(b)`` column order is not the model order (``ivregress`` lists
    endogenous regressors first), so every comparison has to reorder explicitly
    instead of trusting position.
    """

    idx = [cap.names.index(name) for name in order]
    return (
        np.asarray(cap.b, dtype=float)[idx],
        np.sqrt(np.diag(cap.V))[idx],
    )
