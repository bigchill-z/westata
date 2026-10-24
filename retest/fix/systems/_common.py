"""Shared helpers for the ``systems`` group's repro scripts.

Each ``repro_S*.py`` script imports from here so the Stata session, the data
builders and the comparison helpers are written once.  Run any script from the
repository root, e.g.::

    .venv\\Scripts\\python.exe retest/fix/systems/repro_S1.py

Never put these scripts under ``/tmp``: a stray ``profile.py`` there shadows the
standard library module that ``pystata`` pulls in via IPython.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LIB = os.path.join(ROOT, "retest", "lib")
for path in (LIB, os.path.join(ROOT, "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from stata_probe import Stata, rel_diff, rel_pairs  # noqa: E402

AUTO = os.path.join(ROOT, "tests", "data", "auto.dta")


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def row(label: str, before, after, stata, fmt="{:.10g}") -> None:
    """Print one ``before / after / Stata / rel`` line."""

    b = "n/a" if before is None else fmt.format(before)
    a = "n/a" if after is None else fmt.format(after)
    s = "n/a" if stata is None else fmt.format(stata)
    rb = "n/a" if (before is None or stata is None) else f"{rel_pairs(before, stata):.3e}"
    ra = "n/a" if (after is None or stata is None) else f"{rel_pairs(after, stata):.3e}"
    print(f"  {label:<28} {b:>22} {a:>22} {s:>22}   {rb:>9} -> {ra:>9}")


def header() -> None:
    print(f"  {'quantity':<28} {'westata BEFORE':>22} {'westata AFTER':>22} "
          f"{'Stata':>22}   {'rel':>9}    {'rel':>9}")
    print("  " + "-" * 118)


def auto_frame() -> pd.DataFrame:
    return pd.read_stata(AUTO, convert_categoricals=False)


def treatment_frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """The retest's reference confounded sample (``systems_causal/run.py``)."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    t = (rng.uniform(size=n) < p).astype(int)
    y = 2.0 * t + 1.0 * x1 + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame(dict(y=y, t=t, x1=x1, x2=x2))


def panel_two_period(n_units: int = 60, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        fe = rng.normal(0, 1)
        for period in (1, 2):
            d = 1 if (i >= n_units // 2 and period == 2) else 0
            rows.append(
                dict(unit=i + 1, period=period, treated=d,
                     y=fe + 0.5 * (period - 1) + 1.2 * d + rng.normal(0, 0.7),
                     x1=rng.normal())
            )
    return pd.DataFrame(rows)


def panel_five_period(seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(120):
        fe = rng.normal(0, 1)
        cohort = 3 if i < 60 else 0
        for period in (1, 2, 3, 4, 5):
            d = 1 if (cohort and period >= cohort) else 0
            rows.append(
                dict(unit=i + 1, period=period, treated=d,
                     y=fe + 0.3 * (period - 1) + 1.0 * d + rng.normal(0, 0.6))
            )
    return pd.DataFrame(rows)


__all__ = [
    "AUTO",
    "Stata",
    "auto_frame",
    "banner",
    "header",
    "panel_five_period",
    "panel_two_period",
    "rel_diff",
    "rel_pairs",
    "row",
    "treatment_frame",
]
