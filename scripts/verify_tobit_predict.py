"""Verify tobit e(a,b) and ystar(a,b) with exact row alignment."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import westata as ws  # noqa: E402
from westata.conformance.oracles.stata import StataOracle  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(11)
    n = 600
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 0.65 * x1 + 0.3 * x2 + rng.normal(scale=0.35, size=n)
    y = np.maximum(latent, 2.0)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    with StataOracle() as oracle:
        oracle.send_data(frame)
        # A row id that survives reordering through export delimited.
        oracle.run("gen long __id = _n")
        oracle.run("tobit y x1 x2, ll(2)")
        oracle.run("predict double ehat, e(2,.)")
        oracle.run("predict double ystarhat, ystar(2,.)")
        path = ROOT / "_tobit_check.csv"
        if path.exists():
            path.unlink()
        oracle.run(f'export delimited __id ehat ystarhat using "{path}", replace')
        stata = pd.read_csv(path).sort_values("__id").reset_index(drop=True)
        path.unlink()

    result = ws.Tobit.from_formula("y ~ x1 + x2", frame, lower=2.0).fit()
    westata_e = result.predict(kind="e")
    westata_ystar = result.predict(kind="ystar")

    e_gap = float(np.max(np.abs(westata_e / stata["ehat"].to_numpy() - 1)))
    ystar_gap = float(np.max(np.abs(westata_ystar / stata["ystarhat"].to_numpy() - 1)))

    print(f"e(2,.)     max rel gap = {e_gap:.3e}")
    print(f"ystar(2,.) max rel gap = {ystar_gap:.3e}")
    print()
    print("first 5 Stata e    :", np.round(stata["ehat"].to_numpy()[:5], 6))
    print("first 5 westata e  :", np.round(westata_e[:5], 6))
    print("first 5 Stata ystar:", np.round(stata["ystarhat"].to_numpy()[:5], 6))
    print("first 5 westata yst:", np.round(westata_ystar[:5], 6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
