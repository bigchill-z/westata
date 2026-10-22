"""Investigate reghdfe's e(df_r) under clustering."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402


def main() -> int:
    rng = np.random.default_rng(21)
    n_entities, n_periods = 30, 6
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    period = rng.normal(size=n_periods)[time - 1]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + period + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})

    with StataOracle() as oracle:
        oracle.send_data(frame)
        for command in [
            "reghdfe y x1 x2, absorb(firm) cluster(firm)",
            "reghdfe y x1 x2, absorb(firm year) cluster(firm)",
        ]:
            print("=" * 70)
            print(command)
            print("=" * 70)
            oracle.run("discard")
            print(oracle.get_output(command)[:1800])
            capture = oracle.capture()
            print("scalars of interest:")
            for key in ("N", "df_r", "df_m", "r2", "r2_a", "N_clust", "rank"):
                if key in capture.scalars:
                    print(f"  e({key}) = {capture.scalars[key]}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
