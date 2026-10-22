"""Capture reghdfe fixtures (installed from GitHub; SSC was unreachable)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

FIXTURE_DIR = ROOT / "tests" / "conformance" / "stata_fixtures"

SCENARIOS = {
    "reghdfe_firm": "reghdfe y x1 x2, absorb(firm)",
    "reghdfe_firm_year": "reghdfe y x1 x2, absorb(firm year)",
    "reghdfe_firm_robust": "reghdfe y x1 x2, absorb(firm) vce(robust)",
    "reghdfe_firm_cluster": "reghdfe y x1 x2, absorb(firm) cluster(firm)",
    "reghdfe_firm_year_cluster": "reghdfe y x1 x2, absorb(firm year) cluster(firm)",
}


def panel_frame(n_entities: int = 30, n_periods: int = 6, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    period = rng.normal(size=n_periods)[time - 1]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + period + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = panel_frame()

    with StataOracle() as oracle:
        oracle.send_data(frame)
        for name, command in SCENARIOS.items():
            oracle.run("discard")
            oracle.run(command)
            capture = oracle.capture()
            path = capture.save(FIXTURE_DIR / f"{name}.json")
            print(f"{name:28s} -> {path.name}  names={capture.names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
