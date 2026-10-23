"""Capture IPS reference values, reading r() scalars through display."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

FIXTURES = ROOT / "tests" / "conformance" / "stata_fixtures"

SCALARS = (
    "tbar",
    "ttildebar",
    "zt",
    "zttildebar",
    "p_zttildebar",
    "N_g",
    "N_t",
    "N",
    "cv_1",
    "cv_5",
    "cv_10",
)


def panel(n_firms: int = 30, n_years: int = 12, seed: int = 501) -> pd.DataFrame:
    """Two thirds mean-reverting, one third near a random walk."""

    rng = np.random.default_rng(seed)
    rows = []
    for firm in range(1, n_firms + 1):
        phi = 0.6 if firm % 3 else 0.95
        value = rng.normal()
        for year in range(1, n_years + 1):
            value = phi * value + rng.normal()
            rows.append((firm, year, value))
    return pd.DataFrame(rows, columns=["firm", "year", "y"])


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    frame = panel()
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        oracle.send_data(frame)
        oracle.run("xtset firm year")

        for label, command in [
            ("ips_default", "xtunitroot ips y"),
            ("ips_trend", "xtunitroot ips y, trend"),
            ("ips_demean", "xtunitroot ips y, demean"),
            ("ips_lag1", "xtunitroot ips y, lags(1)"),
            ("llc_default", "xtunitroot llc y"),
            ("ht_default", "xtunitroot ht y"),
        ]:
            try:
                captured[label] = f"COMMAND: {command}\n\n" + oracle.get_output(command)
                print(f"  OK   {command}")
            except Exception as exc:  # noqa: BLE001
                captured[label] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"  FAIL {command}: {str(exc)[:90]}")

        try:
            oracle.run("quietly xtunitroot ips y")
            lines = []
            for name in SCALARS:
                lines.append(
                    oracle.get_output(f"display %25.17g r({name})").strip()
                )
            captured["ips_scalars"] = "\n".join(
                f"{name} = {value}" for name, value in zip(SCALARS, lines)
            )
            print("  OK   full-precision r() scalars")
            for name, value in zip(SCALARS, lines):
                print(f"       r({name}) = {value}")
        except Exception as exc:  # noqa: BLE001
            captured["ips_scalars"] = f"FAILED: {exc}"
            print(f"  FAIL scalars: {str(exc)[:90]}")

    path = FIXTURES / "p_ips_outputs.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
