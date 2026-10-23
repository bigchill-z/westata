"""Capture pperron separately (the earlier run lost the frame to the GARCH data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

FIXTURES = ROOT / "tests" / "conformance" / "stata_fixtures"


def main() -> int:
    rng = np.random.default_rng(41)
    n = 400
    rw1 = np.cumsum(rng.normal(size=n))
    rw2 = rw1 + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame({"t": np.arange(1, n + 1), "y1": rw1, "y2": rw2})

    captured: dict[str, str] = {}
    with StataOracle() as oracle:
        oracle.send_data(frame)
        oracle.run("tsset t")
        for label, command in [
            ("pperron_default", "pperron y1"),
            ("pperron_lag1", "pperron y1, lags(1)"),
            ("pperron_lag0", "pperron y1, lags(0)"),
            ("dfuller", "dfuller y1"),
            ("dfuller_lag1", "dfuller y1, lags(1)"),
        ]:
            try:
                output = oracle.get_output(command)
                captured[label] = f"COMMAND: {command}\n\n{output}"
                print(f"  OK   {command}")
            except Exception as exc:  # noqa: BLE001
                captured[label] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"  FAIL {command}: {str(exc)[:80]}")

    path = FIXTURES / "e_upgrade_pperron.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
