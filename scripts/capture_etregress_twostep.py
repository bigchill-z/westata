"""Capture etregress's two-step estimator, which is what westata implements.

The first capture compared westata's control-function estimate against Stata's DEFAULT,
which is maximum likelihood. Those are different estimators, so a gap is expected and
neither is wrong -- comparing them and reporting "5.3e-2 off" would be measuring the
difference between two methods rather than a defect.

Stata can run the two-step version with ``etregress, twostep``, so the honest comparison
is that. Capturing it now.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

FIXTURES = ROOT / "tests" / "conformance" / "stata_fixtures"


def treatreg_frame(n: int = 800, seed: int = 107) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x1 = rng.normal(size=n)
    common = rng.normal(size=n)
    u = 0.6 * common + 0.8 * rng.normal(size=n)
    e = 0.6 * common + 0.8 * rng.normal(size=n)
    treated = (0.6 * z + u) > 0
    y = 1.0 + 2.0 * x1 + 1.5 * treated.astype(float) + e
    return pd.DataFrame({"y": y, "x1": x1, "z": z, "treated": treated.astype(float)})


def capture_matrix(oracle: StataOracle, command: str) -> str:
    oracle.run(f"quietly {command}")
    grabbed = oracle.capture()
    lines = [f"COMMAND: {command}", f"NAMES: {list(grabbed.names)}", f"B: {list(grabbed.b)}"]
    lines.append("V:")
    for row in grabbed.V:
        lines.append(" ".join(repr(float(value)) for value in row))
    return "\n".join(lines)


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    frame = treatreg_frame()
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        oracle.send_data(frame)
        for label, command in [
            ("etregress_twostep", "etregress y x1, treat(treated = z) twostep"),
            ("etregress_ml", "etregress y x1, treat(treated = z)"),
        ]:
            try:
                captured[label] = capture_matrix(oracle, command)
                print(f"  OK   {label}")
            except Exception as exc:  # noqa: BLE001
                captured[label] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"  FAIL {label}: {str(exc)[:90]}")

    path = FIXTURES / "e_upgrade_etregress.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
