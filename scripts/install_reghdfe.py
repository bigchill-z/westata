"""Try to install reghdfe (and its dependency ftools) into the Stata session."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402


def main() -> int:
    with StataOracle() as oracle:
        for command in [
            "ssc install ftools, replace",
            "ssc install reghdfe, replace",
            "which reghdfe",
        ]:
            print(f"--- {command} ---")
            try:
                print(oracle.get_output(command)[:1500])
            except Exception as exc:  # noqa: BLE001
                print(f"FAILED: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
