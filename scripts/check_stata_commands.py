"""Check which Stata commands needed for E3 verification are available."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

CHECKS = [
    ("xtreg", "which xtreg"),
    ("nbreg", "which nbreg"),
    ("ologit", "which ologit"),
    ("oprobit", "which oprobit"),
    ("margins", "which margins"),
    ("predict", "which predict"),
    ("var", "which var"),
    ("vec", "which vec"),
    ("arch", "which arch"),
    ("ivregress", "which ivregress"),
    ("esttab", "which esttab"),
    ("reghdfe", "which reghdfe"),
    ("areg", "which areg"),
    ("xtabond2", "which xtabond2"),
    ("outreg2", "which outreg2"),
]


def main() -> int:
    with StataOracle() as oracle:
        for name, command in CHECKS:
            try:
                output = oracle.get_output(command)
                resolved = output.strip().splitlines()
                first = resolved[0].strip() if resolved else ""
                print(f"{name:12s} OK    {first[:80]}")
            except Exception as exc:  # noqa: BLE001
                print(f"{name:12s} MISSING  {type(exc).__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
