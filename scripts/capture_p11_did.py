"""Re-capture the DID fixtures with a properly time-varying treatment.

The first attempt passed a time-invariant group indicator (`treated`) as the
treatment variable, and Stata rejected it: "The treatment variable treated was
omitted because of collinearity" against the unit fixed effects. That is correct
behaviour on Stata's part -- a variable that never changes within a unit IS the
fixed effect, so there is nothing left to identify.

`didregress`'s treatment variable must vary over time, i.e. it is the interaction
of group and post, not the group indicator alone.
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


def did_frame(
    n_units: int = 200, n_periods: int = 8, seed: int = 223
) -> pd.DataFrame:
    """Balanced panel, half the units treated from period 4 onward.

    ``treated`` here is the time-varying treatment indicator (0 before the unit's
    treatment date, 1 after), which is what didregress expects. ``group`` records
    the time-invariant arm for reference.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        effect = rng.normal(scale=0.6)
        in_treatment_arm = unit % 2 == 0
        start = 4
        for period in range(1, n_periods + 1):
            treated_now = (
                1.0 if (in_treatment_arm and period >= start) else 0.0
            )
            x1 = rng.normal()
            y = (
                1.0
                + 0.5 * x1
                + effect
                + 0.3 * period
                + 1.2 * treated_now
                + rng.normal(scale=0.5)
            )
            rows.append((unit, period, y, x1, treated_now, float(in_treatment_arm)))
    return pd.DataFrame(
        rows, columns=["unit", "period", "y", "x1", "treated", "group"]
    )


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        panel = did_frame()
        oracle.send_data(panel)
        oracle.run("xtset unit period")

        for name, command in [
            (
                "didregress",
                "didregress (y x1) (treated), group(unit) time(period)",
            ),
            (
                "xtdidregress",
                "xtdidregress (y x1) (treated), group(unit) time(period)",
            ),
        ]:
            try:
                output = oracle.get_output(command)
                captured[name] = f"COMMAND: {command}\n\n{output}"
                print(f"=== {name}: OK ===")
            except Exception as exc:  # noqa: BLE001
                captured[name] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"=== {name}: FAILED {str(exc)[:140]} ===")

    path = FIXTURES / "p11_did.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
