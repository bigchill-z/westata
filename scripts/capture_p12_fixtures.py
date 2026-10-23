"""Re-capture the P12 fixtures with a valid fpc specification.

``svyset ..., fpc(N)`` failed with "variable N not found": fpc takes a VARIABLE
naming the population size per stratum, not a literal count. The frame now carries
an ``fpc`` column.
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


def survey_frame(
    n_strata: int = 8, psu_per_stratum: int = 6, n_per_psu: int = 12, seed: int = 301
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    # Population per stratum is larger than the sample, so the fpc is < 1.
    population_per_stratum = psu_per_stratum * n_per_psu + 24
    for stratum in range(1, n_strata + 1):
        stratum_effect = rng.normal(scale=0.5)
        for psu in range(1, psu_per_stratum + 1):
            psu_effect = rng.normal(scale=0.4)
            weight = float(rng.uniform(0.5, 2.5))
            for _ in range(n_per_psu):
                x1 = rng.normal()
                x2 = rng.normal()
                y = (
                    1.0
                    + 0.8 * x1
                    - 0.4 * x2
                    + stratum_effect
                    + psu_effect
                    + rng.normal(scale=0.6)
                )
                rows.append(
                    (
                        stratum,
                        psu,
                        y,
                        x1,
                        x2,
                        weight,
                        float(y > 1.0),
                        population_per_stratum,
                    )
                )
    return pd.DataFrame(
        rows, columns=["stratum", "psu", "y", "x1", "x2", "w", "binary", "fpc"]
    )


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    frame = survey_frame()
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        oracle.send_data(frame)

        # ---- design A: stratified, with replacement (no fpc) ----
        print("design A: stratified, with replacement")
        oracle.run("svyset psu [pw=w], strata(stratum) singleunit(certainty)")
        for position, command in enumerate(
            [
                "svy: mean y",
                "svy: total y",
                "svy: proportion binary",
                "svy: regress y x1 x2",
                "svy: logit binary x1 x2",
            ]
        ):
            try:
                output = oracle.get_output(command)
                captured[f"design_a_{position}"] = f"COMMAND: {command}\n\n{output}"
                print(f"  OK   {command}")
            except Exception as exc:  # noqa: BLE001
                captured[f"design_a_{position}"] = (
                    f"COMMAND: {command}\n\nFAILED: {exc}"
                )
                print(f"  FAIL {command}: {str(exc)[:70]}")

        # ---- design B: finite population correction ----
        print("design B: with fpc")
        oracle.run(
            "svyset psu [pw=w], strata(stratum) fpc(fpc) singleunit(certainty)"
        )
        for position, command in enumerate(["svy: mean y", "svy: regress y x1"]):
            try:
                output = oracle.get_output(command)
                captured[f"design_b_{position}"] = f"COMMAND: {command}\n\n{output}"
                print(f"  OK   {command}")
            except Exception as exc:  # noqa: BLE001
                captured[f"design_b_{position}"] = (
                    f"COMMAND: {command}\n\nFAILED: {exc}"
                )
                print(f"  FAIL {command}: {str(exc)[:70]}")

        # ---- full-precision e(b)/e(V) for design A ----
        print("full precision captures")
        oracle.run("svyset psu [pw=w], strata(stratum) singleunit(certainty)")
        for label, command in [
            ("regress", "svy: regress y x1 x2"),
            ("mean", "svy: mean y"),
            ("logit", "svy: logit binary x1 x2"),
            ("total", "svy: total y"),
        ]:
            try:
                oracle.run(f"quietly {command}")
                capture = oracle.capture()
                lines = [
                    f"COMMAND: {command}",
                    f"NAMES: {list(capture.names)}",
                    f"B: {list(capture.b)}",
                    "V:",
                ]
                for row in capture.V:
                    lines.append(" ".join(repr(float(value)) for value in row))
                captured[f"full_{label}"] = "\n".join(lines)
                print(f"  OK   {label}")
            except Exception as exc:  # noqa: BLE001
                captured[f"full_{label}"] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"  FAIL {label}: {str(exc)[:80]}")

    path = FIXTURES / "p12_outputs.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
