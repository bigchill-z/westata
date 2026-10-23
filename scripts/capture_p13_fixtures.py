"""Capture Stata references for the P13 resampling commands.

Scope:

    bootstrap   nonparametric bootstrap of a statistic or an estimator's VCE
    jackknife   delete-one jackknife
    permute     permutation test
    bs          bootstrap with full control over the resampling

The difficulty here is not any single formula -- jackknife and bootstrap variances
are textbook -- but the DETERMINISM. Bootstrap results depend on the random number
stream, so a fixture only means something if the seed is recorded and the same
replicate draws can be reproduced. Stata's RNG is not numpy's, so an exact
replicate-for-replicate match is impossible by construction; what IS testable is:

  * the jackknife, which is deterministic (no RNG at all) and must match exactly
  * the bootstrap's asymptotic behaviour, and its agreement with the analytic VCE
    on a linear statistic
  * the point estimate, which never depends on resampling

So this capture records the jackknife at full precision (an exact target) and records
bootstrap output with its seed for the magnitude checks.
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


def resample_frame(n: int = 400, seed: int = 401) -> pd.DataFrame:
    """Heteroskedastic regression sample, so robust and OLS variances differ."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scale = 0.5 + 0.6 * np.abs(x1)
    y = 1.0 + 2.0 * x1 - 0.8 * x2 + scale * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    frame = resample_frame()
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        oracle.send_data(frame)

        # ---- jackknife: deterministic, so it is an exact target ----
        print("jackknife (deterministic)")
        cases = [
            ("jack_mean", "jackknife, cluster(id): mean y"),
        ]
        # The cluster form needs an id variable; build one first.
        oracle.run("gen long id = _n")
        oracle.run("set seed 20261023")

        for label, command in [
            ("bootstrap_mean", "bootstrap r(mean), reps(200) seed(20261023): summarize y"),
            ("bootstrap_regress", "bootstrap, reps(200) seed(20261023): regress y x1 x2"),
            ("jackknife_regress", "jackknife: regress y x1 x2"),
            ("permute", "permute (x1) _b[x1], reps(200) seed(20261023): regress y x1 x2"),
        ]:
            try:
                output = oracle.get_output(command)
                captured[label] = f"COMMAND: {command}\n\n{output}"
                print(f"  OK   {command}")
            except Exception as exc:  # noqa: BLE001
                captured[label] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"  FAIL {command[:60]}: {str(exc)[:90]}")

        # ---- full-precision jackknife VCE ----
        print("full-precision jackknife e(b)/e(V)")
        for label, command in [
            ("full_jack_regress", "jackknife: regress y x1 x2"),
            ("full_jack_mean", "jackknife: mean y"),
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
                captured[label] = "\n".join(lines)
                print(f"  OK   {label}")
            except Exception as exc:  # noqa: BLE001
                captured[label] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"  FAIL {label}: {str(exc)[:80]}")

        # ---- the analytic VCEs these should be compared against ----
        print("analytic baselines")
        for label, command in [
            ("analytic_regress", "regress y x1 x2"),
            ("analytic_regress_robust", "regress y x1 x2, vce(robust)"),
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
                captured[label] = "\n".join(lines)
                print(f"  OK   {label}")
            except Exception as exc:  # noqa: BLE001
                captured[label] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"  FAIL {label}: {str(exc)[:80]}")

    path = FIXTURES / "p13_outputs.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
