"""Capture Stata references for the P11 causal-inference commands.

Scope:

  didregress        two-way fixed-effects difference-in-differences
  xtdidregress      panel difference-in-differences
  teffects ra       regression adjustment
  teffects ipw      inverse-probability weighting
  teffects aipw     augmented IPW
  teffects nnmatch  nearest-neighbour matching
  teffects psmatch  propensity-score matching

The hard part of this batch is not the estimators -- every one is a few lines of
algebra -- but the ESTIMAND and the standard errors. ``teffects`` defaults to the
average treatment effect on the treated (ATE vs ATET changes every number), and
its standard errors come from influence functions, which are easy to get subtly
wrong while the point estimate still looks right. So the fixtures record both the
potential-outcome means and their standard errors, not just the difference.
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


def treatment_frame(n: int = 1200, seed: int = 211) -> pd.DataFrame:
    """Observational study with confounding between x1 and the treatment."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Treatment probability depends on x1 -- that is the confounding.
    propensity = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1 - 0.4 * x2)))
    treated = (rng.random(size=n) < propensity).astype(float)
    y = 1.0 + 2.0 * treated + 1.5 * x1 - 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "treated": treated, "x1": x1, "x2": x2})


def did_frame(n_units: int = 200, n_periods: int = 8, seed: int = 223) -> pd.DataFrame:
    """Balanced panel with staggered treatment timing."""

    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        effect = rng.normal(scale=0.6)
        # Half the units are treated, starting somewhere in the middle.
        is_treated = unit % 2 == 0
        start = 4 if is_treated else None
        for period in range(1, n_periods + 1):
            post = 1.0 if (start is not None and period >= start) else 0.0
            x1 = rng.normal()
            y = 1.0 + 0.5 * x1 + effect + 1.2 * post + rng.normal(scale=0.5)
            rows.append((unit, period, y, x1, float(is_treated), post))
    return pd.DataFrame(
        rows, columns=["unit", "period", "y", "x1", "treated", "post"]
    )


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        # ---- teffects on the observational sample ----
        oracle.send_data(treatment_frame())
        teffects_commands = [
            ("teffects_ra", "teffects ra (y x1 x2) (treated)"),
            ("teffects_ipw", "teffects ipw (y) (treated x1 x2)"),
            ("teffects_aipw", "teffects aipw (y x1 x2) (treated x1 x2)"),
            ("teffects_ra_atet", "teffects ra (y x1 x2) (treated), atet"),
            (
                "teffects_nnmatch",
                "teffects nnmatch (y x1 x2) (treated), nneighbor(1)",
            ),
        ]
        for name, command in teffects_commands:
            try:
                output = oracle.get_output(command)
                captured[name] = f"COMMAND: {command}\n\n{output}"
                print(f"=== {name}: OK ===")
            except Exception as exc:  # noqa: BLE001
                captured[name] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"=== {name}: FAILED {str(exc)[:110]} ===")

        # ---- difference-in-differences ----
        panel = did_frame()
        oracle.send_data(panel)
        oracle.run("xtset unit period")
        for name, command in [
            ("didregress", "didregress (y x1) (treated), group(unit) time(period)"),
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
                print(f"=== {name}: FAILED {str(exc)[:110]} ===")

    path = FIXTURES / "p11_outputs.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
