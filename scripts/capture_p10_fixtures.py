"""Capture Stata references for the P10 multi-equation commands.

Scope:

  sureg       seemingly unrelated regression
  mvreg       multivariate regression
  nl          non-linear least squares
  nlsur       non-linear SUR
  gmm         generalised method of moments
  eivreg      errors-in-variables regression
  manova      multivariate analysis of variance
  frontier    stochastic frontier

The interesting difficulty here is not any single estimator -- most are closed
form -- but the CROSS-EQUATION covariance. sureg's entire point is that the
equations' errors are correlated, and getting that correlation wrong leaves every
point estimate identical while changing every standard error. So the fixtures
record both the coefficients and the VCE.
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


def sur_frame(n: int = 400, seed: int = 101) -> pd.DataFrame:
    """Two outcomes whose errors are strongly correlated -- the sureg case."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    common = rng.normal(size=n)
    e1 = 0.7 * common + 0.7 * rng.normal(size=n)
    e2 = 0.7 * common + 0.7 * rng.normal(size=n)
    y1 = 1.0 + 2.0 * x1 - 0.5 * x2 + e1
    y2 = -0.5 + 1.0 * x2 + 0.8 * x3 + e2
    return pd.DataFrame({"y1": y1, "y2": y2, "x1": x1, "x2": x2, "x3": x3})


def nl_frame(n: int = 300, seed: int = 103) -> pd.DataFrame:
    """Data from a known exponential decay, for nl."""

    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0.1, 4.0, size=n))
    y = 5.0 * np.exp(-0.8 * x) + rng.normal(scale=0.15, size=n)
    return pd.DataFrame({"y": y, "x": x})


def eiv_frame(n: int = 500, seed: int = 107) -> pd.DataFrame:
    """Measurement error in x1, with a second noisy measure as the instrument."""

    rng = np.random.default_rng(seed)
    true_x = rng.normal(size=n)
    noise = rng.normal(scale=0.7, size=n)
    observed = true_x + noise
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * true_x - 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": observed, "x2": x2})


def manova_frame(n: int = 240, seed: int = 109) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 3, size=n)
    effect = np.array([0.0, 1.0, 2.0])[group]
    y1 = 1.0 + effect + rng.normal(scale=0.8, size=n)
    y2 = 2.0 + 0.5 * effect + rng.normal(scale=0.8, size=n)
    return pd.DataFrame({"y1": y1, "y2": y2, "g": group})


def frontier_frame(n: int = 400, seed: int = 113) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    # Production frontier with a one-sided inefficiency term.
    inefficiency = np.abs(rng.normal(scale=0.4, size=n))
    y = 1.0 + 0.8 * x1 + 0.5 * x2 - inefficiency + rng.normal(scale=0.2, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        cases: list[tuple[str, pd.DataFrame, list[str]]] = [
            (
                "sureg",
                sur_frame(),
                [
                    "sureg (y1 x1 x2) (y2 x2 x3)",
                    "sureg (y1 x1 x2) (y2 x2 x3), isure",
                ],
            ),
            ("mvreg", sur_frame(), ["mvreg y1 y2 = x1 x2 x3"]),
            ("nl", nl_frame(), ["nl (y = {b0} * exp(-{b1} * x))"]),
            ("eivreg", eiv_frame(), ["eivreg y x2, reliab(x1 0.7)"]),
            ("manova", manova_frame(), ["manova y1 y2 = g"]),
            ("frontier", frontier_frame(), ["frontier y x1 x2"]),
        ]

        for name, frame, commands in cases:
            oracle.send_data(frame)
            for position, command in enumerate(commands):
                key = f"{name}_{position}"
                label = "sureg_breusch" if (name == "sureg" and position == 1) else key
                try:
                    output = oracle.get_output(command)
                    captured[label] = f"COMMAND: {command}\n\n{output}"
                    print(f"=== {label}: OK ({len(frame)} obs) ===")
                except Exception as exc:  # noqa: BLE001
                    captured[label] = f"COMMAND: {command}\n\nFAILED: {exc}"
                    print(f"=== {label}: FAILED {str(exc)[:110]} ===")

        # Breusch-Pagan test of cross-equation correlation, the statistic that
        # makes sureg worth doing at all.
        try:
            oracle.run("quietly sureg (y1 x1 x2) (y2 x2 x3)")
            captured["sureg_bp"] = oracle.get_output("estat corr")
            print("=== sureg estat corr: OK ===")
        except Exception as exc:  # noqa: BLE001
            captured["sureg_bp"] = f"FAILED: {exc}"
            print(f"=== sureg estat corr: FAILED {str(exc)[:80]} ===")

    path = FIXTURES / "p10_outputs.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
