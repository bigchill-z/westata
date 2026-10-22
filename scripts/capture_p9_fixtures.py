"""Capture Stata references for the P9 panel extensions.

Scope: the non-linear and dynamic panel commands that have no westata
counterpart yet.

  xtlogit, fe / re / pa      panel binary
  xtprobit, re               panel binary
  xttobit                    panel censored
  xtpoisson, fe / re         panel count
  xtnbreg, fe                panel count
  xtivreg, fe / re           panel IV
  xtabond                    Arellano-Bond dynamic panel
  xtdpd                      Arellano-Bond via xtdpd syntax
  xtdpdsys                   Blundell-Bond system GMM
  xtgls                      panel GLS
  xtpcse                     panel-corrected standard errors
  xtunitroot                 panel unit root

Everything is captured against one balanced panel so the sample sizes are
comparable across commands.

Note on xtabond / xtdpd / xtdpdsys: these are the reason this batch is hard. They
differ in how many instruments they use and in whether differences are instrumented
by levels (system GMM), and the Stata output reports Sargan/Hansen statistics whose
construction is not fully documented. Only the coefficients and standard errors are
comparable to a future westata implementation; the test statistics may not be.
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


def binary_panel(n_firms: int = 120, n_years: int = 8, seed: int = 71) -> pd.DataFrame:
    """Balanced panel with a binary outcome and a firm effect."""

    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = n_firms * n_years
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(scale=0.8, size=n_firms)[firm]
    latent = 0.4 * x1 - 0.3 * x2 + effect + rng.logistic(size=n)
    return pd.DataFrame(
        {"firm": firm, "year": year, "x1": x1, "x2": x2, "y": (latent > 0).astype(float)}
    )


def count_panel(n_firms: int = 100, n_years: int = 7, seed: int = 73) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = n_firms * n_years
    x1 = rng.normal(size=n)
    effect = rng.normal(scale=0.4, size=n_firms)[firm]
    mu = np.exp(0.3 + 0.5 * x1 + effect)
    return pd.DataFrame(
        {"firm": firm, "year": year, "x1": x1, "y": rng.poisson(mu).astype(float)}
    )


def dynamic_panel(n_firms: int = 90, n_years: int = 9, seed: int = 79) -> pd.DataFrame:
    """AR(1) panel with a firm effect, the classic Arellano-Bond DGP."""

    rng = np.random.default_rng(seed)
    rows = []
    for firm in range(n_firms):
        effect = rng.normal(scale=0.5)
        lagged = rng.normal()
        for year in range(1, n_years + 1):
            x1 = rng.normal()
            value = 0.5 * lagged + 0.8 * x1 + effect + rng.normal(scale=0.5)
            rows.append((firm, year, value, x1))
            lagged = value
    frame = pd.DataFrame(rows, columns=["firm", "year", "y", "x1"])
    return frame


def censored_panel(n_firms: int = 100, n_years: int = 7, seed: int = 83) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = n_firms * n_years
    x1 = rng.normal(size=n)
    effect = rng.normal(scale=0.5, size=n_firms)[firm]
    latent = 1.0 + 0.9 * x1 + effect + rng.normal(scale=0.6, size=n)
    return pd.DataFrame(
        {"firm": firm, "year": year, "x1": x1, "y": np.maximum(latent, 0.5)}
    )


def iv_panel(n_firms: int = 100, n_years: int = 7, seed: int = 89) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    firm = np.repeat(np.arange(n_firms), n_years)
    year = np.tile(np.arange(1, n_years + 1), n_firms)
    n = n_firms * n_years
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    endog = 0.7 * z + 0.6 * u + rng.normal(size=n)
    effect = rng.normal(scale=0.4, size=n_firms)[firm]
    y = 1.0 + 1.2 * endog + 0.5 * z + effect + u
    return pd.DataFrame(
        {"firm": firm, "year": year, "y": y, "endog": endog, "z": z}
    )


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        cases: list[tuple[str, pd.DataFrame, list[str]]] = [
            (
                "xtlogit",
                binary_panel(),
                ["xtlogit y x1 x2, fe", "xtlogit y x1 x2, re", "xtlogit y x1 x2, pa"],
            ),
            ("xtprobit", binary_panel(), ["xtprobit y x1 x2, re"]),
            ("xttobit", censored_panel(), ["xttobit y x1, ll(.5)"]),
            (
                "xtpoisson",
                count_panel(),
                ["xtpoisson y x1, fe", "xtpoisson y x1, re"],
            ),
            ("xtnbreg", count_panel(), ["xtnbreg y x1, fe"]),
            (
                "xtivreg",
                iv_panel(),
                ["xtivreg y z (endog = z), fe", "xtivreg y z (endog = z), re"],
            ),
            (
                "xtabond",
                dynamic_panel(),
                ["xtabond y x1, lags(1)", "xtabond y x1, lags(2)"],
            ),
            ("xtdpd", dynamic_panel(), ["xtdpd y x1, dgmmiv(y)"]),
            ("xtdpdsys", dynamic_panel(), ["xtdpdsys y x1"]),
            ("xtgls", count_panel(), ["xtgls y x1, panels(heteroskedastic)"]),
            ("xtpcse", count_panel(), ["xtpcse y x1"]),
            ("xtunitroot", dynamic_panel(), ["xtunitroot ips y", "xtunitroot llc y"]),
        ]

        for name, frame, commands in cases:
            oracle.send_data(frame)
            oracle.run("xtset firm year")
            for position, command in enumerate(commands):
                key = f"{name}_{position}"
                try:
                    output = oracle.get_output(command)
                    captured[key] = f"COMMAND: {command}\n\n{output}"
                    print(f"=== {key}: OK ({len(frame)} obs) ===")
                except Exception as exc:  # noqa: BLE001
                    captured[key] = f"COMMAND: {command}\n\nFAILED: {exc}"
                    print(f"=== {key}: FAILED {str(exc)[:120]} ===")

    path = FIXTURES / "p9_outputs.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
