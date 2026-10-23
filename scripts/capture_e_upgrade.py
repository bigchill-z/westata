"""Capture full-precision Stata references for every E1/E2 item that can be upgraded.

The nine candidates are the ones with no fundamental obstacle:

    etregress        control function, just never diffed
    xtlogit, pa      GEE, just never diffed
    xtprobit, re     quadrature, just never diffed
    xttobit          quadrature, just never diffed
    xtpoisson, re    quadrature, just never diffed
    xtnbreg, fe      quasi-likelihood, just never diffed
    vec              coefficient estimates beyond the rank test
    vecrank, lags(1) the 0.4% boundary case
    ivregress gmm    coefficients exact, robust SE 2.9e-2 apart
    arch             alpha 6.9e-3, mean constant 3.6e-2
    pperron          1.6e-2, plausibly a lag-truncation difference

Everything is captured as full-precision e(b)/e(V), not the printed table, because the
previous batches showed repeatedly that the variance is where the disagreement hides.
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


def treatreg_frame(n: int = 800, seed: int = 107) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x1 = rng.normal(size=n)
    common = rng.normal(size=n)
    u = 0.6 * common + 0.8 * rng.normal(size=n)
    e = 0.6 * common + 0.8 * rng.normal(size=n)
    treated = (0.6 * z + u) > 0
    y = 1.0 + 2.0 * x1 + 1.5 * treated.astype(float) + e
    return pd.DataFrame(
        {"y": y, "x1": x1, "z": z, "treated": treated.astype(float)}
    )


def capture_matrix(oracle: StataOracle, label: str, command: str) -> str:
    """Run quietly and return a block with names, b and V at full precision."""

    oracle.run(f"quietly {command}")
    grabbed = oracle.capture()
    lines = [f"COMMAND: {command}", f"NAMES: {list(grabbed.names)}", f"B: {list(grabbed.b)}"]
    lines.append("V:")
    for row in grabbed.V:
        lines.append(" ".join(repr(float(value)) for value in row))
    return "\n".join(lines)


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        # ---- panel binary / censored / count ----------------------------
        print("panel models")
        binary = binary_panel()
        oracle.send_data(binary)
        oracle.run("xtset firm year")
        for label, command in [
            ("xtlogit_pa", "xtlogit y x1 x2, pa"),
            ("xtlogit_re", "xtlogit y x1 x2, re"),
            ("xtprobit_re", "xtprobit y x1 x2, re"),
        ]:
            try:
                captured[label] = capture_matrix(oracle, label, command)
                print(f"  OK   {label}")
            except Exception as exc:  # noqa: BLE001
                captured[label] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"  FAIL {label}: {str(exc)[:80]}")

        censored = censored_panel()
        oracle.send_data(censored)
        oracle.run("xtset firm year")
        try:
            captured["xttobit"] = capture_matrix(oracle, "xttobit", "xttobit y x1, ll(.5)")
            print("  OK   xttobit")
        except Exception as exc:  # noqa: BLE001
            captured["xttobit"] = f"FAILED: {exc}"
            print(f"  FAIL xttobit: {str(exc)[:80]}")

        counts = count_panel()
        oracle.send_data(counts)
        oracle.run("xtset firm year")
        for label, command in [
            ("xtpoisson_re", "xtpoisson y x1, re"),
            ("xtnbreg_fe", "xtnbreg y x1, fe"),
        ]:
            try:
                captured[label] = capture_matrix(oracle, label, command)
                print(f"  OK   {label}")
            except Exception as exc:  # noqa: BLE001
                captured[label] = f"COMMAND: {command}\n\nFAILED: {exc}"
                print(f"  FAIL {label}: {str(exc)[:80]}")

        # ---- etregress -------------------------------------------------
        print("etregress")
        treat = treatreg_frame()
        oracle.send_data(treat)
        try:
            captured["etregress"] = capture_matrix(
                oracle, "etregress", "etregress y x1, treat(treated = z)"
            )
            print("  OK   etregress")
        except Exception as exc:  # noqa: BLE001
            captured["etregress"] = f"FAILED: {exc}"
            print(f"  FAIL etregress: {str(exc)[:90]}")

        # ---- time series ----------------------------------------------
        print("time series")
        rng = np.random.default_rng(41)
        n = 400
        rw1 = np.cumsum(rng.normal(size=n))
        rw2 = rw1 + rng.normal(scale=0.4, size=n)
        series = pd.DataFrame({"t": np.arange(1, n + 1), "y1": rw1, "y2": rw2})
        oracle.send_data(series)
        oracle.run("tsset t")

        try:
            captured["vec"] = capture_matrix(
                oracle, "vec", "vec y1 y2, lags(1) rank(1)"
            )
            print("  OK   vec")
        except Exception as exc:  # noqa: BLE001
            captured["vec"] = f"FAILED: {exc}"
            print(f"  FAIL vec: {str(exc)[:90]}")

        # GARCH series
        rng = np.random.default_rng(47)
        m = 800
        innovations = rng.normal(size=m)
        sigma2 = np.zeros(m)
        returns = np.zeros(m)
        sigma2[0] = 1.0
        for index in range(1, m):
            sigma2[index] = 0.05 + 0.10 * returns[index - 1] ** 2 + 0.85 * sigma2[index - 1]
            returns[index] = np.sqrt(sigma2[index]) * innovations[index]
        garch = pd.DataFrame({"t": np.arange(1, m + 1), "r": returns})
        oracle.send_data(garch)
        oracle.run("tsset t")
        try:
            captured["arch"] = capture_matrix(oracle, "arch", "arch r, arch(1) garch(1)")
            print("  OK   arch")
        except Exception as exc:  # noqa: BLE001
            captured["arch"] = f"FAILED: {exc}"
            print(f"  FAIL arch: {str(exc)[:90]}")

        try:
            captured["pperron"] = capture_matrix(oracle, "pperron", "pperron y1")
            print("  OK   pperron")
        except Exception as exc:  # noqa: BLE001
            captured["pperron"] = f"FAILED: {exc}"
            print(f"  FAIL pperron: {str(exc)[:90]}")

    path = FIXTURES / "e_upgrade_outputs.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
