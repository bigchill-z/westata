"""Capture Stata fixtures for the extended E3 checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

FIXTURE_DIR = ROOT / "tests" / "conformance" / "stata_fixtures"


def panel_frame(n_entities: int = 30, n_periods: int = 6, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def iv_frame(n: int = 500, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.8 * z1 + 0.5 * z2 + 0.6 * v
    y = 1.0 + 2.0 * x + 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "x2": x2, "z1": z1, "z2": z2})


def nbreg_frame(n: int = 600, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.3 + 0.5 * x1)
    alpha = 0.4
    lam = rng.gamma(shape=1.0 / alpha, scale=mu * alpha)
    y = rng.poisson(lam).astype(float)
    return pd.DataFrame({"y": y, "x1": x1})


def ordered_frame(n: int = 700, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 0.8 * x1 - 0.4 * x2 + rng.normal(size=n)
    y = pd.cut(latent, bins=[-np.inf, -0.7, 0.4, np.inf], labels=[1, 2, 3]).astype(float)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def var_frame(n: int = 400, seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = np.array([[0.5, 0.1], [0.2, 0.4]])
    data = np.zeros((n, 2))
    shocks = rng.normal(size=(n, 2))
    for t in range(1, n):
        data[t] = a @ data[t - 1] + shocks[t]
    return pd.DataFrame({"y1": data[:, 0], "y2": data[:, 1], "t": np.arange(1, n + 1)})


#: name -> (data, commands)
SCENARIOS: dict[str, tuple[pd.DataFrame | None, list[str]]] = {
    "xtreg_fe": (panel_frame(), ["xtset firm year", "xtreg y x1 x2, fe"]),
    "xtreg_re": (panel_frame(), ["xtset firm year", "xtreg y x1 x2, re"]),
    "xtreg_be": (panel_frame(), ["xtset firm year", "xtreg y x1 x2, be"]),
    "areg": (panel_frame(), ["areg y x1 x2, absorb(firm)"]),
    "iv_liml": (iv_frame(), ["ivregress liml y x2 (x = z1 z2)"]),
    "iv_gmm": (iv_frame(), ["ivregress gmm y x2 (x = z1 z2)"]),
    "nbreg": (nbreg_frame(), ["nbreg y x1"]),
    "ologit": (ordered_frame(), ["ologit y x1 x2"]),
    "oprobit": (ordered_frame(), ["oprobit y x1 x2"]),
    "var": (var_frame(), ["tsset t", "var y1 y2, lags(1)"]),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=sorted(SCENARIOS))
    args = parser.parse_args(argv)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    selected = args.only or sorted(SCENARIOS)

    with StataOracle() as oracle:
        for name in selected:
            data, commands = SCENARIOS[name]
            oracle.send_data(data)
            for command in commands:
                oracle.run(command)
            capture = oracle.capture()
            path = capture.save(FIXTURE_DIR / f"{name}.json")
            print(f"{name:12s} -> {path.name}  b={capture.b.size}  names={capture.names}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
