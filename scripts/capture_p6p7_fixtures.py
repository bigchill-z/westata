"""Fix the two failing P6/P7 Stata captures and re-run all of them."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

FIXTURES = ROOT / "tests" / "conformance" / "stata_fixtures"


def truncated_frame(n: int = 800, seed: int = 31) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 1.0 + 1.5 * x1 - 0.5 * x2 + rng.normal(scale=0.8, size=n)
    keep = latent > 0.0
    return pd.DataFrame({"y": latent[keep], "x1": x1[keep], "x2": x2[keep]})


def interval_frame(n: int = 600, seed: int = 37) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    latent = 2.0 + 1.2 * x1 + rng.normal(scale=0.9, size=n)
    edges = np.array([-np.inf, 0.5, 1.5, 2.5, 3.5, np.inf])
    index = np.searchsorted(edges, latent, side="right") - 1
    return pd.DataFrame({"lo": edges[index], "hi": edges[index + 1], "x1": x1})


def selection_frame(n: int = 1500, seed: int = 41) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = rng.normal(size=n)
    common = rng.normal(size=n)
    u = 0.6 * common + 0.8 * rng.normal(size=n)
    e = 0.6 * common + 0.8 * rng.normal(size=n)
    selected = (0.8 * z + u) > 0
    outcome = 1.0 + 2.0 * x + e
    return pd.DataFrame(
        {
            "y": np.where(selected, outcome, np.nan),
            "x": x,
            "z": z,
            "selected": selected.astype(float),
        }
    )


def count_frame(n: int = 1200, seed: int = 43) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.4 + 0.6 * x1)
    structural_zero = rng.random(size=n) < 0.25
    counts = np.where(structural_zero, 0, rng.poisson(mu)).astype(float)
    return pd.DataFrame({"y": counts, "x1": x1})


def overdispersed_frame(n: int = 1200, seed: int = 47) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    mu = np.exp(0.4 + 0.6 * x1)
    alpha = 0.5
    lam = rng.gamma(shape=1.0 / alpha, scale=mu * alpha)
    return pd.DataFrame({"y": rng.poisson(lam).astype(float), "x1": x1})


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    captured: dict[str, str] = {}

    with StataOracle() as oracle:
        cases = [
            ("truncreg", truncated_frame, "truncreg y x1 x2, ll(0)"),
            ("intreg", interval_frame, "intreg lo hi x1"),
            ("heckman", selection_frame, "heckman y x, select(selected = z)"),
            # zinb needs `inflate()` on a model that converges; the earlier
            # failure came from an over-specified inflation equation.
            ("zip", count_frame, "zip y x1, inflate(x1)"),
            ("zinb", overdispersed_frame, "zinb y x1, inflate(_cons)"),
            (
                "ztp",
                lambda: count_frame().query("y > 0").reset_index(drop=True),
                "ztp y x1",
            ),
            # ll(0) not ll(1): Stata requires the limit to be below the minimum y,
            # and the positive-count sample starts at 1.
            (
                "tnbreg",
                lambda: overdispersed_frame().query("y > 0").reset_index(drop=True),
                "tnbreg y x1, ll(0)",
            ),
        ]

        for name, builder, command in cases:
            frame = builder()
            try:
                oracle.send_data(frame)
                output = oracle.get_output(command)
                captured[name] = output
                print(f"=== {name}: OK ({len(frame)} obs, {len(output)} chars) ===")
            except Exception as exc:  # noqa: BLE001
                captured[name] = f"FAILED: {exc}"
                print(f"=== {name}: FAILED {str(exc)[:160]} ===")

    path = FIXTURES / "p6p7_outputs.txt"
    with path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {path} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
