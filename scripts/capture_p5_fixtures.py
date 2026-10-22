"""Capture Stata references for the P5 postestimation additions.

Covers nlcom, testnl, lrtest, hausman, contrast, pwcompare and predict stdp/stdf.
Run with: .venv\\Scripts\\python.exe scripts\\capture_p5_fixtures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from westata.conformance.oracles.stata import StataOracle  # noqa: E402

FIXTURES = ROOT / "tests" / "conformance" / "stata_fixtures"


def regression_frame(n: int = 200, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 1.0 * x2 + 0.5 * x3 + rng.normal(scale=0.6, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})


def panel_frame(n_entities: int = 40, n_periods: int = 6, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def factor_frame(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 3, size=n)
    x = rng.normal(size=n)
    # Three distinct group effects so the contrasts are meaningfully different.
    effect = np.array([0.0, 0.8, -0.5])[group]
    y = 1.0 + 2.0 * x + effect + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"y": y, "x": x, "g": group})
    df["g_label"] = df["g"].map({0: "a", 1: "b", 2: "c"})
    return df


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    with StataOracle() as oracle:
        # ---- nlcom / testnl ----
        frame = regression_frame()
        oracle.send_data(frame)
        oracle.run("regress y x1 x2 x3")
        for expression, label in [
            ("exp(_b[x1])", "exp_x1"),
            ("_b[x1] / _b[x2]", "ratio"),
            ("_b[x1] + _b[x2]", "sum"),
        ]:
            output = oracle.get_output(f"nlcom {expression}")
            rows.append({"check": "nlcom", "label": label, "output": output})
            print(f"nlcom {expression}:\n{output}")

        for expression, label in [
            ("exp(_b[x1]) = 2", "exp_eq_2"),
            ("_b[x1] = _b[x2]", "x1_eq_x2"),
        ]:
            output = oracle.get_output(f"testnl {expression}")
            rows.append({"check": "testnl", "label": label, "output": output})
            print(f"testnl {expression}:\n{output}")

        # ---- lrtest ----
        oracle.run("regress y x1", quietly=False)
        oracle.run("estimates store small")
        oracle.run("regress y x1 x2 x3", quietly=False)
        oracle.run("estimates store big")
        output = oracle.get_output("lrtest small big")
        rows.append({"check": "lrtest", "label": "small_vs_big", "output": output})
        print(f"lrtest:\n{output}")

        # ---- hausman ----
        panel = panel_frame()
        oracle.send_data(panel)
        oracle.run("xtset firm year")
        oracle.run("xtreg y x1 x2, fe")
        oracle.run("estimates store fe")
        oracle.run("xtreg y x1 x2, re")
        oracle.run("estimates store re")
        output = oracle.get_output("hausman fe re")
        rows.append({"check": "hausman", "label": "fe_vs_re", "output": output})
        print(f"hausman:\n{output}")

        # ---- contrast / pwcompare ----
        factor = factor_frame()
        oracle.send_data(factor)
        oracle.run("regress y x i.g")
        for method, label in [("r.", "reference"), ("g.", "grand"), ("a.", "adjacent")]:
            output = oracle.get_output(f"contrast {method}g")
            rows.append({"check": "contrast", "label": label, "output": output})
            print(f"contrast {method}g:\n{output}")

        output = oracle.get_output("pwcompare g")
        rows.append({"check": "pwcompare", "label": "all_pairs", "output": output})
        print(f"pwcompare:\n{output}")

        # ---- predict stdp / stdf ----
        oracle.run("sysuse auto, clear")
        oracle.run("regress mpg weight foreign")
        oracle.run("predict double stdp_v, stdp")
        oracle.run("predict double stdf_v, stdf")
        path = ROOT / "_p5_predict.csv"
        if path.exists():
            path.unlink()
        oracle.run(f'export delimited stdp_v stdf_v using "{path}", replace')
        values = pd.read_csv(path)
        path.unlink()
        values.to_csv(FIXTURES / "predict_stdp_stdf.csv", index=False)
        print(f"predict stdp/stdf captured: {values.shape}")

    # Persist the text outputs for reference; the structured values are parsed
    # in the tests.
    text_path = FIXTURES / "p5_postestimation_outputs.txt"
    with text_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f"=== {row['check']} :: {row['label']} ===\n")
            handle.write(str(row["output"]))
            handle.write("\n\n")
    print(f"wrote {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
