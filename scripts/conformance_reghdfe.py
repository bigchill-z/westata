"""Compare FEOLS with a real reghdfe (installed from GitHub, since SSC is blocked)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import westata as ws  # noqa: E402
from westata.conformance.oracles.stata import StataOracle  # noqa: E402


def rel(actual, expected) -> float:
    a = np.asarray(actual, dtype=float).reshape(-1)
    e = np.asarray(expected, dtype=float).reshape(-1)
    return float(np.max(np.abs(a - e) / np.maximum(np.abs(e), 1e-12)))


def panel_frame(n_entities: int = 30, n_periods: int = 6, seed: int = 21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    entity = np.repeat(np.arange(n_entities), n_periods)
    time = np.tile(np.arange(1, n_periods + 1), n_entities)
    n = n_entities * n_periods
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(size=n_entities)[entity]
    # A period effect as well, so the two-way case is meaningful.
    period = rng.normal(size=n_periods)[time - 1]
    y = 1.0 + 1.5 * x1 - 0.5 * x2 + effect + period + rng.normal(scale=0.4, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})


def align(result, stata_names):
    """Return (positions in westata, positions in stata) for shared parameters.

    reghdfe reports an intercept (``_cons``) that westata's FEOLS omits when the
    model absorbs fixed effects, so the comparison is restricted to the
    coefficients both estimate and the dropped ones are reported.
    """

    westata_positions = []
    stata_positions = []
    skipped = []
    for index, name in enumerate(stata_names):
        key = "Intercept" if name == "_cons" else name
        if key in result.param_names:
            westata_positions.append(list(result.param_names).index(key))
            stata_positions.append(index)
        else:
            skipped.append(name)
    return (
        np.asarray(westata_positions, dtype=int),
        np.asarray(stata_positions, dtype=int),
        skipped,
    )


def main() -> int:
    frame = panel_frame()
    rows = []

    with StataOracle() as oracle:
        oracle.send_data(frame)

        cases = [
            (
                "reghdfe, absorb(firm)",
                "y ~ x1 + x2 | firm",
                "reghdfe y x1 x2, absorb(firm)",
            ),
            (
                "reghdfe, absorb(firm year)",
                "y ~ x1 + x2 | firm + year",
                "reghdfe y x1 x2, absorb(firm year)",
            ),
            (
                "reghdfe, absorb(firm) vce(robust)",
                "y ~ x1 + x2 | firm",
                "reghdfe y x1 x2, absorb(firm) vce(robust)",
            ),
            (
                "reghdfe, absorb(firm) cluster(firm)",
                "y ~ x1 + x2 | firm",
                "reghdfe y x1 x2, absorb(firm) cluster(firm)",
            ),
            (
                "reghdfe, absorb(firm year) cluster(firm)",
                "y ~ x1 + x2 | firm + year",
                "reghdfe y x1 x2, absorb(firm year) cluster(firm)",
            ),
        ]

        for label, formula, command in cases:
            oracle.run("discard")
            oracle.run(command)
            capture = oracle.capture()

            cov_type = "nonrobust"
            cov_kwds = None
            if "vce(robust)" in command:
                cov_type = "HC1"
            elif "cluster(firm)" in command:
                cov_type = "cluster"
                cov_kwds = {"cluster": frame["firm"]}

            result = ws.FEOLS.from_formula(formula, frame).fit(
                cov_type=cov_type, cov_kwds=cov_kwds
            )
            westata_positions, stata_positions, skipped = align(result, capture.names)

            parameter_gap = rel(
                result.params.to_numpy()[westata_positions],
                np.asarray(capture.b, dtype=float)[stata_positions],
            )
            se_gap = rel(
                result.bse.to_numpy()[westata_positions],
                np.sqrt(np.diag(capture.V))[stata_positions],
            )
            df_gap = (
                abs(result.e["df_r"] - capture.scalars.get("df_r", np.nan))
                if np.isfinite(capture.scalars.get("df_r", np.nan))
                else float("nan")
            )
            r2_gap = (
                abs(result.e["r2"] - capture.scalars.get("r2", np.nan))
                if np.isfinite(capture.scalars.get("r2", np.nan))
                else float("nan")
            )

            print(f"\n=== {label} ===")
            print(f"  Stata names   : {capture.names}")
            print(f"  westata names : {list(result.param_names)}")
            if skipped:
                print(f"  not compared  : {skipped} (absorbed-model intercept)")
            print(f"  parameters    : {parameter_gap:.3e}")
            print(f"  std. errors   : {se_gap:.3e}")
            print(f"  e(df_r) delta : {df_gap:g}")
            print(f"  e(r2) delta   : {r2_gap:g}")
            print(f"  Stata e(N)={capture.scalars.get('N')} e(df_r)={capture.scalars.get('df_r')} "
                  f"e(r2)={capture.scalars.get('r2')}")
            print(f"  westata e(N)={result.e['N']} e(df_r)={result.e['df_r']} "
                  f"e(r2)={result.e['r2']}")

            rows.append(
                {
                    "case": label,
                    "param_rel": parameter_gap,
                    "se_rel": se_gap,
                    "df_r_delta": df_gap,
                    "r2_delta": r2_gap,
                }
            )

    table = pd.DataFrame(rows)
    print()
    print(table.to_string(index=False))
    table.to_csv(ROOT / "reghdfe_conformance.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
