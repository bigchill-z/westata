"""Capture Stata references for the P8 quantile/robust regression commands.

Covers qreg, sqreg, bsqreg, rreg and their postestimation shapes. Run with
``.venv\\Scripts\\python.exe scripts\\capture_p8_fixtures.py``.

Why a dedicated script: quantile regression's covariance matrix depends on the
sparsity estimate ``f(0)``, and Stata computes it from a kernel density of the
residuals. Getting that wrong shifts every standard error without touching a
single coefficient, which is exactly the class of bug the fixtures exist to
catch.
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


def quantile_frame(n: int = 600, seed: int = 61) -> pd.DataFrame:
    """Heteroskedastic DGP so different quantiles have different slopes."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scale = 0.5 + 0.4 * np.abs(x1)
    y = 1.0 + 2.0 * x1 - 0.8 * x2 + scale * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def outlier_frame(n: int = 400, seed: int = 67) -> pd.DataFrame:
    """A few gross outliers, so rreg and qreg diverge from OLS visibly."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    # Contaminate 5% of the sample.
    kontam = rng.random(size=n) < 0.05
    y = np.where(kontam, y + rng.normal(scale=30.0, size=n), y)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "kontam": kontam.astype(float)})


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    captured: dict[str, str] = {}
    values: list[dict[str, object]] = []

    with StataOracle() as oracle:
        frame = quantile_frame()
        oracle.send_data(frame)

        # ---- qreg at several quantiles ----
        for q in (0.25, 0.5, 0.75, 0.9):
            command = f"qreg y x1 x2, quantile({q})"
            output = oracle.get_output(command)
            captured[f"qreg_{q}"] = output
            print(f"=== qreg q={q} ===\n{output}")

        # ---- sqreg: joint VCE across quantiles ----
        captured["sqreg"] = oracle.get_output(
            "sqreg y x1 x2, quantiles(.25 .5 .75) reps(200)"
        )
        print("=== sqreg ===\n" + captured["sqreg"])

        # ---- bsqreg ----
        captured["bsqreg"] = oracle.get_output(
            "bsqreg y x1 x2, quantile(.5) reps(200)"
        )
        print("=== bsqreg ===\n" + captured["bsqreg"])

        # ---- rreg ----
        outliers = outlier_frame()
        oracle.send_data(outliers)
        captured["rreg"] = oracle.get_output("rreg y x1 x2")
        print("=== rreg ===\n" + captured["rreg"])

        # ---- predict types available for qreg ----
        oracle.send_data(frame)
        oracle.run("qreg y x1 x2, quantile(.5)")
        for kind in ("xb", "residuals", "stdp", "stdf"):
            try:
                oracle.run(f"cap drop __t_{kind}")
                oracle.run(f"predict double __t_{kind}, {kind}")
                values.append({"check": "qreg_predict", "kind": kind, "status": "OK"})
                print(f"qreg predict {kind}: OK")
            except Exception as exc:  # noqa: BLE001
                values.append(
                    {"check": "qreg_predict", "kind": kind, "status": "REFUSED"}
                )
                print(f"qreg predict {kind}: REFUSED ({str(exc)[:60]})")

        # ---- does qreg support robust VCE? ----
        try:
            captured["qreg_robust"] = oracle.get_output("qreg y x1 x2, quantile(.5) vce(robust)")
            print("=== qreg vce(robust) accepted ===")
        except Exception as exc:  # noqa: BLE001
            captured["qreg_robust"] = f"REFUSED: {exc}"
            print(f"qreg vce(robust): REFUSED ({str(exc)[:80]})")

    text_path = FIXTURES / "p8_outputs.txt"
    with text_path.open("w", encoding="utf-8") as handle:
        for name, output in captured.items():
            handle.write(f"=== {name} ===\n{output}\n\n")
    print(f"\nwrote {text_path} ({text_path.stat().st_size} bytes)")

    matrix_path = FIXTURES / "p8_predict_types.csv"
    pd.DataFrame(values).to_csv(matrix_path, index=False)
    print(f"wrote {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
