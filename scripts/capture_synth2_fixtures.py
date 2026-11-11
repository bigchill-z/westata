"""Capture live StataNow 19.5 ``synth2`` e() results as conformance fixtures.

Run with the project venv (needs a Stata licence + ``synth2`` installed in the
user's PLUS directory)::

    .venv/Scripts/python.exe scripts/capture_synth2_fixtures.py

Writes one JSON per scenario into ``tests/conformance/stata_fixtures/``.
The JSON files are committed so the Python test suite needs no licence.

Shape contract: the repo-wide fixture guards require a top-level ``cases``
mapping whose keys contain ``|``, plus ``ado`` / ``stata_version`` / ``dataset``.
Each case stores every e() matrix at FULL precision (``sfi.Matrix.get``) with
row/column names, every e() scalar (``sfi.Scalar.getValue``), and the
row-filtered sample description.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "conformance" / "stata_fixtures"
DATA = Path(r"D:\myCodeProject\test_westata\data\synth_smoking.dta")

# California / ADH 2010 specification from synth2.sthlp:216-231.
FULL_PRED = ("cigsale lnincome age15to24 beer retprice "
             "cigsale(1988) cigsale(1980) cigsale(1975)")
# In-time placebo must drop cigsale(1988) and use an earlier xperiod
# (synth2.sthlp:223-224).
SHORT_PRED = ("cigsale lnincome age15to24 beer retprice "
              "cigsale(1980) cigsale(1975)")

SCENARIOS: list[tuple[str, str]] = [
    (
        "main",
        f"synth2 {FULL_PRED}, trunit(3) trperiod(1989) "
        "xperiod(1980(1)1988) nofigure",
    ),
    (
        "placebo_unit",
        f"synth2 {FULL_PRED}, trunit(3) trperiod(1989) "
        "xperiod(1980(1)1988) nofigure placebo(unit)",
    ),
    (
        "placebo_unit_cut2",
        f"synth2 {FULL_PRED}, trunit(3) trperiod(1989) "
        "xperiod(1980(1)1988) nofigure placebo(unit cut(2))",
    ),
    (
        "placebo_period_1985",
        f"synth2 {SHORT_PRED}, trunit(3) trperiod(1989) "
        "xperiod(1980(1)1984) nofigure placebo(period(1985))",
    ),
    (
        "loo",
        f"synth2 {FULL_PRED}, trunit(3) trperiod(1989) "
        "xperiod(1980(1)1988) nofigure loo frame(loofr)",
    ),
]

# LOO's numeric output lives in FRAME variables, not e().  These are the four
# min/max envelope columns created by synth2.ado:286-293 (symbol=1 -> "·").
LOO_ENV_COLS = (
    "pred·cigsale·loomin", "pred·cigsale·loomax",
    "tr·cigsale·loomin", "tr·cigsale·loomax",
)

SCALARS = ("N", "T", "T0", "T1", "K", "J", "rmse", "r2", "att")
MATRICES = ("bal", "U_wt", "V_wt", "mspe", "pval")


def main() -> int:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.path.append(r"D:\Program Files\StataNow19\utilities")
    from pystata import config

    config.init("mp")
    from pystata import stata
    import sfi

    if not DATA.exists():
        print(f"FATAL: {DATA} not found")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    stata.run(f'use "{DATA.as_posix()}", clear', echo=False)
    stata.run("xtset state year", echo=False)
    stata.run("set linesize 255", echo=False)
    version = "19.5"
    try:
        stata.run('mata: st_strscalar("dsh_ver", c(stata_version))', echo=False)
        version = sfi.Scalar.getString("dsh_ver")
    except Exception:
        version = "19.5"

    failures = 0
    for name, cmd in SCENARIOS:
        print(f"\n=== {name} ===", flush=True)
        # Re-load the data for every scenario: synth2/placebo/loo mutate it.
        stata.run(f'use "{DATA.as_posix()}", clear', echo=False)
        stata.run("xtset state year", echo=False)
        stata.run("set linesize 255", echo=False)
        stata.run("capture " + cmd, echo=False)
        # ``_rc`` is a Stata macro-like scalar, not a Mata c() item.
        stata.run("scalar dsh_rc = _rc", echo=False)
        rc = int(float(sfi.Scalar.getValue("dsh_rc")))
        if rc != 0:
            print(f"  FAILED rc={rc} cmd={cmd}", flush=True)
            failures += 1
            continue

        case: dict = {
            "id": f"synth2|{name}",
            "command": cmd,
            "dataset": DATA.as_posix(),
            "stata_version": version,
            "stata_release": "19.5",
            "ado": "synth2.ado v2.1.0 (Guanpeng Yan & Qiang Chen 05/10/2023)",
            "ok": True,
            "scalars": {},
            "matrices": {},
            "matrix_names": {},
        }
        for sc in SCALARS:
            try:
                case["scalars"][sc] = float(sfi.Scalar.getValue(f"e({sc})"))
            except Exception as exc:  # pragma: no cover
                case["scalars"][sc] = None
                print(f"  scalar e({sc}) missing: {exc}", flush=True)
        for mat in MATRICES:
            key = f"e({mat})"
            try:
                values = sfi.Matrix.get(key)
                # Stata's ``capture ereturn matrix`` stores an absent matrix as
                # a 1x1 MISSING sentinel (8.98846567431158e307), not a real pval.
                arr = np.asarray(values, dtype=float)
                if arr.size == 1 and arr.ravel()[0] > 1e300:
                    # Stata missing sentinel -> treat as absent.
                    continue
                case["matrices"][mat] = values
                case["matrix_names"][mat] = {
                    "rows": sfi.Matrix.getRowNames(key),
                    "cols": sfi.Matrix.getColNames(key),
                }
            except Exception:
                # e(mspe)/e(pval) legitimately absent without in-space placebo.
                pass

        # LOO: save the named frame and record the min/max envelope columns.
        if name == "loo":
            import tempfile
            frame_path = Path(tempfile.gettempdir()) / "dsh_synth2_loofr.dta"
            stata.run(
                f'capture frame loofr: save "{frame_path.as_posix()}", replace',
                echo=False,
            )
            try:
                import pandas as pd
                fr = pd.read_stata(frame_path)
                envs = {}
                for col in LOO_ENV_COLS:
                    if col in fr.columns:
                        envs[col] = [float(x) if np.isfinite(x) else None
                                     for x in fr[col].to_numpy()]
                case["loo_envelopes"] = envs
                case["loo_frame_columns"] = list(fr.columns)
                case["loo_n_runs"] = sum(
                    1 for c in fr.columns if "·rmv" in c and "pred" in c
                )
            except Exception as exc:  # pragma: no cover
                case["loo_envelopes_error"] = str(exc)

        path = OUT / f"synth2_{name}.json"
        payload = {
            "_comment": (
                "Full-precision synth2 e() capture from live StataNow 19.5 MP. "
                "Evidence artifact only; asserts no conformance level."
            ),
            "ado": case["ado"],
            "stata_version": version,
            "stata_release": "19.5",
            "dataset": DATA.as_posix(),
            "captured": "2026-11-11",
            "cases": {f"synth2|{name}": case},
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        s = case["scalars"]
        print(f"  rmse={s.get('rmse')} r2={s.get('r2')} att={s.get('att')}",
              flush=True)
        print(f"  matrices: " + ", ".join(
            f"{k}={len(v)}x{len(v[0]) if v else 0}"
            for k, v in case["matrices"].items()
        ), flush=True)
        print(f"  wrote {path.name} ({path.stat().st_size} bytes)", flush=True)

    print(f"\nDONE: {len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios "
          f"captured, {failures} failure(s)", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
