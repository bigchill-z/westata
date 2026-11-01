"""Estimate the AJR (2001) specifications with StataNow: the self-consistency baseline.

This driver is the **reference** for the replication, which is unusual here.  Every other
paper in this repository had a log or a table to check against; this package has neither, so
the reference is Stata itself running the authors' own commands.  The report states that
plainly rather than implying a table match.

Beyond the point estimates this captures the joint test the authors report alongside each
column (``test fpresence1850 fpresence1875 fpresence1900``) and the first-stage-free
degrees-of-freedom fields, because the replication's claim is that westata reproduces the
*full* reported object, not only the coefficient column.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "retest" / "lib"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ajr2001_specs import FPRESENCE, SPECS, Spec  # noqa: E402
from stata_probe import Capture, Stata  # noqa: E402

OUT = HERE / "out"

#: The joint test each column reports: the three post-1850 terms are jointly zero?
JOINT_TERMS = ("fpresence1850", "fpresence1875", "fpresence1900")


def sample(frame: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """Restrict to the column's sample before estimation.

    Doing this in pandas rather than via a Stata ``if`` keeps both sides on identical rows:
    the Stata command still carries the ``if`` (so the reported output is the authors'), and
    the frame handed to westata is already restricted to the same observations.
    """

    frame = frame[frame["year"].isin([1700, 1750, 1800, 1850, 1875, 1900])].copy()
    if spec.west_only:
        frame = frame[frame["westelbe"] == 1]
    if spec.drop_ids:
        frame = frame[~frame["id"].isin(spec.drop_ids)]
    needed = ["urbrate", *spec.regressors]
    if spec.weighted:
        needed.append("totalpop1750")
    return frame.dropna(subset=needed).reset_index(drop=True)


def _capture(stata: Stata, command: str) -> Capture:
    cap = stata.estimate(command)
    for key in ("N", "N_g", "df_a", "df_r", "r2", "r2_w", "rmse", "F"):
        value = stata.scalar(f"e({key})")
        if value is not None:
            cap.scalars[key] = float(value)
    cap.nobs = int(cap.scalars.get("N", cap.nobs))

    # The authors' joint test, captured the same way their do-file does.
    stata.quiet("test " + " ".join(JOINT_TERMS))
    cap.scalars["joint_F"] = float(stata.scalar("r(F)") or float("nan"))
    cap.scalars["joint_p"] = float(stata.scalar("r(p)") or float("nan"))
    return cap


def _serialise(cap: Capture) -> dict:
    return {
        "command": cap.command,
        "names": list(cap.names),
        "b": [float(x) for x in np.asarray(cap.b, dtype=float)],
        "se": [float(x) for x in np.sqrt(np.diag(np.asarray(cap.V, dtype=float)))],
        "scalars": {k: float(v) for k, v in cap.scalars.items()},
        "nobs": int(cap.nobs),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from ajr2001_acquire import workfile

    raw = pd.read_stata(workfile(), convert_categoricals=False)
    payload: dict = {"specs": {}, "meta": {}}

    with Stata() as stata:
        payload["meta"]["stata_version"] = stata.version
        payload["meta"]["role"] = "self-consistency baseline (no published table available)"
        # One frame for the whole run; each spec's Stata command carries its own `if`, and
        # `run_westata.py` applies the identical restriction in pandas.
        stata.send(raw[raw["year"].isin([1700, 1750, 1800, 1850, 1875, 1900])])
        stata.quiet("xtset id year")

        for spec in SPECS:
            cap = _capture(stata, spec.stata_command())
            payload["specs"][spec.id] = _serialise(cap)
            print(f"[stata] {spec.id}: N={cap.nobs}  {spec.table}{spec.column}", flush=True)

    (OUT / "stata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'stata_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
