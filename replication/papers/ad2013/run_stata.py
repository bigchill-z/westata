"""Estimate the Autor & Dorn specifications with StataNow through ``pystata``.

The authors' do-file uses ``xi i.statefip`` to generate ``_Istatefip_*`` columns and then
regresses on them.  ``i.statefip`` produces the same design matrix and is what this driver
writes; ``gen_do.py`` renders the same commands into a runnable ``.do`` file so a reader
can reproduce the tables in the Stata GUI.

Reference values come from the authors' own ``log/czone_analysis.log``, which ships in the
archive.  That log is full precision (seven decimals), so agreement is checked tightly
rather than against three-decimal table entries.
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

from specs import SPECS, Spec  # noqa: E402
from stata_probe import Capture, Stata  # noqa: E402

OUT = HERE / "out"


def _capture(stata: Stata, command: str) -> Capture:
    cap = stata.estimate(command)
    for key in ("N", "r2", "r2_a", "rmse", "df_r", "df_m", "F"):
        value = stata.scalar(f"e({key})")
        if value is not None:
            cap.scalars[key] = float(value)
    cap.nobs = int(cap.scalars.get("N", cap.nobs))
    return cap


def _serialise(cap: Capture) -> dict:
    return {
        "command": cap.command,
        "names": list(cap.names),
        "b": [float(x) for x in np.asarray(cap.b, dtype=float)],
        "se": [float(x) for x in np.sqrt(np.diag(np.asarray(cap.V, dtype=float)))],
        "scalars": {k: float(v) for k, v in cap.scalars.items()},
        "nobs": int(cap.nobs),
        "stata_version": cap.version,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from acquire import workfile

    payload: dict = {"specs": {}, "meta": {}}
    with Stata() as stata:
        payload["meta"]["stata_version"] = stata.version
        payload["meta"]["edition"] = "mp"
        stata.run(f'use "{workfile().as_posix()}", clear')
        payload["meta"]["nobs_in_memory"] = stata.nobs()

        for spec in SPECS:
            cap = _capture(stata, spec.stata_command())
            payload["specs"][spec.id] = _serialise(cap)
            print(f"[stata] {spec.id}: N={cap.nobs}  {spec.table}{spec.row}", flush=True)

    (OUT / "stata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'stata_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
