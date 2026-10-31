"""Run the paper's Stata code through ``pystata`` and capture full-precision results.

This is the "Stata side" of the replication.  It is not a re-reading of the paper's
do-file output: it starts a live StataNow 19.5 (MP) session and re-estimates every
column, capturing ``e(b)``/``e(V)`` at full precision rather than the three decimals the
paper prints.

Two deliberate choices:

* **The do-file is generated, not hand-written.**  ``gen_do.py`` renders the same
  :mod:`specs` into a ``.do`` file a human can run in the Stata GUI.  Generating both from
  one description is what stops the "artefact a reader runs" and "the numbers in the
  report" from being two different experiments.
* **The first stage is run as its own ``regress``.**  ``ivregress ..., first`` prints the
  first stage but leaves the structural equation in ``e(b)``, so it cannot be captured
  through the API.  Running ``regress`` gives the same coefficient -- and reveals a
  version difference in the *standard error*, which the report treats as a finding
  rather than smoothing over (see ``REPORT.md``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "retest" / "lib"))

import numpy as np  # noqa: E402

from specs import SPECS, Spec  # noqa: E402
from stata_probe import Capture, Stata  # noqa: E402

OUT = HERE / "out"


def capture_of(stata: Stata, command: str) -> Capture:
    """Estimate, then attach the scalars the report quotes.

    ``Capture.nobs`` from ``stata_probe`` is the number of rows *in memory*, which is
    not the number of rows *in the estimation* -- Table 2's columns 1 and 2 restrict to
    one period and report 722 observations out of 1,444 rows.  ``e(N)`` is the estimate's
    own count, so that is what gets recorded and compared.
    """

    cap = stata.estimate(command)
    for key in ("N", "r2", "r2_a", "rmse", "df_r", "F", "chi2", "df_m", "ll"):
        value = stata.scalar(f"e({key})")
        if value is not None:
            cap.scalars[key] = float(value)
    cap.nobs = int(cap.scalars.get("N", cap.nobs))
    return cap


def serialise(cap: Capture) -> dict:
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
    from data import workfile

    payload: dict[str, dict] = {"specs": {}, "first_stage": {}, "meta": {}}
    with Stata() as stata:
        payload["meta"]["stata_version"] = stata.version
        payload["meta"]["edition"] = "mp"
        stata.run(f'use "{workfile().as_posix()}", clear')
        payload["meta"]["nobs_in_memory"] = stata.nobs()

        for spec in SPECS:
            cap = capture_of(stata, spec.stata_command())
            payload["specs"][spec.id] = serialise(cap)
            print(f"[stata] {spec.id}: N={cap.nobs} {spec.stata_command()[:70]}...", flush=True)

            if spec.endog:
                fs = capture_of(stata, spec.first_stage_command())
                payload["first_stage"][spec.id] = serialise(fs)
                # The same first stage with a heteroskedasticity-robust rather than
                # clustered covariance.  It is captured because the published panel II
                # standard errors match this one, not the clustered one -- a fact the
                # report has to state and cannot state without the number.
                robust = spec.first_stage_command().replace(
                    f", cluster({spec.cluster})", ", vce(robust)"
                )
                payload["first_stage"][f"{spec.id}_robust"] = serialise(
                    capture_of(stata, robust)
                )

    target = OUT / "stata_results.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def _unused(spec: Spec) -> None:  # pragma: no cover - keeps the import honest
    _ = spec
