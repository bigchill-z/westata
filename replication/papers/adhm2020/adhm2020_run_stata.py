"""Estimate the ADHM (2020) Table 3 specifications in StataNow.

Three estimators per column, because they answer different questions
--------------------------------------------------------------------
1. **``ivreg2`` with two-way clustering** -- what the authors ran.  This produces the paper's
   standard errors and is the reference for everything else.
2. **``ivregress 2sls`` with ``cluster(czone)``** -- stock Stata.  Same coefficient, one-way
   standard error.
3. **``ivregress 2sls`` with ``cluster(congressionaldistrict)``** -- the other dimension.

Why all three
-------------
Stock ``ivregress`` cannot do multiway clustering::

    . ivregress 2sls ..., vce(cluster czone cdist)
    multiway cluster VCE not allowed

So the paper's number is only reachable through ``ivreg2``.  Capturing the single-way
estimates as well is what lets the report quantify the cost of *not* having multiway
support -- the single-way standard errors sit 1.3 to 11 percent below the two-way ones,
which changes whether the estimate looks significant.

An earlier version of this driver skipped ``ivreg2`` to keep the run offline.  That left the
paper's own standard errors unreproduced, which is precisely the quantity this replication
exists to check, so the dependency is now installed explicitly by
``adhm2020_install.py``.
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

from adhm2020_specs import CLUSTERS, ENDOG, INSTRUMENT, SPECS, WEIGHT, Spec  # noqa: E402
from stata_probe import Capture, Stata  # noqa: E402

OUT = HERE / "out"


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the numeric district id that ``ivreg2`` needs inside ``cluster()``.

    ``congressionaldistrict`` arrives as a string (values like ``NY-1``), and Stata's
    ``cluster()`` only accepts numeric variables.  Encoding to a dense integer preserves the
    grouping exactly -- clustering depends on which rows share a value, never on the value.
    """

    frame = frame.copy()
    frame["cdist"] = frame["congressionaldistrict"].astype("category").cat.codes + 1
    return frame


def command_ivreg2(spec: Spec) -> str:
    """The authors' command: ``ivreg2`` with a two-way cluster."""

    controls = " ".join(spec.regressors)
    parts = [
        "ivreg2", spec.depvar, f"({ENDOG}={INSTRUMENT})", controls,
        f"[aw={WEIGHT}]", f", cluster({' '.join(CLUSTERS)})",
    ]
    return " ".join(part for part in parts if part)


def command_ivregress(spec: Spec, cluster: str) -> str:
    """Stock Stata, one cluster dimension at a time."""

    controls = " ".join(spec.regressors)
    cluster_var = "cdist" if cluster == "congressionaldistrict" else cluster
    parts = [
        "ivregress 2sls", spec.depvar, f"({ENDOG}={INSTRUMENT})", controls,
        f"[aw={WEIGHT}]", f", cluster({cluster_var})",
    ]
    return " ".join(part for part in parts if part)


def _capture(stata: Stata, cmd: str) -> Capture:
    cap = stata.estimate(cmd)
    for key in ("N", "r2", "rmse", "chi2", "df_m", "F"):
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
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import adhm2020_install
    from adhm2020_acquire import workfile

    frame = pd.read_stata(workfile(), convert_categoricals=False)
    payload: dict = {"ivreg2": {}, "ivregress": {}, "meta": {}}

    with Stata() as stata:
        payload["meta"]["stata_version"] = stata.version
        payload["meta"]["packages"] = adhm2020_install.ensure(stata)
        if any("MISSING" in note for note in payload["meta"]["packages"].values()):
            raise RuntimeError(
                "ivreg2 is required for the two-way clustered estimates and is not "
                f"available: {payload['meta']['packages']}.  Without it the paper's "
                "standard errors cannot be reproduced at all."
            )

        stata.send(_prepare(frame))
        for spec in SPECS:
            cap = _capture(stata, command_ivreg2(spec))
            payload["ivreg2"][spec.id] = _serialise(cap)
            print(f"[ivreg2 2-way]    {spec.id}: N={cap.nobs}", flush=True)

            payload["ivregress"][spec.id] = {}
            for cluster in CLUSTERS:
                alt = _capture(stata, command_ivregress(spec, cluster))
                payload["ivregress"][spec.id][cluster] = _serialise(alt)
                print(f"[ivregress {cluster[:9]:9s}] {spec.id}: N={alt.nobs}", flush=True)

    (OUT / "stata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'stata_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
