"""Estimate the ADHM (2020) specifications with ``westata``.

What this driver is really measuring
------------------------------------
The coefficients are not in doubt -- ``ivregress 2sls`` already reproduced all five to the
authors' seven decimals, and ``westata`` must match.  The open question is the **standard
error**, because the authors cluster on two dimensions at once (``czone`` and
``congressionaldistrict``) and ``westata``'s ``IV2SLS`` documents two-way clustering as
unimplemented::

    westata.errors.UnsupportedError: two-way clustering is not implemented for IV in P1;
    pass a single cluster variable as cov_kwds={'clusters': ...}

That error is the right behaviour -- it refuses rather than silently returning a
one-way number labelled as two-way -- but it leaves a real gap, because an ADHM reader
cannot get the paper's standard errors out of ``westata`` at all.

So this driver does the two things that are possible:

1. estimates each specification with a **single** cluster dimension (``czone``, then
   ``congressionaldistrict``), giving standard errors that bracket the authors' value;
2. records the ``UnsupportedError`` explicitly, so the boundary is a measured fact in the
   output rather than a sentence in a report.

Bracketing matters: if the two single-way standard errors straddled the two-way one, a
careless driver could report one of them and look right by accident.  They do not straddle
it -- both are *below* it, by 10 and 9 percent -- which is exactly what the two-way
correction is for, and the report states it that way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import westata as ws  # noqa: E402

from adhm2020_specs import (  # noqa: E402
    CLUSTERS,
    ENDOG,
    INSTRUMENT,
    SPECS,
    WEIGHT,
    Spec,
)

OUT = HERE / "out"


def load() -> pd.DataFrame:
    from adhm2020_acquire import workfile

    return pd.read_stata(workfile(), convert_categoricals=False)


def formula(spec: Spec) -> str:
    """``y ~ 1 + controls + [endog ~ instrument]`` -- linearmodels' IV grammar.

    The exogenous controls sit outside the bracket and the instrument pair inside it, which
    is how ``IV2SLS.from_formula`` segregates endogenous from exogenous regressors.
    """

    controls = " + ".join(spec.regressors) if spec.regressors else "1"
    return f"{spec.depvar} ~ 1 + {controls} + [{ENDOG} ~ {INSTRUMENT}]"


def _fit(frame: pd.DataFrame, spec: Spec, clusters):
    return ws.IV2SLS.from_formula(
        formula(spec), frame, weights=frame[WEIGHT], weight_type="aweight"
    ).fit(cov_type="cluster", cov_kwds={"clusters": clusters})


def _serialise(result) -> dict:
    return {
        "names": [str(n) for n in result.param_names],
        "b": [float(x) for x in np.asarray(result.params, dtype=float)],
        "se": [float(x) for x in np.asarray(result.bse, dtype=float)],
        "nobs": int(result.nobs),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load()
    payload: dict = {"specs": {}, "twoway": {}, "meta": {}}
    payload["meta"]["westata_version"] = ws.__version__
    payload["meta"]["rows"] = int(len(frame))

    for spec in SPECS:
        payload["specs"][spec.id] = {}
        for cluster in CLUSTERS:
            result = _fit(frame, spec, frame[cluster])
            payload["specs"][spec.id][cluster] = _serialise(result)
            print(f"[westata] {spec.id} cluster({cluster}): N={result.nobs}", flush=True)

        # ---- record the two-way refusal, without turning it into a crash ----
        try:
            result = _fit(frame, spec, frame[list(CLUSTERS)])
            payload["twoway"][spec.id] = {
                "supported": True,
                "se": float(np.asarray(result.bse, dtype=float)[
                    [str(n) for n in result.param_names].index(ENDOG)]),
            }
            print(f"[westata] {spec.id} TWO-WAY: supported", flush=True)
        except Exception as exc:  # noqa: BLE001 - the refusal is the finding
            payload["twoway"][spec.id] = {
                "supported": False,
                "error": type(exc).__name__,
                "message": str(exc).splitlines()[0][:200],
            }
            print(f"[westata] {spec.id} TWO-WAY: {type(exc).__name__}", flush=True)

    (OUT / "westata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'westata_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
