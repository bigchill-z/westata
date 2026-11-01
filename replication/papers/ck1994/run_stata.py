"""Estimate the Card & Krueger specifications with StataNow through ``pystata``.

The paper's own code is SAS, not Stata, so there is no do-file to adapt.  These commands
are written here to match the paper's Table 4 exactly: ``regress``, plain OLS, with the
control sets from ``specs.py``.
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

from ck1994_data import build, table4_sample  # noqa: E402
from ck1994_specs import SPECS, Spec  # noqa: E402
from stata_probe import Capture, Stata  # noqa: E402

OUT = HERE / "out"


def _frame_for(spec: Spec):
    frame = build()
    return table4_sample(frame) if spec.table4_sample else frame


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


def _difference_in_differences() -> dict:
    """Table 3's raw DiD, computed both by hand and by a regression, so the two agree.

    The paper quotes 2.76 FTE (13%, t = 2.03).  Computing the group means directly *and*
    fitting the equivalent regression is a cheap internal consistency check -- if the
    two disagree, the sample restriction or the weighting is wrong.
    """

    frame = build()
    balanced = frame.loc[frame["fte"].notna() & frame["fte2"].notna()]
    means = balanced.groupby("nj")["d_fte"].mean()
    did = float(means[1] - means[0])
    # Standard error of the DiD, by the usual two-sample formula.
    counts = balanced.groupby("nj")["d_fte"].count()
    variances = balanced.groupby("nj")["d_fte"].var()
    se = float(np.sqrt(variances[1] / counts[1] + variances[0] / counts[0]))

    # Same quantity as a regression coefficient, as a cross-check.
    import westata as ws

    fit = ws.OLS.from_formula("d_fte ~ nj", balanced.reset_index(drop=True)).fit()
    index = list(fit.param_names).index("nj")

    zero_temp = frame.loc[frame["fte"].notna() & frame["d_fte_zero_temp"].notna()]
    means0 = zero_temp.groupby("nj")["d_fte_zero_temp"].mean()

    return {
        "method": "Table 3 rows 4-5: change in mean FTE employment, NJ vs PA",
        "nj_change": float(means[1]),
        "pa_change": float(means[0]),
        "did": did,
        "se_hand": se,
        "t_stat": did / se if se else float("nan"),
        "percent_of_nj_base": float(100 * did / balanced.loc[balanced.nj == 1, "fte"].mean()),
        "regression_check_b": float(np.asarray(fit.params, dtype=float)[index]),
        "regression_check_se": float(np.asarray(fit.bse, dtype=float)[index]),
        "n": int(len(balanced)),
        "did_temp_closed_as_zero": float(means0[1] - means0[0]),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {"specs": {}, "table3": {}, "meta": {}}

    with Stata() as stata:
        payload["meta"]["stata_version"] = stata.version
        payload["meta"]["edition"] = "mp"

        for spec in SPECS:
            frame = _frame_for(spec)
            stata.send(frame)
            cap = _capture(stata, spec.stata_command())
            payload["specs"][spec.id] = _serialise(cap)
            print(f"[stata] {spec.id}: N={cap.nobs}", flush=True)

    payload["table3"] = _difference_in_differences()
    (OUT / "stata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'stata_results.json'}")
    print(f"Table 3 DiD = {payload['table3']['did']:.4f} (t = {payload['table3']['t_stat']:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
