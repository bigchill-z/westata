"""Estimate the Card & Krueger specifications with ``westata``.

The point of running the same five regressions through two independent implementations
is that agreement is evidence and disagreement is a lead.  Both sides read the
specification from ``specs.py`` and the sample from ``data.py``, so any difference in the
output is a difference between the *estimators*, not between two hand-typed control
lists.

Card & Krueger is an unusually clean test case for this: every specification is plain
unweighted OLS on 357 observations, with no factor variables beyond the 0/1 dummies that
are already numeric columns.  So the comparison isolates the linear algebra, arithmetic
and degrees-of-freedom conventions, with nothing to hide in.

``westata`` results carry ``params``/``bse`` in patsy order (intercept first, then
alphabetical); the Stata side reports the order written in the command with ``_cons``
last.  ``run_compare.py`` pairs them by name, never by position -- a positional pairing
on these models would silently compare ``nj`` against ``const``.
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

import westata as ws  # noqa: E402

from ck1994_data import build, table4_sample  # noqa: E402
from ck1994_specs import SPECS, Spec  # noqa: E402

OUT = HERE / "out"


def _frame_for(spec: Spec):
    frame = build()
    return table4_sample(frame) if spec.table4_sample else frame


def _serialise(spec: Spec, result) -> dict:
    return {
        "spec": spec.id,
        "names": [str(n) for n in result.param_names],
        "b": [float(x) for x in np.asarray(result.params, dtype=float)],
        "se": [float(x) for x in np.asarray(result.bse, dtype=float)],
        "nobs": int(result.nobs),
        "scalars": {
            k: float(v)
            for k, v in (getattr(result, "e", {}) or {}).items()
            if isinstance(v, (int, float, np.floating))
        },
        "estimator": type(result).__name__,
    }


def _difference_in_differences() -> dict:
    """The Table 3 number, computed with westata so it is comparable to the Stata side."""

    frame = build()
    balanced = frame.loc[frame["fte"].notna() & frame["fte2"].notna()].reset_index(drop=True)
    result = ws.OLS.from_formula("d_fte ~ nj", balanced).fit()
    index = [str(n) for n in result.param_names].index("nj")
    means = balanced.groupby("nj")["d_fte"].mean()
    return {
        "nj_change": float(means[1]),
        "pa_change": float(means[0]),
        "did": float(means[1] - means[0]),
        "regression_b": float(np.asarray(result.params, dtype=float)[index]),
        "regression_se": float(np.asarray(result.bse, dtype=float)[index]),
        "n": int(len(balanced)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payload: dict = {"specs": {}, "table3": {}, "meta": {}}
    payload["meta"]["westata_version"] = ws.__version__
    payload["meta"]["westata_source"] = str(Path(ws.__file__).parent)

    for spec in SPECS:
        frame = _frame_for(spec)
        result = ws.OLS.from_formula(spec.formula, frame).fit()
        payload["specs"][spec.id] = _serialise(spec, result)
        print(f"[westata] {spec.id}: N={result.nobs}", flush=True)

    payload["table3"] = _difference_in_differences()
    (OUT / "westata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'westata_results.json'}")
    print(f"Table 3 DiD = {payload['table3']['did']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
