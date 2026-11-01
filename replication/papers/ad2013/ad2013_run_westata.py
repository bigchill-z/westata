"""Estimate the Autor & Dorn specifications with ``westata``.

The alignment problem, and why this driver drops missing rows first
-------------------------------------------------------------------
The obvious call is::

    ws.OLS.from_formula(formula, frame, weights=frame["timepwt48"], weight_type="aweight")

On this data it fails with ``WeightError: weights have 722 entries but the data have 675
rows``.  The cause is not the data: ``d_rpc`` has 47 missing values, patsy drops those rows
while building the design matrix (722 -> 675), and the weight vector keeps its original
length.  ``OLS.from_formula`` passes ``weights`` straight through without restricting it
to ``kept_index``.

That is a real defect in the library's contract -- ``statsmodels``' own ``from_formula``
accepts a weight *column name* and drops rows consistently, and Stata's ``[aw=w]`` drops
them with the rest of the estimation sample.  It is reported in ``ad2013_gaps.md`` rather
than worked around silently.

**This driver therefore pre-drops the rows the model would drop anyway**, so that the
weight vector and the design matrix are built from the same frame.  The restriction is
derived from the formula and the condition rather than hard-coded: the same columns the
model uses are the ones checked for missingness.  The result is identical to what Stata
computes -- verified against the authors' own log -- which is what makes the workaround
acceptable here and *not* a substitute for fixing the library.

State dummies
-------------
The paper partials out a full set of state fixed effects (``xi i.statefip``).  ``westata``
takes patsy formulas, so the dummies are materialised as 0/1 columns with the lowest state
as the omitted category -- exactly what ``i.statefip`` does.  Building them explicitly
rather than silently is deliberate: a state dummy that is accidentally dropped is a
different regression, and the ``N`` and coefficient checks in ``run_compare.py`` would
catch it, but only if the omitted state is visible in the code.
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

from ad2013_specs import SPECS, Spec  # noqa: E402

OUT = HERE / "out"

#: Columns the model touches, beyond the state dummies which never have missings.
BASE_COLUMNS = ("timepwt48", "statefip")


def _state_dummies(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Materialise ``i.statefip`` as 0/1 columns with the lowest state omitted."""

    frame = frame.copy()
    states = sorted(frame["statefip"].dropna().unique())
    names = []
    for state in states[1:]:
        name = f"st_{int(state)}"
        frame[name] = (frame["statefip"] == state).astype(float)
        names.append(name)
    return frame, names


def _prepare(spec: Spec, raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Restrict to the estimation sample *before* building the model.

    The set of columns checked for missingness is derived from the specification, so a
    control added to ``specs.py`` is automatically included and the two sides cannot
    disagree about which rows survive.
    """

    frame, dummy_names = _state_dummies(raw)
    needed = [spec.depvar, spec.term, *spec.extra_regressors, *BASE_COLUMNS]
    frame = frame.dropna(subset=needed)
    if spec.condition:
        frame = frame.query(spec.condition)
    return frame.reset_index(drop=True), dummy_names


def _formula(spec: Spec, dummy_names: list[str]) -> str:
    terms = [spec.term, *spec.extra_regressors, *dummy_names]
    return f"{spec.depvar} ~ " + " + ".join(terms)


def _serialise(spec: Spec, result, dummy_count: int) -> dict:
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
        "state_dummies": dummy_count,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from ad2013_acquire import workfile

    raw = pd.read_stata(workfile(), convert_categoricals=False)
    payload: dict = {"specs": {}, "meta": {}}
    payload["meta"]["westata_version"] = ws.__version__
    payload["meta"]["westata_source"] = str(Path(ws.__file__).parent)
    payload["meta"]["rows_in_file"] = int(len(raw))

    for spec in SPECS:
        frame, dummy_names = _prepare(spec, raw)
        result = ws.OLS.from_formula(
            _formula(spec, dummy_names),
            frame,
            weights=frame["timepwt48"],
            weight_type="aweight",
        ).fit(cov_type="cluster", cov_kwds={"groups": frame["statefip"]})
        payload["specs"][spec.id] = _serialise(spec, result, len(dummy_names))
        print(f"[westata] {spec.id}: N={result.nobs}  {spec.table}{spec.row}", flush=True)

    (OUT / "westata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'westata_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
