"""Estimate the Nunn & Qian specifications with ``westata``.

The finding this driver exists to expose
----------------------------------------
``areg`` and ``xtreg, fe`` in Stata give the **same point estimate and different standard
errors** on this data (4.5 percent apart), because they count the absorbed country effects
differently in the small-sample cluster adjustment.  Both are Stata.  So "does westata
match the paper" is ambiguous until you say *which Stata command* is the benchmark.

This driver therefore reports both of westata's panel paths:

* ``FEOLS`` -- the ``reghdfe`` equivalent, absorbing the country effects internally;
* ``PanelOLS`` -- the ``xtreg, fe`` equivalent.

Both are compared against both Stata commands in ``run_compare.py``, so the report can say
which convention westata implements rather than asserting a single "match".

Year effects
------------
The specifications control for year fixed effects (``i.year`` in the authors' ``areg``).
``westata`` takes patsy formulas, so the year dummies are materialised as 0/1 columns with
the earliest year omitted -- the same design ``i.year`` produces.
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

from nq2011_specs import SPECS, Spec  # noqa: E402

OUT = HERE / "out"


def prepare() -> tuple[pd.DataFrame, list[str]]:
    """Load the panel, build the do-file's year interactions, and year dummies."""

    from nq2011_acquire import panel

    frame = pd.read_csv(panel(), sep="\t")

    interacted = (
        "ln_wpot", "ln_oworld", "ln_elevation", "ln_tropical", "ln_rugged",
        "ln_all", "ln_nworld", "ln_spot", "ln_silagemaize", "ln_maize",
        "ln_cassava", "ln_export", "malaria", "ln_disteq", "ln_dist_coast",
    )
    year_cols = {}
    for year in (1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1750, 1800, 1850, 1900):
        dummy = frame[f"ydum{year}"]
        for base in interacted:
            if base in frame.columns:
                year_cols[f"{base}_{year}"] = frame[base] * dummy
    frame = pd.concat([frame, pd.DataFrame(year_cols)], axis=1)

    # i.year with the earliest year as the omitted category, exactly as Stata does.
    years = sorted(frame["year"].unique())
    dummy_names = []
    dummies = {}
    for year in years[1:]:
        name = f"yr_{int(year)}"
        dummies[name] = (frame["year"] == year).astype(float)
        dummy_names.append(name)
    return pd.concat([frame, pd.DataFrame(dummies)], axis=1), dummy_names


def _formula(spec: Spec, dummy_names: list[str]) -> str:
    return f"{spec.depvar} ~ " + " + ".join([*spec.all_regressors, *dummy_names])


def _serialise(result) -> dict:
    return {
        "names": [str(n) for n in result.param_names],
        "b": [float(x) for x in np.asarray(result.params, dtype=float)],
        "se": [float(x) for x in np.asarray(result.bse, dtype=float)],
        "nobs": int(result.nobs),
        "estimator": type(result).__name__,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame, dummy_names = prepare()
    payload: dict = {"feols": {}, "panelols": {}, "meta": {}}
    payload["meta"]["westata_version"] = ws.__version__
    payload["meta"]["rows"] = int(len(frame))
    payload["meta"]["year_dummies"] = len(dummy_names)

    for spec in SPECS:
        formula = _formula(spec, dummy_names)

        # reghdfe-equivalent: country effects absorbed, year effects as regressors.
        fe = ws.FEOLS.from_formula(f"{formula} | isocode", frame).fit(
            cov_type="cluster", cov_kwds={"cluster": "isocode"}
        )
        payload["feols"][spec.id] = _serialise(fe)
        print(f"[westata FEOLS]    {spec.id}: N={fe.nobs}", flush=True)

        # xtreg-equivalent: same design through the panel estimator.
        po = ws.PanelOLS.from_formula(
            formula, frame, entity="isocode", time="year"
        ).fit(cov_type="cluster", cov_kwds={"groups": frame["isocode"]})
        payload["panelols"][spec.id] = _serialise(po)
        print(f"[westata PanelOLS] {spec.id}: N={po.nobs}", flush=True)

    (OUT / "westata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'westata_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
