"""Estimate the Nunn & Qian specifications with StataNow through ``pystata``.

Two things are captured for every column, deliberately:

1. the ``areg`` result the authors ran;
2. the ``xtreg, fe`` result on the same design.

They share a point estimate and differ in the standard error, and the reason is a Stata
convention rather than anything about this data.  Capturing both is what lets the report
say "we match xtreg exactly and areg by 4.5 percent because of a documented difference"
instead of "the standard error does not match", which would be true but useless.
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

from nq2011_specs import POTATO_YEARS, SPECS, Spec  # noqa: E402
from stata_probe import Capture, Stata  # noqa: E402

OUT = HERE / "out"


def prepare() -> pd.DataFrame:
    """Load the panel and build the year-interaction columns the do-file creates.

    The do-file writes ``ln_wpot_1900 = ln_wpot * ydum1900`` for each census year; this is
    the same multiplication.  Doing it here rather than in Stata keeps the two sides
    building the *same* columns from the *same* arithmetic.
    """

    from nq2011_acquire import panel

    frame = pd.read_csv(panel(), sep="\t")

    # The do-file generates a year interaction for EVERY controlled series, not only
    # the potato stock; building them all here keeps both sides on one dataset.
    interacted = (
        "ln_wpot", "ln_oworld", "ln_elevation", "ln_tropical", "ln_rugged",
        "ln_all", "ln_nworld", "ln_spot", "ln_silagemaize", "ln_maize",
        "ln_cassava", "ln_export", "malaria", "ln_disteq", "ln_dist_coast",
    )
    # Built as one concat rather than ~180 column inserts: pandas emits a
    # PerformanceWarning for each insert, and the fragmented frame is slow to hand to
    # Stata besides.
    year_cols = {}
    for year in (1000, *POTATO_YEARS):
        dummy = frame[f"ydum{year}"]
        for base in interacted:
            if base in frame.columns:
                year_cols[f"{base}_{year}"] = frame[base] * dummy
    frame = pd.concat([frame, pd.DataFrame(year_cols)], axis=1)

    # xtset needs a numeric id; the panel key is a string.
    codes = {c: i + 1 for i, c in enumerate(sorted(frame["isocode"].unique()))}
    frame["panelid"] = frame["isocode"].map(codes)
    return frame


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
    }


def _xtreg_command(spec: Spec) -> str:
    """The same model through ``xtreg, fe`` -- same design, different dof convention."""

    return f"xtreg {spec.depvar} " + " ".join(spec.all_regressors) + " i.year, fe cluster(panelid)"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = prepare()
    payload: dict = {"specs": {}, "xtreg": {}, "meta": {}}

    with Stata() as stata:
        payload["meta"]["stata_version"] = stata.version
        payload["meta"]["edition"] = "mp"
        stata.send(frame)
        payload["meta"]["nobs_in_memory"] = stata.nobs()
        payload["meta"]["countries"] = int(frame["isocode"].nunique())

        stata.quiet("xtset panelid year")
        for spec in SPECS:
            cap = _capture(stata, spec.stata_command())
            payload["specs"][spec.id] = _serialise(cap)
            print(f"[stata areg]  {spec.id}: N={cap.nobs}", flush=True)

            alt = _capture(stata, _xtreg_command(spec))
            payload["xtreg"][spec.id] = _serialise(alt)
            print(f"[stata xtreg] {spec.id}: N={alt.nobs}", flush=True)

    (OUT / "stata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'stata_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
