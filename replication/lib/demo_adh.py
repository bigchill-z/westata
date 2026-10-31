"""End-to-end demo of the replication harness on Autor-Dorn-Hanson (2013).

Run from the repository root::

    .venv/Scripts/python.exe replication/lib/demo_adh.py

What it does
------------
1. Loads ``workfile_china.dta`` into the embedded Stata and estimates the
   first column of the paper's Table 5 panel A -- the stacked 1990/2000
   first-difference model with import exposure, the seven CZ controls, region
   dummies and a period dummy, weighted by ``timepwt48`` and clustered on
   ``statefip``.
2. Estimates the same specification with ``westata.OLS`` from the same
   DataFrame, mapping ``aweight`` to ``weights=.../weight_type="aweight"`` and
   ``cluster(statefip)`` to ``cov_type="cluster"``.
3. Compares the two through :func:`replication.lib.compare.compare_capture` and
   prints the rows and the verdict.

Why the spec is written out in full rather than read from the do-file
--------------------------------------------------------------------
The published log's command is ``reg ... reg* t2 [aw=timepwt48],
cluster(statefip)``, and the ``reg*`` shorthand (region dummies) is expanded by
Stata's wildcard at run time.  The demo keeps the explicit list so the two
estimators are demonstrably given the *same* regressors: a wildcard that
expands differently on one side is exactly the kind of mismatch the harness is
supposed to make impossible, and it cannot be checked if it is invisible.

The coefficients are expected to agree to ~1e-13; the point of the demo is that
the harness reproduces that agreement without a positional comparison.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for entry in (REPO_ROOT, REPO_ROOT / "src"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import pandas as pd  # noqa: E402

import westata as ws  # noqa: E402
from replication.lib.compare import compare_capture, tier  # noqa: E402
from replication.lib.report import (  # noqa: E402
    Published,
    difference_table,
    replication_table,
)
from replication.lib.stata_session import StataSession  # noqa: E402

ARCHIVE = (
    REPO_ROOT
    / "replication"
    / "candidates"
    / "raw"
    / "adh2013"
    / "Autor-Dorn-Hanson-ChinaSyndrome-FileArchive"
)
DTA = ARCHIVE / "dta" / "workfile_china.dta"

#: The reference numbers for *this* specification.  They come from the
#: authors' own Stata log (``czone_analysis_ipw_final.log``, line 5862), which
#: reports this exact reduced-form regression: -0.289323 (0.0429709), N = 1444,
#: R2 = 0.4789.  The paper's *printed* Table 5 panel A column 1 is the 2SLS
#: estimate (-4.230545) of a different estimand, so quoting it here would
#: compare an OLS reduced form against an IV coefficient and manufacture a
#: 14x "discrepancy" that says nothing about either implementation.
SPEC_NAME = "adh2013-tab5-panelA-mfg"
REFERENCE = Published(
    name=SPEC_NAME,
    estimate=-0.289323,
    se=0.0429709,
    n=1444,
    r2=0.4789,
    row="d_traderes_pw_lag",
    note="authors' czone_analysis_ipw_final.log (same command), not a printed table",
)

CONTROLS = [
    "l_shind_manuf_cbp",
    "l_sh_popedu_c",
    "l_sh_popfborn",
    "l_sh_empl_f",
    "l_sh_routine33",
    "l_task_outsource",
]
REGIONS = [
    "reg_midatl",
    "reg_encen",
    "reg_wncen",
    "reg_satl",
    "reg_escen",
    "reg_wscen",
    "reg_mount",
    "reg_pacif",
]
REGRESSORS = ["d_traderes_pw_lag", *CONTROLS, *REGIONS, "t2"]
DEPVAR = "d_sh_empl_mfg"


def stata_command() -> str:
    """The estimation command, written the way the do-file writes it."""
    return (
        f"reg {DEPVAR} " + " ".join(REGRESSORS) + " [aw=timepwt48], cluster(statefip)"
    )


def westata_formula() -> str:
    """The same model as a patsy formula, with patsy's ``C()`` for factors.

    ``reg_midatl`` ... ``reg_pacif`` are already 0/1 dummies in the workfile, so
    they enter as plain columns; ``t2`` is likewise a 0/1 indicator.  Writing
    them as ``C(...)`` would make patsy drop a level Stata keeps, which is a
    real difference in the model, not a labelling difference.
    """
    return f"{DEPVAR} ~ " + " + ".join(REGRESSORS)


def load_frame() -> pd.DataFrame:
    """Read the workfile with the float64 cast the estimators expect.

    The ``.dta`` stores most columns as float32.  Stata computes in double, so
    leaving them at float32 would introduce a ~1e-7 relative difference that
    has nothing to do with either implementation's algebra and would sit right
    on the ``NUMERICALLY_EQUAL`` threshold.
    """
    frame = pd.read_stata(DTA)
    return frame.astype({column: "float64" for column in frame.select_dtypes("float32")})


def run() -> int:
    if not DTA.is_file():
        print(f"dataset not found: {DTA}", file=sys.stderr)
        return 2

    frame = load_frame()
    command = stata_command()
    formula = westata_formula()

    with StataSession() as st:
        st.load_dta(DTA)
        # The demo casts to float64 on the Python side; Stata's own copy is
        # already double, so no conversion is needed there.
        cap = st.estimate(command)
        print(f"Stata {st.version}: e(N) = {cap.nobs}, e(r2) = {cap.get('r2'):.6f}")
        print(f"Stata parameters ({len(cap.names)}): {', '.join(cap.names)}\n")

    model = ws.OLS.from_formula(
        formula,
        frame,
        weights=frame["timepwt48"],
        weight_type="aweight",
    )
    result = model.fit(cov_type="cluster", cov_kwds={"groups": frame["statefip"]})
    print(f"westata: N = {result.nobs}, r2 = {result.e['r2']:.6f}")
    print(f"westata parameters ({len(result.param_names)}): {', '.join(result.param_names)}\n")

    record = compare_capture(SPEC_NAME, cap, result, scalars=("r2", "rmse", "F"))

    print("Per-parameter comparison")
    print(f"{'parameter':<24}{'Stata b':>16}{'westata b':>18}{'rel':>12}"
          f"{'Stata se':>14}{'westata se':>16}{'rel':>12}")
    for row in record.rows:
        rel_b = f"{row.b_rel:.2e}" if math.isfinite(row.b_rel) else "n/a"
        rel_se = f"{row.se_rel:.2e}" if math.isfinite(row.se_rel) else "n/a"
        print(
            f"{row.name:<24}{row.stata_b:>16.9f}{row.westata_b:>18.9f}{rel_b:>12}"
            f"{row.stata_se:>14.9f}{row.westata_se:>16.9f}{rel_se:>12}"
        )

    print()
    print(f"max coefficient rel diff : {record.max_b_rel:.3e} (tier {tier(record)})")
    print(f"max standard-error rel diff: {record.max_se_rel:.3e}")
    print(f"N: Stata {record.stata_n} / westata {record.westata_n}")
    for key, entry in record.scalars.items():
        print(
            f"e({key}): stata {entry['stata']:.12g}, westata {entry['westata']:.12g}, "
            f"rel {entry['rel']:.3e}"
        )
    for note in record.notes:
        print(f"note: {note}")
    print(f"VERDICT: {record.verdict}\n")

    print(replication_table([record], [REFERENCE], title="ADH 2013, Table 5 panel A, column 1"))
    print(difference_table([record]))

    return 0 if record.verdict == "IDENTICAL" else 1


if __name__ == "__main__":
    raise SystemExit(run())
