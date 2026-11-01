"""Probe westata's surface for what the Card & Krueger replication actually needed.

What this module is for
----------------------
The ADH replication produced ``ck1994_gaps.md``: a list of things westata could improve,
each backed by an executable probe.  Upstream has since fixed all seven entries, so that
list now reports **zero gaps** and every entry has become a control.

This second replication asks a different question.  Card & Krueger needs *no* exotic
estimator -- five unweighted OLS regressions on 357 rows.  So the interesting finding is
not "westata cannot estimate X"; it is **how much of a real replication this library does
not touch at all**.  The probe therefore measures two things:

1. the estimator surface (which passes trivially, and is reported so the triviality is
   visible rather than implied);
2. the *data* surface -- the eight transformations this replication performed, none of
   which westata offers.  That absence is documented policy (README: "本库不做数据管理"),
   so the probe records it as a **boundary**, not a gap.  The distinction matters: a gap
   is something to fix, a boundary is something to plan around.

Outcomes
--------
``OK``        the call works
``BOUNDARY``  westata does not offer this, by design -- recorded, not counted as a gap
``GAP``       westata claims to offer it, or a statsmodels-shaped call should work, and it
              does not -- this is something to fix
``ERROR``     unexpected; the probe itself is wrong or an exception type changed
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import westata as ws  # noqa: E402


@dataclass
class Probe:
    kind: str          # "estimator" or "data"
    name: str
    claim: str
    exercise: Callable[[], str]

    def run(self) -> tuple[str, str]:
        try:
            return "OK", self.exercise()
        except _Boundary as exc:
            return "BOUNDARY", str(exc)
        except _Gap as exc:
            return "GAP", str(exc)
        except Exception as exc:  # noqa: BLE001 - a probe must never crash the sweep
            return "ERROR", f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


class _Boundary(Exception):
    """By design, westata does not do this."""


class _Gap(Exception):
    """westata should do this and does not."""


def _frame() -> pd.DataFrame:
    from ck1994_data import build, table4_sample

    return table4_sample(build())


# --------------------------------------------------------------- estimator surface
def _ols_works() -> str:
    frame = _frame()
    result = ws.OLS.from_formula("d_fte ~ nj", frame).fit()
    index = list(result.param_names).index("nj")
    coef = float(np.asarray(result.params, dtype=float)[index])
    if abs(coef - 2.3258) > 1e-3:
        raise _Gap(f"NJ coefficient drifted to {coef!r}")
    return f"coefficient {coef:.4f}, matching the paper's 2.33"


def _stata_labels_available() -> str:
    """The Stata-facing labels exist, so a user need not build the rename map by hand.

    This probe is here because I got it wrong first: I assumed westata only exposed
    patsy's ``Intercept`` and wrote a ``_cons`` rename into the comparison layer.  The
    library already publishes ``stata_param_names`` / ``stata_params``; the rename was my
    own duplication, not a missing feature.  Recorded so the next reader does not repeat
    the detour.
    """

    frame = _frame()
    result = ws.OLS.from_formula("d_fte ~ nj", frame).fit()
    stata_names = [str(n) for n in result.stata_param_names]
    if "_cons" not in stata_names:
        raise _Gap(f"stata_param_names has no _cons: {stata_names}")
    labelled = result.stata_params
    if hasattr(labelled, "index") and "_cons" not in list(labelled.index):
        raise _Gap("stata_params is not labelled with Stata names")
    return f"stata_param_names = {stata_names}"


def _weights_accepted() -> str:
    """CK does not need weights, but ADH did; this confirms that capability survived."""

    frame = _frame()
    result = ws.OLS.from_formula(
        "d_fte ~ nj", frame, weights=frame["fte"], weight_type="aweight"
    ).fit()
    if not result.nobs:
        raise _Gap("weighted OLS returned no observations")
    return f"weighted OLS runs (N={int(result.nobs)})"


ESTIMATOR_PROBES: tuple[Probe, ...] = (
    Probe("estimator", "plain OLS", "OLS on 357 rows reproduces the paper's coefficient",
          _ols_works),
    Probe("estimator", "Stata labels", "stata_param_names/stata_params expose _cons",
          _stata_labels_available),
    Probe("estimator", "weighted OLS", "weights= and weight_type= still accepted",
          _weights_accepted),
)


# -------------------------------------------------------------------- data surface
#: The eight transformations this replication performed on the raw fixed-width file.
#: Each is something a Stata user would do with ``infile``/``destring``/``gen``/``replace``.
DATA_STEPS = (
    ("fixed-width parse", "read 46 fields by column position"),
    ("missing '.' to NaN", "coerce sentinel to missing"),
    ("derive FTE", "empft + 0.5*emppt + nmgrs"),
    ("conditional replace", "permanently closed stores to zero"),
    ("conditional replace", "temporarily closed stores to missing"),
    ("derive GAP", "max(5.05 - wage, 0) / wage, zero in PA"),
    ("sample restriction", "mask to the 357 stores"),
    ("dummy construction", "chain2..chain4 indicator columns"),
)

#: Names a Stata user would look for.  Their absence is documented policy, so each one
#: raises ``_Boundary`` rather than ``_Gap`` -- the probe's job is to count how much of
#: the work sits outside the library, and to keep that count honest as the library grows.
DATA_NAMES = ("gen", "egen", "replace", "drop", "keep", "merge", "reshape", "collapse",
              "bysort", "append", "infile", "destring", "encode", "decode")


def _data_surface() -> str:
    present = [n for n in DATA_NAMES if hasattr(ws, n)]
    if present:
        raise _Boundary(
            f"westata now exposes {present}; README's 'no data management' claim and "
            "this probe both need updating"
        )
    raise _Boundary(
        f"none of {len(DATA_NAMES)} Stata data verbs exist; all {len(DATA_STEPS)} "
        "transformations in this replication were done in pandas"
    )


DATA_PROBES: tuple[Probe, ...] = (
    Probe("data", "data verbs", "Stata data verbs absent by documented policy",
          _data_surface),
)


# ------------------------------------------------------------------------ controls
def _control_estimator_surface() -> str:
    """If the public surface shrinks unexpectedly, the probes above stop being meaningful."""

    names = [n for n in dir(ws) if not n.startswith("_")]
    if len(names) < 100:
        raise _Gap(f"public surface has only {len(names)} names; something was removed")
    return f"{len(names)} public names"


CONTROLS: tuple[Probe, ...] = (
    Probe("control", "public surface", "the estimator surface is intact",
          _control_estimator_surface),
)


def main() -> int:
    print(f"westata {ws.__version__} from {Path(ws.__file__).parent}\n")

    counts = {"OK": 0, "BOUNDARY": 0, "GAP": 0, "ERROR": 0}
    for group, probes in (("estimator surface", ESTIMATOR_PROBES),
                          ("data surface", DATA_PROBES),
                          ("controls", CONTROLS)):
        print(f"-- {group} --")
        for probe in probes:
            status, detail = probe.run()
            counts[status] += 1
            print(f"  [{status:8s}] {probe.name:18s} {detail}")
        print()

    print(f"{counts['OK']} ok, {counts['BOUNDARY']} boundary, "
          f"{counts['GAP']} gap, {counts['ERROR']} error")

    if counts["GAP"] or counts["ERROR"]:
        print("\nA GAP here is something to fix; see ../../gaps/ck1994_gaps.md.")
    print(
        "\nThe data-surface entry is a BOUNDARY, not a gap: README states the library "
        "does no data management, and this replication is evidence of what that costs "
        "in practice."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
