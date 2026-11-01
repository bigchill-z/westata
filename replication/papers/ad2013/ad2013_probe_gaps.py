"""Probe westata's surface for what the Autor & Dorn replication needed.

Third replication, third question
---------------------------------
The ADH replication produced a list of **7 gaps**, all since fixed upstream.  The Card &
Krueger replication found **0 gaps and 1 boundary** (no data layer, by documented policy).

This one finds something different again: a **contract defect**.  Not a missing feature and
not a declared boundary, but a call that a statsmodels-shaped API should accept and that
fails in a way the user cannot reasonably anticipate.  That is the category the probe is
built around here, and it is the one the earlier two replications did not exercise.

The defect
----------
::

    ws.OLS.from_formula(formula, frame, weights=frame["w"], weight_type="aweight")

raises ``WeightError: weights have 722 entries but the data have 675 rows`` whenever the
formula's columns contain missing values.  Patsy drops those rows while building the design
matrix; the weight vector keeps its original length; ``from_formula`` forwards ``weights``
to the constructor without restricting it to ``kept_index``.

Both comparators handle this without the user doing anything:

* Stata's ``[aw=w]`` drops rows together with the rest of the estimation sample;
* ``statsmodels.formula.api.ols`` accepts a **column name** and applies ``dropna`` across
  the model frame, so the weight column shrinks with everything else.

So the expectation is not exotic -- it is what both reference implementations do.

Outcomes
--------
``OK``     the call works
``GAP``    it does not, and it should
``ERROR``  unexpected
"""

from __future__ import annotations

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


class _Gap(Exception):
    """westata should do this and does not."""


@dataclass
class Probe:
    name: str
    claim: str
    exercise: Callable[[], str]

    def run(self) -> tuple[str, str]:
        try:
            return "OK", self.exercise()
        except _Gap as exc:
            return "GAP", str(exc)
        except Exception as exc:  # noqa: BLE001
            return "ERROR", f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


def _raw() -> pd.DataFrame:
    from ad2013_acquire import workfile

    return pd.read_stata(workfile(), convert_categoricals=False)


def _dummies(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    frame = frame.copy()
    states = sorted(frame["statefip"].dropna().unique())
    names = []
    for state in states[1:]:
        name = f"st_{int(state)}"
        frame[name] = (frame["statefip"] == state).astype(float)
        names.append(name)
    return frame, names


def _straight_call() -> str:
    """The natural call, on a frame that still has missing values in the dependent variable.

    ``d_rpc`` has 47 missings out of 722 rows, so patsy keeps 675.  If the weight vector is
    aligned with the design matrix this must simply work.
    """

    raw = _raw()
    frame, names = _dummies(raw)
    frame = frame.query("t1==1").reset_index(drop=True)
    formula = "d_rpc ~ l_sh_routine33a + " + " + ".join(names)

    present = int(frame["d_rpc"].notna().sum())
    if present == len(frame):
        raise _Gap(
            "the test frame has no missing values, so this probe cannot exercise the "
            "alignment path; pick a variable that does"
        )

    try:
        result = ws.OLS.from_formula(
            formula, frame, weights=frame["timepwt48"], weight_type="aweight"
        ).fit()
    except Exception as exc:
        raise _Gap(
            f"from_formula rejects a weight vector aligned with `frame` when the formula "
            f"drops {len(frame) - present} rows: {type(exc).__name__}: {exc}. "
            "Stata and statsmodels both restrict the weights to the estimation sample, so "
            "the fix is to slice `weights` by patsy's kept_index before constructing the "
            "model."
        ) from exc

    index = [str(n) for n in result.param_names].index("l_sh_routine33a")
    coef = float(np.asarray(result.params, dtype=float)[index])
    if abs(coef - 0.6950717) > 1e-6:
        raise _Gap(f"the aligned call ran but returned {coef!r}, not the paper's 0.6950717")
    return f"aligned weights accepted; coefficient {coef:.7f}"


def _weight_by_name() -> str:
    """Does the API accept a weight *column name*, the way statsmodels does?

    This is the ergonomic half of the same issue.  Passing an array works if the caller
    aligns it; passing a name would remove the need to try.
    """

    raw = _raw()
    frame, names = _dummies(raw)
    frame = frame.query("t1==1").reset_index(drop=True)
    formula = "d_rpc ~ l_sh_routine33a + " + " + ".join(names)
    try:
        ws.OLS.from_formula(formula, frame, weights="timepwt48", weight_type="aweight").fit()
    except Exception as exc:
        raise _Gap(
            f"weights as a column name is not supported ({type(exc).__name__}). "
            "statsmodels' from_formula accepts a name, which is what makes its dropna "
            "handling transparent; an array forces the caller to replicate patsy's "
            "row-dropping by hand."
        ) from exc
    return "weights accepted as a column name"


def _documented_behaviour() -> str:
    """The docstring should say what happens to weights when rows are dropped."""

    doc = (ws.OLS.from_formula.__doc__ or "")
    if "missing" not in doc.lower() and "kept_index" not in doc:
        raise _Gap(
            "OLS.from_formula's docstring does not mention that it drops rows with missing "
            "values, nor what happens to `weights` when it does. A caller following the "
            "signature has no way to learn that the two can disagree about length."
        )
    return "the docstring documents the row-dropping behaviour"


PROBES: tuple[Probe, ...] = (
    Probe("aligned weights", "weights are restricted to the estimation sample",
          _straight_call),
    Probe("weights by name", "a weight column name is accepted", _weight_by_name),
    Probe("documentation", "the docstring warns about row-dropping", _documented_behaviour),
)


# ---------------------------------------------------------------- controls
def _control_plain_ols() -> str:
    """The replication's own numbers must still come out."""

    raw = _raw()
    frame, names = _dummies(raw)
    frame = frame.dropna(subset=["d_rpc"]).query("t1==1").reset_index(drop=True)
    formula = "d_rpc ~ l_sh_routine33a + " + " + ".join(names)
    result = ws.OLS.from_formula(
        formula, frame, weights=frame["timepwt48"], weight_type="aweight"
    ).fit(cov_type="cluster", cov_kwds={"groups": frame["statefip"]})
    index = [str(n) for n in result.param_names].index("l_sh_routine33a")
    coef = float(np.asarray(result.params, dtype=float)[index])
    if abs(coef - 0.6950717) > 1e-6:
        raise _Gap(f"the replication coefficient drifted to {coef!r}")
    return f"pre-aligned call reproduces {coef:.7f}"


CONTROLS: tuple[Probe, ...] = (
    Probe("pre-aligned call", "dropping missing rows first still works",
          _control_plain_ols),
)


def main() -> int:
    print(f"westata {ws.__version__} from {Path(ws.__file__).parent}\n")
    counts = {"OK": 0, "GAP": 0, "ERROR": 0}
    for group, probes in (("probes", PROBES), ("controls", CONTROLS)):
        print(f"-- {group} --")
        for probe in probes:
            status, detail = probe.run()
            counts[status] += 1
            print(f"  [{status:5s}] {probe.name:18s} {detail}")
        print()
    print(f"{counts['OK']} ok, {counts['GAP']} gap, {counts['ERROR']} error")
    if counts["GAP"]:
        print("\nA GAP is a contract defect to fix, not a boundary; see ../../gaps/ad2013_gaps.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
