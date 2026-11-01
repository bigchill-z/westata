"""Estimate the AJR (2001) specifications with ``westata``.

Which estimator, and why it is not ``PanelOLS``
-----------------------------------------------
The paper's command is ``xtreg ..., fe [aweight=...] cluster(id)``.  The obvious westata
mapping is ``PanelOLS`` (its documented ``xtreg`` equivalent), but **``PanelOLS`` accepts no
``weights`` argument at all** -- its signature is
``from_formula(formula, data, entity=, time=, subset=)``.  So it cannot express the paper's
weighted columns, which are four of the five replicated here.

``FEOLS`` takes ``weights``/``weight_type`` and absorbs the entity fixed effects through the
same projection, so it produces the same within estimator.  This driver therefore uses
``FEOLS`` with ``| id`` for the entity effects and the year dummies as explicit regressors,
matching ``xtreg, fe``'s parameterisation.

``Areg`` is also available and also takes weights, but its clustered standard errors differ
from ``xtreg``'s on this data (measured: 0.452923 against 0.407511 for the 1900 term), so
using it here would confound the comparison.  That difference is recorded in the paper's gap
list rather than worked around silently.

The point of this replication
-----------------------------
Stata is the reference, so the claim being tested is narrow and checkable: **westata
reproduces Stata's coefficients, clustered standard errors and the authors' joint test on the
authors' own specifications.**  It is not a claim about the published table, which was not
available.
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

from ajr2001_specs import SPECS, Spec  # noqa: E402

OUT = HERE / "out"

JOINT_TERMS = ("fpresence1850", "fpresence1875", "fpresence1900")


def load() -> pd.DataFrame:
    from ajr2001_acquire import workfile

    return pd.read_stata(workfile(), convert_categoricals=False)


def sample(frame: pd.DataFrame, spec: Spec) -> pd.DataFrame:
    """The same restriction ``run_stata.sample`` applies, so both sides estimate one sample."""

    frame = frame[frame["year"].isin([1700, 1750, 1800, 1850, 1875, 1900])].copy()
    if spec.west_only:
        frame = frame[frame["westelbe"] == 1]
    if spec.drop_ids:
        frame = frame[~frame["id"].isin(spec.drop_ids)]
    needed = ["urbrate", *spec.regressors]
    if spec.weighted:
        needed.append("totalpop1750")
    return frame.dropna(subset=needed).reset_index(drop=True)


def formula(spec: Spec) -> str:
    """``urbrate ~ fpresence* + year dummies | id`` -- the within estimator."""

    return f"urbrate ~ " + " + ".join(spec.regressors) + " | id"


def _fit(spec: Spec, frame: pd.DataFrame):
    kwargs = {}
    if spec.weighted:
        kwargs = {"weights": frame["totalpop1750"], "weight_type": "aweight"}
    return ws.FEOLS.from_formula(formula(spec), frame, **kwargs).fit(
        cov_type="cluster", cov_kwds={"cluster": "id"}
    )


def _joint_test(result) -> tuple[float, float]:
    """The authors' joint test, through westata's own Stata-style ``test`` command.

    ``result.test("a = 0, b = 0")`` returns a :class:`WaldTest` carrying ``statistic`` and
    ``pvalue``, which is what the do-file's subsequent ``test fpresence1850 ...`` produces.
    """

    names = [str(n) for n in result.param_names]
    terms = [term for term in JOINT_TERMS if term in names]
    wald = result.test(", ".join(f"{term} = 0" for term in terms))
    return float(wald.statistic), float(wald.pvalue)


def _serialise(result) -> dict:
    statistic, pvalue = _joint_test(result)
    return {
        "names": [str(n) for n in result.param_names],
        "b": [float(x) for x in np.asarray(result.params, dtype=float)],
        "se": [float(x) for x in np.asarray(result.bse, dtype=float)],
        "nobs": int(result.nobs),
        "joint_F": statistic,
        "joint_p": pvalue,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = load()
    payload: dict = {"specs": {}, "meta": {}}
    payload["meta"]["westata_version"] = ws.__version__
    payload["meta"]["estimator"] = "FEOLS (PanelOLS cannot take weights)"

    for spec in SPECS:
        frame = sample(raw, spec)
        result = _fit(spec, frame)
        payload["specs"][spec.id] = _serialise(result)
        print(f"[westata FEOLS] {spec.id}: N={result.nobs}  {spec.table}{spec.column}",
              flush=True)

    (OUT / "westata_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'westata_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
