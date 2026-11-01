"""Probe westata for what the AJR replication needed: ``xtreg, fe`` with aweight.

One finding, and it is about which estimator the documentation points at rather than about
numerical accuracy:

* ``xtreg`` maps to ``PanelOLS``, but ``PanelOLS`` accepts no ``weights``, so it cannot
  estimate four of this paper's five replicated columns;
* ``FEOLS`` can, and reproduces Stata exactly -- which is why the replication uses it.

The probe also measures ``Areg``'s weighted clustered standard error, which does not match
``xtreg``'s on this data.  That is recorded rather than silently avoided: the replication
uses ``FEOLS`` partly because of it.
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

from ajr2001_specs import SPECS  # noqa: E402


class _Gap(Exception):
    """westata should do this and does not."""


class _Note(Exception):
    """A difference worth recording that is not a defect."""


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
        except _Note as exc:
            return "NOTE", str(exc)
        except Exception as exc:  # noqa: BLE001
            return "ERROR", f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


_FRAME = None


def _frame() -> pd.DataFrame:
    global _FRAME
    if _FRAME is None:
        from ajr2001_acquire import workfile

        _FRAME = pd.read_stata(workfile(), convert_categoricals=False)
    return _FRAME


def _spec():
    return next(s for s in SPECS if s.id == "T3_c1")


def _sample() -> pd.DataFrame:
    from ajr2001_run_westata import sample

    spec = _spec()
    return sample(_frame(), spec)


def _panelols_weights() -> str:
    """The gap: ``PanelOLS`` -- the documented ``xtreg`` equivalent -- takes no weights.

    Four of this paper's five replicated columns are weighted, so the documented mapping
    cannot express them.
    """

    import inspect

    signature = inspect.signature(ws.PanelOLS.from_formula)
    if "weights" in signature.parameters:
        raise _Gap(
            "PanelOLS now accepts weights; this probe and ajr2001_gaps.md section 1 are stale"
        )

    frame = _sample()
    try:
        ws.PanelOLS.from_formula(
            "urbrate ~ " + " + ".join(_spec().regressors),
            frame, entity="id", time="year",
            weights=frame["totalpop1750"],
        )
    except TypeError as exc:
        raise _Gap(
            f"PanelOLS.from_formula does not accept weights ({exc}). The paper's command is "
            "xtreg ... [aweight=totalpop1750], and 4 of the 5 replicated columns are weighted, "
            "so the documented xtreg equivalent cannot estimate them. FEOLS can, and matches "
            "Stata, so this is a mapping gap rather than a capability gap."
        ) from None
    raise _Gap("PanelOLS accepted weights via **kwargs; the gap may already be closed")


def _feols_weighted() -> str:
    """The workaround must reproduce Stata's coefficient and clustered SE."""

    from ajr2001_run_westata import _fit

    frame = _sample()
    result = _fit(_spec(), frame)
    index = [str(n) for n in result.param_names].index("fpresence1900")
    coefficient = float(np.asarray(result.params, dtype=float)[index])
    se = float(np.asarray(result.bse, dtype=float)[index])

    # Stata's values for this specification, captured by run_stata into the results file.
    target_b, target_se = 0.633869, 0.407511
    if abs(coefficient - target_b) > 1e-5:
        raise _Gap(f"FEOLS coefficient {coefficient!r} is not Stata's {target_b}")
    if abs(se - target_se) / target_se > 1e-4:
        raise _Gap(f"FEOLS clustered SE {se:.6f} is not Stata's {target_se}")
    return f"coefficient {coefficient:.6f}, clustered SE {se:.6f}"


def _areg_weighted_se() -> str:
    """``Areg`` takes weights but its clustered SE does not match ``xtreg`` on this data.

    Recorded as a NOTE rather than a GAP because the paper's own diagnostics (the joint test)
    do not depend on it, and because the underlying cause was not isolated here -- only the
    symptom is measured.  ``FEOLS`` is used precisely because of this.
    """

    frame = _sample()
    spec = _spec()
    result = ws.Areg.from_formula(
        "urbrate ~ " + " + ".join(spec.regressors),
        frame, absorb="id",
        weights=frame["totalpop1750"], weight_type="aweight",
    ).fit(cov_type="cluster", cov_kwds={"cluster": "id"})
    index = [str(n) for n in result.param_names].index("fpresence1900")
    coefficient = float(np.asarray(result.params, dtype=float)[index])
    se = float(np.asarray(result.bse, dtype=float)[index])

    if abs(coefficient - 0.633869) > 1e-5:
        raise _Gap(f"Areg's coefficient {coefficient!r} no longer matches; revisit this probe")
    if abs(se - 0.407511) / 0.407511 < 1e-4:
        raise _Gap("Areg now matches xtreg's clustered SE; the NOTE is stale")
    raise _Note(
        f"Areg coefficient matches (0.633869) but its clustered SE is {se:.6f} against "
        "xtreg's 0.407511 (11% apart). The replication uses FEOLS for this reason; the "
        "cause was not isolated."
    )


def _public_surface() -> str:
    names = [n for n in dir(ws) if not n.startswith("_")]
    if len(names) < 100:
        raise _Gap(f"public surface shrank to {len(names)} names")
    return f"{len(names)} public names"


PROBES: tuple[Probe, ...] = (
    Probe("PanelOLS weights", "the documented xtreg equivalent can take weights",
          _panelols_weights),
    Probe("FEOLS weighted FE", "the workaround reproduces Stata", _feols_weighted),
    Probe("Areg weighted SE", "Areg's clustered SE matches xtreg", _areg_weighted_se),
)

CONTROLS: tuple[Probe, ...] = (
    Probe("public surface", "the estimator surface is intact", _public_surface),
)


def main() -> int:
    print(f"westata {ws.__version__} from {Path(ws.__file__).parent}\n")
    counts = {"OK": 0, "NOTE": 0, "GAP": 0, "ERROR": 0}
    for group, probes in (("estimator surface", PROBES), ("controls", CONTROLS)):
        print(f"-- {group} --")
        for probe in probes:
            status, detail = probe.run()
            counts[status] += 1
            print(f"  [{status:5s}] {probe.name:22s} {detail}")
        print()
    print(f"{counts['OK']} ok, {counts['NOTE']} note, {counts['GAP']} gap, "
          f"{counts['ERROR']} error")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
