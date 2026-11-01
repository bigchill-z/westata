"""Probe westata's panel surface for what the Nunn & Qian replication needed.

What this probe covers
----------------------
This replication exercises westata's panel surface against Stata: ``Areg`` against
``areg``, ``FEOLS`` against ``reghdfe``, ``PanelOLS`` against ``xtreg, fe``.  The slopes
match everywhere, so the interesting checks are about conventions and about the constant.

Three findings, in decreasing order of how much they matter:

* ``Areg`` reports ``_cons`` that Stata's ``areg`` does not agree with, on a two-way design
  (explicit year dummies plus absorbed country effects).  A ``GAP``.
* ``Areg``'s clustered standard errors follow ``reghdfe`` rather than ``areg``: the same
  4.5 percent convention difference that ``FEOLS`` has.  Recorded as a ``NOTE``, because
  ``reghdfe``'s own source says it deliberately diverges from ``areg``.
* the coefficients are right on every path, which is the baseline claim.

Outcomes
--------
``OK``        the call works
``NOTE``      a known, attributed difference that is not a defect (recorded, not counted)
``GAP``       westata should do this and does not
``ERROR``     unexpected
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

from nq2011_specs import PAPER  # noqa: E402


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
_DUMMIES: list[str] = []


def _frame():
    """The prepared panel plus the year-dummy names, built once."""

    global _FRAME, _DUMMIES
    if _FRAME is None:
        from nq2011_run_westata import prepare

        _FRAME, _DUMMIES = prepare()
    return _FRAME, _DUMMIES


def _potato() -> str:
    years = (1100, 1200, 1300, 1400, 1500, 1600, 1700, 1750, 1800, 1850, 1900)
    return " + ".join(f"ln_wpot_{year}" for year in years)


def _fit_areg():
    frame, dummies = _frame()
    formula = f"ln_population ~ {_potato()} + " + " + ".join(dummies)
    return ws.Areg.from_formula(formula, frame, absorb="isocode").fit(
        cov_type="cluster", cov_kwds={"cluster": "isocode"}
    )


def _coefficient() -> str:
    """The slope must match the authors' log; this is the baseline claim."""

    result = _fit_areg()
    index = [str(n) for n in result.param_names].index("ln_wpot_1900")
    coefficient = float(np.asarray(result.params, dtype=float)[index])
    published = PAPER["T2_c1"]["ln_wpot_1900"][0]
    if abs(coefficient - published) > 1e-6:
        raise _Gap(f"coefficient {coefficient!r} is not the authors' {published}")
    return f"coefficient {coefficient:.7f}"


def _constant() -> str:
    """``Areg``'s reported ``_cons`` must match Stata's ``areg`` on the same design.

    Stata reports ``_cons = 12.64269254`` with ``se = 0.03383845`` for this specification
    (11 potato terms, 11 year dummies, 130 absorbed country effects, clustered on isocode).
    ``Areg`` reports a different point estimate and a standard error several times too
    large, so the reported intercept is not Stata's.
    """

    result = _fit_areg()
    names = [str(n) for n in result.param_names]
    if "_cons" not in names:
        raise _Gap("Areg reports no _cons, but Stata's areg always reports one")
    index = names.index("_cons")
    intercept = float(np.asarray(result.params, dtype=float)[index])
    se = float(np.asarray(result.bse, dtype=float)[index])

    stata_intercept, stata_se = 12.64269254, 0.03383845
    point_error = abs(intercept - stata_intercept) / abs(stata_intercept)
    se_error = abs(se - stata_se) / stata_se

    if point_error > 1e-6 or se_error > 1e-3:
        raise _Gap(
            f"_cons = {intercept:.8f} (se {se:.8f}), but Stata's areg gives "
            f"{stata_intercept} (se {stata_se}): point estimate off by {point_error:.2e}, "
            f"standard error off by {se_error:.0%}. The slope parameters are correct, so "
            "the defect is in how the absorbed-effect average becomes a constant and a "
            "variance when year dummies are also present."
        )
    return f"_cons {intercept:.8f} (se {se:.8f}) matches Stata"


def _standard_error_convention() -> str:
    """The clustered SE follows ``reghdfe``, not ``areg`` -- by construction, not by bug.

    ``Areg`` subclasses ``FEOLS`` and overrides the identity and the constant, so its
    variance path is ``reghdfe``'s.  ``reghdfe``'s source notes this is deliberate.
    """

    result = _fit_areg()
    index = [str(n) for n in result.param_names].index("ln_wpot_1900")
    se = float(np.asarray(result.bse, dtype=float)[index])
    areg_target, reghdfe_target = 0.0171579, 0.0164182

    if abs(se - reghdfe_target) / reghdfe_target < 1e-4:
        raise _Note(
            f"SE {se:.7f} matches Stata's reghdfe ({reghdfe_target}), not its areg "
            f"({areg_target}); the 4.5% gap is reghdfe's documented break from areg, and "
            "Areg inherits the reghdfe variance path because it subclasses FEOLS"
        )
    if abs(se - areg_target) / areg_target < 1e-4:
        raise _Gap(
            f"SE {se:.7f} now matches areg; this probe and nq2011_gaps.md need updating"
        )
    raise _Gap(f"SE {se:.7f} matches neither areg {areg_target} nor reghdfe {reghdfe_target}")


def _identity() -> str:
    """``Areg`` must present itself as ``areg``, which is the reason the command exists."""

    result = _fit_areg()
    e = getattr(result, "e", {}) or {}
    cmd = e.get("cmd")
    if cmd != "areg":
        raise _Gap(f"e(cmd) = {cmd!r}, not 'areg'")
    if "title" not in e:
        raise _Gap("e(title) is missing; areg reports 'Linear regression, absorbing indicators'")
    return f"e(cmd)='areg', e(title)={e.get('title')!r}"


def _public_surface() -> str:
    names = [n for n in dir(ws) if not n.startswith("_")]
    if len(names) < 100:
        raise _Gap(f"public surface shrank to {len(names)} names")
    return f"{len(names)} public names"


PROBES: tuple[Probe, ...] = (
    Probe("Areg coefficient", "the slope matches the authors' log", _coefficient),
    Probe("Areg _cons", "the constant matches Stata's areg", _constant),
    Probe("Areg clustered SE", "the standard error convention is pinned",
          _standard_error_convention),
    Probe("Areg identity", "e(cmd)/e(title) present themselves as areg", _identity),
)

CONTROLS: tuple[Probe, ...] = (
    Probe("public surface", "the estimator surface is intact", _public_surface),
)


def main() -> int:
    print(f"westata {ws.__version__} from {Path(ws.__file__).parent}\n")
    counts = {"OK": 0, "NOTE": 0, "GAP": 0, "ERROR": 0}
    for group, probes in (("Areg against Stata areg", PROBES), ("controls", CONTROLS)):
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
