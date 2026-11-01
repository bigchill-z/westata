"""Probe westata's panel surface for what the Nunn & Qian replication needed.

What this probe covers
----------------------
This replication exercises westata's **panel** surface: ``FEOLS`` and ``PanelOLS`` against
Stata's ``areg`` and ``xtreg, fe``.  The coefficients match everywhere, so the interesting
checks are about conventions:

* the ``areg`` versus ``xtreg`` standard-error difference is a Stata-internal
  degrees-of-freedom disagreement, not a westata defect (a ``NOTE``);
* but westata's own ``rmse`` and standard error use **two different** freedoms, so the
  output is not internally consistent (a ``GAP``).

The second one is why this file does not simply report "no gaps": a first pass that only
compared the coefficient and the ``xtreg`` standard error missed it, because each number on
its own matches some Stata command.

Five outcomes, because "not a gap" is not one thing
---------------------------------------------------
``OK``        the call works
``NOTE``      a known, attributed difference that is not a defect (recorded, not counted)
``GAP``       westata should do this and does not
``BOUNDARY``  westata does not do this, by design
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

import westata as ws  # noqa: E402


class _Note(Exception):
    """A known difference that is not a defect."""


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
        except _Note as exc:
            return "NOTE", str(exc)
        except _Gap as exc:
            return "GAP", str(exc)
        except Exception as exc:  # noqa: BLE001
            return "ERROR", f"{type(exc).__name__}: {str(exc).splitlines()[0][:90]}"


_FRAME = None


def _frame():
    """The prepared panel, computed once -- it builds ~180 interaction columns."""

    global _FRAME
    if _FRAME is None:
        from nq2011_run_westata import prepare

        _FRAME = prepare()
    return _FRAME


def _fit(kind: str):
    """Fit one specification through both westata panel paths."""

    from nq2011_specs import SPECS

    frame, dummies = _frame()
    spec = next(s for s in SPECS if s.id == "T2_c1")
    formula = f"{spec.depvar} ~ " + " + ".join([*spec.all_regressors, *dummies])
    if kind == "feols":
        return ws.FEOLS.from_formula(f"{formula} | isocode", frame).fit(
            cov_type="cluster", cov_kwds={"cluster": "isocode"}
        )
    return ws.PanelOLS.from_formula(formula, frame, entity="isocode", time="year").fit(
        cov_type="cluster", cov_kwds={"groups": frame["isocode"]}
    )


def _coefficient(kind: str) -> str:
    result = _fit(kind)
    index = [str(n) for n in result.param_names].index("ln_wpot_1900")
    coef = float(np.asarray(result.params, dtype=float)[index])
    if abs(coef - 0.1213388) > 1e-6:
        raise _Gap(f"{kind} gives {coef!r}, not the authors' 0.1213388")
    return f"coefficient {coef:.7f}"


def _estimator_labels() -> str:
    """The result must expose ``stata_param_names``.  It need not contain ``_cons``.

    A first version of this probe demanded ``_cons`` and reported a GAP.  That was wrong:
    ``FEOLS`` absorbs the intercept into the fixed effects instead of estimating it, so its
    parameter list legitimately has no constant.  The check now asks for the attribute and
    for a length that matches ``params``, which is the property that actually matters.
    """

    result = _fit("feols")
    if not hasattr(result, "stata_param_names"):
        raise _Gap("the panel result has no stata_param_names; Stata labels are unavailable")
    names = [str(n) for n in result.stata_param_names]
    if len(names) != len(result.params):
        raise _Gap(
            f"stata_param_names has {len(names)} entries but params has "
            f"{len(result.params)}; the labels cannot be trusted"
        )
    return f"{len(names)} parameters carry Stata-style labels (no _cons: absorbed)"


def _convention_note() -> str:
    """The expected ``areg``/``xtreg`` gap, attributed rather than treated as a defect.

    Both of westata's panel paths follow ``xtreg``.  If a future version made them follow
    ``areg`` instead, the number would change by 4.5 percent with no other symptom -- so
    the probe pins the ratio and explains it.
    """

    results = {kind: _fit(kind) for kind in ("feols", "panelols")}
    ses = {}
    for kind, result in results.items():
        index = [str(n) for n in result.param_names].index("ln_wpot_1900")
        ses[kind] = float(np.asarray(result.bse, dtype=float)[index])

    if abs(ses["feols"] - ses["panelols"]) / ses["feols"] > 1e-9:
        raise _Gap(
            f"FEOLS and PanelOLS disagree on the standard error "
            f"({ses['feols']:.7f} vs {ses['panelols']:.7f}); they are meant to implement "
            "the same convention"
        )

    # xtreg gives 0.0164182; areg gives 0.0171579.  Westata must sit on the xtreg value.
    expected_xtreg = 0.0164182
    if abs(ses["feols"] - expected_xtreg) / expected_xtreg > 1e-4:
        raise _Gap(
            f"the panel standard error is {ses['feols']:.7f}, not xtreg's {expected_xtreg}; "
            "the convention documented in ../../gaps/nq2011_gaps.md no longer holds"
        )
    raise _Note(
        f"SE {ses['feols']:.7f} matches Stata's xtreg (0.0164182). "
        "Stata's areg reports 0.0171579, a 4.5% gap from a degrees-of-freedom convention; "
        "that is Stata's internal disagreement, not a westata defect"
    )


def _rmse_se_convention_consistent() -> str:
    """Does ``e(rmse)`` use the same degrees of freedom as the standard errors?

    On this data ``areg`` and ``xtreg, fe`` disagree because they divide the residual
    variance by different denominators (``N-152`` versus ``N-23``).  westata should pick
    one convention and use it consistently.  It currently does not:

    * ``e(rmse)`` follows ``areg`` (0.2608791, the N-152 denominator);
    * ``bse`` follows ``xtreg`` (0.0164182, the N-23 denominator).

    Both numbers match a real Stata command, so neither looks wrong in isolation, and the
    inconsistency is only visible when you check that they must come from the same model.
    """

    result = _fit("feols")
    e = getattr(result, "e", {}) or {}
    rmse = e.get("rmse")
    if rmse is None:
        raise _Gap("FEOLS no longer reports e(rmse), so this check cannot run")

    # areg divides by N-152 = 1400; xtreg by N-23 = 1529.  Which one does rmse use?
    frame, dummies = _frame()
    n = len(frame)
    rss = float(np.sum(_residuals(result) ** 2))
    denom = rss / float(rmse) ** 2
    areg_denom = n - 152
    xtreg_denom = n - 23
    uses_areg = abs(denom - areg_denom) < 1
    uses_xtreg = abs(denom - xtreg_denom) < 1

    if uses_areg:
        raise _Gap(
            f"e(rmse)={rmse:.7f} implies a denominator of {denom:.0f} = N-152, the areg "
            f"convention, while the standard error follows xtreg (denominator N-23={xtreg_denom}). "
            "The two are computed under different freedoms, so rmse, ll and r2_a are not "
            "consistent with bse. Pick one convention and thread it through both."
        )
    if uses_xtreg:
        return f"rmse and bse both use the xtreg denominator {xtreg_denom}"
    raise _Gap(
        f"e(rmse)={rmse:.7f} implies a denominator of {denom:.0f}, which is neither "
        f"areg's {areg_denom} nor xtreg's {xtreg_denom}"
    )


def _residuals(result):
    """Residuals of a FEOLS fit, falling back to y - xb when no residual accessor exists."""

    for attribute in ("resids", "resid", "residuals"):
        if hasattr(result, attribute):
            return np.asarray(getattr(result, attribute), dtype=float).reshape(-1)
    raise _Gap("the FEOLS result exposes no residual accessor; the probe cannot check rmse")


PROBES: tuple[Probe, ...] = (
    Probe("rmse vs se convention", "rmse and the standard errors share one df convention",
          _rmse_se_convention_consistent),
    Probe("areg-equivalent (FEOLS)", "FEOLS reproduces the authors' coefficient",
          lambda: _coefficient("feols")),
    Probe("xtreg-equivalent (PanelOLS)", "PanelOLS reproduces the same coefficient",
          lambda: _coefficient("panelols")),
    Probe("Stata labels", "the panel result exposes Stata-style parameter names",
          _estimator_labels),
    Probe("areg versus xtreg", "the standard-error convention is pinned and attributed",
          _convention_note),
)


def _control_public_surface() -> str:
    names = [n for n in dir(ws) if not n.startswith("_")]
    if len(names) < 100:
        raise _Gap(f"public surface shrank to {len(names)} names")
    return f"{len(names)} public names"


CONTROLS: tuple[Probe, ...] = (
    Probe("public surface", "the estimator surface is intact", _control_public_surface),
)


def main() -> int:
    print(f"westata {ws.__version__} from {Path(ws.__file__).parent}\n")
    counts = {"OK": 0, "NOTE": 0, "GAP": 0, "ERROR": 0}
    for group, probes in (("estimator surface", PROBES[:3]), ("agreement", PROBES[3:]),
                          ("controls", CONTROLS)):
        print(f"-- {group} --")
        for probe in probes:
            status, detail = probe.run()
            counts[status] += 1
            print(f"  [{status:5s}] {probe.name:26s} {detail}")
        print()

    print(f"{counts['OK']} ok, {counts['NOTE']} note, {counts['GAP']} gap, "
          f"{counts['ERROR']} error")
    print(
        "\nA NOTE is a known, attributed difference rather than a defect: westata follows "
        "Stata's xtreg/reghdfe convention, and areg differs from those by 4.5 percent."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
