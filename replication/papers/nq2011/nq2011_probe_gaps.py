"""Probe westata's panel surface for what the Nunn & Qian replication needed.

What this probe covers
----------------------
This replication exercises westata's **panel** surface: ``FEOLS`` and ``PanelOLS`` against
Stata's ``areg`` and ``xtreg, fe``.  The coefficients match everywhere, so the interesting
checks are about conventions:

* the ``areg`` versus ``xtreg`` standard-error difference is a Stata-internal
  degrees-of-freedom disagreement, not a westata defect (a ``NOTE``);
* westata's ``e(rmse)`` and its standard errors use **different** denominators -- and that
  is ``reghdfe``'s own behaviour, which westata reproduces (an ``OK``).

The second one used to be recorded as a ``GAP``: "rmse and the standard errors must share one
freedom, so the output is internally inconsistent".  That was wrong, and the correction is
the most useful thing in this file: the contract is *reproduce the reference implementation*,
not *obey the identity I expected*.  A live Stata capture of the same specification
(``reghdfe ..., absorb(isocode) vce(cluster isocode)``) reports the same pair, and moving
``rmse`` onto the nested-excluded denominator would put it 8.6 percent away from Stata.
See ``../../gaps/nq2011_gaps.md`` section 3 for the full record.

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
    """Does ``e(rmse)`` sit on the same convention as Stata ``reghdfe``?

    This probe used to report a GAP here, on the reasoning that ``rmse`` and the standard
    errors must share one denominator.  **That reasoning was wrong**, and the fix is
    recorded rather than quietly applied: a live Stata capture of the very same
    specification shows ``reghdfe`` itself reporting ``e(rmse)`` on the full-rank
    denominator while ``e(df_r)``/``bse`` follow the nested-excluded one::

        reghdfe …, absorb(isocode) vce(cluster isocode)
            e(rmse) = 0.2608790670546   -> denominator rss/rmse^2 = 1400
            e(df_r) = 129
            e(rank) = 22

    The offline fixtures show the same split with ``rmse`` held fixed while clustering
    moves ``e(df_r)`` from 148 to 29.  ``reghdfe.mata`` (:3742, :3751) computes
    ``rmse = sqrt(rss / used_df_r)`` with ``used_df_r`` on the full rank, and its source
    comments that this "breaks consistency with areg but ensures internal consistency
    between vce(robust) and vce(cluster _n)".

    So the contract is "reproduce reghdfe", not "obey the identity I expected".  Rewriting
    ``rmse`` onto the nested-excluded denominator would move it 8.6 percent away from
    Stata.  This probe therefore pins reghdfe's actual convention.
    """

    result = _fit("feols")
    e = getattr(result, "e", {}) or {}
    rmse = e.get("rmse")
    if rmse is None:
        raise _Gap("FEOLS no longer reports e(rmse), so this check cannot run")

    # 1400 = N - rank_full (all absorbed levels); 1529 = N - rank with the nested effect
    # excluded.  reghdfe (and therefore westata) uses the former for rmse.
    frame, dummies = _frame()
    n = len(frame)
    rss = float(np.sum(_residuals(result) ** 2))
    denom = rss / float(rmse) ** 2
    full_rank_denom = n - 152
    nested_excluded_denom = n - float(e.get("rank", n - 23))

    if abs(denom - full_rank_denom) < 1:
        return (
            f"e(rmse) keeps the full-rank denominator {full_rank_denom:g}, as reghdfe does, "
            f"while e(df_r)={e.get('df_r')} follows the clustered one; "
            "this is reghdfe's documented behaviour, not an inconsistency"
        )
    if abs(denom - nested_excluded_denom) < 1:
        raise _Gap(
            f"e(rmse)={rmse:.7f} now uses the nested-excluded denominator "
            f"{nested_excluded_denom:g} instead of reghdfe's {full_rank_denom}. That is the "
            "'fix' this probe used to ask for, and it would put rmse 8.6 percent away from "
            "Stata's reghdfe on the fixtures (reghdfe_firm_cluster.json). Reconcile with "
            "tests/conformance/test_stata_reghdfe.py before changing anything"
        )
    raise _Gap(
        f"e(rmse)={rmse:.7f} implies a denominator of {denom:.0f}, which is neither "
        f"reghdfe's full-rank {full_rank_denom} nor the nested-excluded "
        f"{nested_excluded_denom}"
    )


def _residuals(result):
    """Residuals of a FEOLS fit, falling back to y - xb when no residual accessor exists."""

    for attribute in ("resids", "resid", "residuals"):
        if hasattr(result, attribute):
            return np.asarray(getattr(result, attribute), dtype=float).reshape(-1)
    raise _Gap("the FEOLS result exposes no residual accessor; the probe cannot check rmse")


PROBES: tuple[Probe, ...] = (
    Probe("rmse vs se convention", "rmse follows reghdfe's full-rank denominator",
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
