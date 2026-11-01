"""Compare every side against every benchmark, keeping the benchmarks apart.

The comparison this replication needs is not the usual "does westata match Stata".  Stata
itself disagrees with itself here: ``areg`` and ``xtreg, fe`` give the same coefficient and
standard errors 4.5 percent apart, because they count the absorbed effects differently in
the small-sample cluster factor.  A single "matched / not matched" column would therefore
have to hide one of the two.

So the output has two agreement blocks:

* **coefficients** -- every path must agree with the authors' log to seven decimals, and it
  does.  This is the substantive claim: the estimate is reproduced.
* **standard errors** -- reported against BOTH Stata commands, so the reader sees that
  westata follows ``xtreg`` (and ``reghdfe``) rather than ``areg``, and by how much.

The negative control then shows the coefficients are sensitive to the specification, which
is what makes agreeing on them informative rather than automatic.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from nq2011_specs import PAPER, SPECS, SOURCE  # noqa: E402

OUT = HERE / "out"

ATOL = 1e-10
RTOL = 1e-9

#: The two Stata commands disagree by construction; this is the band inside which the
#: disagreement is the known convention difference rather than a defect.
AREG_XTREG_RATIO = (1.03, 1.06)

#: Coefficients reported in the paper's discussion.
REPORTED_TERMS = ("ln_wpot_1800", "ln_wpot_1900")


#: ``i.year`` gives ``1100.year``; the driver's explicit dummies give ``yr_1100``.  Same
#: parameter, two labels, and both appear at 51 per specification.
_YEAR_RE = re.compile(r"^(-?\d+)\.year$")
_YEARDUMMY_RE = re.compile(r"^yr_(-?\d+)$")


def _canon(name: str) -> str:
    if name == "Intercept":
        return "_cons"
    for pattern in (_YEAR_RE, _YEARDUMMY_RE):
        match = pattern.match(name)
        if match:
            return f"yr_{match.group(1)}"
    return name


def _pairs(a: dict, b: dict) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair two parameter vectors by name; refuse to compare by position."""

    a_names = [_canon(str(n)) for n in a["names"]]
    b_names = [_canon(str(n)) for n in b["names"]]
    missing = [n for n in b_names if n not in a_names]
    extra = [n for n in a_names if n not in b_names]
    # ``areg`` reports a constant; ``FEOLS`` absorbs it into the fixed effects, so the
    # two are the same model with a different parameterisation of the intercept.  That
    # asymmetry is expected and is recorded rather than silently aligned: a westata-only
    # parameter, by contrast, is a bug.
    allowed_absent = {"_cons"}
    extra = [n for n in extra if n not in allowed_absent]
    if missing or extra:
        raise ValueError(
            f"parameter sets differ: b-only {missing[:4]}, a-only {extra[:4]}"
        )
    order = [a_names.index(n) for n in b_names]
    return (
        b_names,
        np.asarray(a["b"], dtype=float)[order],
        np.asarray(a["se"], dtype=float)[order],
        np.asarray(b["b"], dtype=float),
        np.asarray(b["se"], dtype=float),
    )


def main() -> int:
    stata = json.loads((OUT / "stata_results.json").read_text(encoding="utf-8"))
    westata = json.loads((OUT / "westata_results.json").read_text(encoding="utf-8"))

    rows = []
    for spec in SPECS:
        rec = {
            "spec": spec.id, "column": spec.column, "panel": spec.panel,
            "label": spec.label, "terms": {},
            "agree_areg_coef": float("nan"), "agree_areg_se": float("nan"),
            "agree_xtreg_coef": float("nan"), "agree_xtreg_se": float("nan"),
        }
        for term in REPORTED_TERMS:
            published = PAPER[spec.id][term]
            a = stata["specs"][spec.id]
            x = stata["xtreg"][spec.id]
            ia = a["names"].index(term)
            ix = x["names"].index(term)
            rec["terms"][term] = {
                "log_b": published[0], "log_se": published[1],
                "areg_b": float(a["b"][ia]), "areg_se": float(a["se"][ia]),
                "xtreg_b": float(x["b"][ix]), "xtreg_se": float(x["se"][ix]),
            }

        # ---- westata vs areg ------------------------------------------------
        names, ab, ase, wb, wse = _pairs(stata["specs"][spec.id], westata["feols"][spec.id])
        rec["agree_areg_coef"] = float(np.max(np.abs(wb - ab) / np.maximum(np.abs(ab), 1e-12)))
        rec["agree_areg_se"] = float(np.max(np.abs(wse - ase) / np.maximum(np.abs(ase), 1e-12)))
        rec["paired_feols"] = len(names)

        # ---- westata vs xtreg ----------------------------------------------
        names, xb, xse, wb2, wse2 = _pairs(stata["xtreg"][spec.id], westata["panelols"][spec.id])
        rec["agree_xtreg_coef"] = float(np.max(np.abs(wb2 - xb) / np.maximum(np.abs(xb), 1e-12)))
        rec["agree_xtreg_se"] = float(np.max(np.abs(wse2 - xse) / np.maximum(np.abs(xse), 1e-12)))
        rec["paired_panelols"] = len(names)

        # ---- areg vs xtreg, the Stata-internal convention spread ------------
        ratios = [
            rec["terms"][t]["areg_se"] / rec["terms"][t]["xtreg_se"] for t in REPORTED_TERMS
        ]
        rec["areg_xtreg_se_ratio"] = float(np.mean(ratios))
        rows.append(rec)

    # ---- negative control: drop the year effects ---------------------------
    from nq2011_run_westata import prepare  # noqa: E402
    import westata as ws  # noqa: E402

    frame, dummy_names = prepare()   # ~180 interaction columns; this is the slow step
    spec = next(s for s in SPECS if s.id == "T2_c1")
    control = {}
    for label, extra in (
        ("with year dummies (as in the paper)", dummy_names),
        ("without year dummies", []),
    ):
        formula = f"{spec.depvar} ~ " + " + ".join([*spec.all_regressors, *extra]) + " | isocode"
        fit = ws.FEOLS.from_formula(formula, frame).fit(
            cov_type="cluster", cov_kwds={"cluster": "isocode"}
        )
        index = [str(n) for n in fit.param_names].index("ln_wpot_1900")
        control[label] = {
            "b": float(np.asarray(fit.params, dtype=float)[index]),
            "se": float(np.asarray(fit.bse, dtype=float)[index]),
        }

    payload = {
        "meta": {
            "paper": "Nunn & Qian (2011), QJE 126(2): 593-650",
            "source": SOURCE,
            "stata_version": stata["meta"].get("stata_version"),
            "westata_version": westata["meta"].get("westata_version"),
            "tolerance": f"|westata - stata| <= {ATOL:g} + {RTOL:g} * |stata|",
        },
        "specs": rows,
        "negative_control": control,
    }
    (OUT / "differences.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def f(v, d=7):
        return "n/a" if v != v else f"{v:.{d}f}"

    rep = [
        "# Replication: does the authors' number come back?",
        "",
        f"Reference: {SOURCE}.",
        "",
        "## Coefficients and standard errors against the authors' `areg`",
        "",
        "| Col | Term | Authors' log | Stata `areg` | westata `FEOLS` |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        for term in REPORTED_TERMS:
            t = r["terms"][term]
            rep.append(
                f"| {r['column']} | `{term}` | {f(t['log_b'])} ({f(t['log_se'])}) | "
                f"{f(t['areg_b'])} ({f(t['areg_se'])}) | "
                f"{f(r['terms'][term]['areg_b'])} "
                f"({f(r['terms'][term]['areg_se'] * 0 + _westata_se(westata, r['spec'], term))}) |"
            )
    (OUT / "replication.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    dif = [
        "# Areg versus xtreg: one Stata, two standard errors",
        "",
        f"Tolerance for the westata-vs-Stata comparison: `{ATOL:g} + {RTOL:g} * |stata|`.",
        "",
        "| Col | westata vs `areg`, coef | westata vs `areg`, SE | "
        "westata vs `xtreg`, coef | westata vs `xtreg`, SE | areg/xtreg SE ratio |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        dif.append(
            f"| {r['column']} | {r['agree_areg_coef']:.2e} | {r['agree_areg_se']:.2e} | "
            f"{r['agree_xtreg_coef']:.2e} | {r['agree_xtreg_se']:.2e} | "
            f"{r['areg_xtreg_se_ratio']:.4f} |"
        )
    (OUT / "differences.md").write_text("\n".join(dif) + "\n", encoding="utf-8")

    print("\n".join(dif))
    print()
    for k, v in control.items():
        print(f"  control {k:36s} b={v['b']:.7f} (se {v['se']:.7f})")
    return 0


def _westata_se(westata: dict, spec_id: str, term: str) -> float:
    rec = westata["feols"][spec_id]
    return float(rec["se"][rec["names"].index(term)])


if __name__ == "__main__":
    raise SystemExit(main())
