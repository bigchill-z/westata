"""Compare the two sides against the authors' own log and against each other.

Three questions, kept apart:

1. **Does the published number come back?**  Here "published" means the authors'
   ``log/czone_analysis.log``, which records seven decimals -- so agreement is checked at
   1e-6 rather than against a rounded table.
2. **Do Stata and westata agree?**  Maximum relative difference over every paired
   coefficient and standard error.  Each regression carries 48 state dummies, so this is
   ~50 parameters per specification, not one.
3. **Is the comparison meaningful?**  A negative control: refit without the state dummies,
   which is a different (and wrong) specification.  If the estimate barely moved, then
   matching the paper would say little about the design matrix.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from specs import PAPER, SPECS, SOURCE  # noqa: E402

OUT = HERE / "out"

ATOL = 1e-10
RTOL = 1e-9

#: Same parameter, three possible labels depending on how the dummies were generated:
#: ``i.statefip`` gives ``4.statefip``, the authors' ``xi`` gives ``_Istatefip_4``, and
#: the driver's explicit columns give ``st_4``.  Canonicalising all three to ``st_<level>``
#: is what lets the comparison be name-based and total.
RENAME = {"Intercept": "_cons"}

_STATE_PATTERNS = (
    re.compile(r"^(-?\d+)\.statefip$"),          # i.statefip
    re.compile(r"^_Istatefip_(-?\d+)$"),          # xi-generated
    re.compile(r"^st_(-?\d+)$"),                  # explicit dummy columns
)


def _canon(name: str, prefix: str = "") -> str:
    name = RENAME.get(name, name)
    for pattern in _STATE_PATTERNS:
        match = pattern.match(name)
        if match:
            return f"st_{match.group(1)}"
    return name


def _pairs(stata: dict, westata: dict) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_names = [_canon(n, "_Istatefip_") for n in stata["names"]]
    w_names = [_canon(n, "_Istatefip_") for n in westata["names"]]
    missing = [n for n in w_names if n not in s_names]
    extra = [n for n in s_names if n not in w_names]
    if missing or extra:
        raise ValueError(
            f"parameter sets differ: westata-only {missing[:5]}, stata-only {extra[:5]}. "
            "Fix the rename map; do not compare by position."
        )
    order = [s_names.index(n) for n in w_names]
    return (
        w_names,
        np.asarray(stata["b"], dtype=float)[order],
        np.asarray(stata["se"], dtype=float)[order],
        np.asarray(westata["b"], dtype=float),
        np.asarray(westata["se"], dtype=float),
    )


def main() -> int:
    stata = json.loads((OUT / "stata_results.json").read_text(encoding="utf-8"))
    westata = json.loads((OUT / "westata_results.json").read_text(encoding="utf-8"))

    rows = []
    for spec in SPECS:
        names, sb, sse, wb, wse = _pairs(stata["specs"][spec.id], westata["specs"][spec.id])
        published = PAPER[spec.id]
        k = names.index(spec.term)
        rows.append({
            "spec": spec.id,
            "table": spec.table,
            "row": spec.row,
            "label": spec.label,
            "term": spec.term,
            "paper_b": published["b"],
            "paper_se": published["se"],
            "paper_N": published["N"],
            "stata_b": float(sb[k]),
            "stata_se": float(sse[k]),
            "westata_b": float(wb[k]),
            "westata_se": float(wse[k]),
            "stata_N": stata["specs"][spec.id]["nobs"],
            "westata_N": westata["specs"][spec.id]["nobs"],
            "max_rel_b": float(np.max(np.abs(wb - sb) / np.maximum(np.abs(sb), 1e-12))),
            "max_rel_se": float(np.max(np.abs(wse - sse) / np.maximum(np.abs(sse), 1e-12))),
            "paired": len(names),
        })

    # ---- negative control: drop the state dummies -----------------------------
    # This is the same dependent variable and treatment, but without the state fixed
    # effects the paper partials out.  If the coefficient did not move, the state dummies
    # would be doing nothing and the agreement above would be a weak test.
    from acquire import workfile  # noqa: E402
    import pandas as pd  # noqa: E402
    import westata as ws  # noqa: E402
    from run_westata import _prepare  # noqa: E402

    raw = pd.read_stata(workfile(), convert_categoricals=False)
    control = {}
    target = next(s for s in SPECS if s.id == "T3_1")
    frame, _ = _prepare(target, raw)
    for label, formula in (
        ("with state dummies (as in the paper)",
         f"{target.depvar} ~ {target.term} + " + " + ".join(
             c for c in frame.columns if c.startswith("st_"))),
        ("without state dummies", f"{target.depvar} ~ {target.term}"),
    ):
        fit = ws.OLS.from_formula(formula, frame, weights=frame["timepwt48"],
                                  weight_type="aweight").fit()
        i = [str(n) for n in fit.param_names].index(target.term)
        control[label] = {
            "b": float(np.asarray(fit.params, dtype=float)[i]),
            "se": float(np.asarray(fit.bse, dtype=float)[i]),
            "n": int(fit.nobs),
        }

    payload = {
        "meta": {
            "paper": "Autor & Dorn (2013), AER 103(5): 1553-1597",
            "source": SOURCE,
            "stata_version": stata["meta"].get("stata_version"),
            "westata_version": westata["meta"].get("westata_version"),
            "tolerance": f"|westata - stata| <= {ATOL:g} + {RTOL:g} * |stata|",
        },
        "specs": rows,
        "negative_control": control,
    }
    (OUT / "differences.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(v, d):
        return "n/a" if v != v else f"{v:.{d}f}"

    rep = [
        "# Replication: does the authors' own number come back?",
        "",
        f"Reference: {SOURCE}.",
        "The log records seven decimals, so agreement is checked at 1e-6 -- far tighter than a printed table allows.",
        "",
        "| Spec | Term | Authors' log | StataNow | westata | N | Matched |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        matched = abs(r["stata_b"] - r["paper_b"]) < 1e-6 and abs(r["stata_se"] - r["paper_se"]) < 1e-6
        rep.append(
            f"| {r['table']}{r['row']} | `{r['term']}` | "
            f"{fmt(r['paper_b'], 7)} ({fmt(r['paper_se'], 7)}) | "
            f"{fmt(r['stata_b'], 7)} ({fmt(r['stata_se'], 7)}) | "
            f"{fmt(r['westata_b'], 7)} ({fmt(r['westata_se'], 7)}) | "
            f"{r['stata_N']} | {'yes' if matched else '**no**'} |"
        )
    (OUT / "replication.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    dif = [
        "# westata vs StataNow on the Autor & Dorn specifications",
        "",
        f"Tolerance: `{ATOL:g} + {RTOL:g} * |stata|`.",
        "Each specification includes ~48 state dummies, so the paired count is far above one.",
        "",
        "| Spec | Params paired | max rel diff, coefs | max rel diff, SEs |",
        "|---|---|---|---|",
    ]
    for r in rows:
        dif.append(f"| {r['spec']} | {r['paired']} | {r['max_rel_b']:.2e} | {r['max_rel_se']:.2e} |")
    (OUT / "differences.md").write_text("\n".join(dif) + "\n", encoding="utf-8")

    print("\n".join(rep))
    print()
    print("\n".join(dif[5:]))
    print()
    for k, v in control.items():
        print(f"  control {k:40s} b={v['b']:8.4f} (se {v['se']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
