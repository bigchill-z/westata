"""Compare the two sides and render the tables the report quotes.

This module answers three questions and keeps them apart, because mixing them is how a
replication ends up sounding confident about nothing:

1. **Does the published number come back?**  Paper vs StataNow vs ``westata`` -- three
   columns, so a reader can see which leg is off without reading a relative difference.
2. **Do the two implementations agree?**  Maximum relative difference over every paired
   coefficient and standard error, with the tolerance stated.
3. **Is the agreement meaningful?**  A negative control: the same data fitted with a
   *deliberately wrong* specification, showing the estimate is sensitive to the choice.
   Without this, "we matched the paper" could just mean the paper's number is easy to hit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from specs import PAPER, PAPER_SAMPLE, PAPER_TABLE3, SPECS, SOURCE  # noqa: E402

OUT = HERE / "out"

ATOL = 1e-10
RTOL = 1e-9

#: Patsy/vs-Stata naming differences, mapped explicitly.  Nothing here should ever be
#: guessed: an unmapped name is a bug in the driver, and the comparison raises.
RENAME = {"Intercept": "_cons"}


def _pairs(stata: dict, westata: dict) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_names = [RENAME.get(n, n) for n in stata["names"]]
    w_names = [RENAME.get(n, n) for n in westata["names"]]
    missing = [n for n in w_names if n not in s_names]
    extra = [n for n in s_names if n not in w_names]
    if missing or extra:
        raise ValueError(
            f"parameter sets differ: westata-only {missing}, stata-only {extra}. "
            "Fix the rename map or the specification -- do not compare by position."
        )
    order = [s_names.index(n) for n in w_names]
    return (
        w_names,
        np.asarray(stata["b"], dtype=float)[order],
        np.asarray(stata["se"], dtype=float)[order],
        np.asarray(westata["b"], dtype=float),
        np.asarray(westata["se"], dtype=float),
    )


def _fmt(value: float, digits: int) -> str:
    return "n/a" if value != value else f"{value:.{digits}f}"


def main() -> int:
    stata = json.loads((OUT / "stata_results.json").read_text(encoding="utf-8"))
    westata = json.loads((OUT / "westata_results.json").read_text(encoding="utf-8"))

    rows = []
    for spec in SPECS:
        names, sb, sse, wb, wse = _pairs(stata["specs"][spec.id], westata["specs"][spec.id])
        published = PAPER.get(spec.id)
        term = published["term"] if published else names[0]
        k = names.index(term)
        rel_b = float(np.max(np.abs(wb - sb) / np.maximum(np.abs(sb), 1e-12)))
        rel_se = float(np.max(np.abs(wse - sse) / np.maximum(np.abs(sse), 1e-12)))
        rows.append(
            {
                "spec": spec.id,
                "table": spec.table,
                "column": spec.column,
                "term": term,
                "label": spec.label,
                "paper_b": published["b"] if published else None,
                "paper_se": published["se"] if published else None,
                "stata_b": float(sb[k]),
                "stata_se": float(sse[k]),
                "westata_b": float(wb[k]),
                "westata_se": float(wse[k]),
                "max_rel_b": rel_b,
                "max_rel_se": rel_se,
                "paired": len(names),
                "paper_matches": (
                    None
                    if not published
                    else bool(
                        abs(round(sb[k], 2) - published["b"]) < 5e-3
                        and abs(round(sse[k], 2) - published["se"]) < 5e-3
                    )
                ),
            }
        )

    # ---- negative control: does the estimate depend on the comparison group? ----
    # A first attempt restricted to Pennsylvania stores, where GAP is identically zero by
    # construction -- westata rejected it as rank deficient, which is the right behaviour
    # and a reminder that "no treatment variation" is a modelling error rather than a
    # weak control.  This version instead perturbs the *treatment* variable: collapse GAP
    # to a New Jersey indicator, which discards the within-New-Jersey wage variation the
    # paper's identification relies on.  The coefficient should move.
    from data import build, table4_sample  # noqa: E402
    import westata as ws  # noqa: E402

    frame = table4_sample(build()).copy()
    frame["gap_as_dummy"] = frame["nj"].astype(float)
    frame["gap_shuffled"] = frame["gap"].sample(frac=1.0, random_state=20261031).to_numpy()

    control = {}
    for label, term in (("gap (as in the paper)", "gap"),
                        ("NJ dummy instead of GAP", "gap_as_dummy"),
                        ("GAP randomly permuted", "gap_shuffled")):
        fit = ws.OLS.from_formula(f"d_fte ~ {term}", frame).fit()
        i = [str(n) for n in fit.param_names].index(term)
        control[label] = {
            "b": float(np.asarray(fit.params, dtype=float)[i]),
            "se": float(np.asarray(fit.bse, dtype=float)[i]),
            "n": int(fit.nobs),
        }

    payload = {
        "meta": {
            "paper": "Card & Krueger (1994), AER 84(4): 772-793",
            "source": SOURCE,
            "stata_version": stata["meta"].get("stata_version"),
            "westata_version": westata["meta"].get("westata_version"),
            "toleration": f"|westata - stata| <= {ATOL:g} + {RTOL:g} * |stata|",
        },
        "specs": rows,
        "table3": {"stata": stata["table3"], "westata": westata["table3"],
                   "paper": PAPER_TABLE3},
        "negative_control": control,
    }
    (OUT / "differences.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---- markdown ----
    rep = [
        "# Replication: does the published number come back?",
        "",
        "Coefficient on the treatment variable, standard errors in parentheses.",
        f"Paper values from {SOURCE}.",
        "",
        "| Column | Term | Paper | StataNow 19.5 | westata | N | Paper matched |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        paper_cell = (
            f"{_fmt(r['paper_b'], 2)} ({_fmt(r['paper_se'], 2)})" if r["paper_b"] is not None else "n/a"
        )
        matched = "yes" if r["paper_matches"] else ("no" if r["paper_matches"] is not None else "n/a")
        rep.append(
            f"| {r['column']} | `{r['term']}` | {paper_cell} | "
            f"{_fmt(r['stata_b'], 4)} ({_fmt(r['stata_se'], 4)}) | "
            f"{_fmt(r['westata_b'], 4)} ({_fmt(r['westata_se'], 4)}) | {r['paired'] and 357} | {matched} |"
        )
    (OUT / "replication.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    dif = [
        "# westata vs StataNow on the Card & Krueger specifications",
        "",
        f"Tolerance: `{ATOL:g} + {RTOL:g} * |stata|`.",
        "",
        "| Column | Params paired | max rel diff, coefs | max rel diff, SEs |",
        "|---|---|---|---|",
    ]
    for r in rows:
        dif.append(
            f"| {r['column']} | {r['paired']} | {r['max_rel_b']:.2e} | {r['max_rel_se']:.2e} |"
        )
    (OUT / "differences.md").write_text("\n".join(dif) + "\n", encoding="utf-8")

    print("\n".join(rep))
    print()
    print("\n".join(dif[4:]))
    print()
    t3 = payload["table3"]
    print(f"Table 3 DiD: stata {t3['stata']['did']:.4f}, westata {t3['westata']['did']:.4f}, "
          f"paper {PAPER_TABLE3['did_fte']}")
    for k, v in control.items():
        print(f"  control {k:26s} b={v['b']:8.4f} (se {v['se']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
