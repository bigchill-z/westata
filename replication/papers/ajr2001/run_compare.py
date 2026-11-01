"""Compare the AJR replication and render the tables the report quotes.

The comparison here is deliberately narrow, and the report says so at the top: Stata is the
reference, not a published table.  So this module answers exactly one question -- *does
westata reproduce Stata on the authors' own specifications?* -- and reports three things
that together answer it:

* every coefficient and clustered standard error, paired by name;
* the joint test the authors report beside each column;
* the sample sizes, because a restriction applied differently on the two sides would show up
  there first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from ajr2001_specs import SPECS, SOURCE  # noqa: E402

OUT = HERE / "out"

ATOL = 1e-10
RTOL = 1e-9


def _pairs(stata: dict, westata: dict) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pair by name; westata may legitimately omit names Stata reports (and vice versa)."""

    a_names = [str(n) for n in stata["names"]]
    b_names = [str(n) for n in westata["names"]]
    shared = [n for n in b_names if n in a_names]
    missing = [n for n in b_names if n not in a_names]
    if missing:
        raise ValueError(f"westata reports parameters Stata does not: {missing[:4]}")
    order = [a_names.index(n) for n in shared]
    return (
        shared,
        np.asarray(stata["b"], dtype=float)[order],
        np.asarray(stata["se"], dtype=float)[order],
        np.asarray(westata["b"], dtype=float)[[b_names.index(n) for n in shared]],
        np.asarray(westata["se"], dtype=float)[[b_names.index(n) for n in shared]],
    )


def main() -> int:
    stata = json.loads((OUT / "stata_results.json").read_text(encoding="utf-8"))
    westata = json.loads((OUT / "westata_results.json").read_text(encoding="utf-8"))

    rows = []
    for spec in SPECS:
        names, sb, sse, wb, wse = _pairs(stata["specs"][spec.id], westata["specs"][spec.id])
        rows.append({
            "spec": spec.id,
            "table": spec.table,
            "column": spec.column,
            "label": spec.label,
            "weighted": spec.weighted,
            "west_only": spec.west_only,
            "paired": len(names),
            "nobs": westata["specs"][spec.id]["nobs"],
            "stata_nobs": stata["specs"][spec.id]["nobs"],
            "max_rel_b": float(np.max(np.abs(wb - sb) / np.maximum(np.abs(sb), 1e-12))),
            "max_rel_se": float(np.max(np.abs(wse - sse) / np.maximum(np.abs(sse), 1e-12))),
            "joint_F_stata": stata["specs"][spec.id]["scalars"].get("joint_F"),
            "joint_F_westata": westata["specs"][spec.id].get("joint_F"),
            "joint_p_stata": stata["specs"][spec.id]["scalars"].get("joint_p"),
            # The 1900 term is the paper's headline coefficient; kept separately so the
            # report can quote a number rather than only a maximum.
            "b1900_stata": float(sb[names.index("fpresence1900")]) if "fpresence1900" in names else None,
            "b1900_westata": float(wb[names.index("fpresence1900")]) if "fpresence1900" in names else None,
            "se1900_stata": float(sse[names.index("fpresence1900")]) if "fpresence1900" in names else None,
            "se1900_westata": float(wse[names.index("fpresence1900")]) if "fpresence1900" in names else None,
        })

    worst_b = max(r["max_rel_b"] for r in rows)
    worst_se = max(r["max_rel_se"] for r in rows)

    payload = {
        "meta": {
            "paper": "Acemoglu, Cantoni, Johnson & Robinson (2001), AER 92(5): 1231-1252",
            "source": SOURCE,
            "baseline_type": "Stata self-consistency (no author log, no accessible article tables)",
            "stata_version": stata["meta"].get("stata_version"),
            "westata_version": westata["meta"].get("westata_version"),
            "westata_estimator": westata["meta"].get("estimator"),
            "worst_rel_b": worst_b,
            "worst_rel_se": worst_se,
            "tolerance": f"|westata - stata| <= {ATOL:g} + {RTOL:g} * |stata|",
        },
        "specs": rows,
    }
    (OUT / "differences.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rep = [
        "# Does westata reproduce Stata on AJR's own specifications?",
        "",
        f"Reference: {SOURCE}.",
        "",
        "**This is a self-consistency replication.**  The deposited package contains no log and",
        "the AER article is paywalled, so the numbers below are checked against Stata running the",
        "authors' commands -- not against a published table.",
        "",
        "| Table | Sample | Weighted | N | 1900 coefficient (Stata / westata) | Params paired | Matched |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        sample = "western states" if r["west_only"] else "all states"
        matched = r["max_rel_b"] <= 1e-9 and r["max_rel_se"] <= 1e-9
        rep.append(
            f"| {r['table']}{r['column']} | {sample} | {'yes' if r['weighted'] else 'no'} | "
            f"{r['nobs']} | {r['b1900_stata']:.6f} / {r['b1900_westata']:.6f} | "
            f"{r['paired']} | {'yes' if matched else '**no**'} |"
        )
    (OUT / "replication.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    dif = [
        "# westata vs StataNow on the AJR specifications",
        "",
        f"Tolerance: `{ATOL:g} + {RTOL:g} * |stata|`.",
        "",
        "| Spec | Params | max rel diff, coefs | max rel diff, SEs | joint F (Stata) | joint F (westata) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        dif.append(
            f"| {r['spec']} | {r['paired']} | {r['max_rel_b']:.2e} | {r['max_rel_se']:.2e} | "
            f"{r['joint_F_stata']:.6f} | {r['joint_F_westata']:.6f} |"
        )
    (OUT / "differences.md").write_text("\n".join(dif) + "\n", encoding="utf-8")

    print("\n".join(rep))
    print()
    print("\n".join(dif))
    print()
    print(f"worst coefficient rel diff: {worst_b:.2e}")
    print(f"worst standard-error rel diff: {worst_se:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
