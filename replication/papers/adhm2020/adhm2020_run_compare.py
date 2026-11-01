"""Compare the ADHM replication and render the tables the report quotes.

What changed once ``ivreg2`` was installed
------------------------------------------
The first version of this module was built around a limitation: stock ``ivregress`` cannot do
multiway clustering, so the paper's two-way standard errors were quoted from the authors' log
and never reproduced.  That made the central question -- "can westata produce this number" --
unanswerable, because nothing in the pipeline produced it either.

With ``ivreg2`` installed, the paper's standard errors are computed in Stata and match the
log.  The comparison therefore now has a real two-way reference:

* **``ivreg2`` two-way** (measured here) against the authors' log -- the replication claim;
* **``ivregress`` one-way** against ``ivreg2`` -- the cost of stock Stata, which is also the
  cost westata inherits by not implementing multiway clustering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from adhm2020_specs import CLUSTERS, ENDOG, PAPER, SPECS, SOURCE  # noqa: E402

OUT = HERE / "out"

ATOL = 1e-10
RTOL = 1e-9


def _coef(record: dict, name: str) -> tuple[float, float]:
    index = record["names"].index(name)
    return float(record["b"][index]), float(record["se"][index])


def main() -> int:
    stata = json.loads((OUT / "stata_results.json").read_text(encoding="utf-8"))
    westata = json.loads((OUT / "westata_results.json").read_text(encoding="utf-8"))

    rows = []
    for spec in SPECS:
        published = PAPER[spec.id]
        two_b, two_se = _coef(stata["ivreg2"][spec.id], ENDOG)
        rec = {
            "spec": spec.id, "column": spec.column, "label": spec.label,
            "controls": len(spec.regressors),
            "paper_b": published["b"], "paper_se": published["se"],
            "ivreg2_b": two_b, "ivreg2_se": two_se,
            "twoway_supported": westata["twoway"][spec.id]["supported"],
            "n_params": len(westata["specs"][spec.id][CLUSTERS[0]]["names"]),
            "clusters": {},
        }
        for cluster in CLUSTERS:
            sb, sse = _coef(stata["ivregress"][spec.id][cluster], ENDOG)
            wb, wse = _coef(westata["specs"][spec.id][cluster], ENDOG)
            rec["clusters"][cluster] = {
                "stata_b": sb, "stata_se": sse, "westata_b": wb, "westata_se": wse,
                "agree_b": abs(wb - sb) / max(abs(sb), 1e-12),
                "agree_se": abs(wse - sse) / max(abs(sse), 1e-12),
            }
        # How much a user loses by falling back to one-way clustering (the only option in
        # stock Stata, and the only option in westata).
        rec["understatement_czone"] = 1.0 - rec["clusters"][CLUSTERS[0]]["stata_se"] / two_se
        rec["understatement_cdist"] = 1.0 - rec["clusters"][CLUSTERS[1]]["stata_se"] / two_se
        rec["coefficient_matches"] = abs(two_b - published["b"]) < 1e-4
        rec["standard_error_matches"] = abs(two_se - published["se"]) / published["se"] < 1e-6
        rows.append(rec)

    worst_b = max(max(r["clusters"][c]["agree_b"] for c in CLUSTERS) for r in rows)
    worst_se = max(max(r["clusters"][c]["agree_se"] for c in CLUSTERS) for r in rows)

    payload = {
        "meta": {
            "paper": "Autor, Dorn, Hanson & Majlesi (2020), AER 110(10): 3139-3183",
            "source": SOURCE,
            "stata_version": stata["meta"].get("stata_version"),
            "westata_version": westata["meta"].get("westata_version"),
            "packages": stata["meta"].get("packages"),
            "worst_rel_b": worst_b,
            "worst_rel_se": worst_se,
            "tolerance": f"|westata - stata| <= {ATOL:g} + {RTOL:g} * |stata|",
        },
        "specs": rows,
    }
    (OUT / "differences.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rep = [
        "# Replication: coefficients and the authors' two-way standard errors",
        "",
        f"Reference: {SOURCE}.",
        "The `ivreg2` column is measured here, on the delivered data, with "
        "`cluster(czone congressionaldistrict)`.",
        "",
        "| Col | Controls | Authors' log | `ivreg2` (2-way) | westata (1-way, czone) | "
        "Coef matched | SE matched |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c = r["clusters"][CLUSTERS[0]]
        rep.append(
            f"| {r['column']} | {r['controls']} | {r['paper_b']:.5f} ({r['paper_se']:.5f}) | "
            f"{r['ivreg2_b']:.5f} ({r['ivreg2_se']:.5f}) | "
            f"{c['westata_b']:.5f} ({c['westata_se']:.5f}) | "
            f"{'yes' if r['coefficient_matches'] else '**no**'} | "
            f"{'yes' if r['standard_error_matches'] else '**no**'} |"
        )
    (OUT / "replication.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    dif = [
        "# westata vs StataNow, and what two-way clustering costs",
        "",
        f"Tolerance: `{ATOL:g} + {RTOL:g} * |stata|`.",
        "",
        "| Col | Params | rel diff vs `ivregress` (czone) | rel diff vs `ivregress` (c.dist) | "
        "2-way in westata | one-way understates the 2-way SE by (czone / c.dist) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        dif.append(
            f"| {r['column']} | {r['n_params']} | "
            f"{r['clusters'][CLUSTERS[0]]['agree_se']:.2e} | "
            f"{r['clusters'][CLUSTERS[1]]['agree_se']:.2e} | "
            f"{'yes' if r['twoway_supported'] else '**no**'} | "
            f"{r['understatement_czone']:.1%} / {r['understatement_cdist']:.1%} |"
        )
    (OUT / "differences.md").write_text("\n".join(dif) + "\n", encoding="utf-8")

    print("\n".join(rep))
    print()
    print("\n".join(dif))
    print()
    print(f"worst coefficient rel diff: {worst_b:.2e}")
    print(f"worst standard-error rel diff (single-way): {worst_se:.2e}")
    print(f"two-way clustering supported in westata: {all(r['twoway_supported'] for r in rows)}")
    print("ivreg2 two-way SE reproduces the authors' log: "
          f"{all(r['standard_error_matches'] for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
