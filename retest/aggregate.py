"""Assemble every domain's results into one machine-readable table.

Reads ``retest/domains/*/results.jsonl`` (written by each domain's ``run.py``
through ``stata_probe.Record``) and emits ``retest/aggregate.json`` plus a
markdown table fragment the report can quote.  It never edits domain results.

Usage::

    .venv\\Scripts\\python.exe retest/aggregate.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOMAINS = ROOT / "domains"
OUT = ROOT / "aggregate.json"

TIER_ORDER = ["A", "B", "C", "D", "E", "NA"]


def load_domain(path: Path) -> list[dict]:
    results = path / "results.jsonl"
    if not results.exists():
        return []
    items = []
    for line in results.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def main() -> int:
    if not DOMAINS.is_dir():
        print(f"no domains directory at {DOMAINS}", file=sys.stderr)
        return 1
    payload: dict[str, list[dict]] = {}
    for path in sorted(DOMAINS.iterdir()):
        if not path.is_dir():
            continue
        items = load_domain(path)
        if items:
            payload[path.name] = items

    total = sum(len(v) for v in payload.values())
    overall: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    per_domain = {}
    for domain, items in payload.items():
        counts = Counter(i["status"] for i in items)
        overall.update(counts)
        dtier = Counter(i.get("param_tier", "NA") for i in items)
        for i in items:
            if i["status"] != "skip":
                tiers[i.get("param_tier", "NA")] += 1
        per_domain[domain] = {
            "counts": dict(counts),
            "tiers": dict(dtier),
            "max_param_rel": max(
                (
                    i["param_rel"]
                    for i in items
                    if isinstance(i.get("param_rel"), (int, float))
                    and i["param_rel"] == i["param_rel"]
                    and i["status"] not in ("skip", "error")
                ),
                default=None,
            ),
        }

    agg = {
        "total": total,
        "status_counts": dict(overall),
        "tier_counts": dict(tiers),
        "per_domain": per_domain,
        "domains": payload,
    }
    OUT.write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"{'domain':22s} {'n':>4s} {'pass':>5s} {'fail':>5s} {'err':>4s} {'skip':>5s} "
          f"{'A':>4s} {'B':>4s} {'C':>4s} {'D':>4s} {'E':>4s}")
    for domain, d in per_domain.items():
        c = d["counts"]
        t = d["tiers"]
        print(
            f"{domain:22s} {sum(c.values()):4d} {c.get('pass',0):5d} {c.get('fail',0):5d} "
            f"{c.get('error',0):4d} {c.get('skip',0):5d} "
            f"{t.get('A',0):4d} {t.get('B',0):4d} {t.get('C',0):4d} {t.get('D',0):4d} {t.get('E',0):4d}"
        )
    print(f"\ntotal {total}: {dict(overall)}")
    print(f"tiers: {dict(tiers)}")
    print(f"written -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
