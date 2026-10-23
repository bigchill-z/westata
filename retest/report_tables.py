"""Build the tables the final report needs, straight from results.jsonl.

Nothing here computes a new comparison: it only sorts and formats what the
domain scripts recorded, so the report cannot drift from the evidence.

Usage::

    .venv\\Scripts\\python.exe retest/report_tables.py > retest/_tables.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOMAINS = ROOT / "domains"
HARNESS = {"smoke", "smoke2"}          # lead's harness self-checks, not results
SKIP_STATUS = {"skip"}


def load():
    out = {}
    for path in sorted(DOMAINS.iterdir()):
        f = path / "results.jsonl"
        if path.name in HARNESS or not f.is_file():
            continue
        items = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        out[path.name] = items
    return out


def fmt(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    return f"{v:.2e}"


def tier_of(item):
    if item["status"] == "error":
        return "error"
    if item["status"] == "skip":
        return "skip"
    return item.get("param_tier", "NA")


def main():
    data = load()
    print("## 领域汇总表\n")
    print("| 领域 | 对比点 | pass | fail | error | skip | A | B | C | D | E | 最大系数相对差 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    total = Counter()
    tiers = Counter()
    for dom, items in data.items():
        c = Counter(i["status"] for i in items)
        t = Counter(tier_of(i) for i in items if i["status"] not in SKIP_STATUS)
        vals = [
            i["param_rel"]
            for i in items
            if i["status"] not in ("skip", "error")
            and isinstance(i.get("param_rel"), (int, float))
            and i["param_rel"] == i["param_rel"]
        ]
        total.update(c)
        tiers.update(t)
        print(
            f"| {dom} | {len(items)} | {c.get('pass',0)} | {c.get('fail',0)} | "
            f"{c.get('error',0)} | {c.get('skip',0)} | {t.get('A',0)} | {t.get('B',0)} | "
            f"{t.get('C',0)} | {t.get('D',0)} | {t.get('E',0)} | {max(vals, default=float('nan')):.2e} |"
        )
    n = sum(total.values())
    print(
        f"| **合计** | **{n}** | **{total.get('pass',0)}** | **{total.get('fail',0)}** | "
        f"**{total.get('error',0)}** | **{total.get('skip',0)}** | **{tiers.get('A',0)}** | "
        f"**{tiers.get('B',0)}** | **{tiers.get('C',0)}** | **{tiers.get('D',0)}** | "
        f"**{tiers.get('E',0)}** | — |"
    )

    graded = sum(tiers[k] for k in ("A", "B", "C", "D", "E", "NA"))
    print(f"\n可判定条目 {graded}：A {tiers.get('A',0)} ({tiers.get('A',0)/graded*100:.1f}%)、"
          f"B {tiers.get('B',0)}、C {tiers.get('C',0)}、D {tiers.get('D',0)}、E {tiers.get('E',0)}、"
          f"NA {tiers.get('NA',0)}；另 error {total.get('error',0)}、skip {total.get('skip',0)}。\n")

    print("## 按领域：A 档（≤1e-9）条目\n")
    for dom, items in data.items():
        rows = [
            i for i in items
            if i["status"] == "pass" and i.get("param_tier") == "A"
        ]
        if not rows:
            print(f"**{dom}**：无。\n")
            continue
        print(f"**{dom}**（{len(rows)} 条）\n")
        print("| 命令 | 系数相对差 | SE 相对差 |")
        print("|---|---|---|")
        for i in rows:
            print(f"| {i['name']} | {fmt(i.get('param_rel'))} | {fmt(i.get('se_rel'))} |")
        print()

    print("## 按领域：E 档（>1e-2）与 error 条目\n")
    for dom, items in data.items():
        rows = [
            i for i in items
            if tier_of(i) in ("E", "error")
        ]
        print(f"**{dom}**\n")
        if not rows:
            print("（无）\n")
            continue
        print("| 命令 | 档位 | 系数相对差 | SE 相对差 | 备注 |")
        print("|---|---|---|---|---|")
        for i in rows:
            note = "；".join(i.get("notes", [])[:2]).replace("|", "/")[:110]
            if i["status"] == "error":
                note = (i.get("error") or "")[:110].replace("|", "/")
            print(
                f"| {i['name']} | {tier_of(i)} | {fmt(i.get('param_rel'))} | "
                f"{fmt(i.get('se_rel'))} | {note} |"
            )
        print()


if __name__ == "__main__":
    main()
