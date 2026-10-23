"""List every failing or badly-matching item across domains, for the report.

Usage::

    .venv\\Scripts\\python.exe retest/failures.py
    .venv\\Scripts\\python.exe retest/failures.py --tier E D --status fail error
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOMAINS = ROOT / "domains"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", nargs="*", default=None)
    ap.add_argument("--status", nargs="*", default=None)
    ap.add_argument("--domain", nargs="*", default=None)
    args = ap.parse_args()

    for path in sorted(DOMAINS.iterdir()):
        results = path / "results.jsonl"
        if not results.is_file():
            continue
        if args.domain and path.name not in args.domain:
            continue
        items = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines() if line.strip()]
        for d in items:
            tier = d.get("param_tier", "NA")
            status = d["status"]
            if args.tier and tier not in args.tier:
                continue
            if args.status and status not in args.status:
                continue
            pr = d.get("param_rel")
            se = d.get("se_rel")

            def f(v):
                try:
                    return f"{float(v):.3e}"
                except (TypeError, ValueError):
                    return "n/a"

            print(
                f"{path.name:18s} {status:5s} {tier:2s} {d['name'][:62]:62s} "
                f"p={f(pr):9s} se={f(se):9s} {(d.get('error') or '')[:70]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
