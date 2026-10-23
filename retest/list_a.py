"""Print every domain's A-tier command names, one line per domain.

Usage::

    PYTHONIOENCODING=utf-8 .venv\\Scripts\\python.exe retest/list_a.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "domains"
HARNESS = {"smoke", "smoke2"}

for path in sorted(ROOT.iterdir()):
    results = path / "results.jsonl"
    if path.name in HARNESS or not results.is_file():
        continue
    items = [json.loads(l) for l in results.read_text(encoding="utf-8").splitlines() if l.strip()]
    a = [i["name"] for i in items if i["status"] == "pass" and i.get("param_tier") == "A"]
    print(f"== {path.name}: {len(items)} items, A={len(a)}")
    for name in a:
        print("   -", name)
