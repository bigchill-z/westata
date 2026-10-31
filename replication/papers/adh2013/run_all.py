"""Drive one replication end to end: Stata first, then westata, then the comparison.

Why one entry point
-------------------
A replication is only reproducible if the three steps run in a fixed order against a
fixed data snapshot: the Stata capture, the westata capture, and the comparison.  Left as
three scripts a reader can run them in any order and silently compare yesterday's
``stata_results.json`` against today's westata build.  This module runs all three in
sequence, records the provenance of the data and both implementations, and writes a
single ``out/run.json`` whose ``git``/``hash`` fields make a stale comparison detectable.

Usage::

    .venv/Scripts/python.exe replication/papers/adh2013/run_all.py

The individual steps remain runnable on their own (``run_stata.py`` and so on) for the
case where Stata is unavailable on a machine and only the comparison is wanted.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT = HERE / "out"


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    import data
    import gen_do
    import run_compare
    import run_stata
    import run_westata

    started = time.time()
    steps: list[dict] = []

    print("== data ==", flush=True)
    provenance = data.ensure()
    steps.append({"step": "data", "ok": True, "archive_sha256": provenance.sha256})

    print("== do-file ==", flush=True)
    gen_do.main()
    steps.append({"step": "gen_do", "ok": True})

    print("== stata ==", flush=True)
    run_stata.main()
    steps.append({"step": "run_stata", "ok": True})

    print("== westata ==", flush=True)
    run_westata.main()
    steps.append({"step": "run_westata", "ok": True})

    print("== compare ==", flush=True)
    run_compare.main()
    steps.append({"step": "run_compare", "ok": True})

    payload = {
        "paper": "Autor, Dorn & Hanson (2013), AER 103(6): 2121-2168",
        "archive_sha256": provenance.sha256,
        "archive_url": provenance.url,
        "seconds": round(time.time() - started, 1),
        "steps": steps,
        "artifacts": {
            name: _file_hash(OUT / name)
            for name in (
                "stata_results.json",
                "westata_results.json",
                "differences.json",
                "differences.md",
                "replication.md",
                "adh2013_replication.do",
            )
            if (OUT / name).exists()
        },
    }
    (OUT / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nrun complete in {payload['seconds']}s -> {OUT / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
