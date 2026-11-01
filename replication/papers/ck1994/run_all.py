"""Run the whole Card & Krueger replication: data, Stata, westata, comparison.

One entry point so the three steps cannot be run in the wrong order, and so a reader has a
single command to reproduce every number in ``REPORT.md``.
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


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    import ck1994_acquire
    import run_compare
    import run_stata
    import ck1994_run_westata

    started = time.time()
    print("== data ==", flush=True)
    provenance = ck1994_acquire.ensure()

    print("== stata ==", flush=True)
    run_stata.main()

    print("== westata ==", flush=True)
    ck1994_run_westata.main()

    print("== compare ==", flush=True)
    run_compare.main()

    payload = {
        "paper": "Card & Krueger (1994), AER 84(4): 772-793",
        "data_sha256": provenance.sha256,
        "data_url": provenance.url,
        "seconds": round(time.time() - started, 1),
        "artifacts": {
            name: _hash(OUT / name)
            for name in (
                "stata_results.json",
                "westata_results.json",
                "differences.json",
                "differences.md",
                "replication.md",
            )
            if (OUT / name).exists()
        },
    }
    (OUT / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nrun complete in {payload['seconds']}s -> {OUT / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
