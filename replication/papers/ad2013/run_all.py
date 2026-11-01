"""Run the whole Autor & Dorn replication: data, do-file, Stata, westata, comparison."""

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
    import ad2013_acquire
    import gen_do
    import run_compare
    import run_stata
    import ad2013_run_westata

    started = time.time()
    print("== data ==", flush=True)
    provenance = ad2013_acquire.ensure()

    print("== do-file ==", flush=True)
    gen_do.main()

    print("== stata ==", flush=True)
    run_stata.main()

    print("== westata ==", flush=True)
    ad2013_run_westata.main()

    print("== compare ==", flush=True)
    run_compare.main()

    payload = {
        "paper": "Autor & Dorn (2013), AER 103(5): 1553-1597",
        "archive_sha256": provenance.sha256,
        "archive_url": provenance.url,
        "seconds": round(time.time() - started, 1),
        "artifacts": {
            name: _hash(OUT / name)
            for name in (
                "stata_results.json", "westata_results.json", "differences.json",
                "differences.md", "replication.md", "ad2013_replication.do",
            )
            if (OUT / name).exists()
        },
    }
    (OUT / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nrun complete in {payload['seconds']}s -> {OUT / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
