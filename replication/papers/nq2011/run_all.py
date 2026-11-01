"""Run the whole Nunn & Qian replication: data, Stata (areg + xtreg), westata, comparison.

The Stata step estimates each specification twice -- through ``areg`` (what the authors ran)
and through ``xtreg, fe`` -- because those two disagree on the standard error and the
report's central finding is precisely *which* convention westata implements.
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
    import nq2011_acquire
    import run_compare
    import run_stata
    import nq2011_run_westata

    started = time.time()
    print("== data ==", flush=True)
    provenance = nq2011_acquire.ensure()

    print("== stata (areg and xtreg) ==", flush=True)
    run_stata.main()

    print("== westata (FEOLS and PanelOLS) ==", flush=True)
    nq2011_run_westata.main()

    print("== compare ==", flush=True)
    run_compare.main()

    payload = {
        "paper": "Nunn & Qian (2011), QJE 126(2): 593-650",
        "data_sha256": {name: prov.sha256 for name, prov in provenance.items()},
        "seconds": round(time.time() - started, 1),
        "artifacts": {
            name: _hash(OUT / name)
            for name in ("stata_results.json", "westata_results.json", "differences.json",
                         "differences.md", "replication.md")
            if (OUT / name).exists()
        },
    }
    (OUT / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nrun complete in {payload['seconds']}s -> {OUT / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
