"""Run the whole AJR replication: data, Stata baseline, westata, comparison."""

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
    import ajr2001_acquire
    import ajr2001_run_stata
    import ajr2001_run_westata
    import run_compare

    started = time.time()
    print("== data ==", flush=True)
    provenance = ajr2001_acquire.ensure()

    print("== stata (the self-consistency baseline) ==", flush=True)
    ajr2001_run_stata.main()

    print("== westata ==", flush=True)
    ajr2001_run_westata.main()

    print("== compare ==", flush=True)
    run_compare.main()

    payload = {
        "paper": "Acemoglu, Cantoni, Johnson & Robinson (2001), AER 92(5): 1231-1252",
        "baseline_type": "Stata self-consistency: the package has no log and the article is paywalled",
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
