"""Run the whole ADHM (2020) replication: data, Stata, westata, comparison."""

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
    import adhm2020_acquire
    import adhm2020_install
    import adhm2020_run_compare
    import adhm2020_run_stata
    import adhm2020_run_westata

    started = time.time()
    print("== data ==", flush=True)
    provenance = adhm2020_acquire.ensure()

    # ivreg2 is what reproduces the paper's two-way clustered standard errors; without it
    # the replication would silently degrade to single-way clustering, which is the one
    # thing this replication exists to avoid.
    print("== stata packages ==", flush=True)
    from stata_probe import Stata as _Stata

    with _Stata() as _s:
        status = adhm2020_install.ensure(_s)
    for package, note in status.items():
        print(f"  {package}: {note}")
    if any("MISSING" in note for note in status.values()):
        raise SystemExit(
            f"ivreg2 is required but unavailable: {status}. Install it with "
            "'ssc install ivreg2 ranktest ftools' and re-run."
        )

    print("== stata (ivregress, one cluster at a time) ==", flush=True)
    adhm2020_run_stata.main()

    print("== westata (single-way; two-way recorded as unsupported) ==", flush=True)
    adhm2020_run_westata.main()

    print("== compare ==", flush=True)
    adhm2020_run_compare.main()

    payload = {
        "paper": "Autor, Dorn, Hanson & Majlesi (2020), AER 110(10): 3139-3183",
        "data_sha256": {k: v.sha256 for k, v in provenance.items()},
        "seconds": round(time.time() - started, 1),
        "artifacts": {
            name: _hash(OUT / name)
            for name in ("stata_results.json", "westata_results.json",
                         "differences.json", "differences.md", "replication.md")
            if (OUT / name).exists()
        },
    }
    (OUT / "run.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nrun complete in {payload['seconds']}s -> {OUT / 'run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
