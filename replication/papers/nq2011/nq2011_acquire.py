"""Acquire the Nunn & Qian (2011) replication package.

The files live in Harvard Dataverse as individual data files rather than a zip.  The
authors ship their own Stata log, which is what this replication uses as its reference --
so the Dataverse file ids are pinned here: re-uploading a file changes its id, and a
changed id means the reference changed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: ``(local name, dataverse file id, description)``.  Ids are from the dataset
#: ``doi:10.7910/DVN/4RUFZ0``; the panel is a tab-delimited export the authors deposited.
FILES = {
    "country_level_panel.tab": (2411901, "country-year panel, 130 countries x 12 centuries"),
    "Replication_country_level_results.do": (2411893, "the authors' do-file"),
    "Replication_country_level_results.log": (2411892, "the authors' full-precision output"),
}
BASE = "https://dataverse.harvard.edu/api/access/datafile"


@dataclass
class Provenance:
    path: str
    url: str
    sha256: str
    bytes: int


def _curl(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-sSL", "--ssl-no-revoke", "-A", "Mozilla/5.0 (replication)",
         "--max-time", "180", "-o", str(destination), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(f"download failed for {url}: {result.stderr.strip()}")


def ensure(force: bool = False) -> dict[str, Provenance]:
    marker = DATA / "provenance.json"
    if not force and marker.exists() and (DATA / "country_level_panel.tab").exists():
        record = json.loads(marker.read_text(encoding="utf-8"))
        return {name: Provenance(**item) for name, item in record.items()}

    record = {}
    for name, (file_id, _) in FILES.items():
        target = DATA / name
        if force or not target.exists():
            _curl(f"{BASE}/{file_id}", target)
        raw = target.read_bytes()
        record[name] = {
            "path": str(target.relative_to(HERE)),
            "url": f"{BASE}/{file_id}",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }

    DATA.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {name: Provenance(**item) for name, item in record.items()}


def panel() -> Path:
    ensure()
    return DATA / "country_level_panel.tab"


def author_log() -> Path:
    ensure()
    return DATA / "Replication_country_level_results.log"


def main() -> int:
    record = ensure(force="--force" in sys.argv)
    for name, prov in record.items():
        print(f"{name}: {prov.bytes} bytes  sha256={prov.sha256[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
