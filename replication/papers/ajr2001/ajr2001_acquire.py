"""Point the AJR replication at the authors' deposited package.

The package ships no zip on a stable URL, so this records the SHA-256 of both data files
that the replication reads.  A republished dataset would change every number, and the hash
is the cheapest way to notice.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

#: The deposited package, read in place from the candidate download.
SOURCE_DIR = HERE.parents[1] / "candidates" / "raw" / "ajr2001"
DATASET = SOURCE_DIR / "20100816_replication_dataset.dta"
DOFILE = SOURCE_DIR / "rep10" / "20100816_replication10" / "20100816_replication10.do"


@dataclass
class Provenance:
    path: str
    sha256: str
    bytes: int


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure() -> dict[str, Provenance]:
    marker = DATA / "provenance.json"
    if marker.exists() and (DATA / "20100816_replication_dataset.dta").exists():
        return {k: Provenance(**v) for k, v in json.loads(
            marker.read_text(encoding="utf-8")).items()}

    for source in (DATASET, DOFILE):
        if not source.exists():
            raise FileNotFoundError(
                f"{source} not found.  The AJR replication package must be present under "
                f"{SOURCE_DIR}; it is not redistributed here."
            )

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "20100816_replication_dataset.dta").write_bytes(DATASET.read_bytes())
    (DATA / "20100816_replication10.do").write_bytes(DOFILE.read_bytes())

    record = {
        "dataset": asdict(Provenance(
            path="data/20100816_replication_dataset.dta",
            sha256=_sha(DATA / "20100816_replication_dataset.dta"),
            bytes=(DATA / "20100816_replication_dataset.dta").stat().st_size,
        )),
        "dofile": asdict(Provenance(
            path="data/20100816_replication10.do",
            sha256=_sha(DATA / "20100816_replication10.do"),
            bytes=(DATA / "20100816_replication10.do").stat().st_size,
        )),
    }
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {k: Provenance(**v) for k, v in record.items()}


def workfile() -> Path:
    ensure()
    return DATA / "20100816_replication_dataset.dta"


def main() -> int:
    for name, prov in ensure().items():
        print(f"{name}: {prov.bytes} bytes sha256={prov.sha256[:16]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
