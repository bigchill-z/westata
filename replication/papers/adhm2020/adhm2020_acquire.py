"""Acquire the Autor-Dorn-Hanson-Majlesi (2020) replication package.

The authors deposit a complete package (do-files, data, and their own full-precision logs).
The data file this replication uses is ``dta/house_2002_2016.dta``: congressional districts
observed in 2002 and 2010.

The package ships no zip on a stable URL, so the provenance record is the SHA-256 of the
``.dta`` itself: a republished data file would change the numbers, and this is the cheapest
way to notice.
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
SOURCE_DIR = (
    HERE.parents[1] / "candidates" / "raw" / "adhm2020" / "2-FinalDataPackage"
)

WORKFILE = "dta/house_2002_2016.dta"
AUTHOR_LOG = "log/house_2002_2016_analysis.log"


@dataclass
class Provenance:
    path: str
    sha256: str
    bytes: int
    source: str


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure() -> dict[str, Provenance]:
    """Record provenance for the data file and the authors' log."""

    marker = DATA / "provenance.json"
    if marker.exists() and (DATA / "house_2002_2016.dta").exists():
        return {k: Provenance(**v) for k, v in json.loads(
            marker.read_text(encoding="utf-8")).items()}

    for relative in (WORKFILE, AUTHOR_LOG):
        source = SOURCE_DIR / relative
        if not source.exists():
            raise FileNotFoundError(
                f"{source} not found.  The ADHM package must be present under "
                f"{SOURCE_DIR}; it is not redistributed here."
            )
        target = DATA / Path(relative).name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    record = {
        "dta": asdict(Provenance(
            path="data/house_2002_2016.dta",
            sha256=_sha(DATA / "house_2002_2016.dta"),
            bytes=(DATA / "house_2002_2016.dta").stat().st_size,
            source=str(SOURCE_DIR.relative_to(HERE.parents[1].parent)),
        )),
        "log": asdict(Provenance(
            path="data/house_2002_2016_analysis.log",
            sha256=_sha(DATA / "house_2002_2016_analysis.log"),
            bytes=(DATA / "house_2002_2016_analysis.log").stat().st_size,
            source=str(SOURCE_DIR.relative_to(HERE.parents[1].parent)),
        )),
    }
    DATA.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {k: Provenance(**v) for k, v in record.items()}


def workfile() -> Path:
    ensure()
    return DATA / "house_2002_2016.dta"


def author_log() -> Path:
    ensure()
    return DATA / "house_2002_2016_analysis.log"


def main() -> int:
    for name, prov in ensure().items():
        print(f"{name}: {prov.bytes} bytes sha256={prov.sha256[:16]}...")
    print(f"\nworkfile: {workfile()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
