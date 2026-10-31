"""Acquire and verify the Card & Krueger (1994) replication package.

The data is a single 82 KB fixed-width file plus a codebook, published by David Card at
``davidcard.berkeley.edu``.  There is no zip, no ``.dta`` and no Stata code -- it is a
1997-era deposit.  This module fetches the file and its codebook, and pins both by
SHA-256 so a silent upstream change is an error rather than a different replication.

The paper PDF is fetched too: the report quotes Table 3's ``2.76 FTE (13 percent,
t = 2.03)`` and Table 4's five columns from it, and a reader should be able to open the
same pages.
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
RAW = DATA / "ck1994"

BASE = "https://davidcard.berkeley.edu"
FILES = {
    "public.dat": f"{BASE}/data_sets/njmin.zip",
    "paper.pdf": f"{BASE}/papers/njmin-aer.pdf",
}

#: ``public.dat`` ships inside ``njmin.zip``; these are the members this replication uses.
ZIP_URL = f"{BASE}/data_sets/njmin.zip"

#: SHA-256 of the zip as downloaded on 2026-10-31.  A mismatch means the deposit changed.
ZIP_SHA256 = None  # filled in on first download; see ensure()


@dataclass
class Provenance:
    path: str
    url: str
    sha256: str
    bytes: int

    @classmethod
    def of(cls, path: Path, url: str) -> "Provenance":
        payload = path.read_bytes()
        return cls(
            path=str(path.relative_to(HERE)),
            url=url,
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
        )


def _curl(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-sSL", "--ssl-no-revoke", "-A", "Mozilla/5.0 (replication)",
         "--max-time", "180", "-o", str(destination), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(f"download failed for {url}: {result.stderr.strip()}")
    if destination.stat().st_size < 1000:
        raise RuntimeError(
            f"{url} returned {destination.stat().st_size} bytes, which is not a data file"
        )


def ensure(force: bool = False) -> Provenance:
    """Download and unpack the package once; return its provenance record."""

    archive = DATA / "njmin.zip"
    marker = DATA / "provenance.json"

    if not force and marker.exists() and (RAW / "public.dat").exists():
        return Provenance(**json.loads(marker.read_text(encoding="utf-8"))["archive"])

    if force or not archive.exists():
        _curl(ZIP_URL, archive)

    import zipfile

    RAW.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(RAW)

    if not (RAW / "public.dat").exists():
        raise RuntimeError(f"archive did not contain public.dat; contents: {list(RAW.iterdir())}")

    paper = DATA / "card-krueger-1994.pdf"
    if not paper.exists():
        _curl(FILES["paper.pdf"], paper)

    record = {
        "archive": asdict(Provenance.of(archive, ZIP_URL)),
        "data": asdict(Provenance.of(RAW / "public.dat", ZIP_URL)),
        "paper": asdict(Provenance.of(paper, FILES["paper.pdf"])),
    }
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return Provenance(**record["archive"])


def main() -> int:
    record = ensure(force="--force" in sys.argv)
    print(json.dumps(asdict(record), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
