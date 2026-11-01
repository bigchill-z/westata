"""Acquire the Autor & Dorn (2013) replication archive.

The package lives on David Dorn's site as a single zip.  The author's log of every
regression is inside it, which is what makes this replication's reference values
full-precision rather than rounded table entries -- so the archive is pinned by SHA-256:
a republished zip would silently change what "the paper's number" means.
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
#: The zip unpacks to this top-level folder (it is not a bare directory of files).
UNPACKED = "Autor-Dorn-LowSkillServices-FileArchive.zip Folder"

URL = "https://www.ddorn.net/data/Autor-Dorn-LowSkillServices-FileArchive.zip"
#: SHA-256 as downloaded on 2026-10-31.
SHA256 = "a4c9b8200a39d357c1141cfe3cb8b54b7ed586dcf9768e3b1c6689d79c45be0b"
BYTES = 2_741_391

WORKFILE = "dta/workfile2012.dta"
#: The author's own full-precision output; the reference for every replicated number.
AUTHOR_LOG = "log/czone_analysis.log"


@dataclass
class Provenance:
    path: str
    url: str
    sha256: str
    bytes: int


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _curl(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["curl", "-sSL", "--ssl-no-revoke", "-A", "Mozilla/5.0 (replication)",
         "--max-time", "300", "-o", str(destination), url],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(f"download failed for {url}: {result.stderr.strip()}")


def ensure(force: bool = False) -> Provenance:
    archive = DATA / "ad2013.zip"
    marker = DATA / "provenance.json"
    root = DATA / UNPACKED

    if not force and marker.exists() and (root / WORKFILE).exists():
        return Provenance(**json.loads(marker.read_text(encoding="utf-8"))["archive"])

    if force or not archive.exists():
        _curl(URL, archive)

    digest = _sha(archive)
    if SHA256 and digest != SHA256:
        raise RuntimeError(
            f"archive checksum changed: expected {SHA256}, got {digest}. "
            "The authors may have republished; the author's log inside is this "
            "replication's reference, so re-check the tables before proceeding."
        )

    import zipfile

    DATA.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(DATA)
    if not (root / WORKFILE).exists():
        found = list(DATA.rglob(Path(WORKFILE).name))
        raise RuntimeError(
            f"archive did not contain {WORKFILE}; directories found: "
            f"{[d.name for d in DATA.iterdir()]}, workfile candidates: {found}"
        )

    record = {
        "archive": asdict(Provenance(
            path=str(archive.relative_to(HERE)),
            url=URL,
            sha256=digest,
            bytes=archive.stat().st_size,
        ))
    }
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return Provenance(**record["archive"])


def workfile() -> Path:
    ensure()
    return DATA / UNPACKED / WORKFILE


def author_log() -> Path:
    ensure()
    return DATA / UNPACKED / AUTHOR_LOG


def main() -> int:
    record = ensure(force="--force" in sys.argv)
    print(json.dumps(asdict(record), indent=2))
    print(f"\nworkfile: {workfile()}")
    print(f"author log: {author_log()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
