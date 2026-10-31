"""Acquire, verify and stage the paper's replication package.

The archive is published by the authors at ``ddorn.net``; the AEA's own copy
(``aeaweb.org/aer/data/oct2013/20111050_data.zip``) is behind an interstitial page and
returns 404 to a plain client, and openICPSR (``10.3886/E112670``) is a JavaScript-only
search UI.  The authors' mirror is therefore the only machine-readable source, which is
why this module pins it by SHA-256 rather than trusting the URL alone: a silent revision
upstream would otherwise change the replication's numbers without changing its logs.

Nothing here writes into ``src/``, ``tests/`` or ``docs/``.  The staged copy lives under
``data/`` next to this file and is ignored by git -- the repository's policy is not to
redistribute Stata datasets, and a 1.3 MB binary in the history would be paid for by
every clone forever.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

ARCHIVE_URL = "https://www.ddorn.net/data/Autor-Dorn-Hanson-ChinaSyndrome-FileArchive.zip"
PAPER_URL = "https://www.ddorn.net/papers/Autor-Dorn-Hanson-ChinaSyndrome.pdf"
#: SHA-256 of the archive as downloaded on 2026-10-31.  If upstream ever republishes the
#: file, this mismatch is the alarm -- not a reason to skip the check.
ARCHIVE_SHA256 = "1b56c028a75fba784b14477cdf2e89de6ae90a443b9b1b9648179d852048df12"
ARCHIVE_BYTES = 2_953_516

#: The archive unpacks to this directory name.
UNPACKED = "Autor-Dorn-Hanson-ChinaSyndrome-FileArchive"

#: The two estimation samples.  ``workfile_china`` is 722 CZs x 2 periods and is what
#: Tables 2, 3 and 10 use; ``workfile_china_preperiod`` adds the 1970s/1980s
#: falsification rows and is fetched for completeness rather than for the headline.
WORKFILES = ("dta/workfile_china.dta", "dta/workfile_china_preperiod.dta")


@dataclass
class Provenance:
    """One line of the audit trail: what file, from where, with which bytes."""

    path: str
    url: str
    sha256: str
    bytes: int
    source: str

    @classmethod
    def of(cls, path: Path, url: str, source: str) -> "Provenance":
        payload = path.read_bytes()
        return cls(
            path=str(path.relative_to(HERE)),
            url=url,
            sha256=hashlib.sha256(payload).hexdigest(),
            bytes=len(payload),
            source=source,
        )


def _curl(url: str, destination: Path) -> None:
    """Download with the system ``curl``.

    Python's ``urllib`` fails TLS verification on this host for several economics
    domains because the Windows certificate store cannot reach the CRL distribution
    point; ``curl`` with ``--ssl-no-revoke`` works.  The alternative -- disabling
    verification globally -- would be worse, and ``requests`` is not a dependency.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "curl",
            "-sSL",
            "--ssl-no-revoke",
            "-A",
            "Mozilla/5.0 (replication package fetch)",
            "--max-time",
            "180",
            "-o",
            str(destination),
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not destination.exists():
        raise RuntimeError(
            f"download failed for {url}: curl exit {result.returncode}; {result.stderr.strip()}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure(force: bool = False) -> Provenance:
    """Download (once), verify and unpack the replication archive.

    Returns the provenance record that the report quotes.  Re-running on an existing,
    verified copy is a no-op, so a daily replication does not re-download 3 MB.
    """

    archive = DATA / "Aut_Dorn_Hanson_2013.zip"
    unpacked = DATA / UNPACKED
    marker = DATA / "provenance.json"

    if not force and marker.exists() and unpacked.is_dir():
        record = json.loads(marker.read_text(encoding="utf-8"))
        for workfile in WORKFILES:
            if not (unpacked / workfile).exists():
                break
        else:
            return Provenance(**record["archive"])

    if force or not archive.exists() or _sha256(archive) != ARCHIVE_SHA256:
        _curl(ARCHIVE_URL, archive)
    digest = _sha256(archive)
    if digest != ARCHIVE_SHA256:
        raise RuntimeError(
            f"archive checksum changed: expected {ARCHIVE_SHA256}, got {digest}. "
            "The authors may have republished the file; inspect before proceeding, because "
            "the replication's numbers are only meaningful for the bytes the paper used."
        )
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise RuntimeError(
            f"archive size changed: expected {ARCHIVE_BYTES}, got {archive.stat().st_size}"
        )

    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(DATA)

    for workfile in WORKFILES:
        if not (unpacked / workfile).exists():
            raise RuntimeError(f"archive did not contain {workfile}")

    record = {
        "archive": asdict(Provenance.of(archive, ARCHIVE_URL, "authors' replication page")),
        "paper": asdict(Provenance.of(_fetch_paper(), PAPER_URL, "authors' paper page")),
        "workfiles": {
            name: asdict(Provenance.of(unpacked / name, ARCHIVE_URL, "inside the archive"))
            for name in WORKFILES
        },
    }
    marker.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return Provenance(**record["archive"])


def _fetch_paper() -> Path:
    """Keep a local copy of the PDF: the report quotes its table numbers by page."""

    paper = DATA / "Autor-Dorn-Hanson-2013-ChinaSyndrome.pdf"
    if not paper.exists() or paper.stat().st_size < 500_000:
        _curl(PAPER_URL, paper)
    return paper


def workfile(name: str = "workfile_china.dta") -> Path:
    """Absolute path to an estimation sample, downloading first if necessary."""

    ensure()
    path = DATA / UNPACKED / "dta" / name
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run ensure(force=True)")
    return path


def main() -> int:
    record = ensure(force="--force" in sys.argv)
    print(json.dumps(asdict(record), indent=2))
    print(f"\nworkfile: {workfile()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
