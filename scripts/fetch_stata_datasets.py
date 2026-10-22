#!/usr/bin/env python
"""Download public Stata example datasets used by the conformance suite.

Design spec: ``docs/superpowers/specs/2026-10-22-westata-design.md`` section
11.5 — datasets are fetched from Stata's public data server into
``tests/data/`` (gitignored) instead of being vendored in the repository.

Usage::

    python scripts/fetch_stata_datasets.py --help
    python scripts/fetch_stata_datasets.py                # auto.dta
    python scripts/fetch_stata_datasets.py --dataset auto --force

The download only happens from ``main()``; importing this module is inert.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

#: Public Stata 18 example datasets used as E2/E3 fixtures.
DATASETS: dict[str, str] = {
    "auto": "https://www.stata-press.com/data/r18/auto.dta",
}

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "tests" / "data"
TIMEOUT_SECONDS = 60.0
#: Stata 14+ .dta files start with this XML-like magic; guards against saving
#: an HTML error page as if it were data.
DTA_MAGIC = b"<stata_dta>"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_stata_datasets.py",
        description=(
            "Download the public Stata example datasets used by westata's "
            "conformance tests into tests/data/ (gitignored)."
        ),
        epilog="Datasets: " + ", ".join(sorted(DATASETS)),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="auto",
        help="dataset to download (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="destination directory (default: tests/data next to the repo root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download even if the target file already exists",
    )
    return parser


def fetch(dataset: str, out_dir: Path, *, force: bool = False) -> Path:
    """Download ``dataset`` into ``out_dir`` and return the written path.

    Raises
    ------
    ValueError
        If ``dataset`` is unknown or the server response is not a Stata file.
    urllib.error.URLError
        On network/HTTP failures.
    OSError
        On filesystem failures.
    """

    if dataset not in DATASETS:
        raise ValueError(
            f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}"
        )

    out_dir = Path(out_dir)
    target = out_dir / f"{dataset}.dta"
    if target.exists() and not force:
        print(f"already present: {target} (use --force to re-download)")
        return target

    url = DATASETS[dataset]
    print(f"downloading {url}")
    with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
        payload = response.read()
    if not payload:
        raise ValueError(f"empty response downloading {url}")
    if not payload.startswith(DTA_MAGIC):
        raise ValueError(
            f"{url} did not return a Stata .dta file "
            f"(expected magic {DTA_MAGIC!r})"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    try:
        partial.write_bytes(payload)
        partial.replace(target)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"wrote {target} ({len(payload)} bytes)")
    return target


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        fetch(args.dataset, args.out_dir, force=args.force)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
