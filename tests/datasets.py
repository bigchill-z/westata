"""Resolve the reference datasets without hard-coding one machine's paths.

The test suite pins westata against Stata's own output, and the datasets behind
those references live OUTSIDE the repository (they are the Stata example files,
~16 MB in total, including a 15 MB one).  Committing them was not an option, so
the tests point at a directory on the author's machine::

    DATA = r"D:\\myCodeProject\\test_westata\\data\\grilic.dta"

That literal is true on exactly one machine.  On CI it is a FileNotFoundError,
and because some of those tests read the file directly instead of through a
fixture, they CRASHED instead of skipping -- 24 of them in run 43600474823.

This module replaces the literal with a lookup that has a defined answer
everywhere:

1. ``WESTATA_DATA_DIR`` -- an explicit override, for CI or another machine.
2. The historical absolute location, kept so this change cannot break the
   machine the whole suite was written on.
3. ``tests/data`` inside the repository, for the datasets that ARE small enough
   to commit (``auto.dta``).

``dataset("grilic.dta")`` returns a path *whether or not it exists*; callers that
need the file use :func:`require_dataset`, which skips the test with a message
naming the variable to set.  That is the difference between "this machine lacks
the reference data" and "the code is broken", and the suite must not conflate
them.
"""

from __future__ import annotations

import os
import pathlib

import pytest

#: Environment variable pointing at the directory holding the reference ``.dta`` files.
OVERRIDE = "WESTATA_DATA_DIR"

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Searched in order.  The absolute one is the machine the suite was authored on.
_CANDIDATES = (
    pathlib.Path(r"D:\myCodeProject\test_westata\data"),
    REPO_ROOT / "tests" / "data",
)


def data_dir(*, honour_override: bool = True) -> pathlib.Path | None:
    """The first candidate directory that exists, or ``None``.

    ``honour_override=False`` ignores ``WESTATA_DATA_DIR`` and searches the
    built-in candidates only; it exists so a test can assert what the fallback
    would find without the environment leaking in.
    """

    if honour_override:
        override = os.environ.get(OVERRIDE)
        if override:
            candidate = pathlib.Path(override)
            # An override that points nowhere is an ANSWER ("no data here"), not a
            # reason to fall back: a CI run sets it to prove the suite still passes
            # without the reference files, and silently finding them again would
            # make that check vacuous.
            return candidate if candidate.is_dir() else None
    for candidate in _CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def dataset(name: str) -> pathlib.Path:
    """Path to ``name`` under the reference directory (existence NOT guaranteed)."""

    directory = data_dir()
    if directory is None:
        # No directory at all: return a path under the first candidate so the
        # message a caller builds names a concrete, inspectable location.
        return _CANDIDATES[0] / name
    return directory / name


def has(name: str) -> bool:
    """Whether the named reference dataset is readable here.

    ``False`` when there is no data directory at all.  Checking
    ``dataset(name).exists()`` would have been wrong: ``dataset`` falls back to
    the historical absolute path so error messages stay concrete, and that path
    exists on the author's machine, so the fallback made this answer ``True``
    even when the override pointed somewhere empty.
    """

    directory = data_dir()
    if directory is None:
        return False
    return (directory / name).exists()


def require_dataset(name: str) -> pathlib.Path:
    """Path to ``name``, skipping the calling test when it is absent."""

    path = dataset(name)
    if not path.exists():
        pytest.skip(
            f"{name} is not available on this machine (looked in {data_dir() or _CANDIDATES[0]}); "
            f"set {OVERRIDE} to the directory that holds the Stata reference .dta files "
            "to run these checks"
        )
    return path


def reason(name: str) -> str:
    """Skip message without skipping, for use in ``pytestmark``."""

    return (
        f"{name} is not available on this machine; set {OVERRIDE} to the directory "
        "that holds the Stata reference .dta files to run these checks"
    )
