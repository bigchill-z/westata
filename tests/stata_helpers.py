"""Locate the Stata installation's ado directory, or say it is not here.

Many tests read Stata's own ``.ado``/``.sthlp`` files as the authority for a claim --
"the macro is named this in ``regress.sthlp``", "this program exists in ``r/``". That is
the right instinct: reading the source beats trusting a transcription.

The problem is that those paths were HARD-CODED to ``D:\\Program Files\\StataNow19\\...``,
which is true on the machine the tests were written on and false everywhere else. CI runs
on Linux, where every one of those ``read_text()`` calls raised ``FileNotFoundError`` and
took the whole job down -- 20 files, all three Python versions.

``available()`` answers "is there a Stata installation whose ado files I can read here?",
and ``pytest.mark.skipif(not stata.available(), ...)`` turns the local-only checks into
skips on machines without it. The assertion still runs where the evidence exists, so
nothing is weakened -- the test simply stops claiming to have checked something it could
not open.

Resolution order: ``WESTATA_STATA_ADO`` (an explicit override, used by CI or by someone
with a non-standard install), then the usual Windows locations, then the common Unix ones.
"""

from __future__ import annotations

import os
import pathlib

#: Environment variable that points straight at ``ado`` or at the Stata root above it.
OVERRIDE = "WESTATA_STATA_ADO"

_CANDIDATES = (
    # The installation these tests were originally written against.
    r"D:\Program Files\StataNow19",
    r"C:\Program Files\StataNow19",
    r"D:\Program Files\Stata19",
    r"C:\Program Files\Stata19",
    r"D:\Program Files\Stata18",
    r"C:\Program Files\Stata18",
    # Unix layouts, in case a Stata install is ever available there.
    "/usr/local/stata19",
    "/usr/local/stata18",
    "/Applications/StataNow",
)


def _resolve() -> pathlib.Path | None:
    override = os.environ.get(OVERRIDE)
    if override:
        candidate = pathlib.Path(override)
        # Accept either the root or the ado directory itself.
        if candidate.name == "ado" and candidate.is_dir():
            return candidate
        nested = candidate / "ado"
        if nested.is_dir():
            return nested
        return None

    for root in _CANDIDATES:
        ado = pathlib.Path(root) / "ado"
        if ado.is_dir():
            return ado
    return None


def ado_base() -> pathlib.Path:
    """The ``ado`` directory. Raises if there is none -- callers should have skipped first."""

    resolved = _resolve()
    if resolved is None:
        raise RuntimeError(
            "no Stata ado directory was found; guard the test with "
            "pytest.mark.skipif(not stata_helpers.available(), ...)"
        )
    return resolved


def available() -> bool:
    """Whether a readable Stata installation is present on this machine."""

    resolved = _resolve()
    if resolved is None:
        return False
    # A directory named ``ado`` is not enough: the tests read ``base/`` underneath it.
    return (resolved / "base").is_dir()


def path(*parts: str) -> pathlib.Path:
    """A path under ``ado``, e.g. ``path("base", "r", "regress.sthlp")``."""

    return ado_base().joinpath(*parts)


def reason() -> str:
    """A skip message that says what is missing and how to point at it."""

    return (
        "no Stata ado directory found on this machine (looked for StataNow19/19/18 under "
        "the usual Windows locations); set WESTATA_STATA_ADO to the Stata root or its "
        "ado directory to run these checks"
    )
