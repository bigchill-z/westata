"""Capture a Stata command's output BYTE FOR BYTE, so a reference is never hand-typed.

Rounds 81 and 88-89 both lost time to the same mistake: I transcribed an expected line by hand and
got it wrong -- once misreading which impulse/response pair a table block held, once adding a space
that made 45 columns where Stata has 44. In both cases the "mismatch" was in my target, not in the
code under test.

This script is the remedy: it runs a command under a log and prints each captured line with its
length and repr, which is the form in which whitespace differences are visible.

Usage:
    python scripts/capture_stata_output.py "<dataset path>" "<tsset var>" "<command>"

Example:
    python scripts/capture_stata_output.py data/macro_swatson.dta tt "varstable"
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "replication"))

import pandas as pd

from lib.stata_session import StataSession


def capture(dataset: str, tsset: str, command: str) -> list[str]:
    """Run ``command`` in Stata and return the logged lines, verbatim."""

    frame = pd.read_stata(dataset).dropna().reset_index(drop=True)
    frame["__t"] = range(1, len(frame) + 1)
    # Stata writes the log into the PROCESS working directory, not the dataset directory.
    #
    # The name must be UNIQUE PER CALL. A fixed name failed once in the full suite while passing
    # in isolation: two captures raced on the same file, one deleted it, and the other raised
    # FileNotFoundError. The failure looked like a Stata problem and was a test-isolation one.
    log_path = Path(f"capture_stata_output_{os.getpid()}_{uuid.uuid4().hex[:8]}.log")
    try:
        return _run_capture(frame, tsset, command, log_path)
    finally:
        log_path.unlink(missing_ok=True)


def _run_capture(
    frame: pd.DataFrame, tsset: str, command: str, log_path: Path
) -> list[str]:
    """The capture proper; the caller owns the log's lifetime."""

    with StataSession() as session:
        session.push(frame)
        with contextlib.redirect_stdout(io.StringIO()):
            if tsset:
                session.run(f"tsset __t")
            # A command may need an estimation first, so "fit; report" is split on the semicolon
            # and everything but the LAST step runs silently before the log opens.
            steps = [step.strip() for step in command.split(";") if step.strip()]
            for step in steps[:-1]:
                session.run("quietly " + step)
            session.run(f"log using {log_path.name}, text replace name(cap)")
            session.run(steps[-1])
            session.run("log close cap")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return text.splitlines()


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    dataset, tsset, command = argv[0], argv[1], " ".join(argv[2:])
    lines = capture(dataset, tsset, command)
    print(f"# {command}")
    print(f"# {len(lines)} lines")
    for line in lines:
        print(f"{len(line):3} {line!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
