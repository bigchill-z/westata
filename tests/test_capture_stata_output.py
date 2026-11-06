"""The capture tool must return raw bytes, which is what a reference needs to be.

Rounds 81 and 88-89 both lost time to hand-typed expectations: once misreading which table block
held which impulse/response pair, once adding a space that made 45 columns where Stata has 44. In
both cases the "mismatch" was in my target.

scripts/capture_stata_output.py removes the transcription step: it logs a command and prints every
line with its length and repr, the only form in which whitespace differences are visible.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "capture_stata_output.py"
DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.mark.skipif(not SCRIPT.exists(), reason="the capture script is not present")
@pytest.mark.skipif(not Path(DATA).exists(), reason="the dataset is not available")
def test_it_captures_a_boxed_table_verbatim() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), DATA, "tt", "var dinf unem; varstable"],
        capture_output=True,
        text=True,
        timeout=900,
        cwd=str(SCRIPT.parents[1]),
    )
    output = completed.stdout + completed.stderr
    if "tsset" in output and "not found" in output:
        pytest.skip("Stata is not available")
    assert "Eigenvalue stability condition" in output
    # Every printed line is prefixed by its length, and the box rows must be 44 wide.
    assert " 44 '  |     .82182               |    .82182   |'" in output
    # The repr form is what exposes whitespace, so quotes must be present.
    assert "'  |        Eigenvalue        |   Modulus   |'" in output


@pytest.mark.skipif(not SCRIPT.exists(), reason="the capture script is not present")
def test_the_script_does_not_hand_type_expectations() -> None:
    """It must read the log, not contain any literal Stata output."""

    source = SCRIPT.read_text(encoding="utf-8")
    assert "read_text" in source
    assert "Eigenvalue" not in source
    assert ".82182" not in source
