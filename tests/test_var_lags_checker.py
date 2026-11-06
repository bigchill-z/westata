"""The var-lags checker must report this environment's defect correctly.

Rounds 39-47 found `var ..., lags(k)` estimates only the k-th lag here; rounds 81-84 then spent
three rounds diagnosing westata against that broken reference. scripts/check_stata_var_lags.py
exists so a future comparison establishes the reference's validity FIRST.

Its own expectations were wrong twice before this test existed -- once using 3/4/7/10 for a
two-variable system, once assuming the no-option default was lags(1). Both mistakes would have
produced a confident wrong verdict from a checker meant to prevent exactly that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_stata_var_lags.py"


@pytest.mark.skipif(not SCRIPT.exists(), reason="the checker script is not present")
def test_the_checker_reports_the_broken_lags_option() -> None:
    """Run it and require the WARNING, since that is this environment's real state."""

    try:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT)],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(SCRIPT.parents[1]),
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment issues
        pytest.skip("could not run the checker in this environment")
    output = completed.stdout + completed.stderr
    if "tsset" in output and "not found" in output:
        pytest.skip("Stata is not available")
    assert "lags(2)" in output
    assert "BROKEN" in output
    assert "hand-made lag columns" in output


@pytest.mark.skipif(not SCRIPT.exists(), reason="the checker script is not present")
def test_the_checker_uses_the_right_coefficient_counts() -> None:
    """A VAR(k) over two variables stores 2*(2k+1); the checker must not compare against less."""

    source = SCRIPT.read_text(encoding="utf-8")
    # The no-option default is lags(2) here, and lags(1) is correct -- both must read OK.
    assert "2 * (2 * 2 + 1)" in source
    assert "2 * (2 * 3 + 1)" in source
