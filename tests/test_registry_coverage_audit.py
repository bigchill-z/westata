"""Guard: every estimation command in the conformance registry is accounted for.

This test exists because the layout sweep twice reported a completion figure measured
against the wrong denominator:

* once using ``_HEADER_BLOCK``'s 53 entries as if they were the whole registry, while the
  registry holds 94;
* once missing three E3 commands (areg, xtabond, sureg) that had no scan spec at all, so
  they silently fell through to the generic renderer and were never compared.

A green run here means: each registry entry is either exercised by
``scripts/scan_all_layouts.py`` or listed in ``NO_TABLE`` with a stated reason. It does not
mean every command matches Stata -- that is what the scan reports.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_registry_entry_is_accounted_for() -> None:
    script = ROOT / "scripts" / "audit_registry_coverage.py"
    assert script.exists(), "the coverage audit script is missing"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = completed.stdout + completed.stderr
    # The script exits non-zero when an entry is unaccounted for, and prints which ones.
    assert completed.returncode == 0, (
        "some conformance-registry entries are neither scanned nor declared out of "
        f"scope:\n{output}"
    )
    assert "UNACCOUNTED FOR:             0" in output, output
