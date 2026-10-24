"""Run every tseries repair repro in one Stata session-free sequence.

Each repro starts its own embedded StataNow 19.5 session (``stata_probe.Stata``),
so running them from one driver keeps the evidence in a single transcript with a
single pass/fail summary at the end.

Run::

    cd D:\\myCodeProject\\statapy
    .venv\\Scripts\\python.exe retest/fix/tseries/run_all.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

SCRIPTS = [
    ("V1/V2", "repro_v1v2_var.py"),
    ("V3", "repro_v3_dfuller.py"),
    ("V4", "repro_v4_vecrank.py"),
    ("V5", "repro_v5_arima_n.py"),
    ("V6", "repro_v6_arch.py"),
]


def main() -> int:
    failures = 0
    for label, script in SCRIPTS:
        print("\n" + "#" * 118)
        print(f"# {label}  ->  retest/fix/tseries/{script}")
        print("#" * 118, flush=True)
        completed = subprocess.run(
            [sys.executable, str(HERE / script)],
            cwd=str(REPO),
            check=False,
        )
        if completed.returncode != 0:
            failures += 1
            print(f"!! {script} exited with {completed.returncode}")
    print("\n" + "=" * 118)
    print(f"{len(SCRIPTS) - failures}/{len(SCRIPTS)} repro scripts passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
