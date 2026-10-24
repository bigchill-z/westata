"""Execute every code block in docs/GUIDE.md, the way tests/test_readme_examples.py does for the README.

The README's Quick start was once broken in four ways and nobody noticed, because it was proofread
rather than run. This guide contains more code than the README ever did, so the same rule applies: a
documentation example that cannot execute is a defect, and the only way to know it executes is to
execute it.

Scope decisions:

  * Only ```python blocks run. The ```bash blocks are installation instructions, deliberately not
    executed -- running `pip install -e .` inside the test would modify the environment.

  * Blocks run in ONE namespace, in order, because that is how a reader meets them: a later block may
    use `df` or `sim` set up earlier.

  * Some blocks reference variables the prose introduces but no block defines -- `panel`, `sim`, `w`,
    `res1`. Those are pre-seeded here from the committed CSVs, exactly as the guide's own data section
    describes. If a block references something NEITHER defined nor pre-seeded, that is a genuine
    documentation defect and the check fails.

  * The working directory is the repository root, because the guide refers to `docs/data/auto.csv`.
"""

from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(r"D:\myCodeProject\statapy\src")))

REPO = pathlib.Path(r"D:\myCodeProject\statapy")
GUIDE = REPO / "docs" / "GUIDE.md"

text = GUIDE.read_text(encoding="utf-8")
blocks = re.findall(r"```python\n(.*?)```", text, re.S)
print(f"found {len(blocks)} python block(s) in {GUIDE.name}")
print()

# Pre-seeded names the prose introduces. Built from the committed data, so the check is deterministic.
import numpy as np
import pandas as pd

namespace: dict[str, object] = {"__name__": "__guide__"}
auto = pd.read_csv(REPO / "docs" / "data" / "auto.csv")
panel = pd.read_csv(REPO / "docs" / "data" / "panel.csv")
sim = pd.read_csv(REPO / "docs" / "data" / "sim.csv")
namespace.update(
    {
        "pd": pd,
        "np": np,
        "df": auto,
        "auto": auto,
        "panel": panel,
        "sim": sim,
        "w": np.abs(np.random.default_rng(0).normal(size=len(auto))),
        "res1": None,
        "res2": None,
    }
)

failures: list[tuple[int, str, str]] = []

import os

previous = os.getcwd()
os.chdir(REPO)
try:
    for index, block in enumerate(blocks, start=1):
        lines = [line for line in block.splitlines() if line.strip() and not line.strip().startswith("#")]
        if not lines:
            print(f"--- block {index}: no executable lines, skipped")
            continue
        try:
            exec(compile(block, f"<GUIDE block {index}>", "exec"), namespace)  # noqa: S102
            print(f"--- block {index}: OK ({len(lines)} line(s))")
        except Exception as exc:  # noqa: BLE001
            print(f"--- block {index}: FAILED {type(exc).__name__}: {exc}")
            failures.append((index, type(exc).__name__, str(exc)))
finally:
    os.chdir(previous)

print()
if failures:
    print(f"FAILED: {len(failures)} block(s)")
    for index, kind, message in failures:
        print(f"  block {index}: {kind}: {message[:160]}")
    raise SystemExit(1)

print(f"all {len(blocks)} executed cleanly")
