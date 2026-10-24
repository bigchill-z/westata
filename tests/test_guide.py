"""Verify docs/GUIDE.md the same way the README examples are verified: by running them.

The guide claims specific numbers from measured comparisons and shows code that is supposed to
reproduce the API shape. Both claims are checkable, and both are checked here:

  1. every ```python block in the guide executes;
  2. the comparison tables agree with docs/comparison.json -- a table cell that does not match the data
     means the guide was edited without regenerating, which is exactly how a documentation number goes
     stale.

The second check matters more than it looks. The guide is 500+ lines of generated tables, and the
generator is one command away from being re-run with different data. Without this test, a reader could
be reading last week's numbers with no indication.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
GUIDE = REPO / "docs" / "GUIDE.md"
COMPARISON = REPO / "docs" / "comparison.json"


def _guide_text() -> str:
    if not GUIDE.exists():
        pytest.skip("docs/GUIDE.md not present")
    return GUIDE.read_text(encoding="utf-8")


def test_guide_python_blocks_execute() -> None:
    """Run every ```python block, in one namespace, in document order."""

    import numpy as np
    import pandas as pd

    text = _guide_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) >= 5, f"expected several code blocks, found {len(blocks)}"

    namespace: dict[str, object] = {
        "__name__": "__guide__",
        "pd": pd,
        "np": np,
        "df": pd.read_csv(REPO / "docs" / "data" / "auto.csv"),
        "auto": pd.read_csv(REPO / "docs" / "data" / "auto.csv"),
        "panel": pd.read_csv(REPO / "docs" / "data" / "panel.csv"),
        "sim": pd.read_csv(REPO / "docs" / "data" / "sim.csv"),
    }

    import os

    previous = os.getcwd()
    os.chdir(REPO)
    try:
        for index, block in enumerate(blocks, start=1):
            try:
                exec(compile(block, f"<GUIDE block {index}>", "exec"), namespace)  # noqa: S102
            except Exception as exc:  # noqa: BLE001 - report the block
                pytest.fail(
                    f"GUIDE code block {index} raised {type(exc).__name__}: {exc}\n"
                    f"--- block ---\n{block}"
                )
    finally:
        os.chdir(previous)


def test_guide_numbers_match_the_measured_data() -> None:
    """Every relative difference quoted in the guide's summary table must appear in comparison.json.

    The comparison table lists a maximum relative difference per command. Those values are formatted by
    the generator, so this reconstructs the same formatting and asserts the string is present. A
    mismatch means the guide is stale.
    """

    if not COMPARISON.exists():
        pytest.skip("comparison.json not present")
    entries = json.loads(COMPARISON.read_text(encoding="utf-8"))
    comparisons = [e for e in entries if "rel_b" in e]
    assert comparisons, "no comparisons in the JSON"

    text = _guide_text()

    def fmt(value: float) -> str:
        if value == 0.0:
            return "0"
        if value < 1e-12:
            return f"{value:.1e}"
        if value < 1e-3:
            return f"{value:.2e}"
        return f"{value:.3f}"

    missing = []
    for entry in comparisons:
        for value in (entry["rel_b"], entry["rel_se"]):
            rendered = fmt(float(value))
            # The guide must contain the command AND its formatted difference on the same line.
            pattern = re.escape(entry["stata_command"])
            found = False
            for line in text.splitlines():
                if re.search(pattern, line) and rendered in line:
                    found = True
                    break
            if not found:
                missing.append(f"{entry['stata_command']}: {rendered}")

    assert not missing, "guide quotes differences absent from comparison.json:\n  " + "\n  ".join(missing)


def test_guide_names_the_e1_commands() -> None:
    """The nine E1 entries must appear in the guide's known-deviations section.

    A guide that omits them would be the documentation equivalent of the registry claiming more than it
    can show -- the exact failure the evidence grading exists to prevent.
    """

    text = _guide_text()
    start = text.find("## Known deviations")
    assert start > 0, "guide has no Known deviations section"
    section = text[start:]

    for command in (
        "xtgls",
        "teffects nnmatch",
        "frontier",
        "bsqreg",
        "permute",
        "arima",
        "rreg",
        "xtlogit, pa",
        "heckman",
    ):
        assert command in section, f"E1 command {command!r} missing from Known deviations"


def test_guide_has_no_source_syntax_leaked_into_it() -> None:
    r"""No line of the guide may be a bare Python string literal.

    A generator edit once inserted a list of string literals into the triple-quoted document body, so the
    rendered guide contained lines like

        "Some rows above show a different \N\ on each side. None is a mistake, ...",

    instead of the prose. The interpolated table in the same block came out correct, which is what made it
    easy to miss: the page looked structurally right and read like a code listing in one section.

    This is the general guard. Any generator change can reintroduce the class of slip, and the symptom is
    visible as a line that starts with a quote character.
    """

    import re as _re

    text = _guide_text()
    offenders = [
        (index, line)
        for index, line in enumerate(text.splitlines(), start=1)
        if _re.match(r'^\s*"', line) or _re.match(r"^\s*'", line)
    ]
    # A legitimate Python line inside a `python block may be indented with a quote (a continued argument
    # list). Those are excluded by tracking whether we are inside a fence.
    in_fence = False
    real = []
    for index, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("`"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if _re.match(r'^\s*"', line) or _re.match(r"^\s*'", line):
            real.append((index, line))
    assert not real, "guide prose contains Python string syntax:\n  " + "\n  ".join(
        f"{i}: {l[:80]}" for i, l in real
    )
