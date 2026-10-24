"""Extract the README's Quick start code blocks and execute them.

WHY THIS EXISTS. The Quick start was never executed by anyone, and it was broken in four ways at once:

    * it read `auto.dta` and then used columns that do not exist in it -- `y`, `x`, `x1`, `endog`, `z1`, `firm`,
      `year`, `wage`, `edu`, and undefined `res1` / `res2`. `auto` has make, price, mpg, rep78, headroom, trunk,
      weight, length, turn, displacement, gear_ratio, foreign. Roughly 80% of the example could not run.
    * `res.test("weight foreign")` raised KeyError, and the comment beside it printed a specific F statistic that
      nobody could have obtained from it.
    * `res.test("weight = 0, C(foreign)[T.Foreign] = 0")` used a comma as a separator, which `test` does not
      support.
    * `Tobit.from_formula(...).fit(lower=0)` passed `lower` to the wrong method.

Executing the examples also uncovered a LIBRARY bug rather than a documentation one: `resolve_names` treated `[`
as a wildcard character, so `C(foreign)[T.Foreign]` was sent to `fnmatch`, read as a character class, and matched
nothing. Every factor variable was unreachable through `test()` and `lincom()`. That is fixed separately, and this
test is what found it.

The test is deliberately strict about the boundary: it runs the fenced ```python blocks under "## Quick start" and
nothing else, so the shell blocks in "## Install" are not fed to the interpreter.

It executes in ONE namespace, in order, because that is how a reader pastes them -- a later block may rely on `df`
or `res` established earlier.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
README = REPO / "README.md"


def _quick_start_blocks() -> list[str]:
    text = README.read_text(encoding="utf-8-sig")
    start = text.find("## Quick start")
    if start < 0:
        pytest.skip("README has no Quick start section")
    end = text.find("\n## ", start + 5)
    section = text[start : end if end > 0 else len(text)]
    return re.findall(r"```python\n(.*?)```", section, re.S)


def test_readme_quick_start_has_blocks() -> None:
    """A guard against the extraction silently finding nothing and the suite passing vacuously."""

    blocks = _quick_start_blocks()
    assert len(blocks) >= 3, f"expected several Quick start blocks, found {len(blocks)}"


def test_readme_quick_start_executes(tmp_path) -> None:
    """Every block runs, in order, in one namespace.

    The working directory is changed to the repository root because the examples reference
    `tests/data/auto.dta` by a path relative to it.
    """

    import os

    blocks = _quick_start_blocks()
    assert blocks, "no python blocks found under ## Quick start"

    namespace: dict[str, object] = {"__name__": "__readme__"}
    previous = os.getcwd()
    os.chdir(REPO)
    try:
        for index, block in enumerate(blocks, start=1):
            try:
                exec(compile(block, f"<README Quick start block {index}>", "exec"), namespace)  # noqa: S102
            except Exception as exc:  # noqa: BLE001 - surface the block number
                pytest.fail(
                    f"README Quick start block {index} raised {type(exc).__name__}: {exc}\n"
                    f"--- block source ---\n{block}"
                )
    finally:
        os.chdir(previous)

    # The prose makes claims about coefficient naming; check the one that is checkable.
    result = namespace.get("res")
    if result is not None:
        names = list(result.param_names)
        assert "C(foreign)[T.Foreign]" in names, names


def test_factor_levels_are_referencable_in_tests() -> None:
    """Regression test for the `resolve_names` defect found by executing the README.

    `[` in a coefficient name is literal syntax for a factor level, not a wildcard. Treating it as one sent the
    name to `fnmatch`, which read `[T.Foreign]` as a character class and matched nothing -- so a name that was
    present in the `available` list could not be referenced.
    """

    import pandas as pd

    sys.path.insert(0, str(REPO / "src"))
    import westata as ws

    frame = pd.read_stata(REPO / "tests" / "data" / "auto.dta")
    result = ws.OLS.from_formula("mpg ~ weight + C(foreign)", frame).fit()

    # Exact reference to a factor level.
    outcome = result.test("C(foreign)[T.Foreign]")
    assert outcome is not None
    # Equation form.
    assert result.test("C(foreign)[T.Foreign] = 0") is not None
    # And wildcards must still work.
    assert result.test("weight*") is not None
