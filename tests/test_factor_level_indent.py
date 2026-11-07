"""Factor level labels are INDENTED by a formula, and the round-183 expectation right-aligned them.

Round 217 chased the last half-row of C12-09 (the year block's level alignment) and found the
defect was in this package but the EXPECTATION was wrong too. Two live captures fix the rule:

    regress mpg weight i.fxc     '    Foreign  |  -1.444832 ...'   4 leading, 2 trailing
    xtreg y k1 i.year, fe        '       1971  |  -713.0255 ...'   7 leading, 2 trailing

Both put the pipe at column 13 with name_width 12, and both satisfy

    leading = name_width - 1 - len(text)        then text, then two spaces

My two failed attempts, both caught by tests rather than shipped:

  * right-align the text (round 183's behaviour) -- correct only when the label happens to fill
    the field, which is why `Foreign` looked fine standing in for a 12-character slot;
  * hard-code seven leading spaces -- right for `1971`, wrong for `Foreign`.

The round-183 test asserted the right-aligned string, so it passed for months while encoding the
wrong rule. Corrected in place, with the re-captured row and the reason recorded next to it.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = "tests/data/auto.dta"


@pytest.fixture(scope="module")
def factor_fit():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["mpg", "weight", "foreign", "rep78"])
    except FileNotFoundError:  # pragma: no cover - the file ships with the repo
        pytest.skip("auto.dta is missing")
    return ws.OLS.from_formula("mpg ~ weight + C(foreign)", frame.reset_index(drop=True)).fit()


def test_the_level_label_is_indented_not_right_aligned(factor_fit) -> None:
    lines = str(factor_fit.summary(style="stata")).splitlines()
    level = next(line for line in lines if "Foreign" in line)
    assert level.startswith("    Foreign  |"), repr(level[:20])


def test_the_level_indent_follows_the_name_width_rule(factor_fit) -> None:
    """leading = name_width - 1 - len(text); the pipe stays at column 13."""

    lines = str(factor_fit.summary(style="stata")).splitlines()
    level = next(line for line in lines if "Foreign" in line)
    header = next(line for line in lines if "Coefficient" in line)
    name_width = header.index("|") - 1
    leading = len(level) - len(level.lstrip(" "))
    assert leading == name_width - 1 - len("Foreign")
    assert level.index("|") == name_width + 1


def test_the_group_header_is_still_right_aligned(factor_fit) -> None:
    """The TERM row is right-aligned; only the LEVEL row is indented."""

    lines = str(factor_fit.summary(style="stata")).splitlines()
    group = next(line for line in lines if line.strip().startswith("foreign |"))
    assert group.startswith("     foreign |"), repr(group[:20])
