"""`vecstable`'s table is now line-for-line identical to Stata.

Report item C14-08 scores 0.0833 on layout. The eigenvalues were always right; the frame and the
in-cell placement were not.

THE CELL RULE was derived by measuring each row rather than by trying alignments, which had failed
three times:

    '  |          1               |         1   |'
    '  |   .2681229               |   .268123   |'
    '  |  -.1557235 +  .0329841i  |   .159178   |'

A value is RIGHT-ALIGNED TO COLUMN 11 and then padded to the 26-wide field. One rule explains `1`
(ten leading spaces), `.2681229` (three) and `.268123` in the 13-wide modulus field (pivot 10).
Values longer than the pivot start at the field's two-space inset instead, which is what the complex
pair does.

Two other details: Stata drops the leading zero (`.2681229`, not `0.2681229`) and writes a complex
value with one space before the sign and TWO after it.

Also measured: the banner is followed directly by the rule -- there is no blank line between them --
and every line is 44 characters with the pipes at columns 2, 29 and 43.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

#: Captured verbatim from `vecstable` after `vec y x, rank(1) lags(2)`.
STATA_TABLE = [
    "",
    "   Eigenvalue stability condition",
    "  +----------------------------------------+",
    "  |        Eigenvalue        |   Modulus   |",
    "  |--------------------------+-------------|",
    "  |          1               |         1   |",
    "  |   .2681229               |   .268123   |",
    "  |  -.1557235 +  .0329841i  |   .159178   |",
    "  |  -.1557235 -  .0329841i  |   .159178   |",
    "  +----------------------------------------+",
    "   The VECM specification imposes a unit modulus.",
]


@pytest.fixture(scope="module")
def stable():
    rng = np.random.default_rng(9)
    n = 200
    x = np.cumsum(rng.normal(size=n))
    y = 2.0 * x + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x": x})
    fitted = ws.VECM(frame[["y", "x"]], coint_rank=1, lags=2).fit()
    return ws.vecstable(fitted)


def test_the_table_matches_stata_line_by_line(stable) -> None:
    rendered = str(stable).splitlines()

    assert len(rendered) == len(STATA_TABLE)
    problems = [
        (index, theirs, ours)
        for index, (theirs, ours) in enumerate(zip(STATA_TABLE, rendered))
        if theirs != ours
    ]
    assert not problems, problems[:3]


def test_every_row_is_forty_four_characters(stable) -> None:
    for line in str(stable).splitlines()[2:10]:
        assert len(line) == 44, repr(line)


def test_a_short_value_is_right_aligned_to_column_eleven(stable) -> None:
    """The `1` row: ten leading spaces, which a plain left- or right-align does not produce."""

    rows = str(stable).splitlines()
    unit_row = next(line for line in rows if "|          1" in line)
    inner = unit_row[3:29]
    assert inner.index("1") == 10


def test_the_leading_zero_is_dropped(stable) -> None:
    rendered = str(stable)
    assert ".2681229" in rendered
    assert "0.2681229" not in rendered


def test_the_complex_gap_is_one_space_then_two(stable) -> None:
    rendered = str(stable)
    assert "-.1557235 +  .0329841i" in rendered


def test_the_banner_is_not_followed_by_a_blank(stable) -> None:
    """Stata goes straight from the banner to the rule; a blank line made the table one row long."""

    rows = str(stable).splitlines()
    assert rows[1].strip() == "Eigenvalue stability condition"
    assert rows[2].startswith("  +")
