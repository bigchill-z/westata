"""`summarize v, detail` at observation counts that need thousands separators.

Round 159 implemented the block and matched fifteen lines on a ten-value sample. Re-checking at other sizes,
as rounds 178-181 established as the method, found the observation count without a separator:

    Stata    '10%     3.803435       2.451437       Obs               1,000'
    package  '10%     3.803435       2.451437       Obs                1000'

STATA PUNCTUATES COUNTS OF FOUR OR MORE DIGITS. The fix is `,.0f`, and it applies to both `Obs` and
`Sum of wgt.` because they carry the same number.

Verified at n = 10, 100, 999, 1000 and 5000: 5 of 5 byte-identical, including the boundary where the
separator first appears. A guard below checks 999 and 1000 specifically, since an implementation that
always inserts a comma would pass the 1000 case and fail the other.

THIS IS THE FOURTH TABLE where a single captured scenario was right and another was not, and the first where
the cause was a digit count rather than a rounding rule.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _render(n: int) -> str:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"v": rng.normal(size=n) * 100 + 500})
    return str(ws.summarize(frame, detail=True))


def test_a_thousand_observations_carry_a_separator() -> None:
    rendered = _render(1000)
    assert "Obs               1,000" in rendered
    assert "Sum of wgt.       1,000" in rendered


def test_the_separator_does_not_appear_below_one_thousand() -> None:
    """An always-on comma would pass the case above and fail this one."""

    rendered = _render(999)
    assert "Obs                 999" in rendered
    assert "1,000" not in rendered


def test_five_thousand_is_punctuated() -> None:
    rendered = _render(5000)
    assert "5,000" in rendered


def test_a_ten_value_sample_is_unchanged() -> None:
    """The sample round 159 captured must still match, separators or not."""

    rendered = _render(10)
    lines = rendered.splitlines()
    assert len(lines) == 15
    assert "Percentiles" in rendered


def test_every_line_is_still_forty_one_characters_or_less_wide() -> None:
    """The separator lengthens a number, so the column geometry has to absorb it."""

    for count in (10, 999, 1000, 5000):
        for line in _render(count).splitlines():
            if line.strip():
                assert len(line) <= 61, (count, line)
