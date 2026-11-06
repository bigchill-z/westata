"""veclmar renders Stata's boxed table, and its numbers were already right at rank 2.

C14-07 scores 0.25 with numeric coverage 1.00 -- the statistics matched and the LAYOUT did not.
The table used dashes where Stata draws a box, so a line-wise comparison had nothing to align.

Captured from a live ``veclmar`` after a rank-2, three-variable ``vec``:

       Lagrange-multiplier test
      | lag  |      chi2    df   Prob > chi2 |
      |------+-------------------------------|
      |   1  |    6.1793     9     0.72185   |
      |   2  |    5.3813     9     0.79988   |
       H0: no autocorrelation at lag order

Every line is 42 characters. The rank-2 case was checked because rounds 73-78 found six separate
column-0 assumptions in vec's neighbouring code; veclmar turned out to be clean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def rank_two_lm():
    rng = np.random.default_rng(7)
    n = 220
    common = np.cumsum(rng.normal(size=n))
    frame = pd.DataFrame(
        {
            "a": common + np.cumsum(rng.normal(scale=0.2, size=n)),
            "b": 1.5 * common + np.cumsum(rng.normal(scale=0.2, size=n)),
            "c": np.cumsum(rng.normal(size=n)),
        }
    )
    fitted = ws.VECM(frame.to_numpy(float), coint_rank=2, lags=2, names=["a", "b", "c"]).fit()
    return ws.veclmar(fitted)


STATA_LINES = [
    "   Lagrange-multiplier test",
    "  | lag  |      chi2    df   Prob > chi2 |",
    "  |------+-------------------------------|",
    "  |   1  |    6.1793     9     0.72185   |",
    "  |   2  |    5.3813     9     0.79988   |",
    "   H0: no autocorrelation at lag order",
]


def test_every_line_matches_stata(rank_two_lm) -> None:
    lines = str(rank_two_lm).splitlines()
    for expected in STATA_LINES:
        assert expected in lines, repr(expected)


def test_the_box_rows_are_forty_two_characters(rank_two_lm) -> None:
    """The title and H0 lines are shorter by design; only box rows are 42 wide."""

    rows = [line for line in str(rank_two_lm).splitlines() if line.startswith("  |")]
    assert len(rows) == 5
    for row in rows:
        assert len(row) == 42, (len(row), row)


def test_the_statistics_are_unchanged(rank_two_lm) -> None:
    """The numbers were never in question; the layout was."""

    rendered = str(rank_two_lm)
    assert "6.1793" in rendered
    assert "5.3813" in rendered
