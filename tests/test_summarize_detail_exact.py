"""`summarize v, detail` -- 15 of 15 lines match Stata.

Options.md T24 recorded this as missing in round 158. Round 159 implemented it, and this file captures
the result verbatim from a live `summarize v, detail` on ten unevenly spaced values:

    (blank)
                                  v
    -------------------------------------------------------------
          Percentiles      Smallest
     1%            1              1
     5%            1              2
    10%          1.5              7       Obs                  10
    25%            7             11       Sum of wgt.          10
    (blank)
    50%           21                      Mean               36.3
                            Largest       Std. dev.       37.3989
    75%           67             41
    90%           96             67       Variance       1398.678
    95%          103             89       Skewness       .7588594
    99%          103            103       Kurtosis       2.057247

FIFTEEN LINES, ALL EQUAL. The left column skips a row between 25%/50% and again before 75%, and the
right column's statistics sit on their own schedule; both come from the capture rather than from a rule
that looked plausible.

THE SAMPLE IS DELIBERATELY UNEVEN. On values 1..100 every common percentile definition agrees, so a
renderer using numpy's default `linear` would pass -- the same trap that made round 158's first guard
pass with the wrong method.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

SAMPLE = np.array([1.0, 2.0, 7.0, 11.0, 19.0, 23.0, 41.0, 67.0, 89.0, 103.0])

STATA_LINES = [
    "",
    "                              v",
    "-------------------------------------------------------------",
    "      Percentiles      Smallest",
    " 1%            1              1",
    " 5%            1              2",
    "10%          1.5              7       Obs                  10",
    "25%            7             11       Sum of wgt.          10",
    "",
    "50%           21                      Mean               36.3",
    "                        Largest       Std. dev.       37.3989",
    "75%           67             41",
    "90%           96             67       Variance       1398.678",
    "95%          103             89       Skewness       .7588594",
    "99%          103            103       Kurtosis       2.057247",
]


@pytest.fixture(scope="module")
def rendered():
    frame = pd.DataFrame({"v": SAMPLE})
    return str(ws.summarize(frame, detail=True))


@pytest.mark.parametrize("line", STATA_LINES)
def test_each_line_matches_stata(rendered, line) -> None:
    assert line in rendered.splitlines(), line


def test_the_block_is_exactly_fifteen_lines(rendered) -> None:
    assert len(rendered.splitlines()) == 15


def test_the_kurtosis_column_shows_the_original_value(rendered) -> None:
    """2.057247 here: excess kurtosis would be -0.94 for this sample."""

    assert "2.057247" in rendered
    from scipy import stats

    excess = float(stats.kurtosis(SAMPLE))
    assert float(excess + 3.0) == pytest.approx(2.057247, abs=1e-5)


def test_a_larger_sample_keeps_the_layout(rendered) -> None:
    """The grid is fixed width, so a 100-row sample must produce the same 15 lines."""

    frame = pd.DataFrame({"v": np.arange(1, 101, dtype=float)})
    lines = str(ws.summarize(frame, detail=True)).splitlines()
    assert len(lines) == 15
    assert lines[2] == "-" * 61
