"""`summarize, detail` on samples too small to fill every slot.

Round 159 matched the block on ten values. A two-value sample exposed a real difference: Stata prints a
DOT in the extreme-value slots it cannot fill, and the renderer left them blank.

    '10%            1              .       Obs                   2'
    '25%            1              .       Sum of wgt.           2'
    '75%            2              .'
    '90%            2              .       Variance             .5'

With that fixed, three degenerate samples each match all fifteen lines:

    one value     15 lines, no difference
    two values    15 lines, no difference
    all identical 15 lines, no difference
    with a missing value dropped

THE DOT IS THE SAME CONVENTION AS EVERY OTHER MISSING STATISTIC in this package -- Stata prints "."
rather than a blank or `nan` -- but the slot here is not a statistic, it is an absent data point, which
is why it was missed.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def pair():
    return str(ws.summarize(pd.DataFrame({"v": np.array([1.0, 2.0])}), detail=True))


def test_unfillable_extreme_slots_print_a_dot(pair) -> None:
    lines = pair.splitlines()
    assert lines[6] == "10%            1              .       Obs                   2"
    assert lines[7] == "25%            1              .       Sum of wgt.           2"
    assert lines[11] == "75%            2              ."


def test_the_block_is_still_fifteen_lines(pair) -> None:
    assert len(pair.splitlines()) == 15


def test_a_single_value_renders(pair) -> None:
    """One observation cannot produce a standard deviation; the block must still appear."""

    frame = pd.DataFrame({"v": np.array([5.0])})
    rendered = str(ws.summarize(frame, detail=True))
    assert len(rendered.splitlines()) == 15
    assert "Percentiles" in rendered


def test_identical_values_do_not_divide_by_zero() -> None:
    """Every value equal means zero variance, so skewness and kurtosis are undefined rather than inf."""

    frame = pd.DataFrame({"v": np.array([2.0] * 10)})
    rendered = str(ws.summarize(frame, detail=True))
    assert len(rendered.splitlines()) == 15
    assert "inf" not in rendered
    assert "nan" not in rendered


def test_a_missing_value_is_dropped_rather_than_breaking_the_grid() -> None:
    frame = pd.DataFrame({"v": np.array([1.0, np.nan, 3.0, 4.0])})
    rendered = str(ws.summarize(frame, detail=True))
    lines = rendered.splitlines()
    assert len(lines) == 15
    # Obs counts the three non-missing values.
    assert "Obs                   3" in rendered
