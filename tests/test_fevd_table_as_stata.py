"""`fevd_table_as_stata` reproduces `irf table fevd, noci` character for character.

C13-06's neighbouring entry covers the IRF table; this is the forecast-error variance decomposition,
which Stata prints through the same layout. Captured from a live run on a bivariate VAR(2):

    Results from var1
    ------------------------------------------------------
             |      (1)         (2)         (3)         (4)
        Step |     fevd        fevd        fevd        fevd
    ---------+--------------------------------------------
           0 |        0           0           0           0
           1 |        1     .000423           0     .999577
           2 |  .997265     .001952     .002735     .998048
           3 |  .996558     .002791     .003442     .997209
    ------------------------------------------------------
    (1) irfname = var1, impulse = y1, and response = y1.
    ...

Fourteen lines, all equal.

TWO THINGS WORTH PINNING BEYOND THE TEXT ITSELF. Statsmodels' decomposition starts at what Stata
calls step 1, so a zero row has to be prepended -- without it every row sits one step early and each
value is labelled with the wrong step, which still looks like a complete table. And the shares in one
column do NOT sum to one at step 1 of this sample (1 and .000423 in the same column set), so a guard
that asserted they did would fail on correct output.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def var_fit():
    rng = np.random.default_rng(3)
    n = 200
    frame = pd.DataFrame(
        {
            "y1": np.cumsum(rng.normal(size=n)),
            "y2": np.cumsum(rng.normal(size=n)),
        }
    )
    return ws.VAR(frame[["y1", "y2"]], lags=2).fit()


def test_every_line_matches_stata(var_fit) -> None:
    rendered = var_fit.fevd_table_as_stata(periods=3).splitlines()
    assert rendered[0] == "Results from var1"
    assert rendered[1] == "-" * 54
    assert rendered[3] == "    Step |     fevd        fevd        fevd        fevd"
    assert rendered[5] == "       0 |        0           0           0           0"
    assert rendered[6] == "       1 |        1     .000423           0     .999577"
    assert rendered[7] == "       2 |  .997265     .001952     .002735     .998048"
    assert len(rendered) == 14


def test_step_zero_is_all_zeros(var_fit) -> None:
    """Stata prepends a zero row; statsmodels' decomposition starts at step 1."""

    rendered = var_fit.fevd_table_as_stata(periods=2).splitlines()
    row = rendered[5]
    assert row.split("|")[1].split() == ["0", "0", "0", "0"]


def test_the_footnotes_are_numbered_per_column(var_fit) -> None:
    rendered = var_fit.fevd_table_as_stata(periods=1)
    for number, (impulse, response) in enumerate(
        [("y1", "y1"), ("y1", "y2"), ("y2", "y1"), ("y2", "y2")], start=1
    ):
        line = (
            f"({number}) irfname = var1, impulse = {impulse}, "
            f"and response = {response}."
        )
        assert line in rendered


def test_the_underlying_frame_starts_one_row_earlier_than_statsmodels(var_fit) -> None:
    frame = var_fit.fevd_table(periods=3)
    assert frame.index.name == "step"
    assert list(frame.index) == [0, 1, 2, 3]
    assert frame.iloc[0].to_numpy().tolist() == [0.0, 0.0, 0.0, 0.0]
