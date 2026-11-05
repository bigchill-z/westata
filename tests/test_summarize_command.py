"""Pin the standalone ``summarize`` command to Stata's six-column table.

The package already had ``estat_summarize``, but that renders the ESTIMATION-SAMPLE variant --
no ``Obs`` column, headed "Estimation sample regress". The plain ``summarize`` command had no
entry point at all, which is why report cases C05-08 and C09-04 both say "you have to run
summarize yourself" after ``predict``.

Reference from a live ``summarize k l`` on cobb_douglas.dta::

        Variable |        Obs        Mean    Std. dev.       Min        Max
    -------------+---------------------------------------------------------
               k |         24    234.1667    106.2666        100        431
               l |         24    145.7917    29.61636        100        200

Every row is 71 characters, the pipe is at column 13, and the values use %9.0g -- so
``106.2666`` and ``29.61636`` print with DIFFERENT decimal counts in the same column, which a
fixed precision would get wrong for one of them.

Column ends differ between the header and the data for exactly one field: "Std. dev." ends at
50 in the header and 49 in the data rows. The other four coincide (25 / 37 / 60 / 71).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\cobb_douglas.dta"
STATA_BLOCK = [
    "    Variable |        Obs        Mean    Std. dev.       Min        Max",
    "-------------+---------------------------------------------------------",
    "           k |         24    234.1667    106.2666        100        431",
    "           l |         24    145.7917    29.61636        100        200",
]


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("cobb_douglas.dta is not available")


def test_every_line_matches_stata(frame) -> None:
    rendered = str(ws.summarize(frame, ["k", "l"])).splitlines()
    for expected in STATA_BLOCK:
        assert expected in rendered, expected


def test_every_row_is_71_characters(frame) -> None:
    for line in str(ws.summarize(frame, ["k", "l"])).splitlines():
        assert len(line) == 71, line


def test_values_use_percent9_0g_not_a_fixed_precision(frame) -> None:
    """106.2666 and 29.61636 must both appear as Stata prints them."""

    rendered = str(ws.summarize(frame, ["k", "l"]))
    assert "106.2666" in rendered
    assert "29.61636" in rendered
    assert "29.616359" not in rendered


def test_missing_values_are_excluded_from_obs() -> None:
    """Stata's Obs counts non-missing observations, not rows."""

    data = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0]})
    rendered = str(ws.summarize(data))
    assert "3" in rendered
    assert "1.75" not in rendered  # mean over the three kept values is 7/3


def test_a_selection_of_variables_is_honoured(frame) -> None:
    rendered = str(ws.summarize(frame, ["k"]))
    assert "k |" in rendered
    assert "l |" not in rendered
