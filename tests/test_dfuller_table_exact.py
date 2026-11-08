"""`dfuller`'s table is now line-for-line identical to Stata.

C14-01/02/03 scored 0.0 largely on layout. Round 35 got 9 of 12 lines byte-identical and named three
gaps; round 77 closed them:

  * a LEADING BLANK LINE before the banner;
  * an H0 line that names the deterministics -- "Random walk without drift, d = 0" with no options,
    "Random walk with or without drift" with `trend` -- instead of the generic
    "unit root (nonstationary)" the result class stored;
  * U+2013 EN DASH rather than a hyphen in both "Dickey-Fuller" labels.

THE VALUES WERE NEVER WRONG. Verified against a live Stata on a 200-point seeded random walk:

    dfuller y, trend             Z(t) -1.156539   MacKinnon p 0.9190
    dfuller y, lags(9) trend     Z(t) -1.903507   MacKinnon p 0.6528
    dfuller D.y                  Z(t) -13.963700  MacKinnon p 0.0000

The reference block below is a verbatim capture from `dfuller y, trend`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

#: Captured verbatim from `dfuller y, trend` on the seeded walk below.
STATA_TABLE = [
    "",
    "Dickey\u2013Fuller test for unit root          Number of obs  = 199",
    "Variable: y                               Number of lags =   0",
    "",
    "H0: Random walk with or without drift",
    "",
    "                                       Dickey\u2013Fuller",
    "                   Test      -------- critical value ---------",
    "              statistic           1%           5%          10%",
    "--------------------------------------------------------------",
    " Z(t)            -1.157       -4.007       -3.437       -3.137",
    "--------------------------------------------------------------",
    "MacKinnon approximate p-value for Z(t) = 0.9190.",
]


@pytest.fixture(scope="module")
def walk():
    rng = np.random.default_rng(5)
    return pd.Series(np.cumsum(rng.normal(size=200)))


def test_the_table_matches_stata_line_by_line(walk) -> None:
    rendered = str(ws.dfuller(walk, regression="ct")).splitlines()

    assert len(rendered) == len(STATA_TABLE)
    problems = [
        (index, theirs, ours)
        for index, (theirs, ours) in enumerate(zip(STATA_TABLE, rendered))
        if theirs != ours
    ]
    assert not problems, problems[:3]


def test_the_h0_line_depends_on_the_deterministics(walk) -> None:
    """No option vs `trend` name different null hypotheses in Stata."""

    without_trend = str(ws.dfuller(walk, regression="c"))
    with_trend = str(ws.dfuller(walk, regression="ct"))

    assert "H0: Random walk without drift, d = 0" in without_trend
    assert "H0: Random walk with or without drift" in with_trend
    # The generic placeholder must be gone.
    assert "unit root (nonstationary)" not in without_trend
    assert "unit root (nonstationary)" not in with_trend


def test_the_labels_use_an_en_dash(walk) -> None:
    rendered = str(ws.dfuller(walk, regression="ct"))
    assert "Dickey\u2013Fuller" in rendered
    assert "Dickey-Fuller" not in rendered


def test_the_leading_blank_line_is_present(walk) -> None:
    rendered = str(ws.dfuller(walk, regression="ct")).splitlines()
    assert rendered[0] == ""


def test_the_lag_specification_matches(walk) -> None:
    """`lags(9)` gives Z(t) -1.903507 against Stata's -1.903507."""

    rendered = str(ws.dfuller(walk, regression="ct", lags=9))
    assert "Number of lags =   9" in rendered
    assert "-1.904" in rendered


def test_a_differenced_series_matches(walk) -> None:
    """`dfuller D.y` gives -13.963700 against Stata."""

    rendered = str(ws.dfuller(pd.Series(np.diff(walk.to_numpy())), regression="c"))
    assert "-13.964" in rendered
    assert "0.0000" in rendered
