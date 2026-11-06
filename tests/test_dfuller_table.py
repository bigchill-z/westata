"""dfuller's printed table, reproduced line for line.

The report scored C14-01..03 (three ADF cases) at 0.00 line agreement. Every NUMBER already
matched -- Z(t) -3.706, critical values -4.018/-3.441/-3.141, MacKinnon p 0.0220 -- but this
package printed a list while Stata prints a table, so a line-wise comparison had nothing to
align and scored zero.

That makes those three cases a defect in the METRIC as much as in the layout: 0.00 reads as
"completely wrong" when the arithmetic was exact. ``render_dfuller`` closes the layout half.

Details a plausible implementation gets wrong, all measured from a live run rather than assumed:

* the header uses an EN DASH between "Dickey" and "Fuller", not a hyphen
* every number ENDS at a fixed column: 23, 36, 49, 62
* the statistic row begins with a space (" Z(t)") so its text starts at column 1
* the MacKinnon line carries a trailing period
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.tseries import render_dfuller

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

#: Captured from ``dfuller x, trend`` on the inf series, byte for byte.
STATA_LINES = [
    "Dickey\u2013Fuller test for unit root          Number of obs  = 166",
    "Variable: x                               Number of lags =   0",
    "",
    "H0: Random walk with or without drift",
    "",
    "                                       Dickey\u2013Fuller",
    "                   Test      -------- critical value ---------",
    "              statistic           1%           5%          10%",
    "--------------------------------------------------------------",
    " Z(t)            -3.706       -4.018       -3.441       -3.141",
    "--------------------------------------------------------------",
    "MacKinnon approximate p-value for Z(t) = 0.0220.",
]


@pytest.fixture(scope="module")
def result():
    try:
        series = pd.read_stata(DATA)["inf"].dropna().to_numpy(float)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.unitroot(series, regression="ct")


def test_every_line_matches_stata(result) -> None:
    rendered = render_dfuller(result, variable="x", regression="ct").splitlines()
    for line in STATA_LINES:
        assert line in rendered, repr(line)


def test_the_table_is_the_right_width(result) -> None:
    """62 columns at the rule; a stray space anywhere shows up immediately."""

    rendered = render_dfuller(result, variable="x", regression="ct").splitlines()
    rules = [line for line in rendered if line.startswith("---")]
    assert rules and all(len(line) == 62 for line in rules)


def test_the_header_uses_an_en_dash(result) -> None:
    """A hyphen here is the single most likely mis-rendering."""

    rendered = render_dfuller(result, variable="x", regression="ct")
    assert "Dickey\u2013Fuller" in rendered
    assert "Dickey-Fuller" not in rendered


def test_the_numbers_match_stata(result) -> None:
    """The arithmetic was always right; this guards the pairing of number to column."""

    assert float(result.statistic) == pytest.approx(-3.706, abs=5e-4)
    assert float(result.critical["5%"]) == pytest.approx(-3.441, abs=5e-4)


def test_the_no_drift_label_is_used_for_a_different_case() -> None:
    series = np.random.default_rng(3).normal(size=120).cumsum()
    drifted = ws.unitroot(series, regression="c")
    rendered = render_dfuller(drifted, variable="y", regression="c")
    assert "H0: Random walk without drift, d = 0" in rendered
