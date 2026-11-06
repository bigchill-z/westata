"""dfuller / pperron: the names Stata users try, mapped onto unitroot.

Round 131 recorded a usability gap: the package had no top-level `dfuller`, so `ws.dfuller(...)`
raised AttributeError and a Stata user could reasonably conclude the command was missing -- while
the numbers behind `unitroot(test="adf")` were already exact. C14-01 and C14-03 both scored 0.00
with numeric coverage 1.00 for exactly that reason.

Round 132 added thin aliases. They are aliases, not reimplementations: every value must be identical
to the `unitroot` call they delegate to, and the option mapping must be the one Stata uses.

    no option     ->  regression="c"
    , trend       ->  regression="ct"

Reference values from a live Stata on nelson_plosser.dta:

    dfuller lrgnp, trend    Z(t) -2.026
    dfuller d.lrgnp         Z(t) -5.322
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nelson_plosser.dta"

STATA_ADF_TREND = -2.026
STATA_ADF_DIFF = -5.322


@pytest.fixture(scope="module")
def levels():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nelson_plosser.dta is not available")
    values = np.asarray(frame["lrgnp"], dtype=float)
    return values[np.isfinite(values)]


def test_both_names_are_exported() -> None:
    assert hasattr(ws, "dfuller")
    assert hasattr(ws, "pperron")


def test_dfuller_trend_matches_both_stata_and_unitroot(levels) -> None:
    alias = ws.dfuller(levels, trend=True)
    direct = ws.unitroot(levels, test="adf", regression="ct")
    assert float(alias.statistic) == pytest.approx(float(direct.statistic), abs=1e-12)
    assert float(alias.statistic) == pytest.approx(STATA_ADF_TREND, abs=5e-4)


def test_dfuller_on_differences_matches_stata(levels) -> None:
    result = ws.dfuller(np.diff(levels))
    assert float(result.statistic) == pytest.approx(STATA_ADF_DIFF, abs=5e-4)


def test_the_default_is_no_trend(levels) -> None:
    """Stata's no-option case is regression="c", not "ct"."""

    alias = ws.dfuller(levels)
    direct = ws.unitroot(levels, test="adf", regression="c")
    assert float(alias.statistic) == pytest.approx(float(direct.statistic), abs=1e-12)


def test_pperron_delegates_the_same_way(levels) -> None:
    alias = ws.pperron(levels, trend=True)
    direct = ws.unitroot(levels, test="pp", regression="ct")
    assert float(alias.statistic) == pytest.approx(float(direct.statistic), abs=1e-12)


def test_an_explicit_regression_overrides_trend(levels) -> None:
    """So a caller who knows the unitroot spelling is not fought by the alias."""

    alias = ws.dfuller(levels, trend=True, regression="n")
    direct = ws.unitroot(levels, test="adf", regression="n")
    assert float(alias.statistic) == pytest.approx(float(direct.statistic), abs=1e-12)


def test_lags_maps_to_maxlag(levels) -> None:
    alias = ws.dfuller(levels, lags=2)
    direct = ws.unitroot(levels, test="adf", regression="c", maxlag=2)
    assert float(alias.statistic) == pytest.approx(float(direct.statistic), abs=1e-12)
    assert int(alias.lags) == 2
