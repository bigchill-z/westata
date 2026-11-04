"""Pin ``varwle`` against Stata's Wald lag-exclusion table.

Reference from a live ``varwle`` after ``var dinf unem, lags(1/2)`` on macro_swatson.dta::

    Equation dinf : lag 1 =  72.72164 (df 2), lag 2 =  72.9294  (df 2)
    Equation unem : lag 1 = 741.8916  (df 2), lag 2 = 123.1287  (df 2)
    Equation All  : lag 1 = 796.2418  (df 4), lag 2 = 187.9991  (df 4)

The statistic must be built from the ML-scale covariance (``e(V)``), which is what a Stata
``test`` uses. The backend's unbiased covariance gives 764.97 / 73.35 for lags 1 / 2 -- a
different number at both orders, so the choice is not a rescaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.varwle import varwle

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA = {
    ("dinf", 1): 72.72164,
    ("dinf", 2): 72.9294,
    ("unem", 1): 741.8916,
    ("unem", 2): 123.1287,
    ("All", 1): 796.2418,
    ("All", 2): 187.9991,
}


@pytest.fixture(scope="module")
def fitted_var():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()


def test_every_cell_matches_stata(fitted_var) -> None:
    table = varwle(fitted_var)
    seen = {(row.equation, row.lag): row for row in table.rows}
    assert set(seen) == set(STATA), sorted(seen)
    for key, expected in STATA.items():
        row = seen[key]
        assert row.chi2 == pytest.approx(expected, rel=5e-5), key


def test_degrees_of_freedom_match_stata(fitted_var) -> None:
    table = varwle(fitted_var)
    for row in table.rows:
        expected_df = 4 if row.equation == "All" else 2
        assert row.df == expected_df, (row.equation, row.lag)


def test_the_table_is_grouped_per_equation(fitted_var) -> None:
    """``varwle`` prints one block per equation plus an ``All`` block."""

    grouped = varwle(fitted_var).by_equation()
    assert list(grouped) == ["dinf", "unem", "All"]
    for rows in grouped.values():
        assert [row.lag for row in rows] == [1, 2]


def test_all_row_pools_every_equation(fitted_var) -> None:
    """The All statistic must exceed either equation's on its own."""

    table = varwle(fitted_var)
    seen = {(row.equation, row.lag): row.chi2 for row in table.rows}
    for lag in (1, 2):
        assert seen[("All", lag)] > seen[("dinf", lag)]
        assert seen[("All", lag)] > seen[("unem", lag)]
