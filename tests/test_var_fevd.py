"""Pin ``fevd_table`` to Stata's ``irf table fevd`` layout and numbers.

Reference from a live ``var dinf unem, lags(1/2)`` then ``irf table fevd, step(10)``::

    Step | (dinf->dinf) (dinf->unem) (unem->dinf) (unem->unem)
       1 |      1        .005822        0         .994178
       2 |   .878729     .002094     .121271      .997906
       3 |   .876761     .001063     .123239      .998937
      10 |   .877728     .000611     .122272      .999389

Note Stata's numbering: step 1 is the first forecast horizon. The raw statsmodels object
indexes the same numbers from 0.

The underlying NUMBERS already matched; what was missing was a readable layout, and getting the
layout right took two attempts:

1. the raw ``decomp`` is shaped ``(shock, period, response)``, not the intuitive
   ``(period, response, shock)`` -- indexing it the intuitive way reads the wrong cell;
2. Stata's column order varies the RESPONSE fastest -- (dinf->dinf), (dinf->unem),
   (unem->dinf), (unem->unem). Looping the other way gave real numbers in the wrong columns,
   which is the failure mode hardest to notice in a table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA = {
    0: (1.0, 0.005822, 0.0, 0.994178),
    1: (0.878729, 0.002094, 0.121271, 0.997906),
    2: (0.876761, 0.001063, 0.123239, 0.998937),
    9: (0.877728, 0.000611, 0.122272, 0.999389),
}


@pytest.fixture(scope="module")
def table():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    result = ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()
    return result.fevd_table(periods=10)


def test_columns_are_ordered_impulse_then_response(table) -> None:
    """(dinf->dinf), (dinf->unem), (unem->dinf), (unem->unem)."""

    assert list(table.columns) == [
        ("dinf", "dinf"),
        ("dinf", "unem"),
        ("unem", "dinf"),
        ("unem", "unem"),
    ]


def test_values_match_stata(table) -> None:
    for row, expected in STATA.items():
        observed = tuple(float(value) for value in table.iloc[row])
        for got, want in zip(observed, expected):
            assert got == pytest.approx(want, abs=5e-7), (row, got, want)


def test_each_row_sums_to_one(table) -> None:
    """A variance decomposition that does not sum to 1 is not a decomposition."""

    for row in range(len(table)):
        total = (
            float(table.iloc[row][("dinf", "dinf")])
            + float(table.iloc[row][("unem", "dinf")])
        )
        assert total == pytest.approx(1.0, abs=1e-9), row


def test_the_columns_pair_correctly(table) -> None:
    """Guard the transposed-layout bug: swapped columns are still 'real' numbers."""

    row = table.iloc[1]
    # dinf->unem is tiny (.002); unem->dinf is not (.121). Swapping them is detectable.
    assert float(row[("dinf", "unem")]) < 0.01
    assert float(row[("unem", "dinf")]) > 0.1
