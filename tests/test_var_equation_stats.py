"""Pin the VAR's per-equation statistics to Stata's printed column values.

Reference from a live ``var dinf unem, lags(1/2)`` on macro_swatson.dta::

    Equation   Parms     RMSE     R-sq      chi2     P>chi2
    dinf           5  .013815   0.3621   93.11047    0.0000
    unem           5  .002508   0.9731   5926.554    0.0000

Three defects had to be fixed to reproduce those, and each was invisible in a table that
looked plausible:

* ``rmse`` divided by ``df_eq`` instead of ``N - df_eq``: .077907 against .013815;
* ``r2`` centred the FULL series rather than the estimation sample: .3648 against .3621;
* ``chi2`` strided the parameter vector by ``neqs`` although it is laid out equation by
  equation, mixing both equations: 75.34 against 93.11.

The tests that previously existed asserted the first two formulas directly, so they had to be
corrected as well -- a test can pin a bug just as firmly as it can pin a fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA = {
    1: {"parms": 5, "rmse": 0.013815, "r2": 0.3621, "chi2": 93.11047},
    2: {"parms": 5, "rmse": 0.002508, "r2": 0.9731, "chi2": 5926.554},
}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()


def test_rmse_matches_stata(fitted) -> None:
    for index, expected in STATA.items():
        assert fitted.e[f"rmse_{index}"] == pytest.approx(
            expected["rmse"], rel=5e-4
        ), index


def test_rsquared_matches_stata(fitted) -> None:
    for index, expected in STATA.items():
        assert fitted.e[f"r2_{index}"] == pytest.approx(
            expected["r2"], abs=5e-4
        ), index


def test_wald_statistics_match_stata(fitted) -> None:
    for index, expected in STATA.items():
        assert fitted.e[f"chi2_{index}"] == pytest.approx(
            expected["chi2"], rel=5e-5
        ), index


def test_parameter_counts_match_stata(fitted) -> None:
    assert int(fitted.e["df_eq"]) == 5


def test_the_two_equations_are_not_mixed_up(fitted) -> None:
    """The chi2 stride bug produced two numbers from a blend of both equations."""

    first = float(fitted.e["chi2_1"])
    second = float(fitted.e["chi2_2"])
    assert first < second / 10.0, (first, second)
