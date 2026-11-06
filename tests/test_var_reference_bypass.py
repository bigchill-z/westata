"""VAR(2) coefficients match a hand-built Stata regression exactly.

Round 39 found that this Stata's ``var ..., lags(3)`` uses ONLY the third lag -- ``Sample: 4 thru
200``, ``Parms=3``, a single ``L3.`` block -- reproduced on synthetic data, so it is the command,
not the dataset. Every VAR reference in the gap report came through that option, which means
chapter 13's low line-agreement scores (VAR 0.54, varstable 0.27) may be measuring the reference
rather than this package.

To get a trustworthy reference, the lagged columns are built BY HAND and the equation estimated
with ``regress``, which never consults ``lags()``. On macro_swatson the two agree to every printed
digit:

    term        Stata (hand-built)     westata
    _cons         .00847076             .00847076
    L1.inf        .62857689             .62857689
    L1.unem      -1.93905057           -1.93905057
    L2.inf        .32340678             .32340678
    L2.unem       1.83183975            1.83183975

and varstable is internally consistent with those coefficients: the companion matrix built from
them has eigenvalues .951817 / .771230 x2 / .286587, which is exactly what varstable reports.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

#: The same equation fitted in Stata with hand-built lag columns.
STATA_VAR2 = {
    "_cons": 0.00847076,
    "L1.inf": 0.62857689,
    "L1.unem": -1.93905057,
    "L2.inf": 0.32340678,
    "L2.unem": 1.83183975,
}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["inf", "unem"]].dropna().to_numpy(float)
    return ws.VAR(values, lags=2, names=["inf", "unem"]).fit()


def test_the_inf_equation_matches_stata_exactly(fitted) -> None:
    """Every printed digit of the first equation."""

    # params is (constant, L1.inf, L1.unem, L2.inf, L2.unem) by ROW, one column per equation.
    column = np.asarray(fitted.params)[:, 0]
    names = ["_cons", "L1.inf", "L1.unem", "L2.inf", "L2.unem"]
    for name, value in zip(names, column):
        assert float(value) == pytest.approx(STATA_VAR2[name], abs=5e-9), name


def test_the_parameter_count_is_five(fitted) -> None:
    """The reference VAR reports three because of the broken lags(); ours is the real count."""

    assert len(fitted.params) == 5
    assert list(fitted.params.index) == ["const", "L1.inf", "L1.unem", "L2.inf", "L2.unem"]


def test_varstable_is_internally_consistent(fitted) -> None:
    """The reported eigenvalues must equal those of the companion matrix we fitted."""

    coefficients = np.asarray(fitted.params)
    k = 2
    companion = np.zeros((2 * k, 2 * k))
    companion[:k, :k] = coefficients[1:3].T
    companion[:k, k:] = coefficients[3:5].T
    companion[k:, :k] = np.eye(k)
    expected = sorted(np.abs(np.linalg.eigvals(companion)), reverse=True)

    # The table drops the leading zero, as Stata does, so compare against that spelling.
    rendered = str(ws.varstable(fitted))
    for value in expected:
        text_value = f"{value:.6g}"
        if text_value.startswith("0."):
            text_value = text_value[1:]
        assert text_value in rendered, text_value


def test_all_moduli_are_inside_the_unit_circle(fitted) -> None:
    """macro_swatson's VAR(2) is stable, and the verdict line says so."""

    rendered = str(ws.varstable(fitted))
    assert "inside the unit circle" in rendered
