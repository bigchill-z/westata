"""A bare `var a b` is a working VAR(2) reference here -- omit the lags() option.

Rounds 39-47 concluded "`var` is broken in this environment". Round 86 narrowed that to "only an
explicit lags(k>=2) is broken". This round finds the practical consequence: the NO-OPTION form
estimates a complete VAR(2) and can be used as the reference directly.

    var dinf unem          10 coefficients   L.dinf L2.dinf L.unem L2.unem _cons  (x2)
    var dinf unem, lags(2)  6 coefficients   only L2 terms

Against westata's VAR(2) on the first equation:

    term        Stata          westata
    L.dinf     -0.47094365    -0.47094365
    L.unem     -2.24205009    -2.24205009
    L2.dinf    -0.40103104    -0.40103104
    L2.unem     2.03416999     2.03416999
    _cons       0.01237643     0.01237643

every digit identical. So chapter 13's references do not need hand-built lag columns after all --
they need the option OMITTED.

Note the parameter ORDER differs and must be mapped by name, not position: Stata lists
[L, L2] per variable within each equation, while westata lists [L1, L1, L2, L2] across equations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

#: From a bare ``var dinf unem`` (no lags option) -- a complete VAR(2).
STATA_EQUATION_ONE = {
    "L.dinf": -0.47094365,
    "L.unem": -2.24205009,
    "L2.dinf": -0.40103104,
    "L2.unem": 2.03416999,
    "_cons": 0.01237643,
}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna().reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VAR(frame[["dinf", "unem"]].to_numpy(float), lags=2, names=["dinf", "unem"]).fit()


def test_the_first_equation_matches_a_bare_var(fitted) -> None:
    parameters = np.asarray(fitted.params)
    mapping = {
        "L.dinf": parameters[1, 0],
        "L.unem": parameters[2, 0],
        "L2.dinf": parameters[3, 0],
        "L2.unem": parameters[4, 0],
        "_cons": parameters[0, 0],
    }
    for name, expected in STATA_EQUATION_ONE.items():
        assert float(mapping[name]) == pytest.approx(expected, abs=5e-9), name


def test_the_second_equation_matches_too(fitted) -> None:
    parameters = np.asarray(fitted.params)
    # Captured from the same bare var run, second equation.
    expected = {
        "L.dinf": 0.03052226,
        "L.unem": 1.63803069,
        "L2.dinf": -0.01131722,
        "L2.unem": -0.67258072,
        "_cons": 0.00201387,
    }
    mapping = {
        "L.dinf": parameters[1, 1],
        "L.unem": parameters[2, 1],
        "L2.dinf": parameters[3, 1],
        "L2.unem": parameters[4, 1],
        "_cons": parameters[0, 1],
    }
    for name, value in expected.items():
        assert float(mapping[name]) == pytest.approx(value, abs=5e-9), name


def test_the_estimator_is_a_full_var2(fitted) -> None:
    """Five parameters per equation; the broken `var, lags(2)` path stores three."""

    assert len(fitted.params) == 5
    assert np.asarray(fitted.params).shape == (5, 2)
