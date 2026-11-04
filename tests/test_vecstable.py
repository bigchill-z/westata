"""Pin ``vecstable`` against Stata's VECM stability table.

Reference from a live ``vecstable`` after ``vec dinf unem, lags(2) rank(1)``::

       Eigenvalue              Modulus
        1                       1
       -.2233155 + .6276579i    .666201
       -.2233155 - .6276579i    .666201
        .6305164                .630516

    The VECM specification imposes a unit modulus.

All four reproduce, including the imposed unit root -- which is a property of the
specification, not a failure of the fit.

``vecstable.ado`` is six lines and forwards to ``varstable_w2, vec``; the companion is built
in ``varstable_w.ado:95-150`` from the VECM's short-run matrices. Building it from the VAR's
``coefs`` instead -- the obvious shortcut -- returns .82182 / .797022 / .668533 / .668533,
which are the VAR(2) eigenvalues and look perfectly plausible while being the wrong object.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.vecstable import vecstable

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_MODULUS = (1.0, 0.666201, 0.666201, 0.630516)


@pytest.fixture(scope="module")
def fitted_vecm():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VECM(values, coint_rank=1, lags=2, names=["dinf", "unem"]).fit()


def test_modulus_matches_stata(fitted_vecm) -> None:
    result = vecstable(fitted_vecm)
    assert len(result.modulus) == 4
    for got, expected in zip(result.modulus, STATA_MODULUS):
        assert got == pytest.approx(expected, abs=5e-6), (got, expected)


def test_the_complex_pair_matches_stata(fitted_vecm) -> None:
    """-.2233155 +- .6276579i, which no real-valued formula would produce by accident."""

    values = vecstable(fitted_vecm).eigenvalues
    pair = [complex(v) for v in values if abs(complex(v).imag) > 1e-9]
    assert len(pair) == 2
    for value in pair:
        assert value.real == pytest.approx(-0.2233155, abs=5e-7)
        assert abs(value.imag) == pytest.approx(0.6276579, abs=5e-7)


def test_the_imposed_unit_root_is_reported(fitted_vecm) -> None:
    """A VECM of rank 1 in 2 equations must show exactly one unit modulus."""

    result = vecstable(fitted_vecm)
    assert sum(1 for value in result.modulus if abs(value - 1.0) < 1e-9) == 1
    assert result.stable is True


def test_the_companion_is_not_the_var_companion(fitted_vecm) -> None:
    """Guard the wrong-object trap: VAR(2) eigenvalues are different numbers."""

    values = vecstable(fitted_vecm).modulus
    var_like = (0.82182, 0.797022, 0.668533, 0.668533)
    for expected in var_like:
        assert all(abs(value - expected) > 1e-3 for value in values)
