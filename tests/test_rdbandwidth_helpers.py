"""Pins for the two ``rdrobust`` bandwidth helpers that DO reproduce Stata.

The selector as a whole is NOT implemented -- ``rdrobust`` still raises ``UnsupportedError``
when ``bandwidth`` is omitted, and nothing in the public API imports
``westata.estimators.rdbandwidth``. What is tested here are the two pieces that were measured
against Stata's own behaviour and are exact, so a future attempt starts from them rather than
re-deriving them.

Both were errors in earlier attempts:

* the usable sample is 1297, not the 1390 rows of the file -- ``vote`` has missing values and
  Stata drops them pair-wise before anything else. Scaling by N = 1390 with standard
  deviations that counted the missing rows is why rounds 10 and 11 were off by whole
  multiples.
* the pilot is ``C_c * min(1, (iqr/sd)/1.349) * N^(-1/5)`` with ``C_c = 2.576`` for the
  triangular kernel, computed on the standardised running variable because ``stdvars(on)`` is
  the default (``rdbwselect.ado:326-334, 361-369, 459``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.estimators.rdbandwidth import (
    C_C_EPANECHNIKOV,
    C_C_TRIANGULAR,
    C_C_UNIFORM,
    pilot_bandwidth,
    prepare_sample,
)

DATA = r"D:\myCodeProject\test_westata\work\rdrobust_senate.dta"


@pytest.fixture(scope="module")
def senate():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("rdrobust_senate.dta is not available")


def test_the_usable_sample_is_1297_not_1390(senate) -> None:
    """``vote`` has missing values; Stata deletes them before selecting a bandwidth."""

    raw = senate["vote"].to_numpy()
    assert raw.size == 1390
    y, x = prepare_sample(raw, senate["margin"].to_numpy())
    assert y.size == 1297
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(x))


def test_rows_missing_either_variable_are_dropped() -> None:
    y, x = prepare_sample(
        np.array([1.0, np.nan, 3.0, 4.0]), np.array([0.1, 0.2, np.nan, 0.4])
    )
    assert y.size == 2
    assert list(y) == [1.0, 4.0]


def test_the_pilot_matches_statas_internal_value(senate) -> None:
    """0.457666 is what rdbwselect computes internally before the three nesting levels."""

    _, x = prepare_sample(senate["vote"].to_numpy(), senate["margin"].to_numpy())
    assert float(pilot_bandwidth(x)) == pytest.approx(0.457666, abs=5e-7)


def test_the_kernel_constants_are_the_documented_ones() -> None:
    assert C_C_TRIANGULAR == 2.576
    assert C_C_EPANECHNIKOV == 2.34
    assert C_C_UNIFORM == 1.843


def test_rdrobust_still_refuses_to_guess_a_bandwidth(senate) -> None:
    """Guard the honest failure: a 50x-wrong selector must not be silently used."""

    from westata.estimators.rdrobust import rdrobust
    from westata.errors import UnsupportedError

    with pytest.raises(UnsupportedError, match="bandwidth"):
        rdrobust(senate["vote"].to_numpy(), senate["margin"].to_numpy())
