"""``VARResults``' Stata-facing accessors must all be ATTRIBUTES, not methods.

``stata_param_names``, ``bse`` and ``aic`` were properties; ``stata_params`` was the only
one written as a plain method. So three of the four returned values and one returned a
bound function:

    result.stata_param_names    -> list
    result.bse                  -> Series
    result.aic                  -> float
    result.stata_params         -> <bound method>   <-- inconsistent

``result.stata_params[0]`` therefore raised ``TypeError`` on an object that looked
correct, and a caller had to remember which of four neighbouring accessors needed ``()``.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws


def _system(n: int = 200, seed: int = 1300):
    rng = np.random.default_rng(seed)
    return np.column_stack([np.cumsum(rng.normal(size=n)) for _ in range(2)])


@pytest.fixture(scope="module")
def var_fit():
    return ws.VAR(_system(), lags=2, names=["y1", "y2"]).fit()


def test_stata_params_is_a_value_not_a_method(var_fit) -> None:
    """Indexing it must work directly, like every sibling accessor."""

    assert not callable(var_fit.stata_params)
    # And it behaves like a Series.
    assert len(var_fit.stata_params) > 0
    assert var_fit.stata_params.iloc[0] is not None


def test_the_four_accessors_are_consistent(var_fit) -> None:
    """All four return values; none is a bound method."""

    for name in ("stata_param_names", "bse", "aic", "stata_params"):
        value = getattr(var_fit, name)
        assert not callable(value), name


def test_the_names_and_values_are_aligned(var_fit) -> None:
    """``stata_params`` is indexed by ``stata_param_names``."""

    assert list(var_fit.stata_params.index) == list(var_fit.stata_param_names)


def test_the_length_is_the_full_parameter_count(var_fit) -> None:
    """Two equations, each with ``2*2 + 1 = 5`` coefficients."""

    assert len(var_fit.stata_params) == 10
    assert len(var_fit.stata_params) == len(var_fit.stata_param_names)


def test_the_bse_line_up_with_the_params(var_fit) -> None:
    """Both are series over the same index, so they can be divided."""

    assert list(var_fit.bse.index) == list(var_fit.stata_params.index)
    ratio = var_fit.stata_params / var_fit.bse
    assert np.isfinite(np.asarray(ratio, dtype=float)).all()


def test_the_stata_order_is_not_the_raw_order(var_fit) -> None:
    """Guard: if the two orders coincided, the reordering would be untested.

    statsmodels stores the deterministic block first; Stata stores ``_cons`` last in
    each equation, so the flattened vectors differ.
    """

    stata = np.asarray(var_fit.stata_params, dtype=float)
    raw = np.asarray(var_fit.params, dtype=float).ravel()

    assert not np.allclose(stata, raw)


def test_the_constant_comes_last_in_each_equation(var_fit) -> None:
    """Stata puts ``_cons`` after the lag terms, not before them."""

    names = var_fit.stata_param_names

    # Five coefficients per equation: four lag terms then the constant.
    assert names[4] == "_cons"
    assert names[9] == "_cons"
    assert names[0].startswith("L1.")


def test_the_lag_naming_matches_stata(var_fit) -> None:
    """``L1.y1``, ``L2.y1``, ``L1.y2``, ``L2.y2`` -- every lag carries its digit.

    The previous version of this test asserted ``L.y1`` for the FIRST lag and described that
    as "matching Stata". It does not: ``var dinf unem, lags(1/2)`` prints ``L1.`` for the first
    lag row, and the rendered table read ``L.`` where Stata reads ``L1.``. The ordering claim
    in the old docstring (variable outer, lag inner) was right; only the digit was missing.
    """

    assert var_fit.stata_param_names[:4] == ["L1.y1", "L2.y1", "L1.y2", "L2.y2"]