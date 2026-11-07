"""`wald_test` accepts a constraint STRING, as statsmodels does.

The v3 report records this at section 4, item (1): statsmodels' ``wald_test`` takes a formula-like
expression -- ``result.wald_test("s = 0")`` -- and this package converted the argument straight to
a float matrix, so a string raised

    ValueError: could not convert string to float: 's = 0'

The parsing already existed: ``result.test("s = 0")`` speaks Stata's constraint syntax and returns
``statistic``, ``df``, ``df_denom``, ``pvalue`` and ``distribution`` -- the same shape a
``wald_test`` caller wants. So the string form now delegates rather than rejecting.

Both spellings are accepted, because statsmodels allows either:

    result.wald_test("s = 0")
    result.wald_test(["s = 0"])

A matrix argument still goes down the matrix path, unchanged, so nothing that already worked moved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

GIRLIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(GIRLIC).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr", frame).fit()


def test_a_string_constraint_is_accepted(fitted) -> None:
    """It used to raise ValueError: could not convert string to float."""

    result = fitted.wald_test("s = 0")
    assert np.isfinite(float(result.statistic))


def test_a_list_of_strings_is_accepted(fitted) -> None:
    result = fitted.wald_test(["s = 0"])
    assert np.isfinite(float(result.statistic))


def test_the_string_form_agrees_with_test(fitted) -> None:
    """Delegation, not a parallel implementation: the numbers must be identical."""

    from_string = fitted.wald_test("s = 0")
    from_test = fitted.test("s = 0")
    assert float(from_string.statistic) == float(from_test.statistic)
    assert float(from_string.pvalue) == float(from_test.pvalue)


def test_the_result_carries_the_statsmodels_fields(fitted) -> None:
    result = fitted.wald_test("s = 0")
    for field in ("statistic", "pvalue", "df_denom"):
        assert hasattr(result, field), field


def test_a_mathematical_constraint_works_too(fitted) -> None:
    """Stata's syntax includes combinations: `s + expr = 0`."""

    result = fitted.wald_test("s + expr = 0")
    assert np.isfinite(float(result.statistic))


def test_a_matrix_argument_still_works(fitted) -> None:
    """The numeric path is untouched: R must have one column per parameter."""

    names = list(fitted.param_names)
    row = np.zeros((1, len(names)))
    row[0, names.index("s")] = 1.0
    result = fitted.wald_test(row, [0.0])
    assert np.isfinite(float(result.statistic))


def test_a_wrong_shaped_matrix_still_raises(fitted) -> None:
    """The guard that made strings fail must keep rejecting genuinely bad matrices."""

    with pytest.raises(ValueError, match="columns"):
        fitted.wald_test(np.zeros((1, 2)))
