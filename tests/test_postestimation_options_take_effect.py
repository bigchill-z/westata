"""Postestimation options must actually DO something.

Round 138 found ``estat gof, group()`` being silently ignored after ``poisson``: the
argument was accepted, the uncollapsed test ran, and only the label would have hinted
at the difference. That is the dangerous shape -- a plausible number for the wrong
request.

These tests cover the two ways an option can go wrong:

* an UNKNOWN option must be refused, not absorbed by ``**kwargs``;
* a KNOWN option must change the output, checked against data where the change is
  visible.

The ``nosort`` case is the one that hides best: with two nearly equal VIFs the sorted
and unsorted orders can coincide, so a broken ``nosort`` passes. Round 33 hit exactly
that; the fixture here forces clearly different VIFs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def collinear_frame():
    """``x1`` and ``x2`` are near-duplicates; ``x3`` is independent.

    That gives VIFs in the hundreds against one near 1, so descending order is
    unambiguous -- while the MODEL order is ``x3, x1, x2``.
    """

    rng = np.random.default_rng(641)
    n = 400
    x3 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = x1 * 0.95 + rng.normal(size=n) * 0.05
    return pd.DataFrame({"y": rng.normal(size=n), "x3": x3, "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def collinear_result(collinear_frame):
    return ws.OLS.from_formula("y ~ x3 + x1 + x2", collinear_frame).fit()


def test_the_fixture_really_separates_the_orders(collinear_result) -> None:
    """Guard for the fixture: if the orders coincided, the test below proves nothing."""

    sorted_names = list(collinear_result.estat_vif().names)
    unsorted_names = list(collinear_result.estat_vif(nosort=True).names)

    assert sorted_names != unsorted_names


def test_nosort_keeps_the_model_order(collinear_result) -> None:
    """``vif.ado:121-122`` sorts by VIF descending unless ``nosort`` is given."""

    model_order = [
        str(name) for name in collinear_result.param_names if str(name) != "Intercept"
    ]

    assert list(collinear_result.estat_vif(nosort=True).names) == model_order


def test_the_default_sorts_by_descending_vif(collinear_result) -> None:
    """The sorted order must actually be by VIF, not alphabetical or model order."""

    table = collinear_result.estat_vif().to_frame()
    values = list(table["VIF"])

    assert values == sorted(values, reverse=True)
    # The collinear pair outranks the independent regressor.
    assert list(collinear_result.estat_vif().names)[-1] == "x3"


def test_the_unsorted_and_sorted_rows_carry_the_same_numbers(collinear_result) -> None:
    """Sorting rearranges rows; it must not recompute anything."""

    sorted_table = collinear_result.estat_vif().to_frame()
    unsorted_table = collinear_result.estat_vif(nosort=True).to_frame()

    for name in sorted_table.index:
        assert sorted_table.loc[name, "VIF"] == pytest.approx(
            unsorted_table.loc[name, "VIF"], rel=1e-12
        )


@pytest.mark.parametrize(
    "method",
    ["estat_vif", "estat_summarize", "estat_ic"],
)
def test_unknown_options_are_refused(collinear_result, method) -> None:
    """``**kwargs`` must not swallow a misspelled or invented option.

    Accepting it silently is worse than a TypeError: the caller believes the option
    took effect and compares a result that never applied it.
    """

    with pytest.raises(TypeError):
        getattr(collinear_result, method)(bogus_option=1)


def test_estat_vce_refuses_an_unknown_option(collinear_result) -> None:
    """``estat vce`` takes ``correlation``; anything else is a mistake."""

    with pytest.raises(TypeError):
        collinear_result.estat_vce(bogus_option=1)


def test_the_correlation_option_changes_the_matrix(collinear_result) -> None:
    """``estat vce, correlation`` must rescale, not return the covariance again."""

    covariance = np.asarray(collinear_result.estat_vce().to_frame(), dtype=float)
    correlation = np.asarray(
        collinear_result.estat_vce(correlation=True).to_frame(), dtype=float
    )

    assert not np.allclose(covariance, correlation)
    # Correlations are bounded; covariances here are not (the VIFs are large).
    off_diagonal = correlation[~np.eye(correlation.shape[0], dtype=bool)]
    assert np.all(np.abs(off_diagonal) <= 1.0 + 1e-12)


def test_summarize_honours_the_variables_argument(collinear_frame) -> None:
    """``estat summarize`` with an explicit list must report only those variables."""

    result = ws.OLS.from_formula("y ~ x3 + x1", collinear_frame).fit()

    table = result.estat_summarize(variables=["x3"]).table

    assert list(table.index) == ["x3"]