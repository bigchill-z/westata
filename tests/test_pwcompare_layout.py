"""Pin ``pwcompare`` to Stata's table.

Reference from a live ``regress consumption i.grp temp`` then ``pwcompare grp``::

    Pairwise comparisons of marginal linear predictions
    Margins: asbalanced
    --------------------------------------------------------------
                 |                                 Unadjusted
                 |   Contrast   Std. err.     [95% conf. interval]
    -------------+------------------------------------------------
             grp |
         2 vs 1  |  -.0035617   .0194151     -.0434699    .0363465
         3 vs 1  |   .0112906   .0194011     -.0285889    .0511701
         3 vs 2  |   .0148523   .0193778     -.0249792    .0546839
    --------------------------------------------------------------

All three contrasts already matched Stata numerically; what was missing was the table.
``pwcompare`` returned a plain list, so ``str()`` printed a list-of-dataclass repr.

Measured geometry: every table line is 62 characters with the pipe at column 13; the
contrast ends at 25, the standard error at 36, the lower bound at 50 and the upper bound
at 62. The cells use seven decimal places with trailing zeros dropped, like ``margins``.

``PwCompareTable`` subclasses ``list`` so ``isinstance(result, list)`` still holds, which
existing callers rely on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"


@pytest.fixture(scope="module")
def table():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    frame["grp"] = np.tile([1, 2, 3], len(frame))[: len(frame)].astype(int)
    fit = ws.OLS.from_formula("consumption ~ C(grp) + temp", frame).fit()
    return ws.pwcompare(fit, "grp")


RENDERED = [
    "Pairwise comparisons of marginal linear predictions",
    "Margins: asbalanced",
    "-" * 62,
    "             |                                 Unadjusted",
    "             |   Contrast   Std. err.     [95% conf. interval]",
    "-" * 13 + "+" + "-" * 48,
    "         grp |",
    "     2 vs 1  |  -.0035617   .0194151     -.0434699    .0363465",
    "     3 vs 1  |   .0112906   .0194011     -.0285889    .0511701",
    "     3 vs 2  |   .0148523   .0193778     -.0249792    .0546839",
    "-" * 62,
]


def test_every_line_matches_stata(table) -> None:
    assert str(table).splitlines() == RENDERED


def test_it_is_still_a_list(table) -> None:
    """The return shape other callers depend on."""

    assert isinstance(table, list)
    assert len(table) == 3


def test_the_group_and_heading_are_present(table) -> None:
    rendered = str(table)
    assert "Pairwise comparisons of marginal linear predictions" in rendered
    assert "Margins: asbalanced" in rendered
    assert "         grp |" in rendered


def test_the_cells_end_at_25_36_50_and_62(table) -> None:
    rendered = str(table).splitlines()
    row = next(line for line in rendered if line.strip().startswith("2 vs 1"))
    assert row.index("|") == 13
    assert row.index("-.0035617") == 16
    assert len(row) == 62


def test_the_values_are_the_pairwise_differences(table) -> None:
    """3 vs 1 should equal the 3 vs 2 contrast plus the 2 vs 1 one."""

    by_label = {row.label: row.estimate for row in table}
    assert by_label["3 vs 1"] == pytest.approx(
        by_label["2 vs 1"] + by_label["3 vs 2"], rel=1e-12
    )
