"""Pin ``contrast``'s joint Wald block against Stata.

Reference from a live ``contrast i.iqgrp`` after ``reg lnw s expr tenure i.iqgrp`` on
grilic.dta (iq split into quartiles)::

    Contrasts of marginal linear predictions
    Margins: asbalanced
    ------------------------------------------------
                 |         df           F        P>F
    -------------+----------------------------------
           iqgrp |          3        5.90     0.0006
    ------------------------------------------------
     Denominator |        751

The per-comparison rows were already right to every printed digit in both directions
(table below), but Stata prints the JOINT block by default and the rows only under
``nowald``. westata printed the rows unconditionally, so a caller asking for ``contrast``
got a different table from the command of the same name.

    iqgrp      Contrast   Std. err.
    2 vs 1    .1041878   .0363683
    3 vs 1    .0997046   .0383864
    4 vs 1    .1702506   .0413214
    3 vs 2   -.0044832   .0366351
    4 vs 2    .0660628   .0383558
    4 vs 3    .0705460   .0373411
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

STATA_ROWS = {
    "2 vs 1": (0.1041878, 0.0363683),
    "3 vs 1": (0.0997046, 0.0383864),
    "4 vs 1": (0.1702506, 0.0413214),
    "3 vs 2": (-0.0044832, 0.0366351),
    "4 vs 2": (0.0660628, 0.0383558),
    "4 vs 3": (0.0705460, 0.0373411),
}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    frame = frame.copy()
    frame["iqgrp"] = pd.qcut(frame["iq"], 4, labels=False).astype(int) + 1
    return ws.OLS.from_formula("lnw ~ s + expr + tenure + C(iqgrp)", frame).fit()


def test_the_joint_wald_block_matches_stata(fitted) -> None:
    result = ws.contrast(fitted, "C(iqgrp)")
    assert result.df == 3
    assert result.fvalue == pytest.approx(5.90, abs=5e-3)
    assert result.pvalue == pytest.approx(0.0006, abs=5e-5)


def test_nowald_returns_the_per_comparison_rows(fitted) -> None:
    rows = ws.contrast(fitted, "C(iqgrp)", nowald=True)
    assert len(rows) == 3  # three levels against the base


def test_pwcompare_rows_match_stata(fitted) -> None:
    """The pairwise table's numbers were already exact; this pins them."""

    rows = ws.pwcompare(fitted, "C(iqgrp)")
    observed = {row.label: (row.estimate, row.std_err) for row in rows}
    assert set(observed) == set(STATA_ROWS)
    for label, (estimate, std_err) in STATA_ROWS.items():
        assert observed[label][0] == pytest.approx(estimate, abs=5e-8), label
        assert observed[label][1] == pytest.approx(std_err, abs=5e-8), label


def test_the_joint_result_is_still_a_list(fitted) -> None:
    """Backwards compatibility: existing callers iterate over the comparisons."""

    result = ws.contrast(fitted, "C(iqgrp)")
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(hasattr(row, "estimate") for row in result)
