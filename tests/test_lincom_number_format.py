"""lincom's number format: eight significant characters, not seven significant figures.

A comment in lincom.py claimed ``%9.0g`` -- "seven significant figures". Neither the option's
meaning nor the printed output supports that. Captured from two fits against a live Stata:

    cobb_douglas  y ~ k + l, lincom k + l
        '         (1) |   1.075205   .1159063     9.28   0.000     .8341646    1.316245'

    grilic        lnw ~ s + expr + tenure, lincom s
        '         (1) |   .1066109   .0059536    17.91   0.000     .0949233    .1182984'

Below 1 Stata keeps SEVEN decimals (.005953578 -> ".0059536"); at or above 1 it drops one
(1.0752048 -> "1.075205"), so the cell stays eight characters wide.

The old formatter used seven significant figures, which printed ".005953578" -- three characters
wider than Stata's cell, and enough to push the row past 78 columns and break every line-wise
comparison.

Also added: the blank line Stata puts between the "( 1)  s = 0" marker and the table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.lincom import _lincom_g

GIRLIC_BLOCK = [
    " ( 1)  s = 0",
    "",
    "------------------------------------------------------------------------------",
    "         lnw | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]",
    "-------------+----------------------------------------------------------------",
    "         (1) |   .1066109   .0059536    17.91   0.000     .0949233    .1182984",
    "------------------------------------------------------------------------------",
]

COBB_BLOCK = [
    "------------------------------------------------------------------------------",
    "           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]",
    "-------------+----------------------------------------------------------------",
    "         (1) |   1.075205   .1159063     9.28   0.000     .8341646    1.316245",
    "------------------------------------------------------------------------------",
]


@pytest.fixture(scope="module")
def grilic():
    try:
        frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\grilic.dta").dropna(
            subset=["lnw", "s", "expr", "tenure"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return str(ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit().lincom("s"))


@pytest.fixture(scope="module")
def cobb():
    try:
        frame = pd.read_stata(
            r"D:\myCodeProject\test_westata\data\cobb_douglas.dta", convert_categoricals=False
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("cobb_douglas.dta is not available")
    return str(ws.OLS.from_formula("y ~ k + l", frame).fit().lincom("k + l"))


def test_the_grilic_block_matches(grilic) -> None:
    lines = grilic.splitlines()
    for expected in GIRLIC_BLOCK:
        assert expected in lines, repr(expected)


def test_the_cobb_block_matches(cobb) -> None:
    lines = cobb.splitlines()
    for expected in COBB_BLOCK:
        assert expected in lines, repr(expected)


def test_small_values_keep_seven_decimals() -> None:
    assert _lincom_g(0.005953578) == ".0059536"
    assert _lincom_g(0.1066109) == ".1066109"


def test_values_at_or_above_one_drop_a_decimal() -> None:
    """Seven decimals here printed "1.0752048", one character wider than Stata's cell."""

    assert _lincom_g(1.0752048) == "1.075205"
    assert _lincom_g(1.3162451) == "1.316245"


def test_the_cell_is_at_most_eight_characters() -> None:
    for value in (0.005953578, 1.0752048, 0.8341646, 9.011188):
        assert len(_lincom_g(value)) <= 8, (value, _lincom_g(value))
