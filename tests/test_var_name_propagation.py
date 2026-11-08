"""Column names survive into every table derived from a VAR.

Round 162 found that `VAR(dataframe)` discarded the column names and invented y1/y2, so every equation
label in every derived table said `y1` where the caller had passed `dinf`. That is what made varlmar and
varwle look like formatting disagreements.

This file walks the whole family at once, because a name lost in one place is invisible in the others:

    fit.names                     ['dinf', 'unem']
    irf_table footnotes           impulse = dinf, and response = unem.
    fevd_table columns            (dinf, dinf) (dinf, unem) (unem, dinf) (unem, unem)
    vargranger equations          dinf / unem
    varwle equation headings      Equation: dinf / Equation: unem
    varlmar                       no names, and Stata's table has none either
    varstable                     no names, and Stata's table has none either

THE LAST TWO ARE NOT OMISSIONS. Both tables are properties of the SYSTEM rather than of an equation, so
Stata prints no variable names in either -- verified by comparing both against a live run, 11 of 11 lines
for varstable. A guard that demanded names there would be wrong.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

MACRO = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(MACRO).dropna(subset=["dinf", "unem"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("macro_swatson.dta is not available")


@pytest.fixture(scope="module")
def var_fit(frame):
    return ws.VAR(frame[["dinf", "unem"]], lags=2).fit()


def test_the_fit_keeps_the_column_names(var_fit) -> None:
    assert list(var_fit.names) == ["dinf", "unem"]


def test_the_irf_footnotes_use_them(var_fit) -> None:
    rendered = str(var_fit.irf_table(periods=2))
    assert "impulse = dinf, and response = unem." in rendered
    assert "impulse = unem, and response = dinf." in rendered


def test_the_fevd_columns_use_them(var_fit) -> None:
    columns = list(var_fit.fevd_table(periods=2).columns)
    assert columns == [
        ("dinf", "dinf"),
        ("dinf", "unem"),
        ("unem", "dinf"),
        ("unem", "unem"),
    ]


def test_granger_equations_use_them(var_fit) -> None:
    rendered = str(ws.vargranger(var_fit))
    assert "dinf" in rendered
    assert "unem" in rendered


def test_the_wald_table_headings_use_them(var_fit) -> None:
    rendered = str(ws.varwle(var_fit))
    assert "Equation: dinf" in rendered
    assert "Equation: unem" in rendered


def test_the_lm_table_has_no_variable_names(var_fit) -> None:
    """Stata's varlmar is a property of the system, not of an equation -- checked against a live run."""

    rendered = str(ws.varlmar(var_fit))
    assert "dinf" not in rendered
    assert "Lagrange-multiplier test" in rendered


def test_the_stability_table_has_no_variable_names(var_fit) -> None:
    rendered = str(ws.varstable(var_fit))
    assert "dinf" not in rendered
    assert "Eigenvalue" in rendered


def test_an_array_input_still_gets_generated_names(frame) -> None:
    """No labels to keep, so the fallback is correct rather than a defect."""

    fitted = ws.VAR(frame[["dinf", "unem"]].to_numpy(), lags=2).fit()
    assert list(fitted.names) == ["y1", "y2"]
