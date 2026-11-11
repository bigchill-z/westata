"""The collinearity notes `estat ovtest, rhs` prints above its table.

When the auxiliary regression cannot identify a column, Stata says so before the table:

    note: expr2 omitted because of collinearity.
    note: expr2^2 omitted because of collinearity.
    (blank)
    Ramsey RESET test for omitted variables

The package computed the same omission -- that is what makes the df 11 rather than 12 -- but printed
nothing about it, so a reader had no way to see why a power was missing.

THE USER'S REGRESSORS ARE LISTED BEFORE THE POWERS, even though the powers are what enter the model
first. `ovtest.ado:310` builds `ckdrp` from `e(b)`'s column order, so `expr2` precedes the generated
`expr2^2`, and the loop at line 431 prints in that order. Getting this backwards produces the right
count with the wrong sequence.

The default path never emits a note: it takes powers of a single fitted-value vector, which cannot
duplicate a regressor.
"""

import warnings

warnings.filterwarnings("ignore")
import pathlib

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="grilic.dta is not available on this machine",
)


def _frame() -> pd.DataFrame:
    frame = pd.read_stata(DATA)
    frame["expr2"] = frame["expr"] ** 2
    return frame


def _lines(formula: str, **kwargs) -> list[str]:
    fitted = ws.OLS.from_formula(formula, _frame()).fit()
    return [
        line
        for line in str(ws.estat_ovtest(fitted, **kwargs)).splitlines()
    ]


DUPLICATE = "lnw ~ s + expr + expr2 + tenure + smsa + rns"


def test_the_notes_name_both_dropped_columns() -> None:
    lines = _lines(DUPLICATE, rhs=True)
    assert lines[0] == "note: expr2 omitted because of collinearity.", lines[:3]
    assert lines[1] == "note: expr2^2 omitted because of collinearity.", lines[:3]


def test_the_regressor_precedes_the_power() -> None:
    """`ovtest.ado:310` builds the list from e(b) order, so the user column comes first."""

    result = ws.estat_ovtest(ws.OLS.from_formula(DUPLICATE, _frame()).fit(), rhs=True)
    assert result.dropped == ("expr2", "expr2^2"), result.dropped


def test_a_blank_line_follows_the_notes() -> None:
    lines = _lines(DUPLICATE, rhs=True)
    assert lines[2] == "", lines[:4]
    assert lines[3] == "Ramsey RESET test for omitted variables"


def test_the_table_begins_where_it_did_before() -> None:
    """The notes are added, not substituted for an existing line."""

    with_notes = _lines(DUPLICATE, rhs=True)
    index = with_notes.index("Ramsey RESET test for omitted variables")
    assert with_notes[index + 1].startswith("Omitted: Powers of independent variables")


def test_a_model_without_a_duplicate_has_no_notes() -> None:
    lines = _lines("lnw ~ s + expr + tenure + smsa + rns", rhs=True)
    assert not any(line.startswith("note:") for line in lines), lines[:3]
    assert lines[1] == "Ramsey RESET test for omitted variables"


def test_the_default_path_never_notes_anything() -> None:
    """Powers of the fitted values cannot duplicate a regressor."""

    lines = _lines(DUPLICATE)
    assert not any(line.startswith("note:") for line in lines), lines[:3]


def test_the_dropped_attribute_is_empty_on_the_default_path() -> None:
    result = ws.estat_ovtest(ws.OLS.from_formula(DUPLICATE, _frame()).fit())
    assert result.dropped == ()
