"""The notes `hausman` prints, and two formatting rules the table needs.

The user reported that westata's hausman output still differed in its notes. Three things were
missing or wrong, all measured against a live `hausman iv ols`:

1. THE RANK NOTE (printed FIRST, above the coefficient table, only when the differenced matrix's rank
   is smaller than the number of coefficients tested):

       Note: the rank of the differenced variance matrix (1) does not equal the number
               of coefficients being tested (6); be sure this is what you expect, or
               there may be problems computing the test.  Examine the output of your
               estimators for anything unexpected and possibly consider scaling your
               variables so that the coefficients are on a similar scale.

   Continuation lines are indented eight spaces and "test." is followed by TWO spaces.

2. THE NON-POSITIVE-DEFINITE MARKER (the LAST line): "(V_b-V_B is not positive definite)".

3. THE CONSTANT'S LABEL AND PRECISION. The row printed "Intercept" where Stata has "_cons", and the
   decimal count is magnitude-dependent: six places at or above 1, seven below. A fixed seven places
   printed 3.2180433 where Stata has 3.218043.

With those in place the whole block matches Stata line for line under all four option combinations
(20/21/27/28 lines), provided the caller labels the estimates the way `estimates store` would -- see
the last test.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(),
    reason="grilic.dta is not available on this machine",
)


@pytest.fixture(scope="module")
def pair():
    frame = pd.read_stata(DATA)
    ols = ws.OLS.from_formula("lnw ~ iq + s + expr + tenure + rns + smsa", frame).fit()
    iv = ws.IV2SLS.from_formula(
        "lnw ~ s + expr + tenure + rns + smsa + [iq ~ med + kww]", frame
    ).fit()
    return iv, ols


def _render(pair, **kwargs):
    iv, ols = pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return str(ws.hausman(iv, ols, **kwargs)).splitlines()


# ---------------------------------------------------------------- the rank note
def test_the_rank_note_opens_the_output_under_sigmamore(pair) -> None:
    lines = _render(pair, sigmamore=True)
    assert lines[0] == "", lines[:2]
    assert lines[1].startswith("Note: the rank of the differenced variance matrix (1)"), lines[1]


def test_the_rank_note_names_both_numbers(pair) -> None:
    text = "\n".join(_render(pair, sigmamore=True))
    assert "does not equal the number" in text
    assert "of coefficients being tested (6)" in text


def test_the_rank_note_wraps_with_eight_spaces(pair) -> None:
    lines = _render(pair, sigmamore=True)
    for line in lines[2:6]:
        assert line.startswith("        "), repr(line)


def test_the_rank_note_keeps_the_double_space_after_test(pair) -> None:
    text = "\n".join(_render(pair, sigmamore=True))
    assert "computing the test.  Examine" in text


def test_an_exact_rank_prints_no_note(pair) -> None:
    """The default form has rank 6 over 6 coefficients, so Stata says nothing."""

    text = "\n".join(_render(pair, constant=True))
    assert "the rank of the differenced" not in text


# ---------------------------------------------------------------- the marker
def test_the_not_positive_definite_marker_is_the_last_line(pair) -> None:
    lines = _render(pair, sigmamore=True)
    assert lines[-1] == "(V_b-V_B is not positive definite)", lines[-3:]
    assert lines[-2].startswith("Prob > chi2"), lines[-3:]
    assert lines[-3].strip().startswith("="), lines[-3:]


def test_a_positive_definite_difference_has_no_marker(pair) -> None:
    """The default and `constant` forms are invertible directly, so Stata prints no marker."""

    for kwargs in ({}, {"constant": True}):
        text = "\n".join(_render(pair, **kwargs))
        assert "(V_b-V_B is not positive definite)" not in text


# ---------------------------------------------------------------- label and precision
def test_the_constant_row_is_labelled_cons(pair) -> None:
    lines = _render(pair, constant=True)
    rows = [line for line in lines if "|" in line and "Coefficient" not in line]
    constants = [row for row in rows if row.strip().startswith("_cons |")]
    assert constants, [row.strip()[:20] for row in rows]
    assert not any("Intercept" in row for row in lines)


def test_coefficients_above_one_use_six_decimals(pair) -> None:
    """3.2180433224 must print as 3.218043, not 3.2180433."""

    lines = _render(pair, constant=True)
    row = next(line for line in lines if line.strip().startswith("_cons |"))
    assert "3.218043 " in row or "3.218043\n" in row + "\n", row
    assert "3.2180433" not in row, row


def test_coefficients_below_one_use_seven_decimals(pair) -> None:
    lines = _render(pair, constant=True)
    row = next(line for line in lines if line.strip().startswith("iq |"))
    assert "3.218043" not in row
    assert ".0139284" in row, row


# ---------------------------------------------------------------- the store names
def test_estimation_labels_reproduce_the_store_names(pair) -> None:
    """Stata heads the columns with the `estimates store` names, which our results do not know.

    Setting `estimation_label` on each result is the equivalent of `estimates store iv` / `ols`.
    """

    iv, ols = pair
    iv.estimation_label = "iv"
    ols.estimation_label = "ols"
    try:
        header = [line for line in _render(pair) if "Difference" in line and "|" in line]
        assert header and " iv " in header[0] and " ols " in header[0], header
    finally:
        iv.estimation_label = ""
        ols.estimation_label = ""
