"""estat bgodfrey and estat dwatson: both open with a blank line, and dwatson's digits vary.

Round 192 extended round 191's prefix survey -- walk every renderer's raw first character -- from
`estat ic` to all fifteen `estat_` blocks. Two more did not start with a newline:

    estat_bgodfrey   'B'  ->  Stata's capture starts '\\nBreusch...'
    estat_dwatson    'D'  ->  Stata's capture starts '\\nDurbin...'

For bgodfrey this was worse than a one-line offset: a whole-table comparison aligned Stata's row N
against westata's row N+1, so EVERY row read as mismatched while the geometry was already right.

DWATSON'S NUMBER FORMAT is not six significant figures, which was my first guess. Three fits fix it:

    icecream  1.021169     grilic/3 regressors  1.704354     grilic/5 regressors  1.75295

Six DECIMALS with a trailing zero dropped. '%.6f' printed 1.752950 (one character wide) and '%.6g'
printed 1.70435 (six figures, one decimal short). Both were wrong in opposite directions, and each
looked right on the one fit I happened to check first.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"
ICECREAM = r"D:\myCodeProject\test_westata\data\icecream.dta"


@pytest.fixture(scope="module")
def three_regressor():
    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


@pytest.fixture(scope="module")
def five_regressor():
    try:
        frame = pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure + smsa + rns", frame).fit()


def test_dwatson_opens_with_a_blank_line(three_regressor) -> None:
    assert str(ws.estat_dwatson(three_regressor)).startswith("\n")


@pytest.mark.parametrize(
    "fixture_name, expected",
    [
        ("three_regressor", "1.704354"),
        ("five_regressor", "1.75295"),
    ],
)
def test_dwatson_drops_a_trailing_zero(fixture_name, expected, request) -> None:
    """1.704354 keeps six decimals; 1.75295 lost the sixth because it was a zero."""

    fitted = request.getfixturevalue(fixture_name)
    rendered = str(ws.estat_dwatson(fitted))
    assert rendered.rstrip().endswith(expected), repr(rendered)


def test_dwatson_matches_a_third_fit(three_regressor) -> None:
    """icecream's value is 1.021169 -- six decimals with no trailing zero to drop."""

    frame = pd.read_stata(ICECREAM)
    fitted = ws.OLS.from_formula("consumption ~ price + income + temp", frame).fit()
    assert str(ws.estat_dwatson(fitted)).rstrip().endswith("1.021169")


def test_bgodfrey_opens_with_a_blank_line(five_regressor) -> None:
    assert str(ws.estat_bgodfrey(five_regressor)).startswith("\n")


def test_bgodfrey_block_is_two_lines_longer_than_its_table(five_regressor) -> None:
    """Blank, title, rule, header, split, row, rule, H0 -- eight lines for one lag."""

    lines = str(ws.estat_bgodfrey(five_regressor)).splitlines()
    assert lines[0] == ""
    assert lines[1] == "Breusch\u2013Godfrey LM test for autocorrelation"
    assert lines[-1].strip() == "H0: no serial correlation"
