"""Pin ``estat bgodfrey``'s table to Stata.

Reference from a live ``estat bgodfrey`` after ``regress consumption temp price income``
on icecream.dta::

    Breusch–Godfrey LM test for autocorrelation
    ---------------------------------------------------------------------------
        lags(p)  |          chi2               df                 Prob > chi2
    -------------+-------------------------------------------------------------
           1     |          4.237               1                   0.0396
    ---------------------------------------------------------------------------
                            H0: no serial correlation

Values already matched Stata (4.237, df 1, p 0.0396); the table did not. The rules were 74
characters against Stata's 75 and every column was a different width.

Measured geometry: rules 75 characters, pipe at column 13, chi2 ending at 29, df at 45 and
the probability at 70. The lag label is left-aligned inside its field with trailing spaces
before the pipe (``       1     |``), which is not the same as right-aligning to 13.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    fitted = ws.OLS.from_formula(
        "consumption ~ temp + price + income", frame
    ).fit()
    return str(ws.estat_bgodfrey(fitted))


STATA_LINES = [
    "Breusch\u2013Godfrey LM test for autocorrelation",
    "-" * 75,
    "    lags(p)  |          chi2               df                 Prob > chi2",
    "-" * 13 + "+" + "-" * 61,
    "       1     |          4.237               1                   0.0396",
    "-" * 75,
    "                        H0: no serial correlation",
]


def test_every_line_matches_stata(rendered) -> None:
    assert rendered.splitlines() == STATA_LINES


def test_the_rules_are_seventy_five_characters(rendered) -> None:
    rules = [line for line in rendered.splitlines() if set(line) == {"-"}]
    assert len(rules) == 2
    for rule in rules:
        assert len(rule) == 75, rule


def test_the_pipe_sits_at_column_thirteen(rendered) -> None:
    for line in rendered.splitlines():
        if "|" in line:
            assert line.index("|") == 13, line


def test_the_statistic_is_unchanged(rendered) -> None:
    """Only the layout changed; the LM statistic was already right."""

    assert "4.237" in rendered
    assert "0.0396" in rendered
