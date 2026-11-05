"""Pin ``estat ic``'s exact layout: 4-decimal log-likelihoods and full-width edge rules.

Reference from a live ``regress y k`` then ``estat ic`` on cobb_douglas.dta::

    Akaike's information criterion and Bayesian information criterion

    -----------------------------------------------------------------------------
           Model |          N   ll(null)  ll(model)      df        AIC        BIC
    -------------+---------------------------------------------------------------
               . |         24  -124.2294  -102.2761       2   208.5522   210.9083
    -----------------------------------------------------------------------------
    Note: BIC uses N = number of observations. See [R] IC note.

TWO corrections, and the first is the interesting one:

1. The log-likelihoods print FOUR decimals (``-124.2294``), not five. Printing five gave
   ``-124.22935`` -- more precise than Stata, which still fails an exact comparison and makes
   the cells one character wider than the header's column.
2. The first and last rules are FULL WIDTH with no ``+`` column; only the rule under the header
   is the split ``-----+-----`` form. Using the split rule for all three put a stray vertical
   bar on the opening line.

The gap report described this case as "westata adds AICc/CAIC columns and prints the model name
'r...'". Neither is true of the current output -- the model cell is ``.`` as Stata prints, and
there are six columns. What was actually wrong was the decimals and the rules.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\cobb_douglas.dta"
STATA_BLOCK = [
    "Akaike's information criterion and Bayesian information criterion",
    "-" * 77,
    "       Model |          N   ll(null)  ll(model)      df        AIC        BIC",
    "-------------+---------------------------------------------------------------",
    "           . |         24  -124.2294  -102.2761       2   208.5522   210.9083",
    "-" * 77,
    "Note: BIC uses N = number of observations. See [R] IC note.",
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("cobb_douglas.dta is not available")
    return str(ws.OLS.from_formula("y ~ k", frame).fit().estat_ic())


def test_every_line_matches_stata(rendered) -> None:
    lines = rendered.splitlines()
    for expected in STATA_BLOCK:
        assert expected in lines, expected


def test_the_log_likelihoods_have_four_decimals(rendered) -> None:
    """Five decimals is MORE precise than Stata and still wrong for an exact match."""

    assert "-124.2294 " in rendered or "-124.2294 " in rendered + " "
    assert "-124.22935" not in rendered
    assert "-102.2761 " in rendered or "-102.2761" in rendered
    assert "-102.27611" not in rendered


def test_the_edge_rules_are_full_width(rendered) -> None:
    """Only the header separator carries a '+'; the outer rules do not."""

    rules = [line for line in rendered.splitlines() if set(line) == {"-"}]
    assert rules, "no full-width rule found"
    assert len(rules) == 2
    # The split rule sits between the header and the data row.
    assert "-------------+---" in rendered


def test_the_model_cell_is_a_dot(rendered) -> None:
    """Stata prints '.' when no ``estimates store`` name is attached."""

    data_row = next(line for line in rendered.splitlines() if line.lstrip().startswith("."))
    assert data_row.strip().startswith(". |")
