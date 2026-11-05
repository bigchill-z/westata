"""Pin ``estat classification`` to Stata's printed table.

Reference from a live ``logit y x1 x2`` (n = 500, 208/77/61/154)::

    Logistic model for y

                  -------- True --------
    Classified |         D            ~D  |      Total
    -----------+--------------------------+-----------
         +     |       208            77  |        285
         -     |        61           154  |        215
    -----------+--------------------------+-----------
       Total   |       269           231  |        500

    Classified + if predicted Pr(D) >= .5
    True D defined as y != 0
    --------------------------------------------------
    Sensitivity                     Pr( +| D)   77.32%
    ...
    --------------------------------------------------
    Correctly classified                        72.40%
    --------------------------------------------------

Geometry that had to be measured rather than guessed:

* every table line is 50 characters with pipes at columns 11 and 38;
* the cells end at columns 21, 35 and 49;
* row labels are CENTRED in eleven characters ("     +     |"), while the header's
  "Classified |" is left-aligned;
* the nine rates put the label at column 0, the "Pr(...)" code at 32 and the value
  right-aligned in the final six columns;
* the cutoff prints as ".5", not "0.5";
* the title says "Logistic", not "Logit", for BOTH ``logit`` and ``logistic``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame():
    rng = np.random.default_rng(31337)
    n = 500
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 0.9 * x1 - 0.4 * x2 + rng.normal(scale=1.2, size=n)
    return pd.DataFrame({"y": (latent > 0).astype(int), "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def fitted():
    return ws.Logit.from_formula("y ~ x1 + x2", _frame()).fit()


STATA_LINES = [
    "",
    "Logistic model for y",
    "",
    "              -------- True --------",
    "Classified |         D            ~D  |      Total",
    "-----------+--------------------------+-----------",
    "     +     |       208            77  |        285",
    "     -     |        61           154  |        215",
    "-----------+--------------------------+-----------",
    "   Total   |       269           231  |        500",
    "",
    "Classified + if predicted Pr(D) >= .5",
    "True D defined as y != 0",
    "-" * 50,
    "Sensitivity                     Pr( +| D)   77.32%",
    "Specificity                     Pr( -|~D)   66.67%",
    "Positive predictive value       Pr( D| +)   72.98%",
    "Negative predictive value       Pr(~D| -)   71.63%",
    "-" * 50,
    "False + rate for true ~D        Pr( +|~D)   33.33%",
    "False - rate for true D         Pr( -| D)   22.68%",
    "False + rate for classified +   Pr(~D| +)   27.02%",
    "False - rate for classified -   Pr( D| -)   28.37%",
    "-" * 50,
    "Correctly classified                        72.40%",
    "-" * 50,
]


def test_every_line_matches_stata(fitted) -> None:
    rendered = str(ws.estat_classification(fitted)).splitlines()
    assert rendered == STATA_LINES


def test_the_table_cells_end_at_21_35_and_49(fitted) -> None:
    rendered = str(ws.estat_classification(fitted)).splitlines()
    row = next(line for line in rendered if line.strip().startswith("+"))
    assert row.index("|") == 11
    assert row.rindex("|") == 38
    assert row.index("208") + 3 == 22
    assert row.index("77") + 2 == 36
    assert row.rstrip().endswith("285")


def test_the_cutoff_drops_the_leading_zero(fitted) -> None:
    rendered = str(ws.estat_classification(fitted))
    assert ">= .5" in rendered
    assert ">= 0.5" not in rendered


def test_the_title_says_logistic_not_logit(fitted) -> None:
    """Stata prints "Logistic model for y" for both ``logit`` and ``logistic``."""

    rendered = str(ws.estat_classification(fitted))
    assert "Logistic model for y" in rendered
    assert "Logit model" not in rendered


def test_the_dependent_variable_is_named(fitted) -> None:
    """Not "the response" -- Stata uses the actual variable name."""

    rendered = str(ws.estat_classification(fitted))
    assert "True D defined as y != 0" in rendered


def test_all_nine_rates_are_present(fitted) -> None:
    rendered = str(ws.estat_classification(fitted))
    for label in (
        "Sensitivity",
        "Specificity",
        "Positive predictive value",
        "Negative predictive value",
        "False + rate for true ~D",
        "False - rate for true D",
        "False + rate for classified +",
        "False - rate for classified -",
        "Correctly classified",
    ):
        assert label in rendered, label


def test_rates_put_the_code_at_column_32(fitted) -> None:
    rendered = str(ws.estat_classification(fitted)).splitlines()
    line = next(line for line in rendered if line.startswith("Sensitivity"))
    assert line.index("Pr(") == 32
    assert len(line) == 50
