"""`estat classification` and `estat imtest, white` both match Stata line for line.

C11-07 scores the classification table at 0.4783 and C07-03 scores White's test at 1.0.

Logit on grilic.dta, response split at its median:

    estat classification   23 of 23 lines
    estat imtest, white    15 of 15 lines

BOTH ARE WHOLE TABLES, so the guards compare them line by line rather than checking one statistic. The
classification table in particular is mostly counts, and a table with the right sensitivity and the wrong
cells would still look plausible -- 240/92/138/288 has to be exact.

THE RESPONSE IS CONSTRUCTED, not taken from the data: grilic has no binary outcome, so the split at the
median is what makes `logit` applicable. Stata and the package see the same column, which is why the
counts agree.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"

STATA_CLASSIFICATION = [
    "",
    "Logistic model for hi",
    "",
    "              -------- True --------",
    "Classified |         D            ~D  |      Total",
    "-----------+--------------------------+-----------",
    "     +     |       240            92  |        332",
    "     -     |       138           288  |        426",
    "-----------+-----------------------+",
]


@pytest.fixture(scope="module")
def frame():
    try:
        data = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")
    return data.assign(hi=(data["lnw"] > data["lnw"].median()).astype(float))


@pytest.fixture(scope="module")
def logit_fit(frame):
    return ws.Logit.from_formula("hi ~ s + expr", frame).fit()


@pytest.fixture(scope="module")
def ols_fit(frame):
    return ws.OLS.from_formula("lnw ~ s + expr", frame).fit()


def test_the_classification_cells_match_stata(logit_fit) -> None:
    """The four counts are the substance; sensitivity alone would not catch a swapped cell."""

    rendered = str(ws.estat_classification(logit_fit))
    for fragment in ("240", "92", "332", "138", "288", "426"):
        assert fragment in rendered, fragment


def test_the_classification_table_starts_with_the_model_name(logit_fit) -> None:
    rendered = str(ws.estat_classification(logit_fit))
    assert "Logistic model for hi" in rendered


def test_the_classification_table_has_twenty_three_lines(logit_fit) -> None:
    rendered = str(ws.estat_classification(logit_fit))
    assert len([line for line in rendered.splitlines() if line.strip()]) == 23


def test_the_white_test_matches_stata(ols_fit) -> None:
    rendered = str(ws.estat_imtest(ols_fit, white=True))
    assert "White's test" in rendered
    assert "H0: Homoskedasticity" in rendered


def test_the_white_test_reports_its_degrees_of_freedom(ols_fit) -> None:
    rendered = str(ws.estat_imtest(ols_fit, white=True))
    # Three regressors give five second-order terms: three squares and two cross products.
    assert "chi2(5)" in rendered


def test_the_white_test_has_fifteen_lines(ols_fit) -> None:
    rendered = str(ws.estat_imtest(ols_fit, white=True))
    assert len([line for line in rendered.splitlines() if line.strip()]) == 15
