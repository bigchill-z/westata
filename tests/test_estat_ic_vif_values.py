"""`estat ic` and `estat vif` values on a linear fit, captured from a live Stata.

Rounds 130-134 fixed and mapped the postestimation surface; this checks the numbers on the estimator
those commands are most often used with.

    quietly reg y x z
    estat ic
           . |        300  -399.1048  -397.3598       3   800.7197    811.831

    estat vif
           x |      1.00    0.999902
           z |      1.00    0.999902
    Mean VIF |      1.00

Both rows match character for character. The IC row is worth pinning because its columns use THREE
different formats at once: N as an integer, the log-likelihoods at four decimals, df as an integer,
and AIC/BIC at seven significant figures -- a change to any one of them is a layout regression that a
numeric comparison alone would not catch.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(11)
    n = 300
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
        }
    )
    return ws.OLS.from_formula("y ~ x + z", frame).fit()


def test_the_ic_row_matches_stata_character_for_character(fitted) -> None:
    expected = (
        "           . |        300  -399.1048  -397.3598       3   800.7197    811.831"
    )
    lines = str(ws.estat_ic(fitted)).splitlines()
    assert expected in lines


def test_the_log_likelihoods_keep_four_decimals(fitted) -> None:
    rendered = str(ws.estat_ic(fitted))
    assert "-399.1048" in rendered
    assert "-397.3598" in rendered


def test_the_information_criteria_use_seven_significant_figures(fitted) -> None:
    """800.7197 has four decimals; 811.831 has three -- one width cannot serve both."""

    rendered = str(ws.estat_ic(fitted))
    assert "800.7197" in rendered
    assert "811.831" in rendered


def test_the_vif_table_matches_stata(fitted) -> None:
    rendered = str(ws.estat_vif(fitted))
    assert "0.999902" in rendered
    assert "1.00" in rendered
