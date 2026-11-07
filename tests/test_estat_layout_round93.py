"""Three layout/wiring fixes found by capture-and-compare in round 93.

vif       The header carried no trailing spaces, so it was 36 characters where Stata's is 38,
          and an extra leading blank line preceded the table.
ic        Three 77-character rules and blank lines wrapped a table that Stata prints as FOUR
          plain lines. A comment claimed the rules were "measured from a live estat ic"; they
          were not, and the test encoded the same mistake.
dwatson   result.estat_dwatson() raised AttributeError. The implementation in
          postestimation/serial.py was already correct and already matched Stata byte for byte --
          it was simply never wired to the result object.

Captured references:

    Variable |       VIF       1/VIF
-------------+----------------------
        expr |      1.12    0.893267
...
    Mean VIF |      1.06

    Akaike's information criterion and Bayesian information criterion
           Model |          N   ll(null)  ll(model)      df        AIC        BIC
               . |        758  -433.4714  -287.9592       4   583.9184   602.4412
    Note: BIC uses N = number of observations. See [R] IC note.

    Durbin-Watson d-statistic(  4,   758) =  1.704354
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
COLUMNS = ["lnw", "s", "expr", "tenure", "rns", "smsa"]


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA).dropna(subset=COLUMNS)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")


@pytest.fixture(scope="module")
def simple(frame):
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


@pytest.fixture(scope="module")
def extended(frame):
    return ws.OLS.from_formula("lnw ~ s + expr + tenure + smsa + rns", frame).fit()


def test_dwatson_is_callable(simple) -> None:
    """It raised AttributeError before this round."""

    # CORRECTED in round 192: the block opens with a NEWLINE (the raw capture starts with one),
    # so it is two lines -- blank then the statistic -- and the value carries six decimals with a
    # trailing zero dropped.
    rendered = str(simple.estat_dwatson())
    assert rendered == "\nDurbin\u2013Watson d-statistic(  4,   758) =  1.704354"


def test_dwatson_has_the_stata_width(simple) -> None:
    # 50 = 49 characters of statistic plus the leading newline (round 192).
    assert len(str(simple.estat_dwatson())) == 50


def test_ic_has_no_rules(simple) -> None:
    # CORRECTED in round 191: nine lines, THREE 77-wide rules, and a leading blank. This test
    # used to assert four lines and no rules -- the same wrong reading the module comment carried.
    lines = str(simple.estat_ic()).splitlines()
    assert len(lines) == 9
    # Two full rules plus the split rule (which contains a '+'); the data row has digits so it
    # cannot be mistaken for one.
    assert sum(1 for line in lines if set(line) <= {"-", "+"} and len(line) > 10) == 3


def test_ic_matches_stata(simple) -> None:
    # CORRECTED in round 191 for the leading blank line and the rules.
    lines = str(simple.estat_ic()).splitlines()
    assert lines[0] == ""
    assert lines[1] == (
        "Akaike's information criterion and Bayesian information criterion"
    )
    assert lines[2] == ""
    assert lines[4] == (
        "       Model |          N   ll(null)  ll(model)      df        AIC        BIC"
    )
    assert lines[8] == "Note: BIC uses N = number of observations. See [R] IC note."


def test_vif_has_no_leading_blank_line(extended) -> None:
    # CORRECTED in round 193: the block opens with a BLANK line (Stata starts chr(10)), so
    # the header is line 1, not line 0.
    lines = str(extended.estat_vif()).splitlines()
    assert lines[0] == ""
    assert lines[1].startswith("    Variable |")


def test_vif_header_carries_the_trailing_spaces(extended) -> None:
    """Stata's header is 38 characters; ours was 36."""

    header = str(extended.estat_vif()).splitlines()[1]
    assert len(header) == 38, repr(header)
    assert header.endswith("  ")


def test_vif_rows_match_stata(extended) -> None:
    rendered = str(extended.estat_vif())
    assert "        expr |      1.12    0.893267" in rendered
    assert "    Mean VIF |      1.06" in rendered
