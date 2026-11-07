"""test(): Stata's parenthesised hypothesis form was rejected with a KeyError.

Round 160 checked C17-04 (the joint ``test`` command, scored 0.40 / 0.75) and found a real defect
rather than a layout quibble:

    result.test("(s=0) (expr=0)")
        KeyError: coefficient '(s' not found

Stata supports three spellings of the same joint hypothesis, all producing identical output:

    test s expr
    test (s=0) (expr=0)
    test (s = 0) (expr = 0)

and the package handled only the first. The parser split on commas, so the parenthesised string
reached it as ONE spec and ``(s`` was read as a coefficient name.

WHAT IS PINNED HERE:
  * all three spellings give the same F and the same printed constraints;
  * the spacing inside the parentheses does not matter, and adjacent groups without a space work;
  * a constraint containing a comma still works when parenthesised (the reason Stata has the form);
  * the existing non-parenthesised behaviours are unchanged.

Reference from `quietly regress lnw s expr tenure` on grilic.dta:
    test s expr                 F(  2,   754) =  161.61   Prob > F = 0.0000
    test (s=0) (expr=0)         same
    test s = expr               F(  1,   754) =   80.62   Prob > F = 0.0000
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


def test_the_bare_list_matches_stata(fitted) -> None:
    rendered = str(fitted.test("s expr"))
    assert " ( 1)  s = 0" in rendered
    assert " ( 2)  expr = 0" in rendered
    assert "F(  2,   754) =  161.61" in rendered


@pytest.mark.parametrize(
    "spec",
    ["(s=0) (expr=0)", "(s = 0) (expr = 0)", "(s=0)(expr=0)", "( s = 0 ) ( expr = 0 )"],
)
def test_the_parenthesised_form_matches_stata(fitted, spec: str) -> None:
    """The defect fixed in round 160: this raised KeyError before."""

    rendered = str(fitted.test(spec))
    assert " ( 1)  s = 0" in rendered
    assert " ( 2)  expr = 0" in rendered
    assert "F(  2,   754) =  161.61" in rendered


def test_both_spellings_agree(fitted) -> None:
    assert str(fitted.test("s expr")) == str(fitted.test("(s=0) (expr=0)"))


def test_an_equality_chain_still_works(fitted) -> None:
    """The non-parenthesised form must not regress while the new branch is in place."""

    rendered = str(fitted.test("s = expr"))
    assert " ( 1)  s - expr = 0" in rendered
    assert "F(  1,   754) =   80.62" in rendered


def test_a_parenthesised_group_may_contain_a_comma(fitted) -> None:
    """Which is the reason Stata has the parenthesised form at all."""

    rendered = str(fitted.test("(s = expr)"))
    assert "s - expr = 0" in rendered
    assert "F(  1,   754) =   80.62" in rendered


def test_the_comma_separated_form_still_works(fitted) -> None:
    rendered = str(fitted.test("s = 0, expr = 0"))
    assert "F(  2,   754) =  161.61" in rendered
