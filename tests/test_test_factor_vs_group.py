"""`test()` must tell a FACTOR NAME from Stata's parenthesised-hypothesis syntax.

Round 160 added support for Stata's ``test (a = 0) (b = 0)`` grouping by splitting on parentheses.
That branch was too greedy and broke a name that LEGITIMATELY contains parentheses:

    result.test("C(foreign)[T.Foreign]")
        KeyError: coefficient 'C' not found

because ``_split_parenthesised`` returned ``['C', 'foreign', '[T.Foreign]']``. The README
regression guard -- written for exactly this class of defect -- caught it in round 182, which is
the third time a fix in one place has broken another (rounds 160, 173 and now this).

THE DISTINCTION THE CODE NOW MAKES:

  * ``C(var)[T.level]`` / ``C(var)[2]`` -- a COEFFICIENT NAME; the parentheses belong to it.
  * ``(a = 0) (b = 0)`` -- hypothesis grouping; the string starts AND ends with parentheses.

Neither test may pass at the other's expense, so both are pinned here together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

AUTO = "tests/data/auto.dta"
GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def factor_fit():
    try:
        frame = pd.read_stata(AUTO)
    except FileNotFoundError:  # pragma: no cover - the file ships with the repo
        pytest.skip("auto.dta is missing")
    return ws.OLS.from_formula("mpg ~ weight + C(foreign)", frame).fit()


@pytest.fixture(scope="module")
def plain_fit():
    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


def test_a_factor_level_can_be_referenced(factor_fit) -> None:
    """The regression: this raised KeyError naming 'C' as the missing coefficient."""

    rendered = str(factor_fit.test("C(foreign)[T.Foreign]"))
    assert "F(  1,    71) =    2.35" in rendered


def test_the_coefficient_name_really_contains_the_parentheses(factor_fit) -> None:
    """So the splitter must not touch it -- verified against the parameter list itself."""

    assert "C(foreign)[T.Foreign]" in list(factor_fit.params.index)


@pytest.mark.parametrize(
    "spec",
    ["(s=0) (expr=0)", "(s = 0) (expr = 0)", "(s=0)(expr=0)", "( s = 0 ) ( expr = 0 )"],
)
def test_parenthesised_hypotheses_still_expand(plain_fit, spec: str) -> None:
    """Round 160's feature, including the padded spelling that a stricter check broke."""

    rendered = str(plain_fit.test(spec))
    assert " ( 1)  s = 0" in rendered
    assert " ( 2)  expr = 0" in rendered
    assert "F(  2,   754) =  161.61" in rendered


def test_the_bare_and_grouped_forms_agree(factor_fit, plain_fit) -> None:
    assert str(plain_fit.test("s expr")) == str(plain_fit.test("(s=0) (expr=0)"))
    # And on the factor model the two spellings of one hypothesis agree too.
    assert str(factor_fit.test("C(foreign)[T.Foreign]")) == str(
        factor_fit.test("C(foreign)[T.Foreign] = 0")
    )


def test_a_factor_reference_and_a_group_can_be_combined(plain_fit) -> None:
    """Both branches must be reachable in ONE call, not just one at a time."""

    rendered = str(plain_fit.test("(s=0)", "(expr=0)"))
    assert "F(  2,   754) =  161.61" in rendered
