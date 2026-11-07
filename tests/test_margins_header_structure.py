"""margins: header blank lines, the right-aligned count, and six-decimal values.

Round 194 extended the shape survey from the `estat` family to `margins`, which printed TEN lines
where Stata prints FOURTEEN. The numbers were already exact; the header was missing structure:

    (blank)
    Average marginal effects ... Number of obs = 74
    Model VCE: OLS
    (blank)
    Expression: Linear prediction, predict()
    dy/dx wrt:  weight fx
    (blank)
    ----------------------------------------------------------------------

THREE details had to be measured, and two of my first attempts were wrong:

  * "Model VCE: OLS" is printed for an OLS fit even though the result carries no vce_label;
  * "Number of obs = N" is RIGHT-ALIGNED to a 78-wide line, not started at a fixed column.
    regress puts 'Number' at column 60 and logit at 59, because '= 74' is two characters shorter
    than '= 758'. I first used a fixed 59, then a fixed 60 -- each matched one fit and broke the
    other;
  * the margin cells use SEVEN decimals when the integer part is zero and SIX when it is not, the
    same rule verified for tebalance (164) and prais (170). A flat seven decimals printed
    1.0759941 where Stata prints 1.075994.

ONE STALE EXPECTATION was in the suite: it held `1.2276366` for the cobb-douglas upper bound where
a live `margins, dydx(*)` prints `1.227637`.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"
AUTO = "tests/data/auto.dta"
COBB = r"D:\myCodeProject\test_westata\data\cobb_douglas.dta"


@pytest.fixture(scope="module")
def logit_fit():
    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["rns", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.Logit.from_formula("rns ~ s + expr + tenure", frame).fit()


def test_the_header_opens_with_a_blank_line(logit_fit) -> None:
    assert str(ws.margins(logit_fit, dydx=["s"])).startswith("\n")


def test_the_header_has_three_blank_lines(logit_fit) -> None:
    lines = str(ws.margins(logit_fit, dydx=["s"])).splitlines()
    assert lines[0] == ""
    assert lines[2] == "Model VCE: OIM"
    assert lines[3] == ""
    assert lines[4].startswith("Expression:")
    assert lines[6] == ""


def test_the_count_is_right_aligned() -> None:
    """Two data sizes put 'Number' at different columns while both lines end at 78."""

    frame = pd.read_stata(COBB)
    fit = ws.OLS.from_formula("y ~ k + l", frame).fit()
    small_line = str(ws.margins(fit, dydx="*")).splitlines()[1]
    assert len(small_line) == 78
    assert "Number of obs = " in small_line

    large = pd.read_stata(GRILIC).dropna(subset=["rns", "s", "expr", "tenure"])
    big_line = str(
        ws.margins(ws.Logit.from_formula("rns ~ s + expr + tenure", large).fit(), dydx=["s"])
    ).splitlines()[1]
    assert len(big_line) == 78
    assert big_line.endswith("Number of obs = 758")
    # Same end column, different start column.
    assert small_line.index("Number") != big_line.index("Number")


def test_a_large_estimate_keeps_six_decimals() -> None:
    """Stata prints 1.227637 for the cobb-douglas upper bound; the suite held 1.2276366."""

    frame = pd.read_stata(COBB)
    rendered = str(ws.margins(ws.OLS.from_formula("y ~ k + l", frame).fit(), dydx="*"))
    assert "1.227637" in rendered
    assert "1.2276366" not in rendered