"""Pin ``margins, dydx(*) atmeans``: the conditional title and the "At:" block.

Reference from a live ``logit treat age education re74 re75`` then
``margins, dydx(*) atmeans`` on nsw_dw.dta::

    Conditional marginal effects                               Number of obs = 445
    Model VCE: OIM

    Expression: Pr(treat), predict()
    dy/dx wrt:  age education re74 re75
    At: age       = 25.37079 (mean)
        education = 10.19551 (mean)
        re74      = 2102.265 (mean)
        re75      = 1377.138 (mean)

    ------------------------------------------------------------------------------
                 |            Delta-method
                 |      dy/dx   std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
             age |   .0033046   .0033066     1.00   0.318    -.0031761    .0097854

TWO things change beyond the numbers, and the effects themselves already matched:

1. the TITLE becomes "Conditional marginal effects" -- atmeans reports the effect AT a point,
   not averaged over the sample;
2. an "At:" block lists the means it fixed. The name is LEFT-aligned in a 10-wide field
   (unlike coefficient rows, which right-align) and continuation lines are indented four
   spaces so they sit under "At: ".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"


@pytest.fixture(scope="module")
def conditional():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")
    fit = ws.Logit.from_formula("treat ~ age + education + re74 + re75", frame).fit()
    return ws.margins(fit, dydx=["age", "education", "re74", "re75"], atmeans=True)


def test_the_title_becomes_conditional(conditional) -> None:
    rendered = str(conditional)
    assert "Conditional marginal effects" in rendered
    assert "Average marginal effects" not in rendered


def test_the_at_block_lists_the_means(conditional) -> None:
    rendered = str(conditional).splitlines()
    for expected in (
        "At: age       = 25.37079 (mean)",
        "    education = 10.19551 (mean)",
        "    re74      = 2102.265 (mean)",
        "    re75      = 1377.138 (mean)",
    ):
        assert expected in rendered, expected


def test_the_effects_match_stata(conditional) -> None:
    rendered = str(conditional)
    assert "         age |   .0033046   .0033066     1.00   0.318    -.0031761    .0097854" in rendered


def test_the_headline_row_is_78_wide(conditional) -> None:
    headline = next(
        row for row in str(conditional).splitlines() if "Number of obs" in row
    )
    assert len(headline) == 78


def test_a_plain_margins_call_is_unaffected() -> None:
    """Without atmeans there is no At block and the title stays Average."""

    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nsw_dw.dta is not available")
    fit = ws.Logit.from_formula("treat ~ age + education + re74", frame).fit()
    plain = str(ws.margins(fit, dydx=["age"], atmeans=False))
    assert "Average marginal effects" in plain
    assert "(mean)" not in plain
