"""Pin ``margins`` to Stata's table.

Reference from a live run on grilic.dta after ``quietly logit rns s expr tenure``
(n = 758)::

    Average marginal effects                                   Number of obs = 758
    Model VCE: OIM
    Expression: Pr(rns), predict()
    dy/dx wrt:  s
    ------------------------------------------------------------------------------
                 |            Delta-method
                 |      dy/dx   std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
               s |  -.0135532   .0075125    -1.80   0.071    -.0282774     .001171
    ------------------------------------------------------------------------------

and for a plain ``margins``::

    Predictive margins                                         Number of obs = 758
    ...
                 |     Margin   std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
           _cons |   .2691293   .0160621    16.76   0.000     .2376482    .3006103

What the renderer had to get right, each measured rather than guessed:

* every line is 78 characters with the pipe at column 13;
* each value is right-aligned to a fixed END column -- 25, 36, 45, 53, 66, 78 -- which is
  what makes a ten-character estimate and an eight-character one share a heading;
* the cells use SEVEN DECIMAL PLACES with trailing zeros dropped, not significant figures:
  .0075124809 prints as .0075125 and .0011710248 as .001171. A six-figure rule gave
  .00751248 / .00117102 instead;
* a single predictive margin is labelled ``_cons``, not ``margin``;
* the ``wrt:`` line has TWO spaces after the colon.

Before this, ``margins`` printed a bare dataclass repr for a single term and a LIST repr
for several -- it had no table at all, even though all nine numbers were correct.

The return-shape contract is pinned elsewhere and must not change: several rows give a
``list`` (``MarginsTable`` subclasses ``list`` so ``isinstance(x, list)`` still holds), and
one row gives a bare ``MarginsResult`` which is not a list.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
RULE = "-" * 78
COLUMN_RULE = "-" * 13 + "+" + "-" * 64


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.Logit.from_formula("rns ~ s + expr + tenure", frame).fit()


def test_single_marginal_effect_matches_stata(fitted) -> None:
    # CORRECTED in round 194 from a live margins, dydx(s): the block opens with a BLANK line,
    # carries one after "Model VCE:", and another before the table rule.
    expected = [
        "",
        "Average marginal effects                                   Number of obs = 758",
        "Model VCE: OIM",
        "",
        "Expression: Pr(rns), predict()",
        "dy/dx wrt:  s",
        "",
        RULE,
        "             |            Delta-method",
        "             |      dy/dx   std. err.      z    P>|z|     [95% conf. interval]",
        COLUMN_RULE,
        "           s |  -.0135532   .0075125    -1.80   0.071    -.0282774     .001171",
        RULE,
    ]
    assert str(ws.margins(fitted, dydx=["s"])).splitlines() == expected


def test_two_marginal_effects_render_one_table(fitted) -> None:
    rendered = str(ws.margins(fitted, dydx=["s", "expr"])).splitlines()
    # Three header blank lines were added in round 194, shifting this row by three.
    assert rendered[5] == "dy/dx wrt:  s expr"
    assert (
        "        expr |  -.0002224   .0079914    -0.03   0.978    -.0158853    .0154405"
        in rendered
    )
    assert len(rendered) == 14


def test_predictive_margins_matches_stata(fitted) -> None:
    # CORRECTED in round 194 for the three header blank lines.
    expected = [
        "",
        "Predictive margins                                         Number of obs = 758",
        "Model VCE: OIM",
        "",
        "Expression: Pr(rns), predict()",
        "",
        RULE,
        "             |            Delta-method",
        "             |     Margin   std. err.      z    P>|z|     [95% conf. interval]",
        COLUMN_RULE,
        "       _cons |   .2691293   .0160621    16.76   0.000     .2376482    .3006103",
        RULE,
    ]
    assert str(ws.margins(fitted)).splitlines() == expected


def test_every_table_line_is_seventy_eight_characters(fitted) -> None:
    """Rules, the column header and the data rows are 78 characters.

    The heading lines above the table and the "Delta-method" sub-header are naturally
    shorter in Stata too (38 for the sub-header), so they are excluded by construction
    rather than by name.
    """

    for line in str(ws.margins(fitted, dydx=["s", "expr"])).splitlines():
        is_full_width = (
            line.startswith("-")
            or "std. err." in line
            or ("|" in line and any(ch.isdigit() for ch in line))
        )
        if is_full_width:
            assert len(line) == 78, line


def test_a_single_predictive_margin_is_labelled_cons(fitted) -> None:
    """The upstream row says "margin", which no Stata table shows."""

    rendered = str(ws.margins(fitted))
    assert "_cons |" in rendered
    assert "margin |" not in rendered


def test_the_wrt_line_has_two_spaces_after_the_colon(fitted) -> None:
    rendered = str(ws.margins(fitted, dydx=["s"]))
    assert "dy/dx wrt:  s" in rendered


def test_the_cell_formatter_uses_seven_decimals() -> None:
    """Not significant figures: .0075124809 is .0075125, not .00751248."""

    from westata.postestimation.margins import _margin_number

    assert _margin_number(0.007512480908, 36, 26).strip() == ".0075125"
    assert _margin_number(0.001171024855, 78, 71).strip() == ".001171"
    assert _margin_number(0.013553167159, 25, 14).strip() == ".0135532"


def test_several_rows_are_a_list_and_one_row_is_not(fitted) -> None:
    """The shape contract other callers depend on."""

    many = ws.margins(fitted, dydx=["s", "expr"])
    one = ws.margins(fitted, dydx=["s"])
    assert isinstance(many, list)
    assert not isinstance(one, list)
    assert len(many) == 2
