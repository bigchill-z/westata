"""``margins, dydx(*)``: the wildcard form, added in round 4 of the v2 report.

The v2 report listed it as missing (P1, C11-03). ``dydx("*")`` used to raise
``KeyError: column '*' not in [...]``, which says nothing about the wildcard existing.

Stata expands ``*`` to every covariate, i.e. every parameter except the constant. A live
``margins, dydx(*)`` after ``regress y k l`` prints "dy/dx wrt:  k l" -- the constant is not
a marginal effect and is not listed.

Reference, cobb_douglas after ``regress y k l``:

    Average marginal effects                                Number of obs = 24
    dy/dx wrt:  k l

                 |      dy/dx   std. err.      t    P>|t|     [95% conf. interval]
    -------------+----------------------------------------------------------------
               k |   .1585962   .0416823     3.80   0.001     .0719131    .2452793
               l |   .9166087   .1495604     6.13   0.000     .6055807   1.2276366
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\cobb_douglas.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("cobb_douglas.dta is not available")
    return ws.OLS.from_formula("y ~ k + l", frame).fit()


def test_the_wildcard_expands_to_every_non_constant(fitted) -> None:
    rendered = str(fitted.margins(dydx=["*"]))
    assert "dy/dx wrt:  k l" in rendered, rendered


def test_the_wildcard_values_match_stata(fitted) -> None:
    rendered = str(fitted.margins(dydx=["*"]))
    assert "k |   .1585962   .0416823     3.80   0.001     .0719131    .2452793" in rendered
    assert "l |   .9166087   .1495604     6.13   0.000     .6055807    1.227637" in rendered


def test_the_wildcard_equals_naming_them_all(fitted) -> None:
    """The shortest statement of what the wildcard means."""

    wildcard = str(fitted.margins(dydx=["*"]))
    explicit = str(fitted.margins(dydx=["k", "l"]))
    assert wildcard == explicit


def test_the_constant_is_not_a_marginal_effect(fitted) -> None:
    """Intercept must not appear in the dy/dx list."""

    rendered = str(fitted.margins(dydx=["*"]))
    assert "Intercept" not in rendered
    assert "_cons" not in rendered


def test_a_mixed_list_does_not_duplicate(fitted) -> None:
    """``['*', 'k']`` names k twice; Stata prints each covariate once."""

    mixed = str(fitted.margins(dydx=["*", "k"]))
    assert mixed.count("|   ") >= 2  # both rows present
    assert mixed.count("           k |") == 1
