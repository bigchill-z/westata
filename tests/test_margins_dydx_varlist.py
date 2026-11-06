"""margins dydx(): the space-separated varlist spelling Stata accepts.

The report lists "margins dydx('*')" as missing. It is not -- ``dydx='*'`` has worked and is
tested. What did fail was the OTHER spelling Stata accepts, a space-separated varlist inside one
pair of parentheses:

    r.margins(dydx="s expr")     KeyError: column 's expr' not in [...]

which reads like a typo in the data rather than an unsupported spelling. The resolver received
"s expr" as a single name; it now splits on whitespace, so both of these produce the same table:

    r.margins(dydx="s expr")
    r.margins(dydx=["s", "expr"])

and the header reads ``dy/dx wrt:  s expr`` exactly as Stata prints it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
COLUMNS = ["lnw", "s", "expr", "tenure"]


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(subset=COLUMNS)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


def test_the_wildcard_still_works(fitted) -> None:
    """The report calls this missing; it has been supported and stays supported."""

    rendered = str(fitted.margins(dydx="*"))
    assert "dy/dx wrt" in rendered
    assert "s" in rendered and "expr" in rendered and "tenure" in rendered


def test_the_space_separated_varlist_works(fitted) -> None:
    """This spelling raised KeyError on the whole string before."""

    rendered = str(fitted.margins(dydx="s expr"))
    assert "dy/dx wrt:  s expr" in rendered


def test_the_two_spellings_agree(fitted) -> None:
    spaced = str(fitted.margins(dydx="s expr"))
    listed = str(fitted.margins(dydx=["s", "expr"]))
    assert spaced == listed


def test_a_single_name_still_works(fitted) -> None:
    assert "dy/dx wrt:  s" in str(fitted.margins(dydx="s"))


def test_an_unknown_name_still_raises(fitted) -> None:
    """Splitting must not turn a typo into a silent empty result."""

    with pytest.raises(KeyError):
        fitted.margins(dydx="nosuchcolumn")
