"""estat endogenous: the comment AND its test both said "no leading blank line". Both were wrong.

C10-05 scored 0.00 layout / 0.56 numeric. The numbers (28.8042 / 29.8235) were already exact; the
block was two blank lines away from Stata's.

THE INTERESTING PART is not the blank lines. It is that the module docstring asserted
"NO leading blank line -- the block starts at the first line" and a test asserted the same thing,
so the two agreed with each other and neither could catch the other. A live capture shows the raw
output begins with '\\n  ': the block opens with a blank line, and another separates the null
hypothesis from the statistics.

That is a distinct failure mode from the ones this task has been fixing. Elsewhere a comment
claimed something never checked; here a comment and its guard were written from the SAME wrong
reading, which makes the pair look corroborated when it is really one opinion counted twice.

What made it findable was capturing the RAW bytes (``repr(value[:3])`` -> ``'\\n  '``) rather than
comparing ``splitlines()`` of two renderings, because a leading blank line disappears the moment
anything strips.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: From `ivregress 2sls lnw (s = expr tenure age)` then `estat endogenous`, including the
#: leading blank line that the previous version of the module docstring denied.
STATA_LINES = [
    "",
    "  Tests of endogeneity",
    "  H0: Variables are exogenous",
    "",
    "  Durbin (score) chi2(1)          =  28.8042  (p = 0.0000)",
    "  Wu-Hausman F(1,755)             =  29.8235  (p = 0.0000)",
]


@pytest.fixture(scope="module")
def endogenous():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.estat_endogenous(ws.IV2SLS.from_formula("lnw ~ [s ~ expr + tenure + age]", frame).fit())


def test_the_block_matches_stata_byte_for_byte(endogenous) -> None:
    assert str(endogenous).splitlines() == STATA_LINES


def test_the_first_character_is_a_newline(endogenous) -> None:
    """Checked at the RAW string level, where a leading blank line is still visible."""

    assert str(endogenous)[:3] == "\n  "


def test_the_statistics_are_unchanged(endogenous) -> None:
    rendered = str(endogenous)
    assert "28.8042" in rendered
    assert "29.8235" in rendered
