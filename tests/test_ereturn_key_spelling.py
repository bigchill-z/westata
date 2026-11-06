"""e() accepts both ``N`` and ``e(N)`` as a key.

C04-03 scores 0.00 in the report. Checking the key sets, nothing is missing -- our e() carries
25 keys against Stata's 13, and every Stata key is present. The difference is SPELLING:
``ereturn list`` prints ``e(N)``, ``e(r2)``, ``e(F)``, while the mapping stores the bare names.

Anyone reading a name out of that listing, or out of the Stata manual, then writes
``result.e["e(N)"]`` and gets a KeyError that says nothing about the convention. Both spellings
are accepted now.

Worth noting C04-04 and C05-08, which also score 0.00 on predict: those numbers are exact too
(5.419314 / .4806856 against 5.4193145 / 0.4806856). All three zeros are format mismatches, not
arithmetic ones -- 0.00 in a line-wise metric does not mean "wrong".
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


def test_the_bare_key_still_works(fitted) -> None:
    assert float(fitted.e["N"]) == 758.0


def test_the_stata_spelling_works_too(fitted) -> None:
    """What ``ereturn list`` prints, used as a key."""

    assert float(fitted.e["e(N)"]) == 758.0
    assert float(fitted.e["e(N)"]) == float(fitted.e["N"])


def test_a_computed_scalar_through_the_stata_spelling(fitted) -> None:
    assert float(fitted.e["e(r2)"]) == pytest.approx(float(fitted.e["r2"]))


def test_membership_accepts_both_spellings(fitted) -> None:
    assert "N" in fitted.e
    assert "e(N)" in fitted.e


def test_get_accepts_both_spellings(fitted) -> None:
    assert fitted.e.get("e(F)") == fitted.e.get("F")


def test_an_absent_key_is_still_absent(fitted) -> None:
    """The normalisation must not invent keys."""

    assert "e(no_such_scalar)" not in fitted.e
    assert fitted.e.get("e(no_such_scalar)") is None


def test_ordinary_lookups_are_unaffected(fitted) -> None:
    """A key that legitimately contains parentheses is not mangled."""

    assert fitted.e["df_m"] == fitted.e.get("df_m")
