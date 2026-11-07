"""estat ic: the module comment said "FOUR lines, no rules" and called itself byte-exact.

C09-01 scored 0.86 layout / 1.00 numeric -- the best-scoring item audited so far, and still wrong,
because the NUMBERS being right made the shape look settled.

A live `estat ic` on grilic is NINE lines including a leading blank:

    (blank)
    Akaike's information criterion and Bayesian information criterion
    (blank)
    ------------------------------------------------------------------------------   (77)
           Model |          N   ll(null)  ll(model)      df        AIC        BIC
    -------------+----------------------------------------------------------------   (77)
               . |        758  -433.4714  -287.9592       4   583.9184   602.4412
    ------------------------------------------------------------------------------   (77)
    Note: BIC uses N = number of observations. See [R] IC note.

Every content line is 77 wide, rules included.

AN OLDER ROUND HAD REMOVED THE RULES and recorded that removal as a finding -- "the live output has
no rules at all -- the comment and the test encoding it were both wrong". That round was right that
an even earlier version had wrapped them incorrectly, and wrong that the answer was to delete them.
The comment and two tests were then rewritten to agree with the deletion, which is the same
corroboration-of-one-mistake pattern as `estat endogenous` in round 189.

Round 191 found it by walking the PREFIX of every renderer (`str(out)[0]`) rather than by reading
the comment: `estat ic` was the only block that did not start with a newline.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: From `regress lnw s expr tenure` then `estat ic`, captured in round 191.
STATA_LINES = [
    "",
    "Akaike's information criterion and Bayesian information criterion",
    "",
    "-" * 77,
    "       Model |          N   ll(null)  ll(model)      df        AIC        BIC",
    "-" * 13 + "+" + "-" * 63,
    "           . |        758  -433.4714  -287.9592       4   583.9184   602.4412",
    "-" * 77,
    "Note: BIC uses N = number of observations. See [R] IC note.",
]


@pytest.fixture(scope="module")
def ic():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.estat_ic(ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit())


def test_the_block_matches_stata_byte_for_byte(ic) -> None:
    assert str(ic).splitlines() == STATA_LINES


def test_the_block_opens_with_a_blank_line(ic) -> None:
    """Checked at the raw prefix -- the signal that exposed this defect."""

    assert str(ic).startswith("\n")


def test_every_table_line_is_seventy_seven_wide(ic) -> None:
    """The rules, header and data row are 77 wide; the title (65) and Note (59) are not padded."""

    lines = str(ic).splitlines()
    table = lines[3:8]
    assert all(len(line) == 77 for line in table), [len(line) for line in table]
    assert len(lines[1]) == 65
    assert len(lines[8]) == 59


def test_the_numbers_are_unchanged(ic) -> None:
    """They were already right, which is exactly why the shape went unexamined."""

    rendered = str(ic)
    assert "-433.4714" in rendered
    assert "-287.9592" in rendered
    assert "583.9184" in rendered
    assert "602.4412" in rendered
