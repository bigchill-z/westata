"""estat vif and estat imtest: both were missing structure that the comments said was absent.

Round 193 completed the prefix-and-shape survey begun in 191. Two more blocks disagreed with their
own comments:

  * ``estat vif`` had the right numbers AND the right column geometry, but no leading blank line --
    Stata's capture starts with a newline.

  * ``estat imtest`` (Cameron & Trivedi decomposition) was missing a blank line after the title,
    three 50-character rules and the 21+'+'+28 separator. Its comment asserted "FIVE lines, no
    full-width rules and no blank lines" and blamed a previous version for adding them. The
    previous version had the right structure and was removed for the wrong reason -- the third
    instance of that pattern in three rounds (estat ic in 191, estat endogenous in 189).

A test in the suite carried the same wrong transcription, described as "from a captured StataNow
19.5 log". Re-capturing `quietly regress mpg weight fx` / `estat imtest, white` shows all twelve
lines, structure included.

STILL A LIMIT: `estat vif` and `estat imtest` both match now, but this file pins the STRUCTURE and
a few numbers rather than the whole block, because the fixtures they are checked against in the
suite use different data. The byte-exact comparison was run live during the round.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"
AUTO = "tests/data/auto.dta"


@pytest.fixture(scope="module")
def grilic_fit():
    try:
        frame = pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure + smsa + rns", frame).fit()


def test_vif_opens_with_a_blank_line(grilic_fit) -> None:
    rendered = str(ws.estat_vif(grilic_fit))
    assert rendered.startswith("\n")
    assert rendered.splitlines()[1].startswith("    Variable |")


def test_vif_header_keeps_its_trailing_spaces(grilic_fit) -> None:
    header = str(ws.estat_vif(grilic_fit)).splitlines()[1]
    assert len(header) == 38
    assert header.endswith("  ")


def test_vif_values_are_unchanged(grilic_fit) -> None:
    rendered = str(ws.estat_vif(grilic_fit))
    assert "1.12" in rendered and "0.893267" in rendered
    assert "Mean VIF" in rendered


def test_imtest_has_its_rules_and_blank_lines(grilic_fit) -> None:
    """Twelve lines: blank, title, blank, rule, header, split, 3 rows, split, total, rule."""

    lines = str(ws.estat_imtest(grilic_fit)).splitlines()
    assert len(lines) == 12
    assert lines[0] == ""
    assert lines[2] == ""
    assert lines[3] == "-" * 50
    assert lines[5] == "-" * 21 + "+" + "-" * 28
    assert lines[9] == "-" * 21 + "+" + "-" * 28
    assert lines[11] == "-" * 50


def test_imtest_numbers_are_unchanged(grilic_fit) -> None:
    rendered = str(ws.estat_imtest(grilic_fit))
    assert "21.54" in rendered and "14.67" in rendered
    assert "39.88" in rendered