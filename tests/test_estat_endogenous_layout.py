"""estat endogenous: the block starts flush, both header lines indented two spaces.

C10-05 scores 0.00 on line agreement while its numbers were already exact:

    Durbin (score) chi2(1)  = 24.9812   p 0.0000    <- matched
    Wu-Hausman F(1,751)     =  25.594   p 0.0000    <- matched

Three layout details were wrong together, which is why the whole block missed:

    previous                              Stata
    <blank line first>                    starts at line 1
    "Tests of endogeneity"                "  Tests of endogeneity"      two-space indent
    "H0: Variables are exogenous"         "  H0: Variables are exogenous"
    <blank line>                          no blank line
    "  Durbin ..."                        "  Durbin ..."

The old docstring showed the intended output without the indent and WITH the blank lines, so it
described something that was not what Stata prints. Capturing the real block is what settled it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

STATA_LINES = [
    "  Tests of endogeneity",
    "  H0: Variables are exogenous",
    "  Durbin (score) chi2(1)          =  24.9812  (p = 0.0000)",
    "  Wu-Hausman F(1,751)             =   25.594  (p = 0.0000)",
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(DATA).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "iq"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    result = ws.IV2SLS.from_formula(
        "lnw ~ expr + tenure + rns + smsa + [s ~ mrt + iq]", frame
    ).fit()
    return str(result.estat_endogenous())


def test_every_line_matches_stata(rendered) -> None:
    lines = rendered.splitlines()
    for expected in STATA_LINES:
        assert expected in lines, repr(expected)


def test_there_is_a_leading_blank_line(rendered) -> None:
    """CORRECTED in round 189 from a live capture.

    The output begins with '\\n  ', so line 0 IS blank and the title is line 1. This test used to
    assert the opposite, and the module docstring agreed with it -- the two were written together
    from the same wrong reading, which is why neither caught the other.
    """

    lines = rendered.splitlines()
    assert lines[0] == ""
    assert lines[1] == "  Tests of endogeneity"


def test_a_blank_line_separates_the_null_hypothesis_from_the_statistics(rendered) -> None:
    """CORRECTED in round 189: 'H0' at index 2, blank at 3, 'Durbin' at 4."""

    lines = rendered.splitlines()
    assert lines[2] == "  H0: Variables are exogenous"
    assert lines[3] == ""
    assert lines[4].startswith("  Durbin")


def test_the_header_lines_share_the_statistics_indent(rendered) -> None:
    """CORRECTED in round 189: blank lines do not start with two spaces, so the check applies to
    the NON-BLANK ones only."""

    lines = rendered.splitlines()
    assert all(line.startswith("  ") for line in lines if line)
    assert not any(line.startswith("   ") for line in lines)


def test_the_statistics_themselves_are_unchanged(rendered) -> None:
    """A layout fix must not move the numbers."""

    assert "24.9812" in rendered
    assert "25.594" in rendered
