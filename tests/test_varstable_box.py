"""varstable renders Stata's box, byte for byte.

Rounds 40-47 verified the eigenvalue NUMBERS against hand-built lags; rounds 87-88 re-verified them
against a bare `var`. The values were never in doubt. What differed was the presentation: a dash
rule instead of a box, and a leading zero Stata never prints.

Captured byte-exactly from ``varstable`` after ``var dinf unem``:

       Eigenvalue stability condition
      +----------------------------------------+
      |        Eigenvalue        |   Modulus   |
      |--------------------------+-------------|
      |     .82182               |    .82182   |
      |   .7970223               |   .797022   |
      |  -.2258777 +  .6292178i  |   .668533   |
      |  -.2258777 -  .6292178i  |   .668533   |
      +----------------------------------------+

all lines 44 characters.

THE ALIGNMENT RULE, which took several attempts to pin down:

  * bars sit at columns 2, 29 and 43
  * a SHORT eigenvalue is right-aligned to offset 11 within its cell
  * a LONG one (the complex row) carries its own two leading spaces
  * the modulus is right-aligned to offset 10
  * the imaginary part is preceded by TWO spaces after the sign: ".2258777 +  .6292178i"

An earlier attempt right-aligned every value to one column and missed the complex row; another
left-aligned the whole cell and missed the short ones.

A note on the method: I typed the expected lines by hand twice and got them WRONG both times --
first with an extra space (45 characters), then again. The mismatch was in my target, not in the
renderer. Only capturing the log's bytes settled it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_LINES = [
    "   Eigenvalue stability condition",
    "  +----------------------------------------+",
    "  |        Eigenvalue        |   Modulus   |",
    "  |--------------------------+-------------|",
    "  |     .82182               |    .82182   |",
    "  |   .7970223               |   .797022   |",
    "  |  -.2258777 +  .6292178i  |   .668533   |",
    "  |  -.2258777 -  .6292178i  |   .668533   |",
    "  +----------------------------------------+",
]


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    fitted = ws.VAR(frame[["dinf", "unem"]].to_numpy(float), lags=2, names=["dinf", "unem"]).fit()
    return str(ws.varstable(fitted))


def test_every_line_matches_stata(rendered) -> None:
    lines = rendered.splitlines()
    for expected in STATA_LINES:
        assert expected in lines, repr(expected)


def test_the_box_rows_are_forty_four_characters(rendered) -> None:
    for line in rendered.splitlines():
        if line.startswith("  |") or line.startswith("  +"):
            assert len(line) == 44, (len(line), line)


def test_the_leading_zero_is_gone(rendered) -> None:
    """Both columns printed "0.82182" before."""

    assert ".82182" in rendered
    assert "0.82182" not in rendered


def test_the_imaginary_part_has_two_spaces_after_the_sign(rendered) -> None:
    assert "+  .6292178i" in rendered
    assert "-  .6292178i" in rendered
