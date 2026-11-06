"""ereturn list: the name field is 22 wide, not 23.

Captured from a live ``regress lnw s`` / ``ereturn list``:

    scalars:
                      e(N) =  758                 29 characters
                   e(df_m) =  1                    27
                   e(df_r) =  756                  29
                     e(r2) =  .2527458374860051    43

Eighteen spaces, "e(N)" (4), " =  " (4), then the value -- so the name field is 22. The code used
23, which put every scalar line one column past Stata's. Three tests had been written around that
23 and are corrected here rather than left to encode it.

The scalar VALUE format (``%.16g`` with the leading zero dropped) was already right: 255.70396623363246
prints as "255.7039662336325" and .2527458374860051 as ".2527458374860051", both exactly as Stata
shows them. The remaining difference on e(F) is a last-digit computation difference, not formatting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.ereturn import _NAME_WIDTH


@pytest.fixture(scope="module")
def rendered():
    try:
        frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\grilic.dta").dropna(
            subset=["lnw", "s"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.format_ereturn_list(ws.OLS.from_formula("lnw ~ s", frame).fit())


def test_the_name_field_is_twenty_two(rendered) -> None:
    assert _NAME_WIDTH == 22


def test_scalar_lines_match_stata(rendered) -> None:
    lines = rendered.splitlines()
    assert "                  e(N) =  758" in lines
    assert "               e(df_m) =  1" in lines
    assert "               e(df_r) =  756" in lines


def test_the_long_scalar_is_forty_three_characters(rendered) -> None:
    line = next(row for row in rendered.splitlines() if "e(r2)" in row)
    assert len(line) == 43, (len(line), line)
    assert line == "                 e(r2) =  .2527458374860051"


def test_macro_lines_match_stata(rendered) -> None:
    lines = rendered.splitlines()
    assert '              e(title) : "Linear regression"' in lines


def test_matrix_shape_lines_match_stata(rendered) -> None:
    lines = rendered.splitlines()
    assert "                  e(b) :  1 x 2" in lines
    assert "                  e(V) :  2 x 2" in lines


def test_the_scalar_value_format_keeps_sixteen_digits() -> None:
    """The value format was already correct; only the field width was wrong."""

    from westata.postestimation.ereturn import _scalar_value

    assert _scalar_value(255.70396623363246) == "255.7039662336325"
    assert _scalar_value(0.2527458374860051) == ".2527458374860051"
