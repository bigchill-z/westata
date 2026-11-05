"""Pin ``ereturn list`` to Stata's four-block layout.

Reference from a live ``regress y k l`` then ``ereturn list`` on cobb_douglas.dta, measured
character by character::

                      e(N) =  24
                   e(r2) =  .9424429769032044
                 e(rmse) =  10.98532816796313
                    e(mss) =  41495.6071992186
                    e(rss) =  2534.226134114726
                   e(r2_a) =  .9369613556558906
                     e(ll) =  -89.9696019953156
                   e(ll_0) =  -124.2293513940736
                   e(rank) =  3

    macros:
                e(cmdline) : "regress y k l"
                  e(title) : "Linear regression"

    matrices:
                      e(b) :  1 x 3
                      e(V) :  3 x 3

    functions:

The details that decide a match, each taken from that output rather than assumed:

* blocks always appear in the order scalars, macros, matrices, functions, each followed by a
  blank line;
* the name field is right-aligned to 23 characters;
* scalars are ``<name> = <two spaces><value>``;
* macros are ``<name> : "<value>"`` with ONE space before the colon;
* matrices are ``<name> : <two spaces><rows> x <cols>``;
* scalar values print up to 16 significant digits with no leading zero (``.9424...``) and no
  trailing zeros (``24``, not ``24.0000``).
"""

from __future__ import annotations

import numpy as np
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


def test_the_blocks_appear_in_statas_order(fitted) -> None:
    rendered = ws.format_ereturn_list(fitted)
    positions = [
        rendered.index("scalars:"),
        rendered.index("macros:"),
        rendered.index("matrices:"),
        rendered.index("functions:"),
    ]
    assert positions == sorted(positions)


def test_scalar_values_match_stata(fitted) -> None:
    """Values are compared through the rendered text, which is what Stata shows."""

    rendered = ws.format_ereturn_list(fitted)
    for line, expected in (
        ("e(r2)", ".942442976903204"),
        ("e(mss)", "41495.607199"),
        ("e(rss)", "2534.2261341"),
        ("e(ll)", "-89.9696019953"),
        ("e(rank)", "3"),
    ):
        match = [row for row in rendered.splitlines() if expected in row]
        assert match, f"{line} = {expected} not rendered"


def test_the_scalar_spacing_and_width(fitted) -> None:
    rendered = ws.format_ereturn_list(fitted)
    line = next(row for row in rendered.splitlines() if "e(rank)" in row)
    # '                   e(rank) =  3' -- 23-wide name, " =  " then the value.
    assert line == f"{'e(rank)':>23} =  3"


def test_the_macro_separator_differs_from_the_scalar_one(fitted) -> None:
    """Macros use a single space before a colon; scalars use ' =  '."""

    rendered = ws.format_ereturn_list(fitted)
    macro = next(
        row for row in rendered.splitlines() if 'e(title)' in row and ':' in row
    )
    assert macro == f"{'e(title)':>23} : \"Linear regression\""


def test_the_matrix_shapes_are_reported(fitted) -> None:
    rendered = ws.format_ereturn_list(fitted)
    assert f"{'e(b)':>23} :  1 x 3" in rendered
    assert f"{'e(V)':>23} :  3 x 3" in rendered


def test_strings_are_filed_as_macros_and_numbers_as_scalars(fitted) -> None:
    """Stata splits the two by type, not by a hard-coded name list."""

    rendered = ws.format_ereturn_list(fitted)
    scalars_block = rendered.split("macros:")[0]
    macros_block = rendered.split("macros:")[1].split("matrices:")[0]
    assert "e(rank)" in scalars_block
    assert "e(title)" in macros_block
    assert "e(title)" not in scalars_block


def test_missing_values_render_as_a_dot() -> None:
    """Stata prints '.' for a missing scalar; the formatter must not print 'nan'."""

    class _Stub:
        e = {"N": float("nan"), "cmd": "regress"}
        params = np.zeros(2)
        level = 0.95

        def cov_params(self):
            return np.eye(2)

    rendered = ws.format_ereturn_list(_Stub())
    assert "e(N)" in rendered
    assert "nan" not in rendered.lower()
    assert "=  ." in rendered
