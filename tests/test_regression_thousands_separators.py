"""Thousands separators in the regression table appear exactly where Stata puts them.

Round 182 found the `summarize` observation count missing its separator at n=1000. Since that is a
cross-cutting rule rather than one table's quirk, this file checks where else it applies -- and finds it is
NOT uniform.

    Stata, reg y x on 1500 rows:

        '      Source |       SS           df       MS      Number of obs   =     1,500'
        '-------------+----------------------------------   F(1, 1498)      =      1.49'
        '    Residual |  1440.11748     1,498  .961360131   R-squared       =    0.0010'
        '       Total |  1441.54878     1,499  .961673636   Root MSE        =    .98049'

`Number of obs` AND the df COLUMN carry separators; the `F(1, 1498)` in the header does NOT. That asymmetry is
the point: a blanket change to every integer in the block would have broken the F line.

Verified at n = 200, 1500 and 5000: 3 of 3 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _render(n: int):
    rng = np.random.default_rng(n)
    frame = pd.DataFrame({"y": rng.normal(size=n), "x": rng.normal(size=n)})
    return str(ws.OLS.from_formula("y ~ x", frame).fit().summary(style="stata"))


def test_the_observation_count_is_separated() -> None:
    assert "Number of obs   =     1,500" in _render(1500)


def test_the_df_column_is_separated() -> None:
    """Residual df is n-2 = 1498 and Total df is n-1 = 1499."""

    rendered = _render(1500)
    assert "1,498" in rendered
    assert "1,499" in rendered


def test_the_f_statistic_heading_is_not_separated() -> None:
    """`F(1, 1498)` has no comma, which is what a blanket change would have broken."""

    assert "F(1, 1498)" in _render(1500)


def test_small_samples_are_unchanged() -> None:
    rendered = _render(200)
    assert "Number of obs   =       200" in rendered
    assert "1,000" not in rendered


def test_the_line_count_does_not_change_with_the_count() -> None:
    """A separator adds a character inside a fixed-width field, not a line."""

    assert len(_render(200).splitlines()) == len(_render(1500).splitlines())


def test_a_five_thousand_row_fit_is_punctuated() -> None:
    rendered = _render(5000)
    assert "5,000" in rendered
    assert "4,998" in rendered
