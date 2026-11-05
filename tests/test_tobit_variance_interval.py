"""Pin tobit's ``var(e.y)`` interval and Stata's six-character name abbreviation.

Reference from a live ``tobit food_exp income, ll(0)`` on food.dta (n = 40, df_r = 39)::

            food_exp | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
    -------------+----------------------------------------------------------------
          income |   .1020964   .0204026     5.00   0.000     .0608283    .1433646
           _cons |   83.41601   42.31099     1.97   0.056    -2.166051    168.9981
    -------------+----------------------------------------------------------------
    var(e.food~p)|   7612.629   1702.236                      4842.936    11966.32

TWO corrections, both measured.

1. The variance interval's critical value is a t with ``e(df_r)`` freedom, not the normal's
   1.96. Recovering the implied z from Stata's bounds gives 2.022691 on BOTH sides, and
   ``t.ppf(0.975, 39) = 2.022691`` exactly -- that match is what identified the distribution:

       1.96  ->  4911.303   11799.65
       t(39) ->  4842.935   11966.32    (Stata: 4842.936 / 11966.32)

2. The scale row's variable name is abbreviated to a FIXED six characters (first four, a
   tilde, the last one) and its pipe sits at column 13 regardless of the table's name width:

       food_exp      -> food~p
       abcdefghijklm -> abcd~m

   An eight-character name shortens exactly like a thirteen-character one, so this is a fixed
   target width, not an overflow rule. Deriving the row's width from the table's own name
   column produced an 81-character table where Stata prints 78.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.results.summary import _stata_abbreviated

DATA = r"D:\myCodeProject\test_westata\data\food.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("food.dta is not available")
    return ws.Tobit.from_formula("food_exp ~ income", frame, lower=0).fit()


def test_the_variance_interval_matches_stata(fitted) -> None:
    rendered = str(fitted.summary(style="stata"))
    assert "4842.936" in rendered
    assert "11966.32" in rendered


def test_the_critical_value_is_a_t_not_the_normal() -> None:
    """The identification: Stata's own bounds imply 2.022691, which is t(39)."""

    from scipy import stats

    assert float(stats.t.ppf(0.975, 39)) == pytest.approx(2.022691, abs=1e-6)
    # And the normal would have given visibly different bounds.
    normal_lo = 7612.629 * float(np.exp(-1.959963984540054 * 1702.236 / 7612.629))
    assert abs(normal_lo - 4842.936) > 60


def test_the_variance_row_matches_stata_character_for_character(fitted) -> None:
    rendered = str(fitted.summary(style="stata")).splitlines()
    expected = (
        "var(e.food~p)|   7612.629   1702.236"
        "                      4842.936    11966.32"
    )
    assert expected in rendered
    assert len(expected) == 78


def test_the_abbreviation_rule_is_a_fixed_six_characters() -> None:
    assert _stata_abbreviated("food_exp") == "food~p"
    assert _stata_abbreviated("abcdefghijklm") == "abcd~m"
    # Short names pass through untouched.
    assert _stata_abbreviated("income") == "income"
    assert _stata_abbreviated("age") == "age"


def test_the_whole_table_is_78_wide(fitted) -> None:
    """The scale row must not widen the table even though its name is long."""

    for line in str(fitted.summary(style="stata")).splitlines():
        if line.strip():
            assert len(line) == 78, line


def test_the_coefficient_rows_match_stata(fitted) -> None:
    rendered = str(fitted.summary(style="stata"))
    assert "      income |   .1020964   .0204026     5.00   0.000     .0608283    .1433646" in rendered
    assert "       _cons |   83.41601   42.31099     1.97   0.056    -2.166051    168.9981" in rendered
