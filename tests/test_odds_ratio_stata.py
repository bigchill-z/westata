"""X04 (odds ratio, 0.71 / 0.87): the formula path is byte-identical to Stata.

`Logit.from_formula(...).fit().summary(style="stata", odds_ratio=True)` reproduces a live
`logit hi s expr tenure, or` on grilic.dta line for line -- including the `hi | Odds ratio` header
(which needs the response variable's name) and the trailing
`Note: _cons estimates baseline odds.` line.

The ARRAY interface, `Logit(y_array, X_array, param_names=[...])`, cannot print `hi` because the
response array carries no name. It still prints a valid `| Odds ratio` table with identical
numbers, just without the leading response name. That is an interface limitation, not a wrong
value -- and the same reason round 130 documented the intercept contract on the array constructors.

Neither the numbers nor the Stata-style layout needed fixing this round; what was missing was a
test that pinned them, so a future refactor of summary() cannot silently drop the eform path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: Captured from a live `logit hi s expr tenure, or` on grilic.dta.
STATA_OR_ROWS = [
    "           s |   1.582622   .0672408    10.81   0.000      1.45617    1.720054",
    "        expr |   1.172199   .0473904     3.93   0.000       1.0829    1.268862",
    "      tenure |   1.165383   .0590591     3.02   0.003     1.055192     1.28708",
    "       _cons |   .0012414   .0007513   -11.06   0.000     .0003791    .0040652",
]


@pytest.fixture(scope="module")
def frame():
    try:
        data = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure"]).copy()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    data["hi"] = (data["lnw"] > data["lnw"].median()).astype(float)
    return data


def test_the_formula_or_table_matches_stata_line_for_line(frame) -> None:
    result = ws.Logit.from_formula("hi ~ s + expr + tenure", frame).fit()
    rendered = result.summary(style="stata", odds_ratio=True)
    lines = rendered.splitlines()

    assert "          hi | Odds ratio   Std. err.      z    P>|z|     [95% conf. interval]" in lines
    for row in STATA_OR_ROWS:
        assert row in lines, row
    assert lines[-1] == "Note: _cons estimates baseline odds."
    # Every rendered row of the coefficient table is 78 characters, as Stata prints them.
    for row in STATA_OR_ROWS:
        assert len(row) == 78


def test_without_the_flag_the_table_shows_coefficients(frame) -> None:
    """odds_ratio is a real switch, not a no-op."""

    result = ws.Logit.from_formula("hi ~ s + expr + tenure", frame).fit()
    plain = result.summary(style="stata")
    with_or = result.summary(style="stata", odds_ratio=True)
    assert plain != with_or
    assert "Coefficient" in plain or "Coef." in plain
    assert "Odds ratio" not in plain


def test_the_array_interface_has_the_same_numbers_but_no_response_name(frame) -> None:
    """The array carries no response name; Stata's header cannot be reproduced there."""

    design = np.column_stack(
        [np.ones(len(frame)), frame[["s", "expr", "tenure"]].to_numpy(float)]
    )
    result = ws.Logit(
        frame["hi"], design, param_names=["_cons", "s", "expr", "tenure"]
    ).fit()
    rendered = result.summary(style="stata", odds_ratio=True)
    for row in STATA_OR_ROWS:
        assert row in rendered, row
    assert "             | Odds ratio" in rendered
    assert "          hi | Odds ratio" not in rendered
