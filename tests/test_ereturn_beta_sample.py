"""ereturn list: e(beta) and e(sample) -- the last two structural gaps in X05.

Round 147 established that X05's 0.58 was a REAL structural difference, not a mismeasurement: the
four blocks were all present, but ``matrices:`` lacked ``e(beta)`` and ``functions:`` was empty
while Stata prints ``e(sample)``. Both are now implemented.

WHAT e(beta) IS, measured rather than guessed. Two identities agree to 8 decimals on grilic.dta:
re-running the regression on standardized data (sample sd, ddof=1, response and every regressor),
or scaling each coefficient by ``sd(x_k)/sd(y)``. Both give

    s        .55469752
    expr     .18441586
    tenure    .1487096

which are exactly Stata's. An earlier draft of this module claimed the scaling route was a
"single-regressor shortcut" that fails with correlated regressors and quoted .77974961 for
``expr``; that figure was an arithmetic slip of mine, not a property of the formula.

WHAT e(sample) IS: a ``functions:`` entry with a NAME AND NOTHING ELSE --
``'             e(sample)   '``, 25 characters, no value. Nothing is computed for it; printing the
row is the whole implementation.

Both rows now match Stata byte for byte, as do e(b) and e(V) around them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation import format_ereturn_list, standardized_coefficients

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"

#: From `mat list e(beta)` after `regress lnw s expr tenure`.
STATA_BETA = {"s": 0.55469752, "expr": 0.18441586, "tenure": 0.1487096}

#: Captured from `ereturn list` on the same fit.
STATA_MATRIX_LINES = [
    "                  e(b) :  1 x 4",
    "                  e(V) :  4 x 4",
    "               e(beta) :  1 x 3",
]
STATA_SAMPLE_LINE = "             e(sample)   "


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


def test_the_standardized_coefficients_match_stata(fitted) -> None:
    beta = standardized_coefficients(fitted)
    assert list(beta.index) == ["s", "expr", "tenure"]
    for name, expected in STATA_BETA.items():
        assert float(beta[name]) == pytest.approx(expected, abs=5e-8), name


def test_the_scaling_formula_agrees_with_the_standardized_refit(fitted) -> None:
    """``b_k * sd(x_k)/sd(y)`` and re-fitting on standardized data are the SAME thing.

    Pinned because an earlier draft of this module asserted the opposite -- that the scaling
    formula is a single-regressor shortcut that fails with correlated regressors -- and backed it
    with a hand-computed .77974961 for ``expr``. That number was my arithmetic error: the correct
    scaling gives .18441586, exactly Stata's value. The claim was wrong and the test now records
    the identity instead of a false warning.
    """

    frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure"])
    response_sd = frame["lnw"].std(ddof=1)
    beta = standardized_coefficients(fitted)
    for name in ("s", "expr", "tenure"):
        scaled = float(fitted.params[name]) * frame[name].std(ddof=1) / response_sd
        assert scaled == pytest.approx(float(beta[name]), abs=1e-8), name
        assert scaled == pytest.approx(STATA_BETA[name], abs=5e-8), name


def test_the_matrix_block_matches_stata(fitted) -> None:
    lines = format_ereturn_list(fitted).splitlines()
    for expected in STATA_MATRIX_LINES:
        assert expected in lines, repr(expected)


def test_the_functions_block_lists_e_sample_with_no_value(fitted) -> None:
    lines = format_ereturn_list(fitted).splitlines()
    assert "functions:" in lines
    assert STATA_SAMPLE_LINE in lines
    assert len(STATA_SAMPLE_LINE) == 25


def test_a_result_without_regressors_gets_no_beta_row(fitted) -> None:
    """A model whose only parameter is the constant has nothing to standardize."""

    frame = pd.read_stata(DATA).dropna(subset=["lnw"])
    intercept_only = ws.OLS.from_formula("lnw ~ 1", frame).fit()
    lines = format_ereturn_list(intercept_only).splitlines()
    assert not any("e(beta)" in line for line in lines)
