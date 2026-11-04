"""Pin ``estat firststage`` to Stata's two-table output, character by character.

Reference from a live ``quietly ivregress 2sls y x (endog = z1 z2)`` then
``estat firststage`` on a synthetic sample (n = 400)::

    First-stage regression summary statistics
    --------------------------------------------------------------------------
                 |            Adjusted      Partial
        Variable |   R-sq.       R-sq.        R-sq.      F(2,396)   Prob > F
    -------------+------------------------------------------------------------
           endog |  0.6288      0.6260       0.6267       332.355    0.0000
    --------------------------------------------------------------------------

      Minimum eigenvalue statistic = 332.355

      Critical Values                      # of endogenous regressors:    1
      H0: Instruments are weak             # of excluded instruments:     2
      ---------------------------------------------------------------------
                                         |    5%     10%     20%     30%
      2SLS relative bias                 |         (not available)
      -----------------------------------+---------------------------------
                                         |   10%     15%     20%     25%
      2SLS size of nominal 5% Wald test  |  19.93   11.59    8.75    7.25
      LIML size of nominal 5% Wald test  |   8.68    5.33    4.42    3.92
      ---------------------------------------------------------------------

Two things this pins that were previously wrong:

* ``r(mineig)`` is the STATISTIC (332.355), not the raw eigenvalue (0.6267). Confirmed
  against both ``display r(mineig)`` and the identity ``eig/(1-eig)*df_denom/df_num``.
* the Stock-Yogo block was computed but never rendered, so the weak-instrument verdict --
  the reason to run the command -- never appeared.

The first critical-value cell is SEVEN wide and the remaining three are eight; that
asymmetry is Stata's and cost several attempts to discover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame():
    rng = np.random.default_rng(4242)
    n = 400
    w = rng.normal(size=n)
    x = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    endog = 1.0 * z1 + 0.5 * z2 + 0.6 * u + rng.normal(scale=0.5, size=n)
    y = 2.0 * endog + 0.8 * x + u
    return pd.DataFrame({"y": y, "endog": endog, "x": x, "z1": z1, "z2": z2})


@pytest.fixture(scope="module")
def first_stage():
    frame = _frame()
    fit = ws.IV2SLS.from_formula("y ~ x + [endog ~ z1 + z2]", frame).fit()
    return ws.estat_firststage(fit)


STATA_LINES = [
    "First-stage regression summary statistics",
    "--------------------------------------------------------------------------",
    "             |            Adjusted      Partial",
    "    Variable |   R-sq.       R-sq.        R-sq.      F(2,396)   Prob > F",
    "-------------+------------------------------------------------------------",
    "       endog |  0.6288      0.6260       0.6267       332.355    0.0000",
    "--------------------------------------------------------------------------",
]


def _lines(first_stage) -> list[str]:
    rendered = str(first_stage).splitlines()
    return [line for line in rendered if line.strip()]


def test_the_summary_table_matches_stata_line_for_line(first_stage) -> None:
    lines = _lines(first_stage)
    for expected in STATA_LINES:
        assert expected in lines, expected


def test_mineig_is_the_statistic_not_the_eigenvalue(first_stage) -> None:
    """Stata prints 332.355; the partial R-squared is 0.6267."""

    assert first_stage.mineig == pytest.approx(332.355, abs=5e-3)
    assert first_stage.mineig > 1.0, "a [0,1] eigenvalue is the wrong scale"


def test_the_stock_yogo_block_is_rendered(first_stage) -> None:
    """It was computed and stored but never printed, hiding the weak-IV verdict."""

    rendered = str(first_stage)
    assert "Minimum eigenvalue statistic" in rendered
    assert "H0: Instruments are weak" in rendered
    assert "# of endogenous regressors" in rendered
    assert "# of excluded instruments" in rendered


def test_the_critical_values_match_stata(first_stage) -> None:
    rendered = str(first_stage)
    assert "2SLS size of nominal 5% Wald test  |  19.93   11.59    8.75    7.25" in rendered
    assert "LIML size of nominal 5% Wald test  |   8.68    5.33    4.42    3.92" in rendered


def test_the_relative_bias_row_says_not_available(first_stage) -> None:
    rendered = str(first_stage)
    assert "2SLS relative bias                 |         (not available)" in rendered


def test_mineig_satisfies_the_eigenvalue_identity(first_stage) -> None:
    """An independent path to the same number, so the scale is not a coincidence."""

    partial = float(first_stage.partial_r2[0])
    expected = (
        partial / (1.0 - partial)
        * float(first_stage.df_denom[0])
        / float(first_stage.df_num[0])
    )
    assert first_stage.mineig == pytest.approx(expected, rel=1e-12)
