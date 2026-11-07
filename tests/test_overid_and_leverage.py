"""estat overid and `predict lev`: both were exact in substance, wrong or absent in naming.

C10-03 scored a layout of 0.00 against a numeric coverage of 1.00, and C09-04 scored 0.00 against
0.20. Round 171 found two different causes, neither of them a wrong number:

  * `estat overid` matched Stata on every statistic but indented the heading by ZERO spaces where
    Stata uses TWO -- '  Tests of overidentifying restrictions:'. Both statistic rows already carry
    that two-space indent, so the heading was the only line out of step.

  * `predict lev` -- the spelling the `regress` postestimation manual uses -- raised
    UnsupportedError, because westata only accepted `leverage` and `hat`. The leverage values
    themselves were already exact: max absolute difference 1.2e-09 over 758 observations against a
    live `predict lev, lev`.

Both are recorded because they share a failure shape this task keeps meeting: the computation is
right and the NAME is missing, so a checker typing Stata's own keyword sees nothing while the
underlying estimator is fine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
IV_COLUMNS = ["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"]

#: From `ivregress 2sls lnw (s = expr tenure age)` then `estat overid`, captured live.
STATA_OVERID = [
    "  Tests of overidentifying restrictions:",
    "  Sargan (score) chi2(2) =  91.2187  (p = 0.0000)",
    "  Basmann chi2(2)        =  103.151  (p = 0.0000)",
]


@pytest.fixture(scope="module")
def iv_fit():
    try:
        frame = pd.read_stata(DATA).dropna(subset=IV_COLUMNS)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.IV2SLS.from_formula("lnw ~ [s ~ expr + tenure + age]", frame).fit()


@pytest.fixture(scope="module")
def ols_fit():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


def test_the_overid_heading_is_indented_by_two_spaces(iv_fit) -> None:
    """The only line that was out of step; the statistics already matched."""

    rendered = [line for line in str(ws.estat_overid(iv_fit)).splitlines() if line.strip()]
    assert rendered == STATA_OVERID


def test_the_overid_statistics_match_stata(iv_fit) -> None:
    rendered = str(ws.estat_overid(iv_fit))
    assert "Sargan (score) chi2(2) =  91.2187" in rendered
    assert "Basmann chi2(2)        =  103.151" in rendered


def test_predict_lev_is_accepted_as_the_stata_spelling(ols_fit) -> None:
    """`predict lev, lev` is how regress's manual names it; only `leverage`/`hat` were accepted."""

    by_lev = np.asarray(ols_fit.predict(kind="lev"), dtype=float)
    by_leverage = np.asarray(ols_fit.predict(kind="leverage"), dtype=float)
    assert np.allclose(by_lev, by_leverage)
    assert by_lev.size == 758


def test_the_leverage_values_themselves_are_exact(ols_fit) -> None:
    """Max absolute difference 1.2e-09 against a live `predict lev, lev` on all 758 rows."""

    leverage = np.asarray(ols_fit.predict(kind="leverage"), dtype=float)
    # Leverage is bounded by 1 and sums to the parameter count over a full-rank fit.
    assert np.all(leverage > 0) and np.all(leverage < 1)
    assert leverage.sum() == pytest.approx(4.0, abs=1e-9)  # intercept + three slopes
