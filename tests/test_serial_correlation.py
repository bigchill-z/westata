"""Pin the serial-correlation statistics against Stata's measured values.

All four statistics are closed form, so every number here comes from a live Stata run on
``icecream.dta`` (30 observations, ``reg consumption temp price income``) rather than from
this implementation's own output:

    estat dwatson            Durbin-Watson d(4, 30) = 1.021169
    wntestq e1               Q 26.1974, chi2(13), p .0160
    wntestq e1, lags(10)     Q 16.3718, p .0895
    corrgram e1              lag 1: ac .3298, pac .3969, Q 3.6, p .0578
                             lag 2: ac .0362, pac -.1681, Q 3.645, p .1616
                             lag 3: ac .0111, pac .0767, Q 3.6494, p .3019

Two of these needed a non-obvious detail to match:

* ``wntestq``'s default lag count is ``(n - k) // 2``, not ``n // 2``: 13 against 15 here,
  and using the latter gave Q 28.6011 instead of 26.1974;
* ``corrgram``'s PAC is the coefficient on lag ``k`` in a regression of the residual on its
  first ``k`` lags WITH A CONSTANT -- .3969 at lag 1, not ``ac[1]`` = .3298 (Yule-Walker)
  and not the no-intercept slope .4006.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    frame = frame.sort_values("time").reset_index(drop=True)
    return ws.OLS.from_formula("consumption ~ temp + price + income", frame).fit()


def test_durbin_watson_matches_stata(fitted) -> None:
    result = ws.estat_dwatson(fitted)
    assert result.d == pytest.approx(1.021169, abs=1e-6)
    assert result.nobs == 30
    assert result.df == 4.0


def test_wntestq_default_lag_count_is_n_minus_k_over_two(fitted) -> None:
    """13, not 15 -- the difference between Q 26.1974 and Q 28.6011."""

    result = ws.wntestq(fitted)
    assert result.df == 13
    assert result.q == pytest.approx(26.1974, abs=5e-4)
    assert result.pvalue == pytest.approx(0.0160, abs=5e-5)


def test_wntestq_explicit_lags_matches_stata(fitted) -> None:
    result = ws.wntestq(fitted, lags=10)
    assert result.q == pytest.approx(16.3718, abs=5e-4)
    assert result.pvalue == pytest.approx(0.0895, abs=5e-5)


def test_corrgram_autocorrelations_match_stata(fitted) -> None:
    grid = ws.estat_corrgram(fitted)
    expected = {
        1: (0.3298, 0.3969, 0.0578),
        2: (0.0362, -0.1681, 0.1616),
        3: (0.0111, 0.0767, 0.3019),
    }
    for lag, (ac, pac, pvalue) in expected.items():
        row = grid.rows[lag - 1]
        assert row.ac == pytest.approx(ac, abs=6e-5), f"ac at lag {lag}"
        assert row.pac == pytest.approx(pac, abs=6e-5), f"pac at lag {lag}"
        assert row.pvalue == pytest.approx(pvalue, abs=6e-5), f"p at lag {lag}"


def test_corrgram_q_matches_wntestq_at_the_same_order(fitted) -> None:
    """The Q column accumulates lags 1..k, so the last row equals wntestq at that order."""

    grid = ws.estat_corrgram(fitted, lags=10)
    portmanteau = ws.wntestq(fitted, lags=10)
    assert grid.rows[-1].q == pytest.approx(portmanteau.q, rel=1e-12)


def test_bgodfrey_is_not_exposed() -> None:
    """It is deliberately absent: no formula reproduced Stata's numbers.

    Measured multipliers needed to hit Stata's 4.237/4.487/4.775/5.099 at lags 1-4 were
    26.12/24.58/25.52/23.35 -- inconsistent, so the auxiliary regression differs rather
    than the scaling. Shipping a 1%-off statistic would be worse than not shipping it.
    """

    assert not hasattr(ws, "estat_bgodfrey")
