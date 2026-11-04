"""Pin ``unitroot(test="adf")`` to Stata's ``dfuller`` defaults.

Reference from a live ``dfuller inf`` on macro_swatson.dta::

    Dickey-Fuller test for unit root    Number of obs = 166
    Number of lags = 0
    Z(t) = -3.718   p = 0.0039

westata reported -2.571 with p = 0.0992 before this fix -- a different conclusion about
whether the series has a unit root, not a rounding difference. Two defaults were wrong:

* ``autolag`` defaulted to ``"AIC"``, so the lag order was CHOSEN rather than fixed. Stata's
  ``dfuller`` uses ``lags(0)`` unless the caller asks otherwise.
* even with ``autolag=None``, ``maxlag=None`` let statsmodels pick its own default
  (``floor(12*(n/100)**0.25)`` = 14 here) instead of 0.

Both now default to Stata's behaviour, and the statistic reproduces exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def inflation():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return frame["inf"].dropna().to_numpy()


def test_adf_matches_stata_with_no_lags(inflation) -> None:
    result = ws.unitroot(inflation, test="adf")
    assert float(result.statistic) == pytest.approx(-3.718, abs=5e-4)
    assert float(result.pvalue) == pytest.approx(0.0039, abs=5e-5)


def test_the_default_fixes_the_lag_order_at_zero(inflation) -> None:
    """Stata's dfuller uses lags(0); choosing a lag changes the test, not just its precision."""

    stata_default = ws.unitroot(inflation, test="adf")
    explicit_zero = ws.unitroot(inflation, test="adf", maxlag=0, autolag=None)
    assert float(stata_default.statistic) == pytest.approx(
        float(explicit_zero.statistic), rel=1e-12
    )


def test_a_common_random_walk_is_not_rejected(inflation) -> None:
    """Sanity check on the sign convention: a strong trend must NOT look stationary."""

    rng = np.random.default_rng(11)
    walk = np.cumsum(rng.normal(size=200))
    result = ws.unitroot(walk, test="adf")
    assert float(result.pvalue) > 0.05, float(result.pvalue)
