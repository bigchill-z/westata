"""Pin ``estat_bgodfrey`` against Stata.

Reference from a live ``estat bgodfrey, lags(1/4)`` after ``reg consumption temp price income``
on icecream.dta::

    lag 1: chi2 4.237  df 1  p 0.0396
    lag 2: chi2 4.487  df 2  p 0.1061
    lag 3: chi2 4.775  df 3  p 0.1891
    lag 4: chi2 5.099  df 4  p 0.2773

This statistic was twice declared unimplementable earlier in the session, after six guessed
variants all missed. The formula was in ``ado/base/b/bgodfrey.ado`` the whole time, in a
readable ``Calc`` subroutine::

    tsrevar l(1/`lags').`res'
    foreach i of local reslist { qui replace `i' = 0 if `i' >= . & `touse' }
    qui regress `res' `reslist' `varlist' if `touse', `cons'
    return scalar chi2 = e(N)*e(r2)

The decisive line is the ZERO FILL: lagged residuals lose their leading values and those are
set to zero rather than dropped, so the auxiliary regression keeps all ``e(N)`` observations.
Truncating the sample by ``p`` rows gives 4.704 / 5.111 / 5.053 / 5.679 -- close at lag 1,
wrong everywhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"

STATA = {
    1: (4.237, 0.0396),
    2: (4.487, 0.1061),
    3: (4.775, 0.1891),
    4: (5.099, 0.2773),
}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    frame = frame.sort_values("time").reset_index(drop=True)
    return ws.OLS.from_formula("consumption ~ temp + price + income", frame).fit()


def test_every_lag_matches_stata(fitted) -> None:
    table = ws.estat_bgodfrey(fitted, lags=[1, 2, 3, 4])
    assert [row.lags for row in table.rows] == [1, 2, 3, 4]
    for row in table.rows:
        chi2, pvalue = STATA[row.lags]
        assert row.chi2 == pytest.approx(chi2, abs=5e-4), f"chi2 at lag {row.lags}"
        assert row.pvalue == pytest.approx(pvalue, abs=5e-5), f"p at lag {row.lags}"
        assert row.df == row.lags


def test_default_is_lag_one(fitted) -> None:
    table = ws.estat_bgodfrey(fitted)
    assert len(table.rows) == 1
    assert table.rows[0].chi2 == pytest.approx(4.237, abs=5e-4)


def test_the_zero_fill_keeps_the_full_sample(fitted) -> None:
    """e(N) is the whole sample; the lagged residuals are zero-padded, not dropped."""

    table = ws.estat_bgodfrey(fitted, lags=4)
    assert table.nobs == 30


def test_robust_fits_are_refused(fitted) -> None:
    """``bgodfrey.ado:5-10`` rejects a robust regression, and so does this."""

    frame = pd.read_stata(DATA, convert_categoricals=False).sort_values("time")
    robust = ws.OLS.from_formula(
        "consumption ~ temp + price + income", frame.reset_index(drop=True)
    ).fit(cov_type="hc1")
    with pytest.raises(Exception):
        ws.estat_bgodfrey(robust)


def test_the_nonrobust_cov_type_is_not_mistaken_for_robust(fitted) -> None:
    """westata spells the ordinary type "nonrobust"; a substring test would reject it."""

    assert str(fitted.cov_type).lower() == "nonrobust"
    assert ws.estat_bgodfrey(fitted).rows[0].chi2 > 0.0
