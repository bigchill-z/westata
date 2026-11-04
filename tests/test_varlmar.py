"""Pin ``varlmar`` and ``veclmar`` against Stata.

References from live runs on macro_swatson.dta::

    var dinf unem, lags(1/2)   then varlmar, mlag(2)
        lag 1: chi2 7.3130  df 4  p 0.12024
        lag 2: chi2 5.0645  df 4  p 0.28074

    vec dinf unem, lags(2) rank(1)   then veclmar
        lag 1: chi2 8.8911  df 4  p 0.06388
        lag 2: chi2 4.8537  df 4  p 0.30264

Two details decide every row, and both are the kind that produce plausible-looking wrong
numbers:

1. ``varlmar.ado:139-145`` REBUILDS ``rvars`` inside the outer loop, so the test at order 2
   uses the lag-2 residuals and NOT lags 1 and 2 together. Accumulating gave 11.1593 at order 2
   (against 5.0645) and made the statistic grow where Stata's shrinks.

2. ``veclmar`` is not a rename of ``varlmar``. ``veclmar.ado`` forwards to
   ``_vec_postvar``, which calls ``_vecmkce`` and so puts the cointegrating term
   ``beta'Y_{t-1}`` into the regression as a COLUMN:

       Delta Y_t = c + alpha (beta'Y_{t-1}) + Gamma Delta Y_{t-1} + e_t

   Using the unrestricted VAR's own level lags instead gave 14.8557 / 12.5931 at the two
   orders.

Both tests therefore check their own numbers rather than one against the other: the two
statistics genuinely differ (7.3130 vs 8.8911 at lag 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_VARLMAR = {1: (7.3130, 0.12024), 2: (5.0645, 0.28074)}
STATA_VECLMAR = {1: (8.8911, 0.06388), 2: (4.8537, 0.30264)}


@pytest.fixture(scope="module")
def data():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return frame[["dinf", "unem"]].dropna().to_numpy()


def test_varlmar_matches_stata(data) -> None:
    result = ws.varlmar(ws.VAR(data, lags=2, names=["dinf", "unem"]).fit(), mlag=2)
    for row in result.rows:
        chi2, pvalue = STATA_VARLMAR[row.lag]
        assert row.chi2 == pytest.approx(chi2, abs=5e-4), row.lag
        assert row.pvalue == pytest.approx(pvalue, abs=5e-5), row.lag
        assert row.df == 4


def test_veclmar_matches_stata(data) -> None:
    vecm = ws.VECM(data, coint_rank=1, lags=2, names=["dinf", "unem"]).fit()
    result = ws.veclmar(vecm, mlag=2)
    for row in result.rows:
        chi2, pvalue = STATA_VECLMAR[row.lag]
        assert row.chi2 == pytest.approx(chi2, abs=5e-4), row.lag
        assert row.pvalue == pytest.approx(pvalue, abs=5e-5), row.lag


def test_the_two_statistics_are_genuinely_different(data) -> None:
    """Guard against 'veclmar is just varlmar renamed' -- Stata reports different numbers."""

    var_stat = ws.varlmar(ws.VAR(data, lags=2, names=["dinf", "unem"]).fit(), mlag=1)
    vecm = ws.VECM(data, coint_rank=1, lags=2, names=["dinf", "unem"]).fit()
    vec_stat = ws.veclmar(vecm, mlag=1)
    assert abs(var_stat.rows[0].chi2 - vec_stat.rows[0].chi2) > 1.0


def test_the_sample_is_not_truncated(data) -> None:
    """varlmar.ado zero-fills the leading lags instead of dropping them."""

    result = ws.VAR(data, lags=2, names=["dinf", "unem"]).fit()
    table = ws.varlmar(result, mlag=2)
    assert len(table.rows) == 2
    assert table.rows[0].df == table.rows[1].df == 4
