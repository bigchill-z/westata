"""Pin ``vecrank`` against Stata's Johansen table.

Reference from a live ``vecrank dinf unem, lags(2)`` on macro_swatson.dta::

    Johansen tests for cointegration
    Trend: Constant                Number of obs = 164
                                   Number of lags =  2

    rank  Params       LL        Eigenvalue   Trace   5% crit
       0    6       1132.9103        .        183.0918  15.41
       1    9       1220.5657     0.65664       7.7811   3.76
       2   10       1224.4562     0.04634

Every cell reproduces.

``vecrank.ado`` is display-only -- it reads ``e(trace)``/``e(max)`` from the builtin
``_vecu``, which has no source. The mapping was therefore established by measurement:

    lags(2)  ->  det_order=0, k_ar_diff=1   trace 183.09176 / 7.7810575
    lags(3)  ->  det_order=0, k_ar_diff=2   trace 74.151124 / 7.3883271

Both match Stata. ``lags(1)`` does NOT: it maps to ``k_ar_diff=0``, where the backend reports
173.8827 / 2.8945 against Stata's 172.50341 / 1.2744366, and shifting the sample start lands at
172.4773 / 2.7275 -- right on the first statistic, wrong on the second. It raises instead of
returning the nearly-right answer.

Two relations were needed for the remaining columns:

* ``Params`` = ``k*(k*(lags-1)) + k + r*(2k-r)``, giving 6 / 9 / 10;
* ``LL(r)`` = ``LL_unrestricted - trace(r)/2``, because LR = 2*dLL and the trace statistic IS
  that likelihood ratio. The unrestricted value 1224.4562 comes from the VAR on the same
  sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.vecrank import vecrank

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA = {
    0: {"params": 6, "ll": 1132.9103, "eigenvalue": None, "trace": 183.0918, "critical": 15.41},
    1: {"params": 9, "ll": 1220.5657, "eigenvalue": 0.65664, "trace": 7.7811, "critical": 3.76},
    2: {"params": 10, "ll": 1224.4562, "eigenvalue": 0.04634, "trace": None, "critical": None},
}


@pytest.fixture(scope="module")
def data():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return frame[["dinf", "unem"]].dropna().to_numpy()


@pytest.fixture(scope="module")
def table(data):
    return vecrank(data, lags=2)


def test_sample_and_lag_count_match_stata(table) -> None:
    assert table.nobs == 164
    assert table.lags == 2
    assert table.equations == 2


def test_parameter_counts_match_stata(table) -> None:
    for row in table.rows:
        assert row.params == STATA[row.rank]["params"], row.rank


def test_log_likelihoods_match_stata(table) -> None:
    for row in table.rows:
        assert row.loglikelihood == pytest.approx(
            STATA[row.rank]["ll"], abs=5e-4
        ), row.rank


def test_eigenvalues_match_stata(table) -> None:
    """Stata prints each rank's eigenvalue on ITS row, from rank 1 down."""

    assert table.rows[1].eigenvalue == pytest.approx(STATA[1]["eigenvalue"], abs=5e-6)
    assert table.rows[2].eigenvalue == pytest.approx(STATA[2]["eigenvalue"], abs=5e-6)


def test_trace_statistics_match_stata(table) -> None:
    for rank in (0, 1):
        row = table.rows[rank]
        assert row.trace == pytest.approx(STATA[rank]["trace"], abs=5e-4), rank
        assert row.trace_critical == pytest.approx(STATA[rank]["critical"], abs=5e-4), rank


def test_a_third_order_rank_has_no_further_statistic(table) -> None:
    """Rank 2 is the last row; Stata prints no trace against it."""

    assert np.isnan(table.rows[2].trace)


def test_lags_one_is_refused_rather_than_approximated(data) -> None:
    """The lag-1 convention was not pinned down, so it raises instead of guessing."""

    with pytest.raises(Exception):
        vecrank(data, lags=1)
