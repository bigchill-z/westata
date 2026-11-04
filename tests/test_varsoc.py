"""Pin ``varsoc`` against Stata's lag-order selection table.

Reference from a live ``varsoc dinf unem, maxlag(4)`` on ``macro_swatson.dta``
(N = 162 after the longest lag)::

     Lag |    LL      LR      df    p     FPE       AIC      HQIC      SBIC
       0 |  881.581                     6.6e-08   -10.859  -10.8435  -10.8209
       1 |   1141.4  519.63    4  0.000 2.8e-09  -14.0172  -13.9708  -13.9029
       2 |  1213.86  144.93*   4  0.000 1.2e-09* -14.8625* -14.7851* -14.6719*
       3 |  1217.32  6.9224    4  0.140 1.2e-09  -14.8558  -14.7475   -14.589
       4 |  1219.45  4.2537    4  0.373 1.2e-09  -14.8327  -14.6934  -14.4896

The detail everything hinges on: all candidate lag orders are scored on the SAME estimation
sample, the ``n - maxlag`` observations left by the longest lag. Scoring each order on
``n - p`` observations instead gives -10.8815 / -14.8104 for p = 0 / 2 against Stata's
-10.859 / -14.8625 -- the trend looks right and every value is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.postestimation.varsoc import varsoc

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_LL = [881.581, 1141.4, 1213.86, 1217.32, 1219.45]
STATA_LR = [None, 519.63, 144.93, 6.9224, 4.2537]
STATA_AIC = [-10.859, -14.0172, -14.8625, -14.8558, -14.8327]
STATA_HQIC = [-10.8435, -13.9708, -14.7851, -14.7475, -14.6934]
STATA_SBIC = [-10.8209, -13.9029, -14.6719, -14.589, -14.4896]
STATA_FPE = [6.6e-08, 2.8e-09, 1.2e-09, 1.2e-09, 1.2e-09]


@pytest.fixture(scope="module")
def selection():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return varsoc(values, maxlag=4)


def test_sample_size_matches_stata(selection) -> None:
    """162 = 166 - 4, shared by every lag order."""

    assert selection.nobs == 162
    assert selection.equations == 2


def test_log_likelihoods_match_stata(selection) -> None:
    for row, expected in zip(selection.rows, STATA_LL):
        assert row.ll == pytest.approx(expected, rel=5e-5), f"lag {row.lag}"


def test_likelihood_ratio_statistics_match_stata(selection) -> None:
    for row, expected in zip(selection.rows, STATA_LR):
        if expected is None:
            assert row.lr == 0.0
            continue
        assert row.lr == pytest.approx(expected, rel=5e-5), f"lag {row.lag}"
        assert row.lr_df == 4


def test_every_information_criterion_matches_stata(selection) -> None:
    for row, aic, hqic, sbic in zip(selection.rows, STATA_AIC, STATA_HQIC, STATA_SBIC):
        assert row.aic == pytest.approx(aic, abs=5e-5), f"aic lag {row.lag}"
        assert row.hqic == pytest.approx(hqic, abs=5e-5), f"hqic lag {row.lag}"
        assert row.sbic == pytest.approx(sbic, abs=5e-5), f"sbic lag {row.lag}"


def test_fpe_matches_stata(selection) -> None:
    for row, expected in zip(selection.rows, STATA_FPE):
        assert row.fpe == pytest.approx(expected, rel=0.06), f"lag {row.lag}"


def test_optimal_lag_matches_stata(selection) -> None:
    assert selection.optimal == 2


def test_scoring_each_order_on_its_own_sample_would_be_wrong() -> None:
    """Guard the shared-sample rule: the per-order sample gives different numbers.

    Recomputing lag 2 on its own 164 observations yields -14.8104 against Stata's -14.8625.
    If a future change reintroduces per-order samples this test still passes, but the value
    assertions above will fail -- which is the point of keeping both.
    """

    frame = pd.read_stata(DATA, convert_categoricals=False)
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    shared = varsoc(values, maxlag=4)
    assert shared.nobs == len(values) - 4
    # A different maxlag rescales the WHOLE table, which is itself the shared-sample effect.
    smaller = varsoc(values, maxlag=2)
    assert smaller.rows[2].aic != pytest.approx(shared.rows[2].aic, abs=1e-12)
