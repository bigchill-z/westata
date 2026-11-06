"""``vecrank, lags(2)`` now reproduces Stata, and the reason round 14 failed is one line.

The bug: LAGGED DIFFERENCES MUST ENTER BOTH RESIDUALISATIONS. Putting DY_{t-1} into the model
for DY_t alone -- the natural reading of "short-run dynamics" -- leaves it in Y_{t-1}'s equation
and inflates every statistic: 51.4854 / 7.5897 against Stata's 183.0918 / 7.7811.

Reference from live ``vecrank dinf unem, lags(2) max`` on macro_swatson::

    rank  Params           LL  Eigenvalue      Trace      5%
       0       6    1132.9103               183.0918   15.41
       1       9    1220.5657     0.65664     7.7811    3.76
       2      10    1224.4562     0.04634

Round 14's mistake was methodological, and this file pins the outcome so it cannot recur: write
the general version only after making the KNOWN-GOOD ``lags(1)`` path pass as a running test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")


def test_lags_one_is_still_exact(frame) -> None:
    """The running test that round 14 should have kept: this must never regress."""

    values = frame[["dinf", "unem"]].dropna().to_numpy()
    rendered = str(ws.vecrank(values, lags=1))
    assert "172.5034" in rendered
    assert "1.2744" in rendered
    assert "0.64575" in rendered
    assert "0.00769" in rendered


def test_lags_two_matches_stata(frame) -> None:
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    rendered = str(ws.vecrank(values, lags=2))
    assert "183.0918" in rendered
    assert "7.7811" in rendered
    assert "0.65664" in rendered
    assert "0.04634" in rendered
    assert "1132.9103" in rendered
    assert "1224.4562" in rendered


def test_the_lagged_difference_is_in_both_residualisations(frame) -> None:
    """Guard the specific error: dropping it from R1's model inflates the trace to 51.4854."""

    from westata.postestimation.vecrank import _johansen_general

    values = frame[["dinf", "unem"]].dropna().to_numpy()
    eigenvalues, trace = _johansen_general(values, 2)
    assert float(trace[0]) == pytest.approx(183.0918, abs=1e-4)
    assert abs(float(trace[0]) - 51.4854) > 100.0
    assert float(eigenvalues[0]) == pytest.approx(0.65664, abs=1e-5)
    assert np.all(eigenvalues <= 1.0)
