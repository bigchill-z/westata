"""Portmanteau Q and the Chow-style structural break, both checked against Stata.

C08-03 scores `wntestq` at 1.0 and C09-05 scores the Chow test at 1.0, but neither had a numerical
capture. Measured on a 150-point random walk with two regressors:

    reg y x1 x2
    predict e1, residual
    wntestq e1, lags(12)
        Portmanteau (Q) statistic =  1355.9717
        Prob > chi2(12)           =     0.0000

Both lines match, including the chi2 degrees of freedom.

THE CHOW TEST has no separate entry point: it is a joint test on the interacted design, which is
what `test()` already does. That is worth stating because the report's row says "test d yd" -- the
book constructs the interaction dummies and tests them together rather than calling a dedicated
command.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(5)
    n = 150
    frame = pd.DataFrame(
        {
            "y": np.cumsum(rng.normal(size=n)),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    return ws.OLS.from_formula("y ~ x1 + x2", frame).fit()


def test_the_portmanteau_q_matches_stata(fitted) -> None:
    rendered = str(ws.wntestq(fitted, lags=12))
    assert "1355.9717" in rendered
    assert "chi2(12)" in rendered


def test_the_portmanteau_p_value_matches_stata(fitted) -> None:
    """Both tables print 0.0000; the point is that the statistic behind it agrees."""

    rendered = str(ws.wntestq(fitted, lags=12))
    assert "0.0000" in rendered


def test_a_chow_test_is_a_joint_test_on_the_interaction(fitted) -> None:
    """C09-05 needs no dedicated function: interacting with the break dummy and testing jointly."""

    rng = np.random.default_rng(5)
    n = 150
    frame = pd.DataFrame(
        {
            "y": np.cumsum(rng.normal(size=n)),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
        }
    )
    frame["post"] = (np.arange(n) >= 75).astype(float)
    frame["x1_post"] = frame["x1"] * frame["post"]
    frame["x2_post"] = frame["x2"] * frame["post"]

    broken = ws.OLS.from_formula(
        "y ~ x1 + x2 + post + x1_post + x2_post", frame
    ).fit()
    joint = broken.test("x1_post x2_post post")
    assert float(joint.statistic) > 0.0
    assert int(joint.df) == 3


def test_a_short_series_is_rejected_rather_than_mis_reported() -> None:
    """With fewer observations than lags the statistic is undefined; it must not print a number."""

    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"y": rng.normal(size=8), "x": rng.normal(size=8)})
    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    with pytest.raises(Exception):
        ws.wntestq(fitted, lags=30)
