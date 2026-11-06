"""The per-equation summary block at rank 1 AND rank 2.

Round 76 fixed the covariance scale; this round found three more literals in the same block that
were correct only because df_eq happens to be 4 at rank 1:

  * residual_dof = N - 4, twice (the printed RMSE and the e() payload). At rank 2 df_eq is 6, so
    RMSE came out .934955 against Stata's .939355;
  * chi2's p-value used chi2.sf(chi2, 3) -- the wrong distribution entirely. At rank 2 it gave
    0.0483 where Stata prints 0.2461.

Two formatters were also wrong in opposite directions:

  * RMSE switches format by MAGNITUDE in Stata: .0137721869 -> ".013772" (six decimals) but
    1.370735 -> "1.37074" (six significant figures);
  * chi2 is SEVEN SIGNIFICANT figures throughout: 463.5890975 -> "463.5891" and
    7.8916758 -> "7.891676".

Both blocks now match:

    rank 1: D_dinf  4  .013772  0.7434  463.5891  0.0000
            D_unem  4   .00256  0.4062  109.4497  0.0000

    rank 2: D_a     6  .939355  0.0359  7.891676  0.2461
            D_b     6  1.37074  0.0429  9.497462  0.1475
            D_c     6  1.01344  0.0190  4.113776  0.6613
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.estimators.tseries import _vec_chi2, _vec_rmse


@pytest.fixture(scope="module")
def rank_one():
    try:
        frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\macro_swatson.dta")[
            ["dinf", "unem"]
        ].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VECM(
        frame[["dinf", "unem"]].to_numpy(float), coint_rank=1, lags=2, names=["dinf", "unem"]
    ).fit()


@pytest.fixture(scope="module")
def rank_two():
    rng = np.random.default_rng(11)
    n = 220
    common = np.cumsum(rng.normal(size=n))
    frame = pd.DataFrame(
        {
            "a": common + np.cumsum(rng.normal(scale=0.2, size=n)),
            "b": 1.5 * common + np.cumsum(rng.normal(scale=0.2, size=n)),
            "c": np.cumsum(rng.normal(size=n)),
        }
    )
    return ws.VECM(frame.to_numpy(float), coint_rank=2, lags=2, names=["a", "b", "c"]).fit()


def test_rank_one_rows_match(rank_one) -> None:
    lines = str(rank_one.summary()).splitlines()
    assert "D_dinf                4     .013772   0.7434   463.5891   0.0000" in lines
    assert "D_unem                4      .00256   0.4062   109.4497   0.0000" in lines


def test_rank_two_rows_match(rank_two) -> None:
    lines = str(rank_two.summary()).splitlines()
    assert "D_a                   6     .939355   0.0359   7.891676   0.2461" in lines
    assert "D_b                   6     1.37074   0.0429   9.497462   0.1475" in lines
    assert "D_c                   6     1.01344   0.0190   4.113776   0.6613" in lines


def test_rank_two_rmse_uses_df_eq(rank_two) -> None:
    """N - 4 gave .934955; N - df_eq gives .939355."""

    assert float(rank_two.e["rmse_1"]) == pytest.approx(0.9393551564, rel=1e-9)


def test_rank_two_chi2_matches(rank_two) -> None:
    assert float(rank_two.e["chi2_1"]) == pytest.approx(7.8916758454, rel=1e-9)


def test_the_p_value_uses_df_eq(rank_two) -> None:
    """It used chi2.sf(chi2, 3) and printed 0.0483 against Stata's 0.2461."""

    rendered = str(rank_two.summary())
    assert "0.2461" in rendered
    assert "0.0483" not in rendered


def test_rmse_switches_format_by_magnitude() -> None:
    assert _vec_rmse(0.0137721869) == ".013772"
    assert _vec_rmse(1.370735) == "1.37074"
    assert _vec_rmse(0.0025600170) == ".00256"


def test_chi2_is_seven_significant_figures() -> None:
    assert _vec_chi2(463.5890975) == "463.5891"
    assert _vec_chi2(7.891675845) == "7.891676"
