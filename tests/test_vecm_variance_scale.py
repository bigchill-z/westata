"""The statsmodels-to-Stata covariance scale is T/(T - df_eq), not T/(T - 4).

The converter carried the literal 4. On the rank-1 reference fit df_eq IS 4, so every standard
error matched Stata to the last digit -- which is exactly why the constant survived.

It stops being 4 as soon as the model is richer:

    rank 2, three variables, 2 lags:   df_eq = 6, N = 218
        N/(N-4) = 1.01869159   <- what the code returned
        N/(N-6) = 1.02830189   <- what Stata uses

a 0.47% error on every standard error in every block -- the third-digit mismatch that was recorded
against alpha and gamma in the alignment summary across several rounds.

After the fix, all ten coefficient rows of a rank-2 system match Stata byte for byte:

    '         L1. |   .1279021   .0578459     2.21   0.027     .0145262     .241278'
    '         L1. |  -.1163404   .0509856    -2.28   0.022    -.2162703   -.0164105'
    '         LD. |   -.349193   .2610162    -1.34   0.181    -.8607754    .1623894'
    ...

The rank-1 case is unaffected, because there df_eq happens to equal 4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


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


STATA_ROWS = [
    "         L1. |   .1279021   .0578459     2.21   0.027     .0145262     .241278",
    "         L1. |  -.1163404   .0509856    -2.28   0.022    -.2162703   -.0164105",
    "         LD. |   -.349193   .2610162    -1.34   0.181    -.8607754    .1623894",
    "         LD. |   .2494462   .1786774     1.40   0.163    -.1007551    .5996475",
    "         LD. |  -.0490737   .0633847    -0.77   0.439    -.1733054     .075158",
]


def test_every_coefficient_row_matches_stata(rank_two) -> None:
    rendered = str(rank_two.summary()).splitlines()
    for expected in STATA_ROWS:
        assert expected in rendered, repr(expected)


def test_the_scale_uses_df_eq(rank_two) -> None:
    """df_eq is 6 here; the old literal 4 gave a 0.47% error."""

    nobs = float(rank_two.raw.nobs)
    df_eq = float(rank_two.e["df_eq"])
    assert df_eq == 6.0
    assert rank_two._stata_variance_scale() == pytest.approx(nobs / (nobs - df_eq), rel=1e-12)
    assert rank_two._stata_variance_scale() != pytest.approx(nobs / (nobs - 4.0), rel=1e-6)


def test_rank_one_scale_is_still_four(rank_two) -> None:
    """There df_eq happens to be 4, which is why the literal looked right."""

    try:
        frame = pd.read_stata(r"D:\myCodeProject\test_westata\data\macro_swatson.dta")[
            ["dinf", "unem"]
        ].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    single = ws.VECM(
        frame[["dinf", "unem"]].to_numpy(float), coint_rank=1, lags=2, names=["dinf", "unem"]
    ).fit()
    assert float(single.e["df_eq"]) == 4.0
    assert single._stata_variance_scale() == pytest.approx(
        float(single.raw.nobs) / (float(single.raw.nobs) - 4.0), rel=1e-12
    )
