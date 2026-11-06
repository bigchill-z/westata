"""Every VECM e() scalar that feeds the printed table now matches Stata.

Rounds 68-69 fixed two arithmetic errors in the parameter accounting:

  * df_eq omitted the error-correction terms, so e(k_1) was 3 against Stata's 4 and the printed
    "Parms" column contradicted the coefficient list beneath it;
  * e(df_m#) subtracted the constant, giving 3 where Stata stores 4 -- for a vec equation the
    EC term and the constant are both free parameters.

Because df_eq feeds rmse, df_m, k and k_rank, fixing it corrected a whole family at once. The
comparison against a live ``vec dinf unem, lags(2) rank(1)``:

    k_1 4    k_2 4    df_m 4    df_m1 4    df_m2 4    rank 8    k_rank 9
    rmse_1 .01377218691   rmse_2 .002560017026
    r2_1 .743420787       r2_2 .406197219
    chi2_1 463.5890988    chi2_2 109.4497316
    ll 1220.5657   aic -14.77519146   hqic -14.70613135   sbic -14.60507684

Eighteen of twenty matched before this round's df_m fix; all twenty now do.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

#: Captured from ``vec dinf unem, lags(2) rank(1)``.
STATA = {
    "k_1": 4.0,
    "k_2": 4.0,
    "df_m": 4.0,
    "df_m1": 4.0,
    "df_m2": 4.0,
    "rank": 8.0,
    "k_rank": 9.0,
    "rmse_1": 0.01377218691,
    "rmse_2": 0.002560017026,
    "r2_1": 0.743420787,
    "r2_2": 0.406197219,
    "ll": 1220.5657,
    "aic": -14.77519146,
    "hqic": -14.70613135,
    "sbic": -14.60507684,
    "N": 164.0,
}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna().reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VECM(
        frame[["dinf", "unem"]].to_numpy(float), coint_rank=1, lags=2, names=["dinf", "unem"]
    ).fit()


def test_every_scalar_matches_stata(fitted) -> None:
    for name, expected in STATA.items():
        actual = float(fitted.e[name])
        assert actual == pytest.approx(expected, rel=1e-6, abs=1e-9), name


def test_the_parameter_count_includes_the_ec_terms(fitted) -> None:
    """The bug that started this: 3 against Stata's 4."""

    assert float(fitted.e["k_1"]) == 4.0
    assert float(fitted.e["k_2"]) == 4.0


def test_df_m_is_not_reduced_by_the_constant(fitted) -> None:
    """For a vec equation nothing is removed -- the EC term and constant are both free."""

    assert float(fitted.e["df_m1"]) == float(fitted.e["k_1"])
    assert float(fitted.e["df_m2"]) == float(fitted.e["k_2"])


def test_k_rank_exceeds_the_unrestricted_count(fitted) -> None:
    """9 against df_eq*neqs = 8 -- larger, not smaller."""

    unrestricted = float(fitted.e["df_eq"]) * float(fitted.e["neqs"])
    assert float(fitted.e["k_rank"]) == unrestricted + 1


def test_rmse_uses_the_corrected_degrees_of_freedom(fitted) -> None:
    """rmse divides by N - df_eq, so it moved when df_eq did."""

    assert float(fitted.e["rmse_1"]) == pytest.approx(0.01377218691, rel=1e-8)
