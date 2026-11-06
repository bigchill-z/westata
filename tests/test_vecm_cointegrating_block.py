"""vec's cointegrating-equation block: the chi2 was fabricated, now it is computed.

The row used to print

    _ce1                  1   1.000    0.0000

for EVERY model: the "chi2" was e(beta_icnt), which is the identification FLAG (always 1), and
the p-value was hard-coded to 0.0.

Against a live ``vec dinf unem, lags(2) rank(1)``:

    Cointegrating equations

    Equation           Parms    chi2     P>chi2
    -------------------------------------------
    _ce1                  1   8.358789   0.0038
    -------------------------------------------

The statistic is the Wald test that beta's free coefficients are zero (the normalised column
drops out -- its variance is 0):

    beta = [1, .11191198], stderr_beta = [0, .0382334]

and the standard errors need the same sqrt(T/(T-4)) scale the rest of the table uses, otherwise
the statistic comes out 8.5678 against Stata's 8.358789. With the scale: 8.358791.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

STATA_BLOCK = [
    "Cointegrating equations",
    "",
    "Equation           Parms    chi2     P>chi2",
    "-------------------------------------------",
    "_ce1                  1   8.358789   0.0038",
]


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)[["dinf", "unem"]].dropna().reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VECM(
        frame[["dinf", "unem"]].to_numpy(float), coint_rank=1, lags=2, names=["dinf", "unem"]
    ).fit()


def test_the_block_matches_stata(fitted) -> None:
    lines = str(fitted.summary()).splitlines()
    for expected in STATA_BLOCK:
        assert expected in lines, repr(expected)


def test_the_chi2_is_computed_not_a_flag(fitted) -> None:
    """It used to read 1.000 for every model, because it printed e(beta_icnt)."""

    assert float(fitted.e["beta_icnt"]) == 1.0
    rendered = str(fitted.summary())
    assert "8.358789" in rendered
    assert "1.000    0.0000" not in rendered


def test_the_pvalue_is_derived_from_the_statistic(fitted) -> None:
    """It used to be hard-coded to 0.0."""

    assert "0.0038" in str(fitted.summary())


def test_the_wald_scale_is_the_same_one_the_table_uses(fitted) -> None:
    """Without the sqrt(T/(T-4)) factor the statistic would be 8.5678, not 8.3588."""

    beta = np.asarray(fitted.raw.beta, dtype=float).reshape(-1)
    errors = np.asarray(fitted.raw.stderr_beta, dtype=float).reshape(-1)
    observations = float(fitted.e["N"])
    unscaled = float(np.sum((beta[1:] / errors[1:]) ** 2))
    assert unscaled == pytest.approx(8.5678, abs=1e-3)
    scaled = unscaled * (observations - 4.0) / observations
    assert scaled == pytest.approx(8.3588, abs=1e-3)
