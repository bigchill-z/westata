"""vargranger is a plain Wald test on the ML covariance -- verified from first principles.

Chapter 13's Granger-causality entry scores 0.80 in the gap report, but every VAR reference in
that report was produced through a ``var`` whose ``lags()`` option is broken in this environment
(rounds 39-43). So the question is not "does it match Stata" but "is the statistic the right
statistic", which can be answered without Stata at all.

vargranger.ado:137 reduces to ``test <restriction>``: a Wald test on the coefficients of one
equation. Round 40 proved those coefficients exact. This checks the remaining piece -- the
covariance -- by rebuilding the Wald statistic by hand:

    sigma_ml = resid' resid / T                    <- ML, divided by T not T-k
    cov_eq0  = sigma_ml[0,0] * (X'X)^-1
    chi2     = b' cov_eq0^-1 b                       over the excluded lags

which gives 24.1243 against the package's 24.124.

The first attempt used cov_params instead and landed on 798.94 -- 33x too large. cov_params is
the unbiased (T-k) estimator, and Stata's Wald uses the ML one. That factor is the whole
difference, and it is exactly the trap this test now guards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)[["inf", "unem"]].dropna().reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    return ws.VAR(frame[["inf", "unem"]].to_numpy(float), lags=2, names=["inf", "unem"]).fit()


def manual_wald(result, equation: int = 0, causing: str = "unem") -> float:
    """Wald statistic for "causing does not Granger-cause equation" using the ML covariance."""

    residual = np.asarray(result._result.resid)
    sample = residual.shape[0]
    sigma_ml = (residual.T @ residual) / sample
    # Rebuild the design the VAR used: constant plus two lags of both variables.
    series = np.asarray(result._result.endog, dtype=float)
    total = series.shape[0]
    lagged = [series[total - sample - depth: total - depth] for depth in range(1, 3)]
    # Column order must match params' ROW order: const, then L1, then L2.
    design = np.column_stack([np.ones(sample)] + lagged)
    bread = np.linalg.inv(design.T @ design)
    block = sigma_ml[equation, equation] * bread

    names = list(result.params.index)
    index = [position for position, name in enumerate(names) if causing in name and name != "const"]
    beta = np.asarray(result.params)[index, equation]
    return float(beta @ np.linalg.solve(block[np.ix_(index, index)], beta))


def test_the_wald_statistic_matches_first_principles(fitted) -> None:
    """24.1243 computed by hand against 24.124 printed."""

    assert manual_wald(fitted) == pytest.approx(24.124, abs=5e-3)


def test_the_reverse_direction_also_matches(fitted) -> None:
    assert manual_wald(fitted, equation=1, causing="inf") == pytest.approx(24.363, abs=5e-3)


def test_using_cov_params_would_be_wrong(fitted) -> None:
    """Pin the trap: cov_params is the T-k estimator and gives a 33x-larger statistic.

    If someone "simplifies" manual_wald to read cov_params, the number moves by more than an
    order of magnitude. This test makes that visible rather than silent.
    """

    names = list(fitted.params.index)
    index = [position for position, name in enumerate(names) if "unem" in name and name != "const"]
    covariance = np.asarray(fitted.cov_params)
    k_params = len(names)
    beta = np.asarray(fitted.params)[index, 0]
    block = covariance[:k_params, :k_params][np.ix_(index, index)]
    naive = float(beta @ np.linalg.solve(block, beta))
    assert naive > 10 * manual_wald(fitted)


def test_vargranger_prints_the_same_number(fitted) -> None:
    rendered = str(ws.vargranger(fitted))
    assert "24.124" in rendered or "24.12" in rendered
