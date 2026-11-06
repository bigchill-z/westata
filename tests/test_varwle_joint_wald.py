"""varwle is a JOINT Wald across equations at each lag -- verified from first principles.

varwle.ado:69-93 collects "[eq]Li.var" for every equation and every variable and runs ONE test
over all of them, so each row has df = neqs * neqs and accounts for cross-equation covariance.

The distinction is not cosmetic. Summing per-equation Wald statistics gives 735.17 at lag 1
against the package's 733.07 -- wrong by 0.3% and, worse, systematically wrong in a way that
looks like ordinary numerical noise.

With the joint form:
    lag 1: 733.0650   package 733.0650
    lag 2: 125.9184   package 125.9184
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


def joint_wald(fitted, lag: int) -> tuple[float, int]:
    """Joint Wald that the whole lag-``lag`` coefficient block is zero, ML covariance."""

    values = np.asarray(fitted._result.endog, dtype=float)
    residual = np.asarray(fitted._result.resid)
    sample = residual.shape[0]
    sigma_ml = (residual.T @ residual) / sample
    total = values.shape[0]
    lagged = [
        values[total - sample - depth: total - depth]
        for depth in range(1, int(fitted.lags) + 1)
    ]
    design = np.column_stack([np.ones(sample)] + lagged)
    bread = np.linalg.inv(design.T @ design)

    names = list(fitted.params.index)
    params = np.asarray(fitted.params)
    equations = params.shape[1]
    k_params = len(names)
    rows = [index for index, name in enumerate(names) if name.startswith(f"L{lag}.")]

    stacked = []
    for equation in range(equations):
        stacked.extend([equation * k_params + index for index in rows])
    stacked = np.array(stacked)

    beta = params.T.reshape(-1)[stacked]
    covariance = np.kron(sigma_ml, bread)
    variance = covariance[np.ix_(stacked, stacked)]
    return float(beta @ np.linalg.solve(variance, beta)), len(rows) * equations


def test_lag_one_matches(fitted) -> None:
    statistic, degrees = joint_wald(fitted, 1)
    assert statistic == pytest.approx(733.0650, abs=5e-3)
    assert degrees == 4


def test_lag_two_matches(fitted) -> None:
    statistic, degrees = joint_wald(fitted, 2)
    assert statistic == pytest.approx(125.9184, abs=5e-3)
    assert degrees == 4


def test_the_package_prints_those_numbers(fitted) -> None:
    rendered = str(ws.varwle(fitted))
    assert "733.0650" in rendered or "733.065" in rendered
    assert "125.9184" in rendered or "125.918" in rendered


def test_summing_per_equation_would_be_wrong(fitted) -> None:
    """Pin the trap: a sum ignores cross-equation covariance and lands ~0.3% high."""

    values = np.asarray(fitted._result.endog, dtype=float)
    residual = np.asarray(fitted._result.resid)
    sample = residual.shape[0]
    sigma_ml = (residual.T @ residual) / sample
    total = values.shape[0]
    lagged = [values[total - sample - depth: total - depth] for depth in range(1, 3)]
    design = np.column_stack([np.ones(sample)] + lagged)
    bread = np.linalg.inv(design.T @ design)

    names = list(fitted.params.index)
    params = np.asarray(fitted.params)
    rows = [index for index, name in enumerate(names) if name.startswith("L1.")]
    total_sum = 0.0
    for equation in range(params.shape[1]):
        beta = params[rows, equation]
        block = sigma_ml[equation, equation] * bread
        total_sum += float(beta @ np.linalg.solve(block[np.ix_(rows, rows)], beta))
    assert total_sum > 734.0 and total_sum != pytest.approx(733.0650, abs=1e-3)
