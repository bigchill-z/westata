"""Pin ``mlogit`` to Stata: coefficients, log-likelihood and the base-outcome convention.

The fixture is a three-level multinomial logit generated from a known DGP on lin_1992.dta, so
the reference is reproducible without shipping an extra data file:

    rng = np.random.default_rng(20240101)
    X standardised from land and mci
    beta = [[0, 0], [0.7, -0.4], [-0.3, 0.5]]   (level 1 is the base)
    choice from the MNL probabilities

Reference from a live ``mlogit y land mci, base(1)``::

    e(ll) = -491.487956370835

    e(b)  = 1:land 1:mci 1:_cons | 2:land 2:mci 2:_cons | 3:land 3:mci 3:_cons
                  0     0      0  | .00001555 -1.3655805 1.1794885 |
              -.00001018 .52624552 -.31499565

Two conventions matter for an exact match and both are pinned here:

1. Stata stores ``levels`` blocks of coefficients, with the BASE outcome's block left as ZEROS
   and still printed. statsmodels estimates only ``levels - 1`` blocks, so the base row has to
   be inserted rather than dropped -- otherwise every later coefficient shifts position.
2. The predictor order is the model's own, so ``e(b)`` is (level, predictor) row-major. Passing
   the design as [land, mci, _cons] reproduces Stata's order directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
STATA_LL = -491.487956370835
STATA_PARAMS = [
    0.0, 0.0, 0.0,
    0.00001555, -1.3655805, 1.1794885,
    -0.00001018, 0.52624552, -0.31499565,
]


@pytest.fixture(scope="module")
def sample():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    frame = frame[["sownarea", "land", "mci"]].dropna().reset_index(drop=True)
    count = len(frame)
    rng = np.random.default_rng(20240101)
    covariates = frame[["land", "mci"]].to_numpy(float)
    standardised = (covariates - covariates.mean(axis=0)) / covariates.std(axis=0)
    beta = np.array([[0.0, 0.0], [0.7, -0.4], [-0.3, 0.5]])
    utility = standardised @ beta.T
    utility -= utility.max(axis=1, keepdims=True)
    probability = np.exp(utility)
    probability /= probability.sum(axis=1, keepdims=True)
    draw = rng.random(count)[:, None]
    choice = (draw > np.cumsum(probability, axis=1)).sum(axis=1) + 1
    outcome = np.clip(choice, 1, 3).astype(float)
    # Column order matches Stata's predictor order.
    design = np.column_stack(
        [frame["land"].to_numpy(float), frame["mci"].to_numpy(float), np.ones(count)]
    )
    return outcome, design


@pytest.fixture(scope="module")
def fitted(sample):
    outcome, design = sample
    return ws.MultinomialLogit(
        outcome, design, param_names=["land", "mci", "_cons"]
    ).fit()


def test_the_log_likelihood_matches_stata(fitted) -> None:
    assert float(fitted.e["ll"]) == pytest.approx(STATA_LL, abs=1e-6)


def test_every_coefficient_matches_stata(fitted) -> None:
    values = np.asarray(fitted.params, dtype=float)
    assert values.size == len(STATA_PARAMS)
    for got, expected in zip(values, STATA_PARAMS):
        assert got == pytest.approx(expected, abs=2e-6)


def test_the_base_outcome_block_is_zero_and_kept(fitted) -> None:
    """Stata prints the base row as zeros; it must not be dropped."""

    values = np.asarray(fitted.params, dtype=float)
    assert np.allclose(values[:3], 0.0)
    assert float(fitted.e["baseoutcome"]) == 1.0


def test_the_parameter_names_are_level_prefixed(fitted) -> None:
    names = list(fitted.params.index)
    assert names[0] == "1:land"
    assert names[1] == "1:mci"
    assert names[2] == "1:_cons"
    assert names[3] == "2:land"
    assert names[-1] == "3:_cons"


def test_a_binary_outcome_is_refused(sample) -> None:
    """Logit is the right model there; mlogit should say so rather than guess."""

    outcome, design = sample
    binary = np.where(outcome > 1, 1.0, 0.0)
    with pytest.raises(Exception, match="at least three levels|Logit"):
        ws.MultinomialLogit(binary, design).fit()


def test_the_covariance_has_the_same_shape_as_the_parameters(fitted) -> None:
    covariance = np.asarray(fitted.cov_params(), dtype=float)
    assert covariance.shape == (9, 9)
    # The base block is fixed at zero, so its variances are zero too.
    assert np.allclose(covariance[:3, :3], 0.0)
