"""Tests for frequency weights in the discrete estimators.

The gap this closes: `logit y x1 x2 [fweight=w]` raised `TypeError: _DiscreteBase.fit() got an unexpected
keyword argument 'weights'` while Stata fit it. It now matches Stata.

Three things are asserted, because getting only some of them produces plausible wrong output rather than an
error:

1. The coefficients and standard errors match Stata on a weighted sample.
2. `e(N)` is the SUM of the weights, not the row count. Stata reports N=1205 here; a weighted fit labelled
   with 600 rows would read as correct and be wrong.
3. Models without weight support REFUSE. `zip` does not implement weighting, and an unweighted fit under a
   weighted call would return valid-looking numbers for the wrong sample -- the failure mode this whole
   package's rejections exist to prevent.

Reference values are from a live Stata session on the sample generated below, so they are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError


def _weighted_sample(seed: int = 20261024, n: int = 600):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    latent = 0.4 + 0.7 * x1 - 0.5 * x2
    binary = rng.binomial(1, 1 / (1 + np.exp(-latent)), size=n).astype(float)
    counts = rng.poisson(np.exp(0.3 + 0.4 * x1 - 0.2 * x2)).astype(float)
    weights = rng.integers(1, 4, size=n).astype(float)
    return pd.DataFrame(
        {"yb": binary, "yc": counts, "x1": x1, "x2": x2, "w": weights}
    )


# Stata: logit yb x1 x2 [fweight=w]  -> order [x1, x2, _cons]
STATA_LOGIT_B = np.array([0.5978586236350707, -0.40194252310901374, 0.29032225186512134])
STATA_LOGIT_SE = np.array([0.06579129946363292, 0.061207630505676576, 0.06154778692636371])


def test_weighted_logit_matches_stata():
    """Coefficients and standard errors under fweight, against Stata's own numbers."""

    frame = _weighted_sample()
    result = ws.Logit.from_formula("yb ~ x1 + x2", frame, freq_weights="w").fit()

    order = [1, 2, 0]  # ours [Intercept, x1, x2] -> Stata [x1, x2, _cons]
    # 1e-7, not machine precision: the weighted fit converges to a slightly different stopping point, which
    # is the same pattern as the unweighted models (GAP_REGISTER 8.11).
    np.testing.assert_allclose(
        np.asarray(result.params, dtype=float)[order], STATA_LOGIT_B, rtol=1e-7
    )
    np.testing.assert_allclose(
        np.asarray(result.bse, dtype=float)[order], STATA_LOGIT_SE, rtol=1e-7
    )


def test_weighted_eN_is_the_sum_of_the_weights():
    """Stata reports the expanded count. A weighted fit labelled with the row count reads as correct."""

    frame = _weighted_sample()
    result = ws.Logit.from_formula("yb ~ x1 + x2", frame, freq_weights="w").fit()
    assert int(result.e["N"]) == int(frame["w"].sum())
    assert int(result.e["N"]) != len(frame)


def test_weight_equals_row_expansion():
    """fweight means each observation counts that many times, so expanding the data must agree exactly."""

    frame = _weighted_sample()
    weighted = ws.Logit.from_formula("yb ~ x1 + x2", frame, freq_weights="w").fit()

    expanded = frame.loc[frame.index.repeat(frame["w"].astype(int))].reset_index(drop=True)
    plain = ws.Logit.from_formula("yb ~ x1 + x2", expanded).fit()

    np.testing.assert_allclose(
        np.asarray(weighted.params, dtype=float),
        np.asarray(plain.params, dtype=float),
        rtol=1e-10,
    )


def test_probit_and_poisson_accept_weights():
    """The opt-in list is not logit-only; the same helper serves both, and both are exercised."""

    frame = _weighted_sample()
    for model, formula in ((ws.Probit, "yb ~ x1 + x2"), (ws.Poisson, "yc ~ x1 + x2")):
        result = model.from_formula(formula, frame, freq_weights="w").fit()
        assert int(result.e["N"]) == int(frame["w"].sum())
        assert np.all(np.isfinite(np.asarray(result.bse, dtype=float)))


def test_models_without_weight_support_refuse():
    """An unsupported model must raise, not fit unweighted.

    `zip` is the case: it has no weighting implemented, and returning an unweighted fit under a weighted call
    would report numbers for the wrong sample with nothing to reveal it.
    """

    frame = _weighted_sample()
    with pytest.raises(ConsistencyError, match="frequency weights"):
        ws.ZeroInflatedPoisson.from_formula("yc ~ x1 + x2", frame, freq_weights="w")


def test_a_weight_column_that_does_not_exist_is_an_error():
    """A typo in the weight column name must not silently mean 'no weights'."""

    frame = _weighted_sample()
    with pytest.raises(ConsistencyError, match="not a column"):
        ws.Logit.from_formula("yb ~ x1 + x2", frame, freq_weights="not_a_column")


def test_negative_weights_are_rejected():
    frame = _weighted_sample()
    frame.loc[0, "w"] = -1.0
    with pytest.raises(ConsistencyError, match="non-negative"):
        ws.Logit.from_formula("yb ~ x1 + x2", frame, freq_weights="w")
