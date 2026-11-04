"""`res.predict()` must default to what Stata and statsmodels both return.

The defect this pins: ``StataResults.predict`` hardcoded ``kind="xb"`` while ``margins``
consulted a per-model table, so a bare ``logit_result.predict()`` returned log-odds --
neither Stata's meaning (Stata assumes ``pr`` and says "(option pr assumed; Pr(...))") nor
statsmodels' (``sm.Logit.predict`` returns probabilities).

It was silent: the values are finite and look like numbers, so feeding them to a classifier
or a Brier score produced confidently wrong answers (gap report 0.0.6 §3.5 measured a Brier
score ~15x too large on the titanic fweight fit).

These tests compare against the OTHER entry points and against closed-form transforms rather
than against the previous output, because agreeing with a buggy default proves nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.predict import DEFAULT_PREDICT, default_predict_kind


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(20261104)
    n = 400
    x = rng.normal(size=n)
    latent = 0.7 * x + rng.normal(size=n)
    return {
        "binary": pd.DataFrame({"y": (latent > 0).astype(int), "x": x}),
        "count": pd.DataFrame(
            {"y": rng.poisson(np.exp(0.5 * x + 0.3), size=n), "x": x}
        ),
        "continuous": pd.DataFrame({"y": 1.0 + 0.5 * x + rng.normal(size=n), "x": x}),
    }


def test_logit_default_is_a_probability(frames) -> None:
    """The headline case: a bare predict() must be inside [0, 1]."""

    result = ws.Logit.from_formula("y ~ x", frames["binary"]).fit()
    predicted = np.asarray(result.predict(), dtype=float)
    assert predicted.min() >= 0.0, predicted.min()
    assert predicted.max() <= 1.0, predicted.max()
    # ... and it is exactly the probability type, not something that merely happens to fit.
    assert np.allclose(predicted, result.predict(kind="pr"))


def test_probit_default_is_a_probability(frames) -> None:
    result = ws.Probit.from_formula("y ~ x", frames["binary"]).fit()
    predicted = np.asarray(result.predict(), dtype=float)
    assert 0.0 <= predicted.min() and predicted.max() <= 1.0
    assert np.allclose(predicted, result.predict(kind="pr"))


def test_poisson_default_is_the_count_mean(frames) -> None:
    """``n`` is exp(xb); it must not be the linear index."""

    result = ws.Poisson.from_formula("y ~ x", frames["count"]).fit()
    predicted = np.asarray(result.predict(), dtype=float)
    assert np.allclose(predicted, result.predict(kind="n"))
    # exp(xb) of the SAME model, computed independently.
    assert np.allclose(predicted, np.exp(result.predict(kind="xb")))


def test_ols_default_stays_xb(frames) -> None:
    result = ws.OLS.from_formula("y ~ x", frames["continuous"]).fit()
    assert np.allclose(result.predict(), result.predict(kind="xb"))


def test_explicit_xb_still_returns_log_odds(frames) -> None:
    """Backwards compatibility: callers that want the linear index still get it."""

    result = ws.Logit.from_formula("y ~ x", frames["binary"]).fit()
    xb = np.asarray(result.predict(kind="xb"), dtype=float)
    # Log-odds are unbounded, so this fixture must contain values outside [0, 1].
    assert np.any(xb < 0.0) or np.any(xb > 1.0), xb


@pytest.mark.parametrize("model_name", sorted(DEFAULT_PREDICT))
def test_every_table_entry_names_a_supported_type(model_name: str) -> None:
    """A default that points at an unsupported type would raise at call time."""

    from westata.postestimation.predict import supported_types

    assert default_predict_kind(model_name) in supported_types(model_name)


def test_reghdfe_is_absent_from_the_default_table() -> None:
    """``reghdfe`` cannot predict, so listing a default for it would be a lie.

    With absorbed fixed effects the stored design matrix keeps an ``Intercept`` column while
    the parameter vector does not (the FE absorb it): on ``y ~ x | g`` the design has 2
    columns and ``params`` has 1, so ``X @ params`` does not dimension-check. An earlier
    revision of this change added a ``reghdfe`` registration and a matching table entry; the
    suite's own "no predict types" contract caught it, and the entries were removed.
    """

    from westata.postestimation.predict import supported_types

    assert "reghdfe" not in DEFAULT_PREDICT
    with pytest.raises(Exception):
        supported_types("reghdfe")


def test_the_two_entry_points_agree() -> None:
    """``StataResults.predict`` and the free function must resolve the same default.

    They disagreed for several releases; the shared table is what prevents a repeat.
    """

    from westata.postestimation.predict import predict as free_predict

    frame = pd.DataFrame(
        {"y": (np.random.default_rng(7).normal(size=200) > 0).astype(int), "x": np.linspace(-2, 2, 200)}
    )
    result = ws.Logit.from_formula("y ~ x", frame).fit()
    design = np.asarray(result.design_matrix[0])
    assert np.allclose(result.predict(), free_predict(result, design))
