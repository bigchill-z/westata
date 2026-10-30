"""``arima`` results need ``forecast``, not the regression ``predict``.

Calling ``result.predict(n=5)`` on an ``arima`` fit failed with

    ConsistencyError: predict: the design matrix was not retained on this result

which describes the WRONG problem: a time-series model has no design matrix to retain.
``predict, xb`` is the regression interface (fitted values from covariates), and Stata's
postestimation command after ``arima`` is ``forecast`` -- future values from the fitted
state, parameterised by a horizon rather than by regressors.

The result now carries ``forecast(steps)``. ``predict`` is deliberately STILL not
available: offering it would require inventing a design matrix, and its presence would
suggest a capability the estimator does not have.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws


def _series(n: int = 200, seed: int = 1400):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n))


@pytest.fixture(scope="module")
def arima_fit():
    return ws.ARIMA(_series(), order=(1, 1, 1)).fit()


def test_forecast_exists(arima_fit) -> None:
    assert hasattr(arima_fit, "forecast")


@pytest.mark.parametrize("horizon", [1, 5, 12])
def test_the_length_is_the_horizon(arima_fit, horizon) -> None:
    """``forecast(h)`` returns exactly ``h`` values."""

    values = arima_fit.forecast(horizon)

    assert np.asarray(values).shape == (horizon,)


def test_the_values_are_finite(arima_fit) -> None:
    assert np.isfinite(np.asarray(arima_fit.forecast(8), dtype=float)).all()


def test_a_longer_horizon_extends_the_shorter_one(arima_fit) -> None:
    """The first ``h`` values must not depend on how far ahead the caller asked.

    Computing each horizon independently from scratch could drift; sharing the state
    means the prefixes agree exactly.
    """

    short = np.asarray(arima_fit.forecast(3), dtype=float)
    long = np.asarray(arima_fit.forecast(10), dtype=float)

    assert np.allclose(short, long[:3], rtol=1e-10)


def test_a_non_positive_horizon_is_refused(arima_fit) -> None:
    """Zero or negative steps is a caller mistake, not an empty result."""

    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            arima_fit.forecast(bad)


def test_predict_is_not_offered(arima_fit) -> None:
    """The regression interface would need a design matrix that does not exist.

    Its error message ("the design matrix was not retained") described the wrong
    problem, so the honest surface is ``forecast`` alone.
    """

    with pytest.raises(Exception):
        arima_fit.predict(n=3)


def test_the_forecast_continues_from_the_observed_level(arima_fit) -> None:
    """A sanity bound on the direction, not just the shape.

    The series is a random walk with positive drift over 200 steps, so the forecast
    should stay in the same neighbourhood as the last observed value rather than
    jumping by orders of magnitude.
    """

    last = float(np.asarray(_series(), dtype=float)[-1])
    forecast = float(np.asarray(arima_fit.forecast(1), dtype=float)[0])

    assert abs(forecast - last) < max(abs(last), 1.0)


def test_forecast_works_for_a_seasonal_model() -> None:
    """Nothing about the binding is specific to a non-seasonal fit."""

    fitted = ws.ARIMA(
        _series(seed=1402), order=(1, 0, 0), seasonal_order=(1, 0, 0, 4)
    ).fit()

    values = np.asarray(fitted.forecast(8), dtype=float)

    assert values.shape == (8,)
    assert np.isfinite(values).all()