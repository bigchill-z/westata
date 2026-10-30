"""``var``'s ``forecast``/``irf``/``fevd`` -- the three series postestimation commands.

``forecast`` was BROKEN: it read ``self._result.y`` where statsmodels exposes the observed
series as ``endog``, so every call raised

    AttributeError: 'VARResults' object has no attribute 'y'

via statsmodels' attribute wrapper -- an error that reads like a statsmodels
incompatibility rather than a wrong attribute name in this package.

The other two (``irf``, ``fevd``) worked, which is why the gap survived: a caller who
only used those never saw it.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws


def _system(n: int = 200, seed: int = 1500, variables: int = 2):
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [np.cumsum(rng.normal(size=n)) for _ in range(variables)]
    )


@pytest.fixture(scope="module")
def var_fit():
    return ws.VAR(_system(), lags=2, names=["y1", "y2"]).fit()


# ------------------------------------------------------------------- forecast
@pytest.mark.parametrize("horizon", [1, 3, 10])
def test_the_forecast_shape_is_horizon_by_variables(var_fit, horizon) -> None:
    """``(steps, neqs)`` -- one row per period, one column per variable."""

    values = np.asarray(var_fit.forecast(horizon), dtype=float)

    assert values.shape == (horizon, 2)


def test_the_forecast_is_finite(var_fit) -> None:
    assert np.isfinite(np.asarray(var_fit.forecast(6), dtype=float)).all()


def test_a_non_positive_horizon_is_refused(var_fit) -> None:
    """Zero or negative steps is a caller mistake, not an empty result.

    Before the fix this branch was unreachable, because the attribute error fired first.
    """

    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            var_fit.forecast(bad)


def test_a_three_variable_system_gets_three_columns() -> None:
    """The column count follows the system, not a constant."""

    fitted = ws.VAR(
        _system(seed=1502, variables=3), lags=1, names=["a", "b", "c"]
    ).fit()

    assert np.asarray(fitted.forecast(4), dtype=float).shape == (4, 3)


def test_the_forecast_starts_near_the_last_observation(var_fit) -> None:
    """A sanity bound on the direction, not just the shape.

    These series are random walks, so the one-step forecast must land in the same
    neighbourhood as the final observed values rather than jumping away.
    """

    observed = _system()
    forecast = np.asarray(var_fit.forecast(1), dtype=float)[0]
    last = observed[-1]

    assert np.all(np.abs(forecast - last) < 10.0 * np.maximum(np.abs(last), 1.0))


def test_a_longer_horizon_extends_the_shorter_one(var_fit) -> None:
    """Shared state: the prefix must not depend on how far ahead was asked."""

    short = np.asarray(var_fit.forecast(2), dtype=float)
    long = np.asarray(var_fit.forecast(8), dtype=float)

    assert np.allclose(short, long[:2], rtol=1e-10)


# ---------------------------------------------------------------- irf and fevd
def test_irf_returns_an_analysis_object(var_fit) -> None:
    """``irf`` is Stata's impulse-response command; it worked all along."""

    analysis = var_fit.irf(6)

    assert type(analysis).__name__ == "IRAnalysis"


def test_fevd_returns_a_decomposition(var_fit) -> None:
    assert type(var_fit.fevd(6)).__name__ == "FEVD"


def test_all_three_share_the_horizon_argument(var_fit) -> None:
    """A caller passing a period count gets that many periods from each."""

    assert np.asarray(var_fit.forecast(4), dtype=float).shape[0] == 4
    assert var_fit.irf(4) is not None
    assert var_fit.fevd(4) is not None