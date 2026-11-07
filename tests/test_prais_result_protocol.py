"""`PraisResult` now implements the statsmodels result protocol.

v3 report section 4, item (4): result types are inconsistent across the package. The concrete
symptom here was that `PraisResult` carried its confidence bounds PRIVATELY (``_conf_int``) and
exposed no public method at all, so

    result.conf_int()

raised ``AttributeError: 'PraisResult' object has no attribute 'conf_int'`` while every estimator
that returns a `StataResults` answered it happily. Porting statsmodels code across the two surfaces
therefore failed on this one class.

THREE METHODS ADDED:

  * ``conf_int(alpha=0.05)`` returns a DataFrame with ``lower``/``upper`` columns -- the same names
    every other result class here uses, which is the other half of the report's complaint. A level
    other than 95% recomputes from the t quantile rather than linearly rescaling the stored bounds;
  * ``cov_params()`` reconstructs the matrix from the reported standard errors. ONLY the diagonal
    is recoverable, so the off-diagonals are NaN rather than zero -- a zero would assert the
    estimates are uncorrelated, which is a different and false claim;
  * ``summary(style=...)`` accepts the argument for protocol parity.

WHY ONLY THIS CLASS. Of the package's non-``StataResults`` result types, `PraisResult` is the one
that genuinely has coefficients and standard errors. `SummarizeResult`, `OverIdResult`,
`MarginsTable` and `FirstStageResult` describe tables and tests, not a coefficient vector, so
forcing the estimator protocol onto them would add methods that could only lie. `cov_params()` on
`StataResults` is deliberately untouched -- the user asked that it stay as it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

ICECREAM = r"D:\myCodeProject\test_westata\data\icecream.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(ICECREAM)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("icecream.dta is not available")
    return ws.prais(frame, "consumption ~ temp + price + income", corc=True)


def test_conf_int_exists(fitted) -> None:
    """It used to raise AttributeError."""

    intervals = fitted.conf_int()
    assert isinstance(intervals, pd.DataFrame)


def test_the_columns_are_lower_and_upper(fitted) -> None:
    """The report's other half: column names must match the rest of the package."""

    assert list(fitted.conf_int().columns) == ["lower", "upper"]


def test_the_bounds_match_stata(fitted) -> None:
    """From `prais consumption temp price income, corc`: price is -.2.562373 .. .777581."""

    intervals = fitted.conf_int()
    assert float(intervals.loc["price", "lower"]) == pytest.approx(-2.562373, abs=5e-7)
    assert float(intervals.loc["price", "upper"]) == pytest.approx(0.777581, abs=5e-7)


def test_the_interval_brackets_the_estimate(fitted) -> None:
    intervals = fitted.conf_int()
    for name in fitted.param_names if hasattr(fitted, "param_names") else intervals.index:
        point = float(fitted.params[name])
        assert float(intervals.loc[name, "lower"]) <= point <= float(intervals.loc[name, "upper"])


def test_a_different_level_is_narrower(fitted) -> None:
    """90% must sit strictly inside 95%, which a rescaled 95% would not guarantee."""

    wide = fitted.conf_int()
    narrow = fitted.conf_int(alpha=0.10)
    assert (
        (narrow["upper"] - narrow["lower"]) < (wide["upper"] - wide["lower"])
    ).all()


def test_cov_params_diagonal_is_the_squared_standard_error(fitted) -> None:
    covariance = fitted.cov_params()
    assert np.allclose(np.diag(covariance.values), np.asarray(fitted.bse) ** 2)


def test_cov_params_off_diagonals_are_nan(fitted) -> None:
    """Not zero: an uncorrelated pair is a claim this class cannot support."""

    values = fitted.cov_params().values
    mask = ~np.eye(values.shape[0], dtype=bool)
    assert np.all(np.isnan(values[mask]))


def test_summary_accepts_the_style_argument(fitted) -> None:
    assert isinstance(fitted.summary(style="stata"), str)
