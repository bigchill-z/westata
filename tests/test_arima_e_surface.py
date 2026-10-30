"""``arima``'s ``e()`` scalars, checked against ``arima.sthlp:561-580``.

The documented block lists ``e(k)``, ``e(N_gaps)``, ``e(ar_max)``, ``e(ma_max)``,
``e(ic)`` and ``e(rc)`` alongside the ones we already stored. All six were absent, so a
caller following the help file found nothing where Stata returns a value.

Two of them carry a definition that is easy to get wrong:

* ``e(ar_max)``/``e(ma_max)`` report the EFFECTIVE lag, so a seasonal term contributes
  ``P * s`` (or ``Q * s``) on top of the non-seasonal order;
* ``e(rc)`` is a return code, not a statistic -- ``0`` means the estimation completed,
  exactly as ``xtreg.sthlp:1222`` documents for its own ``e(rc)``.
"""

from __future__ import annotations

import numpy as np
import pytest

import westata as ws


def _series(n: int = 200, seed: int = 1000):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=n))


@pytest.fixture(scope="module")
def simple_fit():
    return ws.ARIMA(_series(), order=(1, 1, 1)).fit()


@pytest.fixture(scope="module")
def two_ar_fit():
    return ws.ARIMA(_series(), order=(2, 1, 1)).fit()


def test_k_counts_the_estimated_parameters(simple_fit) -> None:
    """``arima.sthlp:563``: "``e(k)`` number of parameters".

    An ARIMA(1,1,1) with a constant has three: the AR term, the MA term and the
    intercept.
    """

    assert float(simple_fit.e["k"]) == 3.0


def test_k_grows_with_the_order(two_ar_fit) -> None:
    """ARIMA(2,1,1) adds one AR coefficient."""

    assert float(two_ar_fit.e["k"]) == 4.0


def test_k_matches_the_parameter_vector(simple_fit) -> None:
    """The scalar and the vector must count the same thing."""

    assert float(simple_fit.e["k"]) == float(len(simple_fit.param_names))


def test_ar_max_is_the_non_seasonal_order(simple_fit) -> None:
    """ARIMA(1,1,1) with no seasonal part: one AR lag, one MA lag."""

    assert float(simple_fit.e["ar_max"]) == 1.0
    assert float(simple_fit.e["ma_max"]) == 1.0


def test_ar_max_follows_the_order(two_ar_fit) -> None:
    assert float(two_ar_fit.e["ar_max"]) == 2.0
    assert float(two_ar_fit.e["ma_max"]) == 1.0


def test_a_seasonal_term_multiplies_into_ar_max() -> None:
    """The trap: ``P * s`` on top of ``p``.

    ARIMA(1,0,0)(1,0,0,4) has a lag at 1 AND at 4, so ``ar_max`` is 5 -- not 1, which is
    what reporting only the non-seasonal order would give.
    """

    fitted = ws.ARIMA(
        _series(), order=(1, 0, 0), seasonal_order=(1, 0, 0, 4)
    ).fit()

    assert float(fitted.e["ar_max"]) == 5.0
    assert float(fitted.e["ma_max"]) == 0.0


def test_a_seasonal_ma_multiplies_into_ma_max() -> None:
    fitted = ws.ARIMA(
        _series(), order=(0, 0, 1), seasonal_order=(0, 0, 1, 12)
    ).fit()

    assert float(fitted.e["ma_max"]) == 13.0
    assert float(fitted.e["ar_max"]) == 0.0


def test_rc_is_a_return_code_zero(simple_fit) -> None:
    """``arima.sthlp:579``: "``e(rc)`` return code".

    Zero means success; it is not a correlation and not a statistic.
    """

    assert float(simple_fit.e["rc"]) == 0.0


def test_n_gaps_is_reported(simple_fit) -> None:
    """``arima.sthlp:562``: "number of gaps".

    This package passes dense arrays, so the count is zero -- but the KEY has to exist,
    because a caller reading Stata's ``e(N_gaps)`` expects a number.
    """

    assert float(simple_fit.e["N_gaps"]) == 0.0


def test_ic_is_a_non_negative_iteration_count(simple_fit) -> None:
    """``arima.sthlp:578``: "number of iterations"."""

    value = float(simple_fit.e["ic"])

    assert np.isfinite(value)
    assert value >= 0.0


def test_the_converged_flag_is_zero_or_one(simple_fit) -> None:
    """``arima.sthlp:580``: 1 if converged, 0 otherwise."""

    assert float(simple_fit.e["converged"]) in (0.0, 1.0)


def test_the_documented_family_is_present(simple_fit) -> None:
    """Every scalar the help file lists for a plain ARIMA fit.

    Grouped so a single missing key names itself in the failure message.
    """

    macros = simple_fit.e
    for key in (
        "N",
        "df_m",
        "ll",
        "sigma",
        "chi2",
        "p",
        "rank",
        "converged",
        "k",
        "N_gaps",
        "ar_max",
        "ma_max",
        "ic",
        "rc",
    ):
        assert key in macros, key
        assert macros[key] is not None, key


# ------------------------------------------------------------------ macros
def test_depvar_is_the_series_name() -> None:
    """``arima.sthlp:585``: "name of dependent variable".

    It was absent entirely, even for a named pandas Series -- and it is the one macro
    every postestimation command reads.
    """

    import pandas as pd

    rng = np.random.default_rng(1002)
    series = pd.Series(
        np.cumsum(rng.normal(size=120)), name="gdp"
    )
    fitted = ws.ARIMA(series, order=(1, 0, 0)).fit()

    assert fitted.e["depvar"] == "gdp"


def test_depvar_falls_back_for_an_unnamed_array() -> None:
    """A bare array carries no name; the macro must still hold a string."""

    rng = np.random.default_rng(1004)
    fitted = ws.ARIMA(np.cumsum(rng.normal(size=120)), order=(1, 0, 0)).fit()

    assert fitted.e["depvar"] == "y"


def test_eqnames_matches_the_single_equation() -> None:
    """``arima.sthlp:564``: one equation, named after the dependent variable."""

    import pandas as pd

    rng = np.random.default_rng(1006)
    series = pd.Series(np.cumsum(rng.normal(size=100)), name="price")
    fitted = ws.ARIMA(series, order=(1, 0, 0)).fit()

    assert fitted.e["eqnames"] == "price"
    assert fitted.e["eqnames"] == fitted.e["depvar"]


def test_covariates_is_an_empty_macro(simple_fit) -> None:
    """``arima.sthlp:586``: this estimator takes no regressors, so it stores "".

    ``None`` would mean "not stored"; the empty string is what Stata records.
    """

    assert simple_fit.e["covariates"] == ""


def test_the_time_range_covers_the_series() -> None:
    """``arima.sthlp:573-574``: ``tmin``/``tmax`` bound an undated series at 1..N."""

    rng = np.random.default_rng(1008)
    n = 150
    fitted = ws.ARIMA(np.cumsum(rng.normal(size=n)), order=(1, 0, 0)).fit()

    assert float(fitted.e["tmin"]) == 1.0
    assert float(fitted.e["tmax"]) == float(n)


def test_the_formatted_time_range_matches_the_numeric_one() -> None:
    """``tmins``/``tmaxs`` are the string forms; they must agree with the numbers."""

    rng = np.random.default_rng(1010)
    n = 88
    fitted = ws.ARIMA(np.cumsum(rng.normal(size=n)), order=(1, 0, 0)).fit()

    assert fitted.e["tmins"] == "1"
    assert fitted.e["tmaxs"] == str(n)
    assert float(fitted.e["tmins"]) == float(fitted.e["tmin"])
    assert float(fitted.e["tmaxs"]) == float(fitted.e["tmax"])