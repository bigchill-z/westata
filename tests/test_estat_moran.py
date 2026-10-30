"""``estat moran`` -- spatial correlation in the residuals.

``estat_moran.sthlp:44-46``:

    estat moran is a postestimation test that can be run after fitting a model using
    regress with spatial data.  It performs the Moran test for spatial correlation among
    the residuals.

Syntax requires a weighting matrix (``sthlp:29-31``)::

    estat moran, errorlag(spmatname) [errorlag(spmatname) ...]

IMPORTANT CAVEAT, recorded here rather than only in a report: ``sthlp:56-59`` points at a
"Methods and formulas" section that is "not included in this help file", and there is no
``estat_moran.ado`` in ``ado/base`` -- the arithmetic is a compiled builtin. Stata's exact
``r(chi2)`` convention therefore cannot be reproduced from local evidence.

What is implemented is the STANDARD randomisation-based Moran test (Moran 1950; Cliff &
Ord 1981). Its p-value is exact for "no spatial correlation"; it may differ from Stata's
number, and the tests below check the statistic against its own definition rather than
against Stata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


def _ring_weights(count: int) -> np.ndarray:
    """Row-standardised first-order contiguity on a ring.

    A ring avoids the boundary rows that a line would have, so every observation gets the
    same neighbour weight -- which makes the expected value checkable by hand.
    """

    matrix = np.zeros((count, count))
    for index in range(count):
        matrix[index, (index + 1) % count] = 1.0
        matrix[index, (index - 1) % count] = 1.0
    return matrix / matrix.sum(axis=1, keepdims=True)


def _frame(count: int = 100, seed: int = 3100):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"y": rng.normal(size=count), "x1": rng.normal(size=count)}
    )


@pytest.fixture(scope="module")
def fitted():
    frame = _frame()
    return frame, ws.OLS.from_formula("y ~ x1", frame).fit()


def _manual_moran(residuals: np.ndarray, weights: np.ndarray):
    """Rebuild ``I``, ``E[I]`` and ``sd[I]`` from their definitions."""

    count = residuals.size
    squared = float(residuals @ residuals)
    s0 = float(weights.sum())
    morans_i = (count / s0) * float(residuals @ weights @ residuals) / squared
    expected = -1.0 / (count - 1)

    transposed = weights.T
    s1 = 0.5 * float(np.sum((weights + transposed) ** 2))
    s2 = float(
        np.sum((weights.sum(axis=1) + weights.sum(axis=0)) ** 2)
    )
    kurtosis = count * float(np.sum(residuals ** 4)) / (squared ** 2)
    numerator = count * (
        (count ** 2 - 3 * count + 3) * s1 - count * s2 + 3 * s0 ** 2
    ) - kurtosis * (count * (count - 1) * s1 - 2 * count * s2 + 6 * s0 ** 2)
    denominator = (count - 1) * (count - 2) * (count - 3) * s0 ** 2
    variance = numerator / denominator - expected ** 2
    return morans_i, expected, float(np.sqrt(variance))


# ------------------------------------------------------------------- the formula
def test_morans_i_matches_its_definition(fitted) -> None:
    frame, result = fitted
    weights = _ring_weights(len(frame))
    expected_i, _, _ = _manual_moran(np.asarray(result.resid), weights)

    reported = result.estat_moran(errorlag=weights)

    assert reported.moran_i == pytest.approx(expected_i, rel=1e-12)


def test_the_expected_value_is_minus_one_over_n_minus_one(fitted) -> None:
    reported = fitted[1].estat_moran(errorlag=_ring_weights(len(fitted[0])))

    assert reported.expected_i == pytest.approx(-1.0 / (len(fitted[0]) - 1))


def test_the_z_statistic_matches_its_definition(fitted) -> None:
    frame, result = fitted
    weights = _ring_weights(len(frame))
    expected_i, expected_expected, expected_sd = _manual_moran(
        np.asarray(result.resid), weights
    )
    expected_z = (expected_i - expected_expected) / expected_sd

    reported = result.estat_moran(errorlag=weights)

    assert reported.z == pytest.approx(expected_z, rel=1e-12)


def test_chi2_is_the_squared_z(fitted) -> None:
    """``chi2 = z^2`` on one degree of freedom -- the convention assumed here."""

    reported = fitted[1].estat_moran(errorlag=_ring_weights(len(fitted[0])))

    assert reported.statistic == pytest.approx(reported.z ** 2, rel=1e-12)
    assert reported.pvalue == pytest.approx(
        float(scipy_stats.chi2.sf(reported.statistic, 1)), rel=1e-12
    )


# ------------------------------------------------------------------- behaviour
def test_positive_autocorrelation_is_detected() -> None:
    """Errors built to be spatially clustered must give a large ``I``.

    The fixture makes each residual correlate with its ring neighbours, so ``e'We`` is
    large and the test rejects independence.
    """

    rng = np.random.default_rng(3102)
    count = 120
    weights = _ring_weights(count)
    shock = np.zeros(count)
    shock[0] = rng.normal()
    # Propagate the shock around the ring so neighbours share it.
    for index in range(1, count):
        shock[index] = 0.9 * shock[index - 1] + rng.normal() * 0.3
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.5 * rng.normal(size=count) + shock,
            "x1": rng.normal(size=count),
        }
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()

    reported = result.estat_moran(errorlag=weights)

    assert reported.moran_i > 0.0
    assert reported.pvalue < 0.01


def test_independent_errors_are_not_flagged() -> None:
    """The other direction: independent errors must not reject."""

    rng = np.random.default_rng(3104)
    count = 120
    frame = pd.DataFrame(
        {"y": rng.normal(size=count), "x1": rng.normal(size=count)}
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()

    reported = result.estat_moran(errorlag=_ring_weights(count))

    assert reported.pvalue > 0.05


def test_a_symmetric_weight_matrix_centres_i_near_its_expectation(fitted) -> None:
    """Guard for the direction tests: with independent errors ``I`` sits near ``E[I]``."""

    reported = fitted[1].estat_moran(errorlag=_ring_weights(len(fitted[0])))

    assert abs(reported.moran_i - reported.expected_i) < 0.1


# ------------------------------------------------------------- higher-order lags
def test_two_matrices_are_summed(fitted) -> None:
    """``sthlp:66-68``: ``errorlag()`` is repeatable for higher-order lags.

    Repeated matrices combine into one lag operator, so the reported ``I`` must equal
    the value computed from their SUM -- not from either alone.
    """

    count = len(fitted[0])
    first = _ring_weights(count)
    rng = np.random.default_rng(3106)
    second = rng.uniform(0.0, 1.0, size=(count, count))
    np.fill_diagonal(second, 0.0)
    second = second / second.sum(axis=1, keepdims=True)

    combined = fitted[1].estat_moran(errorlag=[first, second])
    expected_i, _, _ = _manual_moran(
        np.asarray(fitted[1].resid), first + second
    )

    assert combined.moran_i == pytest.approx(expected_i, rel=1e-12)
    assert combined.errorlags == 2


def test_a_single_matrix_and_a_one_element_list_agree(fitted) -> None:
    weights = _ring_weights(len(fitted[0]))

    single = fitted[1].estat_moran(errorlag=weights)
    wrapped = fitted[1].estat_moran(errorlag=[weights])

    assert single.moran_i == pytest.approx(wrapped.moran_i, rel=1e-12)


# ------------------------------------------------------------------- refusals
def test_the_weighting_matrix_is_required(fitted) -> None:
    """``sthlp:66-67``: "``errorlag()`` is required"."""

    with pytest.raises(ValueError, match="errorlag"):
        fitted[1].estat_moran()


def test_a_wrong_sized_matrix_is_refused(fitted) -> None:
    wrong = _ring_weights(len(fitted[0]) - 5)

    with pytest.raises(ValueError, match="estimation sample"):
        fitted[1].estat_moran(errorlag=wrong)


def test_a_non_square_matrix_is_refused(fitted) -> None:
    with pytest.raises(ValueError, match="square"):
        fitted[1].estat_moran(errorlag=np.zeros((10, 12)))


def test_a_non_regress_fit_is_refused() -> None:
    """``sthlp:44-45``: "after fitting a model using regress"."""

    rng = np.random.default_rng(3108)
    frame = pd.DataFrame(
        {"yb": (rng.normal(size=80) > 0).astype(int), "x1": rng.normal(size=80)}
    )
    logit = ws.Logit.from_formula("yb ~ x1", frame).fit()

    with pytest.raises(NotImplementedError, match="regress"):
        logit.estat_moran(errorlag=_ring_weights(80))


def test_passing_both_aliases_is_refused(fitted) -> None:
    weights = _ring_weights(len(fitted[0]))

    with pytest.raises(ValueError, match="not both"):
        fitted[1].estat_moran(errorlag=weights, weights=weights)


# ---------------------------------------------------------------- the r() names
def test_to_dict_reports_the_r_names(fitted) -> None:
    """``sthlp:105-111``: ``r(chi2)``, ``r(df)``, ``r(p)``."""

    table = fitted[1].estat_moran(errorlag=_ring_weights(len(fitted[0]))).to_dict()

    assert set(table) == {"chi2", "df", "p"}
    for value in table.values():
        assert np.isfinite(value)