"""Tests for the system estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.systems import Eivreg, Manova, Mvreg, Sureg


def _sur(n: int = 400, seed: int = 101) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    common = rng.normal(size=n)
    e1 = 0.7 * common + 0.7 * rng.normal(size=n)
    e2 = 0.7 * common + 0.7 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * x1 - 0.5 * x2 + e1,
            "y2": -0.5 + 1.0 * x2 + 0.8 * x3 + e2,
            "x1": x1,
            "x2": x2,
            "x3": x3,
        }
    )


def _eiv(n: int = 500, seed: int = 107) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    true_x = rng.normal(size=n)
    observed = true_x + rng.normal(scale=0.7, size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame(
        {"y": 1.0 + 2.0 * true_x - 0.5 * x2 + rng.normal(size=n), "x1": observed, "x2": x2}
    )


def _manova(n: int = 240, seed: int = 109) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group = rng.integers(0, 3, size=n)
    effect = np.array([0.0, 1.0, 2.0])[group]
    return pd.DataFrame(
        {
            "y1": 1.0 + effect + rng.normal(scale=0.8, size=n),
            "y2": 2.0 + 0.5 * effect + rng.normal(scale=0.8, size=n),
            "g": group,
        }
    )


# ------------------------------------------------------------------- sureg


def test_sureg_matches_stata_coefficients():
    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()

    assert float(result.params["y1:Intercept"]) == pytest.approx(1.00723, rel=1e-4)
    assert float(result.params["y1:x1"]) == pytest.approx(1.960029, rel=1e-4)
    assert float(result.params["y1:x2"]) == pytest.approx(-0.6210211, rel=1e-4)
    assert float(result.params["y2:Intercept"]) == pytest.approx(-0.4903816, rel=1e-4)
    assert float(result.params["y2:x2"]) == pytest.approx(0.9412217, rel=1e-4)
    assert float(result.params["y2:x3"]) == pytest.approx(0.8024322, rel=1e-4)


def test_sureg_matches_stata_standard_errors():
    """The whole point of SUR is the cross-equation covariance."""

    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()

    assert float(result.bse["y1:Intercept"]) == pytest.approx(0.0492993, rel=1e-3)
    assert float(result.bse["y1:x1"]) == pytest.approx(0.040549, rel=1e-3)
    assert float(result.bse["y1:x2"]) == pytest.approx(0.0476005, rel=1e-3)
    assert float(result.bse["y2:Intercept"]) == pytest.approx(0.0487194, rel=1e-3)
    assert float(result.bse["y2:x2"]) == pytest.approx(0.047139, rel=1e-3)
    assert float(result.bse["y2:x3"]) == pytest.approx(0.039197, rel=1e-3)


def test_sureg_covariance_is_not_block_diagonal():
    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()
    covariance = np.asarray(result.cov_params(), dtype=float)

    assert covariance.shape == (6, 6)
    assert np.any(np.abs(covariance[:3, 3:]) > 1e-12)
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)


def test_sureg_reduces_to_per_equation_ols_when_errors_are_uncorrelated():
    """SUR collapses to OLS only when the cross-equation correlation is zero.

    With independent errors SUR is a no-op on the coefficients, so any difference
    would be a bug. The correlation is asserted to be small first: without that
    guard the test would silently become vacuous if the DGP changed, and worse, it
    would fail for the wrong reason (an earlier version did exactly that, on a DGP
    whose correlation was 0.084).
    """

    rng = np.random.default_rng(5)
    n = 2000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y1": 1.0 + 2.0 * x1 + rng.normal(size=n),
            "y2": -0.5 + 1.0 * x2 + rng.normal(size=n),
            "x1": x1,
            "x2": x2,
        }
    )
    result = Sureg.from_formula(["y1 ~ x1", "y2 ~ x2"], frame).fit()
    first = ws.OLS.from_formula("y1 ~ x1", frame).fit()
    second = ws.OLS.from_formula("y2 ~ x2", frame).fit()

    correlation = float(np.asarray(result.e["corr"])[0, 1])
    assert abs(correlation) < 0.05, (
        f"the two equations are correlated ({correlation:.3f}); the premise of "
        "this test is independent errors"
    )

    assert float(result.params["y1:x1"]) == pytest.approx(
        float(first.params["x1"]), abs=1e-3
    )
    assert float(result.params["y2:x2"]) == pytest.approx(
        float(second.params["x2"]), abs=1e-3
    )


def test_sureg_reports_the_cross_equation_correlation():
    result = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], _sur()).fit()
    corr = np.asarray(result.e["corr"], dtype=float)
    # The DGP shares a common error component, so the correlation is substantial.
    assert 0.3 < corr[0, 1] < 0.8


def test_sureg_rejects_iterated_estimation():
    model = Sureg.from_formula(["y1 ~ x1", "y2 ~ x2"], _sur())
    with pytest.raises(UnsupportedError, match="iterate"):
        model.fit(iterate=True)


# ------------------------------------------------------------------- mvreg


def test_mvreg_matches_stata():
    result = Mvreg.from_formula("y1 y2 = x1 x2 x3", _sur()).fit()

    assert float(result.params["y1:x1"]) == pytest.approx(1.95372, rel=1e-4)
    assert float(result.params["y1:x2"]) == pytest.approx(-0.6277454, rel=1e-4)
    assert float(result.params["y1:x3"]) == pytest.approx(-0.0732678, rel=1e-4)
    assert float(result.params["y2:x2"]) == pytest.approx(0.9381316, rel=1e-4)


def test_mvreg_coefficients_equal_per_equation_ols():
    frame = _sur()
    result = Mvreg.from_formula("y1 y2 = x1 x2 x3", frame).fit()
    first = ws.OLS.from_formula("y1 ~ x1 + x2 + x3", frame).fit()

    assert float(result.params["y1:x1"]) == pytest.approx(float(first.params["x1"]), abs=1e-10)


def test_mvreg_differs_from_sureg_by_specification_not_by_bug():
    """mvreg puts every regressor in every equation; sureg does not.

    Their coefficients therefore differ on the same data, and that is expected.
    """

    frame = _sur()
    sureg = Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x2 + x3"], frame).fit()
    mvreg = Mvreg.from_formula("y1 y2 = x1 x2 x3", frame).fit()

    assert float(sureg.params["y1:x1"]) != pytest.approx(
        float(mvreg.params["y1:x1"]), rel=1e-6
    )


# ------------------------------------------------------------------ eivreg


def test_eivreg_with_perfect_reliability_reduces_to_ols():
    frame = _eiv()
    result = Eivreg.from_formula("y ~ x1 + x2", frame, reliab={"x1": 1.0}).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        result.params[["Intercept", "x1", "x2"]].to_numpy(),
        plain.params[["Intercept", "x1", "x2"]].to_numpy(),
        rtol=1e-8,
    )


def test_eivreg_matches_stata():
    result = Eivreg.from_formula("y ~ x1 + x2", _eiv(), reliab={"x1": 0.7}).fit()

    assert float(result.params["x1"]) == pytest.approx(2.073214, rel=1e-3)
    assert float(result.params["x2"]) == pytest.approx(-0.4678675, rel=1e-3)
    assert float(result.params["Intercept"]) == pytest.approx(0.8649387, rel=1e-3)


def test_eivreg_requires_reliab():
    with pytest.raises(ConsistencyError, match="reliab is required"):
        Eivreg.from_formula("y ~ x1 + x2", _eiv())


def test_eivreg_rejects_a_reliability_for_a_non_regressor():
    with pytest.raises(ConsistencyError, match="not regressors"):
        Eivreg.from_formula("y ~ x1 + x2", _eiv(), reliab={"nonexistent": 0.9})


def test_eivreg_rejects_an_out_of_range_reliability():
    with pytest.raises(ConsistencyError, match="reliability"):
        Eivreg.from_formula("y ~ x1 + x2", _eiv(), reliab={"x1": 1.5})


# ------------------------------------------------------------------ manova


def test_manova_wilks_lambda_matches_stata():
    outcome = Manova.from_formula("y1 y2 = g", _manova()).fit()
    assert float(outcome.wilks_lambda) == pytest.approx(0.4290, abs=5e-3)


def test_manova_reports_an_f_statistic():
    outcome = Manova.from_formula("y1 y2 = g", _manova()).fit()
    assert np.isfinite(outcome.f_statistic)
    assert 0.0 <= outcome.pvalue <= 1.0


def test_manova_rejects_multiple_factors():
    frame = _manova()
    frame["h"] = np.random.default_rng(1).integers(0, 2, size=len(frame))
    with pytest.raises(UnsupportedError, match="one factor"):
        Manova.from_formula("y1 y2 = g + h", frame)
