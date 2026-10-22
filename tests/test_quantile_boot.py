"""Tests for simultaneous and bootstrapped quantile regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.estimators.quantile import QuantileRegression
from westata.estimators.quantile_boot import Bsqreg, Sqreg


def _frame(n: int = 400, seed: int = 61) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scale = 0.5 + 0.4 * np.abs(x1)
    y = 1.0 + 2.0 * x1 - 0.8 * x2 + scale * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


QUANTILES = (0.25, 0.5, 0.75)


# ------------------------------------------------------------------ sqreg


def test_sqreg_coefficients_match_per_quantile_qreg():
    """The point estimates are the plain qreg fits, not a separate estimator."""

    frame = _frame()
    sqreg = Sqreg.from_formula("y ~ x1 + x2", frame, quantiles=QUANTILES).fit(
        reps=30, seed=1
    )

    for position, tau in enumerate(QUANTILES):
        reference = QuantileRegression.from_formula(
            "y ~ x1 + x2", frame, quantile=tau
        ).fit()
        block = sqreg.params[position * 3 : (position + 1) * 3]
        np.testing.assert_allclose(block, reference.params.to_numpy(), rtol=1e-10)


def test_sqreg_covariance_is_symmetric_positive_semidefinite():
    frame = _frame()
    result = Sqreg.from_formula("y ~ x1 + x2", frame, quantiles=QUANTILES).fit(
        reps=50, seed=2
    )
    covariance = np.asarray(result.cov_params(), dtype=float)

    assert covariance.shape == (9, 9)
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)
    eigenvalues = np.linalg.eigvalsh(covariance)
    assert eigenvalues.min() > -1e-10


def test_sqreg_has_cross_quantile_blocks():
    """A simultaneous VCE must not be block diagonal."""

    frame = _frame()
    result = Sqreg.from_formula("y ~ x1 + x2", frame, quantiles=QUANTILES).fit(
        reps=50, seed=3
    )
    covariance = np.asarray(result.cov_params(), dtype=float)
    block_01 = covariance[0:3, 3:6]
    assert np.any(np.abs(block_01) > 1e-12)


def test_sqreg_standard_errors_are_positive_and_finite():
    frame = _frame()
    result = Sqreg.from_formula("y ~ x1 + x2", frame, quantiles=QUANTILES).fit(
        reps=50, seed=4
    )
    errors = result.bse.to_numpy()
    assert np.all(np.isfinite(errors))
    assert np.all(errors > 0)


def test_sqreg_names_quantile_blocks():
    frame = _frame()
    result = Sqreg.from_formula("y ~ x1 + x2", frame, quantiles=(0.25, 0.75)).fit(
        reps=20, seed=5
    )
    names = list(result.param_names)
    assert names[0].startswith("q0.25:")
    assert names[3].startswith("q0.75:")


def test_sqreg_rejects_non_positive_reps():
    frame = _frame()
    model = Sqreg.from_formula("y ~ x1", frame)
    with pytest.raises(ValueError, match="reps"):
        model.fit(reps=0)


def test_sqreg_rejects_a_bad_quantile():
    with pytest.raises(ValueError, match="quantile"):
        Sqreg.from_formula("y ~ x1", _frame(), quantiles=(0.5, 1.5))


def test_sqreg_is_reproducible_with_a_seed():
    frame = _frame()
    first = Sqreg.from_formula("y ~ x1 + x2", frame, quantiles=(0.5,)).fit(
        reps=25, seed=99
    )
    second = Sqreg.from_formula("y ~ x1 + x2", frame, quantiles=(0.5,)).fit(
        reps=25, seed=99
    )
    np.testing.assert_allclose(
        np.asarray(first.cov_params()), np.asarray(second.cov_params()), rtol=1e-12
    )


# ----------------------------------------------------------------- bsqreg


def test_bsqreg_coefficients_equal_qreg_exactly():
    """``bsqreg`` is ``sqreg`` with one quantile, so the point estimate is qreg's."""

    frame = _frame()
    boot = Bsqreg.from_formula("y ~ x1 + x2", frame, quantile=0.5).fit(
        reps=30, seed=7
    )
    reference = QuantileRegression.from_formula(
        "y ~ x1 + x2", frame, quantile=0.5
    ).fit()

    np.testing.assert_allclose(boot.params.to_numpy(), reference.params.to_numpy(), rtol=1e-12)


def test_bsqreg_standard_errors_are_the_same_order_as_the_iid_ones():
    frame = _frame()
    boot = Bsqreg.from_formula("y ~ x1 + x2", frame, quantile=0.5).fit(
        reps=100, seed=11
    )
    reference = QuantileRegression.from_formula(
        "y ~ x1 + x2", frame, quantile=0.5
    ).fit(cov_type="iid")

    ratio = float(boot.bse["x1"]) / float(reference.bse["x1"])
    # A 100-replicate bootstrap carries real Monte Carlo error, so the band is
    # deliberately wide; it still catches a factor-of-several mistake.
    assert 0.5 < ratio < 2.0


def test_bsqreg_uses_plain_parameter_names():
    frame = _frame()
    result = Bsqreg.from_formula("y ~ x1 + x2", frame, quantile=0.5).fit(
        reps=20, seed=13
    )
    assert list(result.param_names) == ["Intercept", "x1", "x2"]


def test_bsqreg_rejects_a_bad_quantile():
    with pytest.raises(ValueError, match="quantile"):
        Bsqreg.from_formula("y ~ x1", _frame(), quantile=0.0)


def test_bsqreg_populates_e_and_renders():
    frame = _frame()
    result = Bsqreg.from_formula("y ~ x1 + x2", frame, quantile=0.5).fit(
        reps=20, seed=17
    )
    for key in ["N", "df_m", "df_r", "reps", "q"]:
        assert key in result.e, key
    assert result.e["q"] == 0.5
    assert "Coef." in result.summary(style="stata")
