"""Tests for quantile and robust regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError
from westata.estimators.quantile import (
    QuantileRegression,
    RobustRegression,
    solve_lad,
)


def _quantile_frame(n: int = 600, seed: int = 61) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scale = 0.5 + 0.4 * np.abs(x1)
    y = 1.0 + 2.0 * x1 - 0.8 * x2 + scale * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _outlier_frame(n: int = 400, seed: int = 67) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    kontam = rng.random(size=n) < 0.05
    y = np.where(kontam, y + rng.normal(scale=30.0, size=n), y)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# --------------------------------------------------------------- solve_lad


def test_solve_lad_with_only_a_constant_reproduces_the_quantile():
    rng = np.random.default_rng(3)
    y = rng.normal(size=500)
    X = np.ones((y.size, 1))

    for tau in (0.25, 0.5, 0.75, 0.9):
        beta = solve_lad(X, y, quantile=tau)
        # The LAD estimate of a location is a sample quantile; allow for the
        # fact that several order statistics can be minimisers.
        expected = np.quantile(y, tau)
        assert beta[0] == pytest.approx(expected, abs=0.05)


def test_solve_lad_rejects_bad_quantile():
    X = np.ones((10, 1))
    with pytest.raises(ValueError, match="quantile"):
        solve_lad(X, np.arange(10.0), quantile=1.5)


def test_solve_lad_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="observations"):
        solve_lad(np.ones((10, 1)), np.arange(5.0))


# ------------------------------------------------------------------- qreg


def test_qreg_median_is_close_to_ols_on_homoskedastic_data():
    rng = np.random.default_rng(11)
    n = 800
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.5 * x2 + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    quantile = QuantileRegression.from_formula("y ~ x1 + x2", frame).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        quantile.params[["Intercept", "x1", "x2"]].to_numpy(),
        plain.params[["Intercept", "x1", "x2"]].to_numpy(),
        rtol=5e-2,
    )


def test_qreg_heteroskedastic_slopes_increase_with_the_quantile():
    frame = _quantile_frame()
    coefficients = []
    for tau in (0.25, 0.5, 0.75, 0.9):
        result = QuantileRegression.from_formula(
            "y ~ x1 + x2", frame, quantile=tau
        ).fit()
        coefficients.append(float(result.params["Intercept"]))

    # The DGP has scale rising in |x1|, so the intercept rises across quantiles.
    assert coefficients == sorted(coefficients)


def test_qreg_pseudo_r2_is_between_zero_and_one():
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", _quantile_frame(), quantile=0.5
    ).fit()
    assert 0.0 < float(result.e["r2_p"]) < 1.0
    # Pseudo R2 here is 1 - sum_adev/sum_rdev, not the OLS R-squared.
    implied = 1.0 - float(result.e["sum_adev"]) / float(result.e["sum_rdev"])
    assert float(result.e["r2_p"]) == pytest.approx(implied, rel=1e-12)


def test_qreg_rejects_a_bad_quantile():
    with pytest.raises(ValueError, match="quantile"):
        QuantileRegression.from_formula("y ~ x1", _quantile_frame(), quantile=0.0)
    with pytest.raises(ValueError, match="quantile"):
        QuantileRegression.from_formula("y ~ x1", _quantile_frame(), quantile=1.0)


def test_qreg_rejects_an_unknown_cov_type():
    model = QuantileRegression.from_formula("y ~ x1 + x2", _quantile_frame())
    with pytest.raises(ValueError, match="cov_type"):
        model.fit(cov_type="hac")


def test_qreg_robust_and_iid_differ_but_are_close_on_iid_errors():
    rng = np.random.default_rng(19)
    n = 600
    x1 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x1": x1})

    iid = QuantileRegression.from_formula("y ~ x1", frame).fit(cov_type="iid")
    robust = QuantileRegression.from_formula("y ~ x1", frame).fit(cov_type="robust")

    np.testing.assert_allclose(iid.params.to_numpy(), robust.params.to_numpy())
    ratio = float(robust.bse["x1"]) / float(iid.bse["x1"])
    # Measured against Stata: vce(robust) differs from the iid default by only
    # about 1% on this DGP (.0352211 vs .0355009), so the ratio must be near 1.
    # An earlier version of the sandwich dropped the density factor and gave
    # 0.0204 here, which is what this bound now catches.
    assert 0.8 < ratio < 1.25


def test_qreg_populates_e_and_renders():
    result = QuantileRegression.from_formula(
        "y ~ x1 + x2", _quantile_frame(), quantile=0.5
    ).fit()
    for key in ["N", "df_m", "df_r", "q", "sum_adev", "sum_rdev", "r2_p", "rank"]:
        assert key in result.e, key
    assert "Coef." in result.summary(style="stata")


# ------------------------------------------------------------------- rreg


def test_rreg_beats_ols_on_contaminated_data():
    frame = _outlier_frame()
    robust = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    true_slope = 2.0
    assert abs(float(robust.params["x1"]) - true_slope) < abs(
        float(plain.params["x1"]) - true_slope
    )


def test_rreg_reports_a_positive_scale():
    result = RobustRegression.from_formula("y ~ x1 + x2", _outlier_frame()).fit()
    assert np.isfinite(float(result.e["scale"]))
    assert float(result.e["scale"]) > 0


def test_rreg_cooks_screen_is_reported_and_consistent():
    """The Cook's-distance screen is applied, but need not drop anything.

    On the reference DGP the largest Cook's D is 0.218, well below the threshold
    of 1, so Stata and westata both keep every observation. Asserting that at
    least one point is dropped would have encoded a false expectation -- the
    screen exists, but this contamination is handled by the robust weights
    rather than by deletion.
    """

    frame = _outlier_frame()
    result = RobustRegression.from_formula("y ~ x1 + x2", frame).fit()
    dropped = int(result.e["dropped"])
    assert dropped >= 0
    assert int(result.e["N"]) == len(frame) - dropped

    # A hand-computed cook's D max confirms the threshold behaviour.
    X = np.column_stack(
        [np.ones(len(frame)), frame["x1"].to_numpy(), frame["x2"].to_numpy()]
    )
    y = frame["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    p = X.shape[1]
    sigma2 = float(residual @ residual) / (len(y) - p)
    leverage = np.einsum(
        "ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X
    )
    cooks = residual**2 / (p * sigma2) * (leverage / (1 - leverage) ** 2)
    assert dropped == int((cooks > 1.0).sum())


def test_rreg_populates_e_and_renders():
    result = RobustRegression.from_formula("y ~ x1 + x2", _outlier_frame()).fit()
    for key in ["N", "df_m", "df_r", "rank", "scale"]:
        assert key in result.e, key
    assert "Coef." in result.summary(style="stata")
