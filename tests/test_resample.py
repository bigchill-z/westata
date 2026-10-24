"""Tests for the resampling estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.resample import Bootstrap, Jackknife, Permute


def _frame(n: int = 200, seed: int = 401) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    scale = 0.5 + 0.6 * np.abs(x1)
    y = 1.0 + 2.0 * x1 - 0.8 * x2 + scale * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


# --------------------------------------------------------------- jackknife


def test_jackknife_coefficients_equal_ols():
    frame = _frame()
    jack = Jackknife.from_formula("y ~ x1 + x2", frame).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        np.sort(np.asarray(jack.params, dtype=float)),
        np.sort(plain.params[["x1", "x2", "Intercept"]].to_numpy()),
        rtol=1e-12,
    )


def test_jackknife_matches_stata_on_the_mean():
    """The mean case has no design matrix, so it is the cleanest exact target."""

    rng = np.random.default_rng(401)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * x1 - 0.8 * x2 + (0.5 + 0.6 * np.abs(x1)) * rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

    jack = Jackknife.from_formula("y ~ 1", frame, estimator="mean").fit()
    # Stata: b 0.7552785610919946, V 0.015102300026884398.
    # The mean's variance is the tightest jackknife target available: it matches to
    # 3.7e-9, i.e. the eighth significant digit. The regression variance does not
    # reach this precision (see the module note), which is why this test uses the
    # mean rather than the regression.
    assert float(jack.params.iloc[0]) == pytest.approx(0.7552785610919946, rel=1e-12)
    assert float(np.asarray(jack.cov_params())[0, 0]) == pytest.approx(
        0.015102300026884398, rel=1e-6
    )


def test_jackknife_variance_is_positive_definite():
    jack = Jackknife.from_formula("y ~ x1 + x2", _frame()).fit()
    covariance = np.asarray(jack.cov_params(), dtype=float)
    np.testing.assert_allclose(covariance, covariance.T, rtol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() > 0


def test_jackknife_mse_and_default_forms_differ():
    frame = _frame()
    default = Jackknife.from_formula("y ~ x1 + x2", frame).fit()
    mse = Jackknife.from_formula("y ~ x1 + x2", frame).fit(mse=True)

    a = float(np.asarray(default.cov_params())[0, 0])
    b = float(np.asarray(mse.cov_params())[0, 0])
    # The two formulas are genuinely different estimators, so they must not agree.
    assert a != pytest.approx(b, rel=1e-9)


def test_jackknife_rejects_an_unknown_estimator():
    with pytest.raises(UnsupportedError, match="regress"):
        Jackknife.from_formula("y ~ x1", _frame(), estimator="logit")


def test_jackknife_uses_n_minus_one_degrees_of_freedom():
    """Stata's jackknife tests with t(N-1), not t(N-k).

    Captured from StataNow 19.5 on ``sysuse auto`` (N=74):
    ``jackknife: regress mpg weight`` stores ``e(df_r) = 73``; so does
    ``jackknife: regress mpg weight foreign`` (still 73, not 71) and
    ``jackknife: mean mpg``.  westata used N-k, so t(72)=1.99346 was used where
    Stata uses t(73)=1.99300 and every interval was ~2.3e-4 relatively too wide.
    """

    auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
    n = len(auto)

    for formula, mse in (
        ("mpg ~ weight", False),
        ("mpg ~ weight + foreign", False),
        ("mpg ~ weight", True),
    ):
        result = Jackknife.from_formula(formula, auto).fit(mse=mse)
        assert result.df_resid == float(n - 1)
        assert float(result.e["df_r"]) == float(n - 1)

    mean = Jackknife.from_formula("mpg ~ weight", auto, estimator="mean").fit()
    assert mean.df_resid == float(n - 1)


# --------------------------------------------------------------- bootstrap


def test_bootstrap_point_estimate_equals_the_original():
    """The reported estimate never depends on the resampling."""

    frame = _frame()
    bootstrap = Bootstrap.from_formula("y ~ x1 + x2", frame, reps=50, seed=7).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        np.sort(np.asarray(bootstrap.params, dtype=float)),
        np.sort(plain.params[["x1", "x2", "Intercept"]].to_numpy()),
        rtol=1e-12,
    )


def test_bootstrap_is_reproducible_with_a_seed():
    frame = _frame()
    first = Bootstrap.from_formula("y ~ x1 + x2", frame, reps=40, seed=11).fit()
    second = Bootstrap.from_formula("y ~ x1 + x2", frame, reps=40, seed=11).fit()
    np.testing.assert_allclose(
        np.asarray(first.cov_params()), np.asarray(second.cov_params()), rtol=1e-12
    )


def test_bootstrap_different_seeds_give_different_variances():
    frame = _frame()
    first = Bootstrap.from_formula("y ~ x1 + x2", frame, reps=40, seed=1).fit()
    second = Bootstrap.from_formula("y ~ x1 + x2", frame, reps=40, seed=2).fit()
    assert float(np.asarray(first.cov_params())[0, 0]) != pytest.approx(
        float(np.asarray(second.cov_params())[0, 0]), rel=1e-9
    )


def test_bootstrap_standard_errors_are_the_right_order():
    """A bootstrap SE should be near the analytic one, not merely finite."""

    frame = _frame(n=300)
    bootstrap = Bootstrap.from_formula("y ~ x1 + x2", frame, reps=300, seed=5).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    ratio = float(bootstrap.bse["x1"]) / float(plain.bse["x1"])
    assert 0.5 < ratio < 2.0


def test_bootstrap_rejects_non_positive_reps():
    with pytest.raises(ValueError, match="reps"):
        Bootstrap.from_formula("y ~ x1", _frame(), reps=0)


def test_bootstrap_rejects_an_unsupported_estimator():
    with pytest.raises(UnsupportedError, match="interpreter"):
        Bootstrap.from_formula("y ~ x1", _frame(), estimator="summarize")


def test_bootstrap_uses_a_normal_reference_distribution():
    """Stata's bootstrap prints "Normal-based" and stores no ``e(df_r)``.

    Captured from StataNow 19.5 on ``sysuse auto``:
    ``bootstrap, reps(100) seed(20241024) nodots: regress mpg weight`` has
    e(df_r) absent, and its 95% CI for ``_cons`` is [35.29428573, 43.58628133],
    which is ``b ± 1.959964 se`` -- not the t(72) interval
    [35.2234, 43.6571] westata used to report (1.7% wider).
    """

    auto = pd.read_stata("tests/data/auto.dta", convert_categoricals=False)
    result = Bootstrap.from_formula(
        "mpg ~ weight", auto, reps=100, seed=20241024).fit()

    assert result.dist == "z"
    lower, upper = result.conf_int().loc["Intercept"].tolist()
    assert lower == pytest.approx(35.29428573, rel=1e-7)
    assert upper == pytest.approx(43.58628133, rel=1e-7)


# ----------------------------------------------------------------- permute


def test_permute_produces_a_pvalue_in_range():
    frame = _frame(n=120)
    result = Permute.from_formula(
        "y ~ x1 + x2", frame, permute="x1", reps=50, seed=3
    ).fit()
    assert 0.0 <= float(result.e["pvalue_permuted"]) <= 1.0


def test_permute_finds_a_real_effect():
    """x1 genuinely drives y here, so nearly every permutation is less extreme."""

    frame = _frame(n=120)
    result = Permute.from_formula(
        "y ~ x1 + x2", frame, permute="x1", reps=200, seed=3
    ).fit()
    assert float(result.e["pvalue_permuted"]) < 0.05


def test_permute_rejects_a_non_regressor():
    with pytest.raises(ConsistencyError, match="not a regressor"):
        Permute.from_formula("y ~ x1", _frame(), permute="z", reps=10)


def test_permute_rejects_non_positive_reps():
    with pytest.raises(ValueError, match="reps"):
        Permute.from_formula("y ~ x1", _frame(), permute="x1", reps=0)
