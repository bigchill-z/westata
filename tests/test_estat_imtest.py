"""Tests for Stata's ``estat imtest`` (Cameron & Trivedi IM-test).

Transcribed from ``imtest.ado`` v2.1.5:

    s2     = e(rss)/e(N)                       # ML residual variance
    rhs2   = centred x_i*x_j, j <= i
    white  : N * R2 of (res^2 ~ rhs + rhs2)    # centred R2
    hetero : N*(1 - rss/uss), y = res^2 - s2,  regress y ~ rhs + rhs2
    skew   : N*(1 - rss/uss), y = res^3 - 3*s2*res,          y ~ rhs
    kurt   : N*(1 - rss/uss), y = res^4 - 6*s2*res^2 + 3*s2^2, y ~ const
    total  = hetero + skew + kurt

``uss`` is the **uncentred** sum of squares of the transformed residual.

Besides the direct transcription, two structural facts are asserted:

* the heteroskedasticity component must equal White's original statistic --
  they are the same test expressed differently, so the equality holds to machine
  precision and would break if either auxiliary regression were built wrongly;
* the total and its df must be the sums of the three components.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws


@pytest.fixture()
def fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length"]].dropna()
    return frame, ws.OLS.from_formula("mpg ~ weight + length", frame).fit()


def _setup(result):
    matrix, _ = result.design_matrix
    matrix = np.asarray(matrix, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    n = resid.size
    slopes = matrix[:, 1:]  # drop the constant
    centred = slopes - slopes.mean(axis=0)
    columns = [
        centred[:, i] * centred[:, j]
        for i, j in combinations_with_replacement(range(centred.shape[1]), 2)
    ]
    rhs2 = np.column_stack(columns)
    s2 = float(resid @ resid) / n
    return matrix, resid, n, slopes, rhs2, s2


def _uncentred_chi2(y, design):
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    rss = float(residual @ residual)
    uss = float(y @ y)
    return y.size * (1.0 - rss / uss), design.shape[1] - 1


def test_white_matches_the_centred_r2_formula(fit) -> None:
    _, result = fit
    _, resid, n, slopes, rhs2, _ = _setup(result)

    target = resid**2
    design = np.column_stack([np.ones(n), slopes, rhs2])
    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ beta
    rss = float(residual @ residual)
    centred = target - target.mean()
    tss = float(centred @ centred)
    expected = n * (1.0 - rss / tss)

    rendered = result.estat_imtest(white=True)

    assert rendered.white == pytest.approx(expected, rel=1e-10)
    assert rendered.white_df == design.shape[1] - 1


def test_heteroskedasticity_component(fit) -> None:
    _, result = fit
    _, resid, n, slopes, rhs2, s2 = _setup(result)

    y = resid**2 - s2
    expected, df = _uncentred_chi2(y, np.column_stack([np.ones(n), slopes, rhs2]))

    rendered = result.estat_imtest()
    assert rendered.heteroskedasticity == pytest.approx(expected, rel=1e-10)
    assert rendered.df_heteroskedasticity == df


def test_skewness_component_regresses_on_the_plain_regressors(fit) -> None:
    _, result = fit
    _, resid, n, slopes, _, s2 = _setup(result)

    y = resid**3 - 3.0 * s2 * resid
    expected, df = _uncentred_chi2(y, np.column_stack([np.ones(n), slopes]))

    rendered = result.estat_imtest()
    assert rendered.skewness == pytest.approx(expected, rel=1e-10)
    assert rendered.df_skewness == df


def test_kurtosis_component_regresses_on_a_constant_only(fit) -> None:
    _, result = fit
    _, resid, n, _, _, s2 = _setup(result)

    y = resid**4 - 6.0 * s2 * resid**2 + 3.0 * s2**2
    expected, _ = _uncentred_chi2(y, np.ones((n, 1)))

    rendered = result.estat_imtest()
    assert rendered.kurtosis == pytest.approx(expected, rel=1e-10)
    assert rendered.df_kurtosis == 1.0


def test_heteroskedasticity_equals_whites_statistic(fit) -> None:
    """Same test written two ways; the equality is a strong internal check.

    If either auxiliary regression were mis-specified the two would diverge.
    """

    rendered = fit[1].estat_imtest(white=True)

    assert rendered.heteroskedasticity == pytest.approx(
        rendered.white, rel=1e-8
    )


def test_total_is_the_sum_of_the_components(fit) -> None:
    rendered = fit[1].estat_imtest()

    assert rendered.total == pytest.approx(
        rendered.heteroskedasticity + rendered.skewness + rendered.kurtosis
    )
    assert rendered.df_total == (
        rendered.df_heteroskedasticity
        + rendered.df_skewness
        + rendered.df_kurtosis
    )


def test_s2_uses_the_ml_residual_variance(fit) -> None:
    """``s2 = e(rss)/e(N)``, not ``rss/(N-K)``.

    The skewness and kurtosis components are sensitive to this scale, so a
    ``ddof`` slip would move both.
    """

    _, result = fit
    _, resid, n, _, _, _ = _setup(result)

    ml_s2 = float(resid @ resid) / n
    df_s2 = float(resid @ resid) / (n - len(result.params))
    assert not np.isclose(ml_s2, df_s2)

    # Recompute skewness with the df-adjusted variance and confirm it differs.
    _, _, n2, slopes, _, _ = _setup(result)
    y_ml = resid**3 - 3.0 * ml_s2 * resid
    y_df = resid**3 - 3.0 * df_s2 * resid
    ml_chi2, _ = _uncentred_chi2(y_ml, np.column_stack([np.ones(n2), slopes]))
    df_chi2, _ = _uncentred_chi2(y_df, np.column_stack([np.ones(n2), slopes]))

    assert not np.isclose(ml_chi2, df_chi2, rtol=1e-6)
    assert result.estat_imtest().skewness == pytest.approx(ml_chi2, rel=1e-10)


def test_heteroskedastic_data_is_detected() -> None:
    rng = np.random.default_rng(3)
    n = 600
    x = rng.uniform(1, 10, size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=x * 0.4, size=n)  # variance grows with x
    frame = pd.DataFrame({"y": y, "x": x})

    rendered = ws.OLS.from_formula("y ~ x", frame).fit().estat_imtest()

    p = float(stats.chi2.sf(rendered.heteroskedasticity, rendered.df_heteroskedasticity))
    assert p < 0.01


def test_homoskedastic_data_is_not_flagged() -> None:
    rng = np.random.default_rng(3)
    n = 600
    x = rng.uniform(1, 10, size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame({"y": y, "x": x})

    rendered = ws.OLS.from_formula("y ~ x", frame).fit().estat_imtest()

    p = float(stats.chi2.sf(rendered.heteroskedasticity, rendered.df_heteroskedasticity))
    assert p > 0.05


def test_logit_is_refused(fit) -> None:
    frame, _ = fit
    df = pd.read_stata("tests/data/auto.dta")
    frame = frame.assign(dom=(df.loc[frame.index, "foreign"] == "Foreign").astype(int))
    logit = ws.Logit.from_formula("dom ~ weight", frame).fit()

    with pytest.raises(NotImplementedError, match="after regress"):
        logit.estat_imtest()


def test_robust_vce_is_refused(fit) -> None:
    frame, _ = fit
    robust = ws.OLS.from_formula("mpg ~ weight + length", frame).fit(cov_type="HC1")

    with pytest.raises(ValueError, match="weighted or clustered"):
        robust.estat_imtest()


def test_to_dict_uses_stata_r_names(fit) -> None:
    payload = fit[1].estat_imtest().to_dict()

    assert set(payload) == {
        "chi2_h",
        "df_h",
        "chi2_s",
        "df_s",
        "chi2_k",
        "df_k",
        "chi2_t",
        "df_t",
    }
    with_white = fit[1].estat_imtest(white=True).to_dict()
    assert "chi2" in with_white and "df" in with_white


def test_output_names_the_three_components(fit) -> None:
    text = str(fit[1].estat_imtest())

    assert "Cameron & Trivedi" in text
    assert "Heteroskedasticity" in text
    assert "Skewness" in text
    assert "Kurtosis" in text
    assert "Total" in text
