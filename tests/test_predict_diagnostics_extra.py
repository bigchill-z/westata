"""Tests for the remaining ``predict`` influence diagnostics.

Formulas from ``regres_p.ado:158-191`` (Stata's own implementation):

    dfits    = rstudent * sqrt(h/(1-h))
    welsch   = dfits * sqrt((N-1)/(1-h))
    covratio = (((N - df_m - t^2 - 1)/(N - df_m - 2))^(df_m+1)) / (1-h)
               where t = resid/(rmse*sqrt(1-h))

and ``stdr`` is the "Std. err. of the residual" (``fpredict.ado:62``):
``rmse * sqrt(1-h)``.

``welsch`` is checked a second way -- against the ratio it must bear to
``dfits`` -- because the two share an intermediate expression and a shared error
would cancel in a direct comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length"]].dropna()
    return frame, ws.OLS.from_formula("mpg ~ weight + length", frame).fit()


def _pieces(result):
    matrix, _ = result.design_matrix
    matrix = np.asarray(matrix, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    n, k = matrix.shape
    rss = float(resid @ resid)
    df_resid = n - k
    s2 = rss / df_resid
    hat = np.einsum(
        "ij,jk,ik->i", matrix, np.linalg.inv(matrix.T @ matrix), matrix
    )
    return matrix, resid, n, k, rss, df_resid, s2, hat


def _rstudent(resid, hat, rss, df_resid):
    s2_j = (rss - resid**2 / (1 - hat)) / (df_resid - 1)
    return (resid / (1 - hat)) / np.sqrt(s2_j)


def test_stdr_is_rmse_times_sqrt_one_minus_h(fit) -> None:
    _, result = fit
    _, _, _, _, _, df_resid, s2, hat = _pieces(result)
    rmse = np.sqrt(s2)

    expected = rmse * np.sqrt(1 - hat)

    np.testing.assert_allclose(
        result.predict(kind="stdr"), expected, rtol=1e-12, atol=1e-14
    )


def test_dfits_is_rstudent_times_the_leverage_scaling(fit) -> None:
    _, result = fit
    _, resid, _, _, rss, df_resid, _, hat = _pieces(result)
    student = _rstudent(resid, hat, rss, df_resid)

    expected = student * np.sqrt(hat / (1 - hat))

    np.testing.assert_allclose(
        result.predict(kind="dfits"), expected, rtol=1e-10, atol=1e-12
    )


def test_welsch_is_dfits_times_the_extra_scaling(fit) -> None:
    _, result = fit
    _, resid, n, _, rss, df_resid, _, hat = _pieces(result)
    student = _rstudent(resid, hat, rss, df_resid)

    expected = student * np.sqrt(hat / (1 - hat)) * np.sqrt((n - 1) / (1 - hat))

    np.testing.assert_allclose(
        result.predict(kind="welsch"), expected, rtol=1e-10, atol=1e-12
    )


def test_welsch_to_dfits_ratio_is_the_documented_factor(fit) -> None:
    """A second, independent check that does not share the dfits expression."""

    _, result = fit
    _, _, n, _, _, _, _, hat = _pieces(result)

    ratio = result.predict(kind="welsch") / result.predict(kind="dfits")
    expected = np.sqrt((n - 1) / (1 - hat))

    np.testing.assert_allclose(ratio, expected, rtol=1e-10, atol=1e-12)


def test_covratio_matches_the_ado_formula(fit) -> None:
    _, result = fit
    _, resid, n, _, _, _, s2, hat = _pieces(result)
    rmse = np.sqrt(s2)
    df_model = float(result.df_model)

    t_stat = resid / (rmse * np.sqrt(1 - hat))
    ratio = (n - df_model - t_stat**2 - 1) / (n - df_model - 2)
    expected = np.power(ratio, df_model + 1) / (1 - hat)

    np.testing.assert_allclose(
        result.predict(kind="covratio"), expected, rtol=1e-10, atol=1e-12
    )


def test_covratio_floats_around_one(fit) -> None:
    """COVRATIO is a ratio of covariance determinants, so it centres on 1."""

    values = fit[1].predict(kind="covratio")

    assert np.all(values > 0)
    assert 0.5 < float(np.mean(values)) < 2.0


def test_all_four_are_refused_under_a_robust_vce(fit) -> None:
    frame, _ = fit
    robust = ws.OLS.from_formula("mpg ~ weight + length", frame).fit(cov_type="HC1")

    for kind in ("stdr", "dfits", "welsch", "covratio"):
        with pytest.raises(ws.UnsupportedError, match="vce\\(ols\\)"):
            robust.predict(kind=kind)


def test_supported_types_register_the_four() -> None:
    from westata.postestimation.predict import supported_types

    types = supported_types("regress")

    for kind in ("stdr", "dfits", "welsch", "covratio"):
        assert kind in types, kind


def test_shape_matches_the_estimation_sample(fit) -> None:
    _, result = fit

    for kind in ("stdr", "dfits", "welsch", "covratio"):
        values = result.predict(kind=kind)
        assert values.shape == (result.nobs,), kind
        assert np.all(np.isfinite(values)), kind
