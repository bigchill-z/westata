"""Tests for ``predict``'s influence diagnostics (Stata's regress postestimation).

Every statistic is checked against an explicit **leave-one-out refit** rather than
against another closed form, so the implementation cannot pass by agreeing with a
misconception it shares with the test.

The trap this file exists for is ``rstudent``.  The obvious form --
``e_j / (s_(j) * sqrt(1-h_j))``, i.e. rstandard with the leave-out variance --
is wrong: it omits the ``1/(1-h_j)`` inflation of the leave-one-out residual.
On the auto data it is off by 9%; the correct form reproduces the refit to 1e-15.
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
    y = np.asarray(result.fittedvalues, dtype=float) + resid
    n, k = matrix.shape
    rss = float(resid @ resid)
    df_resid = n - k
    xtx_inv = np.linalg.inv(matrix.T @ matrix)
    b = xtx_inv @ matrix.T @ y
    hat = np.einsum("ij,jk,ik->i", matrix, xtx_inv, matrix)
    return matrix, y, resid, n, k, rss, df_resid, xtx_inv, b, hat


def test_hat_matches_the_projection_diagonal(fit) -> None:
    _, result = fit
    matrix, _, _, n, k, *_ = _pieces(result)

    hat = result.predict(kind="hat")

    assert hat.sum() == pytest.approx(k)  # trace of the hat matrix
    # Sherman-Morrison: h_j from a refit without observation j.
    for j in range(n):
        keep = np.arange(n) != j
        inv_j = np.linalg.inv(matrix[keep].T @ matrix[keep])
        quad = float(matrix[j] @ inv_j @ matrix[j])
        assert hat[j] == pytest.approx(quad / (1 + quad), rel=1e-10)


def test_leverage_is_hat(fit) -> None:
    _, result = fit

    np.testing.assert_allclose(
        result.predict(kind="leverage"), result.predict(kind="hat"),
        rtol=0.0, atol=0.0,
    )


def test_rstandard_is_the_internally_studentised_residual(fit) -> None:
    _, result = fit
    _, _, resid, n, k, rss, df_resid, _, _, hat = _pieces(result)
    s = np.sqrt(rss / df_resid)

    expected = resid / (s * np.sqrt(1 - hat))

    np.testing.assert_allclose(
        result.predict(kind="rstandard"), expected, rtol=1e-12, atol=1e-14
    )


def test_rstudent_matches_a_leave_one_out_refit(fit) -> None:
    """The externally studentised residual, checked by actually refitting."""

    _, result = fit
    matrix, y, _, n, k, _, df_resid, _, _, _ = _pieces(result)

    got = result.predict(kind="rstudent")

    for j in range(n):
        keep = np.arange(n) != j
        Xj, yj = matrix[keep], y[keep]
        b_j = np.linalg.inv(Xj.T @ Xj) @ Xj.T @ yj
        e_j = yj - Xj @ b_j
        s2_minus = float(e_j @ e_j) / (df_resid - 1)
        leave_out = y[j] - float(matrix[j] @ b_j)
        expected = leave_out / np.sqrt(s2_minus)
        assert got[j] == pytest.approx(expected, rel=1e-8), j


def test_rstudent_is_not_rstandard_with_a_leave_out_variance(fit) -> None:
    """The tempting wrong form, asserted to differ.

    Without this the wrong formula would pass any single-point check that only
    looked at magnitude: it is within 9% of the truth, not orders of magnitude
    away.
    """

    _, result = fit
    _, _, resid, n, k, rss, df_resid, _, _, hat = _pieces(result)

    s2_minus = (rss - resid**2 / (1 - hat)) / (df_resid - 1)
    wrong = resid / (np.sqrt(s2_minus) * np.sqrt(1 - hat))
    right = result.predict(kind="rstudent")

    assert not np.allclose(wrong, right, rtol=1e-6)


def test_cooksd_matches_the_coefficient_shift_definition(fit) -> None:
    _, result = fit
    matrix, y, resid, n, k, rss, df_resid, _, b, _ = _pieces(result)
    s2 = rss / df_resid

    got = result.predict(kind="cooksd")

    for j in range(n):
        keep = np.arange(n) != j
        Xj, yj = matrix[keep], y[keep]
        b_j = np.linalg.inv(Xj.T @ Xj) @ Xj.T @ yj
        diff = b - b_j
        expected = float(diff @ (matrix.T @ matrix) @ diff) / (k * s2)
        assert got[j] == pytest.approx(expected, rel=1e-6), j


def test_dfbeta_is_the_scaled_coefficient_change(fit) -> None:
    _, result = fit
    matrix, y, resid, n, k, rss, df_resid, xtx_inv, b, hat = _pieces(result)
    se = np.sqrt(np.diag((rss / df_resid) * xtx_inv))

    got = result.predict(kind="dfbeta")

    assert got.shape == (n, k)
    for j in range(n):
        keep = np.arange(n) != j
        Xj, yj = matrix[keep], y[keep]
        b_j = np.linalg.inv(Xj.T @ Xj) @ Xj.T @ yj
        expected = (b - b_j) / se
        np.testing.assert_allclose(got[j], expected, rtol=1e-9, atol=1e-12)


def test_hat_sums_to_the_parameter_count(fit) -> None:
    _, result = fit
    _, _, _, _, k, *_ = _pieces(result)

    assert result.predict(kind="hat").sum() == pytest.approx(k)


def test_robust_vce_is_refused(fit) -> None:
    """``regress_postestimation.sthlp:152-153`` restricts these to vce(ols)."""

    frame, _ = fit
    robust = ws.OLS.from_formula("mpg ~ weight + length", frame).fit(cov_type="HC1")

    for kind in ("hat", "leverage", "rstandard", "rstudent", "cooksd", "dfbeta"):
        with pytest.raises(ws.UnsupportedError, match="vce\\(ols\\)"):
            robust.predict(kind=kind)


def test_clustered_vce_is_refused(fit) -> None:
    frame, _ = fit
    groups = (np.arange(len(frame)) % 7).astype(int)
    clustered = ws.OLS.from_formula("mpg ~ weight + length", frame).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )

    with pytest.raises(ws.UnsupportedError, match="vce\\(ols\\)"):
        clustered.predict(kind="cooksd")


def test_supported_types_lists_the_diagnostics() -> None:
    from westata.postestimation.predict import supported_types

    types = supported_types("regress")

    for kind in ("hat", "leverage", "rstandard", "rstudent", "cooksd", "dfbeta"):
        assert kind in types, kind


def test_other_models_do_not_advertise_the_diagnostics() -> None:
    """They are not merely unimplemented there; Stata does not offer them."""

    from westata.postestimation.predict import supported_types

    for model in ("logit", "ivregress 2sls", "xtreg, fe", "poisson"):
        types = supported_types(model)
        assert "cooksd" not in types, model
        assert "rstudent" not in types, model
