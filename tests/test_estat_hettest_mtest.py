"""Tests for ``estat hettest, mtest()`` -- multiple-testing adjustment.

Transcribed from ``hettest.ado:160-190`` and ``_mtest.ado:189-296``.

``hettest`` with ``mtest()`` runs ONE auxiliary regression per regressor and
adjusts the resulting family of p-values:

    bonferroni: adjp[i] = min(1, m * p[i])
    sidak:      adjp[i] = 1 - (1 - p[i])^m
    holm:       step-down with multiplier m, m-1, ..., 1 in ascending p order,
                floored at the PREVIOUS ADJUSTED value (not the previous raw p)

The Holm floor is the part that is easy to omit, because it only binds when the
raw p-values are close together.  One test constructs exactly that case, since a
plain step-down on well-separated p-values looks correct without it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.mtest import MTEST_METHODS, adjust_pvalues


@pytest.fixture()
def fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length", "displacement"]].dropna()
    return frame, ws.OLS.from_formula(
        "mpg ~ weight + length + displacement", frame
    ).fit()


# ------------------------------------------------------------- adjustment maths
def test_bonferroni_matches_the_ado_formula() -> None:
    p = np.array([0.001, 0.02, 0.3, 0.9])

    assert np.allclose(adjust_pvalues(p, "bonferroni"), np.minimum(1.0, p.size * p))


def test_sidak_matches_the_ado_formula() -> None:
    p = np.array([0.001, 0.02, 0.3, 0.9])

    assert np.allclose(adjust_pvalues(p, "sidak"), 1.0 - (1.0 - p) ** p.size)


def test_holm_matches_the_step_down() -> None:
    p = np.array([0.001, 0.02, 0.3, 0.9])
    # Already ascending.
    expected = np.minimum(1.0, p * np.array([4, 3, 2, 1]))

    assert np.allclose(adjust_pvalues(p, "holm"), expected)


def test_holm_monotonicity_floor_actually_binds() -> None:
    """Without the floor the adjustment would DECREASE, which is invalid.

    p = [0.01, 0.013, 0.02] with m = 3 gives naive steps [0.03, 0.026, 0.02].
    ``_mtest_holm`` floors each value at the previous adjusted one, yielding
    [0.03, 0.03, 0.03].
    """

    p = np.array([0.01, 0.013, 0.02])
    naive = p * np.array([3, 2, 1])
    assert np.all(np.diff(naive) < 0), "the raw steps must decrease here"

    expected = np.maximum.accumulate(naive)

    assert np.allclose(adjust_pvalues(p, "holm"), expected)
    # And demonstrably not the naive step-down.
    assert not np.allclose(adjust_pvalues(p, "holm"), naive)


def test_noadjust_returns_the_raw_values() -> None:
    p = np.array([0.04, 0.5, 0.001])

    assert np.allclose(adjust_pvalues(p, "noadjust"), p)


def test_adjustments_preserve_test_order() -> None:
    p = np.array([0.5, 0.001, 0.2, 0.05])
    order = np.argsort(p)

    for method in ("bonferroni", "holm", "sidak"):
        adjusted = adjust_pvalues(p, method)
        assert np.allclose(adjusted[order], adjust_pvalues(p[order], method)), method


def test_unknown_method_is_refused() -> None:
    with pytest.raises(ValueError, match="not supported"):
        adjust_pvalues(np.array([0.1, 0.2]), "fdr")


def test_the_four_methods_are_the_ado_ones() -> None:
    assert set(MTEST_METHODS) == {"bonferroni", "holm", "sidak", "noadjust"}


# ------------------------------------------------------------- end-to-end
def test_per_variable_statistics_do_not_depend_on_the_method(fit) -> None:
    _, result = fit
    raw = result.estat_hettest(mtest="noadjust")

    for method in ("bonferroni", "holm", "sidak"):
        adjusted = result.estat_hettest(mtest=method)
        assert np.allclose(
            adjusted.table["chi2"].to_numpy(), raw.table["chi2"].to_numpy()
        )
        assert np.allclose(
            adjusted.table["p"].to_numpy(), raw.table["p"].to_numpy()
        )


def test_adjusted_values_match_the_standalone_function(fit) -> None:
    _, result = fit
    raw_p = result.estat_hettest(mtest="noadjust").table["p"].to_numpy()

    for method in ("bonferroni", "holm", "sidak"):
        got = result.estat_hettest(mtest=method).table["adjustp"].to_numpy()
        assert np.allclose(got, adjust_pvalues(raw_p, method)), method


def test_each_variable_gets_its_own_auxiliary_regression(fit) -> None:
    """The statistic is that ONE regressor's ``e(mss)/2``, not the full model's."""

    frame, result = fit
    table = result.estat_hettest(mtest="noadjust").table

    residuals = np.asarray(result.resid, dtype=float)
    scale = float(residuals @ residuals) / residuals.size
    squared = residuals**2 / scale

    for name in ("weight", "length", "displacement"):
        column = frame[name].to_numpy(float)
        X = np.column_stack([np.ones(column.size), column])
        beta, *_ = np.linalg.lstsq(X, squared, rcond=None)
        resid = squared - X @ beta
        rss = float(resid @ resid)
        centred = squared - squared.mean()
        mss = float(centred @ centred) - rss
        assert table.loc[name, "chi2"] == pytest.approx(mss / 2.0, rel=1e-10), name


def test_degrees_of_freedom_are_one_per_single_regressor(fit) -> None:
    table = fit[1].estat_hettest(mtest="noadjust").table

    assert np.all(table["df"].to_numpy() == 1.0)


def test_pvalue_is_the_chi2_upper_tail(fit) -> None:
    from scipy import stats

    table = fit[1].estat_hettest(mtest="noadjust").table

    for name, row in table.iterrows():
        assert row["p"] == pytest.approx(
            float(stats.chi2.sf(row["chi2"], row["df"])), rel=1e-12
        ), name


def test_r_matrix_shape(fit) -> None:
    """``r(mtest)`` is nvar x 3, gaining a fourth column when adjusted."""

    _, result = fit

    assert result.estat_hettest(mtest="noadjust").to_matrix().shape == (3, 3)
    assert result.estat_hettest(mtest="holm").to_matrix().shape == (3, 4)


def test_iid_with_mtest_is_refused(fit) -> None:
    """``hettest.ado:49-53`` rejects this combination."""

    _, result = fit

    with pytest.raises(ValueError, match="iid may not be specified"):
        result.estat_hettest(method="iid", mtest="holm")


def test_fstat_with_mtest_is_refused(fit) -> None:
    """``hettest.ado:54-58`` rejects this combination."""

    _, result = fit

    with pytest.raises(ValueError, match="fstat may not be specified"):
        result.estat_hettest(method="fstat", mtest="holm")


def test_unknown_mtest_method_is_refused(fit) -> None:
    _, result = fit

    with pytest.raises(ValueError, match="not supported"):
        result.estat_hettest(mtest="fdr")


def test_robust_vce_is_still_refused_with_mtest(fit) -> None:
    frame, _ = fit
    robust = ws.OLS.from_formula(
        "mpg ~ weight + length + displacement", frame
    ).fit(cov_type="HC1")

    with pytest.raises(ValueError, match="robust"):
        robust.estat_hettest(mtest="holm")


def test_output_lists_every_variable_and_the_adjustment(fit) -> None:
    text = str(fit[1].estat_hettest(mtest="holm"))

    assert "holm" in text
    for name in ("weight", "length", "displacement"):
        assert name in text
    assert "adjustp" in text
