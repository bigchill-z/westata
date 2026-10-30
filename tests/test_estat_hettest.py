"""Tests for Stata's ``estat hettest`` (Breusch-Pagan / Cook-Weisberg).

All statistics are checked against an independent transcription of
``hettest.ado`` v3.4.4 so the implementation cannot pass by agreeing with itself.

The detail that decides correctness is which statistic each variant reports:

* ``normal`` (the default) -- ``e(mss)/2`` from the auxiliary regression of the
  standardised squared residual on the regressors
* ``iid``   -- ``e(N)*e(r2)``
* ``fstat`` -- the auxiliary regression's ``F``

On a given fit the default and the ``iid`` variant give *different* numbers
(22.53 vs 9.54 on the auto data), so substituting one for the other is silent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.postestimation.hettest import estat_hettest

_CONSTANT = {"intercept", "const", "_cons"}


@pytest.fixture()
def fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length", "displacement"]].dropna()
    return frame, ws.OLS.from_formula(
        "mpg ~ weight + length + displacement", frame
    ).fit()


def _auxiliary(result):
    """Independent auxiliary regression of the standardised squared residual."""

    matrix, names = result.design_matrix
    matrix = np.asarray(matrix, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    n = resid.size

    scale = float(resid @ resid) / n
    squared = resid**2 / scale

    slopes = [j for j, name in enumerate(names) if name.lower() not in _CONSTANT]
    design = np.column_stack([np.ones(n), matrix[:, slopes]])
    beta, *_ = np.linalg.lstsq(design, squared, rcond=None)
    residual = squared - design @ beta
    rss = float(residual @ residual)
    centred = squared - squared.mean()
    tss = float(centred @ centred)
    mss = tss - rss
    return {
        "r2": mss / tss,
        "mss": mss,
        "rss": rss,
        "df_m": len(slopes),
        "n": n,
    }


def test_default_uses_mss_over_two(fit) -> None:
    """``hettest.ado`` default branch: ``return scalar chi2 = e(mss)/2``."""

    _, result = fit
    aux = _auxiliary(result)

    rendered = result.estat_hettest()

    assert rendered.method == "normal"
    assert rendered.statistic == pytest.approx(aux["mss"] / 2.0, rel=1e-10)


def test_default_is_not_n_times_r2(fit) -> None:
    """The ``iid`` formula must not be used for the default.

    ``N*R2`` and ``MSS/2`` coincide only when the auxiliary regression is
    uncentred; here they differ by a factor of more than two.
    """

    _, result = fit
    aux = _auxiliary(result)

    rendered = result.estat_hettest()
    iid_value = aux["n"] * aux["r2"]

    assert rendered.statistic != pytest.approx(iid_value, rel=1e-6)


def test_iid_uses_n_times_r2(fit) -> None:
    _, result = fit
    aux = _auxiliary(result)

    rendered = result.estat_hettest(method="iid")

    assert rendered.method == "iid"
    assert rendered.statistic == pytest.approx(aux["n"] * aux["r2"], rel=1e-10)


def test_fstat_reports_the_auxiliary_f(fit) -> None:
    _, result = fit
    aux = _auxiliary(result)
    df_r = aux["n"] - aux["df_m"] - 1

    rendered = result.estat_hettest(method="fstat")

    expected = (aux["mss"] / aux["df_m"]) / (aux["rss"] / df_r)
    assert rendered.method == "fstat"
    assert rendered.statistic == pytest.approx(expected, rel=1e-10)
    assert rendered.df_denom == df_r
    assert rendered.pvalue == pytest.approx(
        float(stats.f.sf(expected, aux["df_m"], df_r)), rel=1e-12
    )


def test_degrees_of_freedom_is_the_slope_count(fit) -> None:
    _, result = fit
    aux = _auxiliary(result)

    assert result.estat_hettest().df == aux["df_m"]


def test_pvalue_is_upper_tail_chi_square(fit) -> None:
    _, result = fit
    rendered = result.estat_hettest()

    assert rendered.pvalue == pytest.approx(
        float(stats.chi2.sf(rendered.statistic, rendered.df)), rel=1e-12
    )
    assert 0.0 <= rendered.pvalue <= 1.0


def test_empty_varlist_uses_the_fitted_values(fit) -> None:
    """``hettest.ado``: no varlist -> ``_predict z`` then regress on ``z``."""

    _, result = fit
    matrix, _ = result.design_matrix
    matrix = np.asarray(matrix, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    n = resid.size

    squared = resid**2 / (float(resid @ resid) / n)
    yhat = matrix @ np.asarray(result.params, dtype=float)
    design = np.column_stack([np.ones(n), yhat])
    beta, *_ = np.linalg.lstsq(design, squared, rcond=None)
    residual = squared - design @ beta
    rss = float(residual @ residual)
    centred = squared - squared.mean()
    tss = float(centred @ centred)
    expected = (tss - rss) / 2.0

    rendered = result.estat_hettest(variables=[])

    assert rendered.variables == []
    assert rendered.statistic == pytest.approx(expected, rel=1e-10)


def test_explicit_variables_subset_the_auxiliary_regression(fit) -> None:
    frame, result = fit
    full = result.estat_hettest()
    subset = result.estat_hettest(variables=["weight"])

    assert subset.variables == ["weight"]
    assert subset.df == 1
    assert subset.statistic != pytest.approx(full.statistic, rel=1e-6)


def test_unknown_variable_is_rejected(fit) -> None:
    _, result = fit

    with pytest.raises(ValueError, match="unknown regressor"):
        result.estat_hettest(variables=["nope"])


# --------------------------------------------------------------- refusals
def test_robust_vce_is_refused(fit) -> None:
    frame, _ = fit
    fit_robust = ws.OLS.from_formula("mpg ~ weight + length", frame).fit(
        cov_type="HC1"
    )

    with pytest.raises(ValueError, match="robust"):
        fit_robust.estat_hettest()


def test_clustered_vce_is_refused(fit) -> None:
    frame, _ = fit
    groups = (np.arange(len(frame)) % 7).astype(int)
    fit_cluster = ws.OLS.from_formula("mpg ~ weight + length", frame).fit(
        cov_type="cluster", cov_kwds={"groups": groups}
    )

    with pytest.raises(ValueError, match="robust or clustered"):
        fit_cluster.estat_hettest()


def test_nonrobust_vce_is_accepted(fit) -> None:
    _, result = fit
    assert result.cov_type == "nonrobust"
    assert np.isfinite(result.estat_hettest().statistic)


def test_unknown_method_is_rejected(fit) -> None:
    _, result = fit

    with pytest.raises(ValueError, match="method must be one of"):
        result.estat_hettest(method="bogus")


# ------------------------------------------------------------ presentation
def test_to_dict_matches_the_variant(fit) -> None:
    _, result = fit

    normal = result.estat_hettest().to_dict()
    assert set(normal) == {"chi2", "df", "p"}

    f_stat = result.estat_hettest(method="fstat").to_dict()
    assert set(f_stat) == {"F", "df_m", "df_r", "p"}


def test_output_mentions_both_assumptions(fit) -> None:
    _, result = fit

    assert "Normal error terms" in str(result.estat_hettest())
    assert "i.i.d. error terms" in str(result.estat_hettest(method="iid"))
    assert "Constant variance" in str(result.estat_hettest())


def test_module_function_matches_the_method(fit) -> None:
    _, result = fit

    assert estat_hettest(result).to_dict() == result.estat_hettest().to_dict()
