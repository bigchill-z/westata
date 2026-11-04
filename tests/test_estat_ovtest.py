"""Tests for Stata's ``estat ovtest`` (Ramsey RESET test).

Transcribed from ``ovtest.ado`` v2.7.0.  The default path regresses the response
on the original regressors plus the 2nd, 3rd and 4th powers of the **rescaled**
fitted values, then tests those three jointly.

The rescaling to ``[0, 1]`` is load-bearing: on the auto data the rescaled
statistic is 1.8374159 while the unrescaled one is 16672.55.  A test therefore
asserts the two differ, so a future "simplification" that drops the rescale
fails immediately instead of returning a large-but-plausible F.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws
from westata.postestimation.ovtest import estat_ovtest


@pytest.fixture()
def fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length", "displacement"]].dropna()
    return frame, ws.OLS.from_formula("mpg ~ weight + length", frame).fit()


def _joint_f(y, design, tested):
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coefficients
    rss = float(resid @ resid)
    nobs, k = design.shape
    df_denom = nobs - k
    cov = rss / df_denom * np.linalg.pinv(design.T @ design)
    sub = list(tested)
    b = coefficients[sub]
    v = cov[np.ix_(sub, sub)]
    wald = float(b @ np.linalg.solve(v, b))
    return wald / len(sub), len(sub), df_denom, rss


def test_default_uses_powers_of_the_rescaled_fitted_values(fit) -> None:
    _, result = fit
    matrix, _ = result.design_matrix
    matrix = np.asarray(matrix, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    yhat = np.asarray(result.fittedvalues, dtype=float)
    y = yhat + resid

    scaled = (yhat - yhat.min()) / (yhat.max() - yhat.min())
    extra = np.column_stack([scaled**2, scaled**3, scaled**4])
    design = np.column_stack([matrix, extra])
    expected, df1, df2, _ = _joint_f(y, design, range(matrix.shape[1], design.shape[1]))

    rendered = result.estat_ovtest()

    assert rendered.statistic == pytest.approx(expected, rel=1e-10)
    assert rendered.df == df1
    assert rendered.df_denom == df2


def test_rescaling_changes_the_statistic(fit) -> None:
    """Without the [0,1] rescale the F is ~9000x larger; both must not agree."""

    _, result = fit
    matrix, _ = result.design_matrix
    matrix = np.asarray(matrix, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    yhat = np.asarray(result.fittedvalues, dtype=float)
    y = yhat + resid

    extra_raw = np.column_stack([yhat**2, yhat**3, yhat**4])
    design_raw = np.column_stack([matrix, extra_raw])
    raw_f, *_ = _joint_f(y, design_raw, range(matrix.shape[1], design_raw.shape[1]))

    assert not np.isclose(result.estat_ovtest().statistic, raw_f, rtol=1e-8), (
        "the fitted values must be rescaled to [0,1] before taking powers"
    )


def test_rhs_option_uses_powers_of_the_regressors(fit) -> None:
    _, result = fit
    matrix, names = result.design_matrix
    matrix = np.asarray(matrix, dtype=float)
    resid = np.asarray(result.resid, dtype=float)
    y = np.asarray(result.fittedvalues, dtype=float) + resid

    columns = []
    for j, name in enumerate(names):
        if name.lower() in ("intercept", "_cons", "const"):
            continue
        values = matrix[:, j]
        scaled = (values - values.min()) / (values.max() - values.min())
        for power in (2, 3, 4):
            columns.append(scaled**power)
    extra = np.column_stack(columns)
    design = np.column_stack([matrix, extra])
    expected, df1, _, _ = _joint_f(y, design, range(matrix.shape[1], design.shape[1]))

    rendered = result.estat_ovtest(rhs=True)

    assert rendered.statistic == pytest.approx(expected, rel=1e-10)
    assert rendered.df == df1
    assert rendered.omitted == "independent variables"


def test_default_omitted_label_names_the_dependent_variable(fit) -> None:
    """Stata prints "Powers of fitted values OF <depvar>", not "Powers of fitted values".

    Measured on icecream.dta: ``Omitted: Powers of fitted values of consumption``. The old
    assertion pinned the shorter phrase, which is a different sentence rather than a
    truncation of the same one.
    """

    _, result = fit
    label = result.estat_ovtest().omitted

    assert label.startswith("fitted values")
    assert " of " in label, label


def test_pvalue_is_upper_tail_f(fit) -> None:
    _, result = fit
    rendered = result.estat_ovtest()

    assert rendered.pvalue == pytest.approx(
        float(stats.f.sf(rendered.statistic, rendered.df, rendered.df_denom)),
        rel=1e-12,
    )
    assert 0.0 <= rendered.pvalue <= 1.0


def test_a_misspecified_model_is_rejected() -> None:
    """A genuine omitted quadratic must produce a small p-value."""

    rng = np.random.default_rng(3)
    n = 400
    x = rng.uniform(-1, 1, size=n)
    y = 1.0 + 2.0 * x + 3.0 * x**2 + rng.normal(scale=0.3, size=n)
    frame = pd.DataFrame({"y": y, "x": x})

    rendered = ws.OLS.from_formula("y ~ x", frame).fit().estat_ovtest()

    assert rendered.pvalue < 0.01
    assert rendered.statistic > 100.0


def test_a_well_specified_model_is_not_rejected() -> None:
    rng = np.random.default_rng(3)
    n = 400
    x = rng.uniform(-1, 1, size=n)
    y = 1.0 + 2.0 * x + rng.normal(scale=0.3, size=n)
    frame = pd.DataFrame({"y": y, "x": x})

    rendered = ws.OLS.from_formula("y ~ x", frame).fit().estat_ovtest()

    assert rendered.pvalue > 0.01


def test_all_indicator_regressors_are_rejected() -> None:
    """Stata reports "powers of all explanatory variables collinear"."""

    rng = np.random.default_rng(7)
    n = 200
    frame = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g1": rng.integers(0, 2, size=n).astype(float),
            "g2": rng.integers(0, 2, size=n).astype(float),
        }
    )
    fit = ws.OLS.from_formula("y ~ g1 + g2", frame).fit()

    # The default (fitted values) still works, because the fitted values are
    # continuous even when every regressor is an indicator.
    assert np.isfinite(fit.estat_ovtest().statistic)

    # The rhs variant has no continuous regressor to take powers of.
    with pytest.raises(ValueError, match="collinear"):
        fit.estat_ovtest(rhs=True)


def test_to_dict_uses_stata_r_names(fit) -> None:
    _, result = fit
    payload = result.estat_ovtest().to_dict()

    assert set(payload) == {"F", "df", "df_r", "p"}
    assert payload["F"] == result.estat_ovtest().statistic


def test_output_has_the_stata_header(fit) -> None:
    _, result = fit
    text = str(result.estat_ovtest())

    assert "Ramsey RESET test for omitted variables" in text
    assert "H0: Model has no omitted variables" in text
    assert "Prob > F" in text


def test_module_function_matches_the_method(fit) -> None:
    _, result = fit

    assert estat_ovtest(result).to_dict() == result.estat_ovtest().to_dict()
