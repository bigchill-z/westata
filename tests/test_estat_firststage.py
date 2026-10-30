"""Tests for Stata's ``estat firststage`` (IV first-stage diagnostics).

Column definitions come from ``ivregress_estat.ado``'s ``FirstStage`` program
(StataNow 19.5):

* R-sq / adjusted R-sq -- the endogenous regressor on **all** exogenous variables
* partial R-sq -- residualised endogenous on the residualised excluded instruments
* F -- the joint test of the excluded instruments

Shea's partial R-squared (``ivregress_estat.ado:700-736``) nets out the other
endogenous regressors before measuring instrument strength.  A test builds
correlated endogenous regressors specifically to show Shea diverging from the
plain partial R-squared: with independent regressors the two coincide, so a test
on such data would be vacuous.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws


@pytest.fixture()
def auto_iv():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length", "displacement"]].dropna()
    return frame, ws.IV2SLS.from_formula(
        "mpg ~ weight + [length ~ displacement]", frame
    ).fit()


def _resid(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def test_r2_is_the_endogenous_regressor_on_all_exogenous(auto_iv) -> None:
    frame, result = auto_iv
    endog = frame["length"].to_numpy(float)
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["weight"].to_numpy(float),
            frame["displacement"].to_numpy(float),
        ]
    )
    resid = _resid(endog, design)
    centred = endog - endog.mean()
    expected = 1.0 - float(resid @ resid) / float(centred @ centred)

    assert result.estat_firststage().r2[0] == pytest.approx(expected, rel=1e-12)


def test_adjusted_r2_uses_squared_on_all_exogenous(auto_iv) -> None:
    frame, result = auto_iv
    fs = result.estat_firststage()
    n = len(frame)
    k = 3  # constant + weight + displacement
    expected = 1.0 - (1.0 - fs.r2[0]) * (n - 1) / (n - k)

    assert fs.adj_r2[0] == pytest.approx(expected, rel=1e-12)


def test_partial_r2_residualises_both_sides(auto_iv) -> None:
    frame, result = auto_iv
    n = len(frame)
    exog = np.column_stack([np.ones(n), frame["weight"].to_numpy(float)])
    endog_res = _resid(frame["length"].to_numpy(float), exog)
    instr_res = _resid(frame["displacement"].to_numpy(float), exog)

    expected = float(np.corrcoef(endog_res, instr_res)[0, 1] ** 2)

    assert result.estat_firststage().partial_r2[0] == pytest.approx(
        expected, rel=1e-10
    )


def test_f_is_the_partial_regression_f(auto_iv) -> None:
    frame, result = auto_iv
    fs = result.estat_firststage()
    r2 = fs.partial_r2[0]
    expected = (r2 / fs.df_num[0]) / ((1 - r2) / fs.df_denom[0])

    assert fs.f[0] == pytest.approx(expected, rel=1e-8)


def test_pvalue_is_upper_tail_f(auto_iv) -> None:
    _, result = auto_iv
    fs = result.estat_firststage()

    assert fs.pvalue[0] == pytest.approx(
        float(stats.f.sf(fs.f[0], fs.df_num[0], fs.df_denom[0])), rel=1e-10
    )


def test_degrees_of_freedom(auto_iv) -> None:
    """``ivregress_estat.ado:581``: ``dfdenom = e(N) - rank - df_a - nendog``.

    This test used to assert ``N - 2`` -- the auxiliary regression's own ``df_resid``,
    which counts the intercept and the instrument but not the endogenous regressor being
    explained.  Stata's denominator also subtracts that regressor via ``rank`` and
    ``nendog``, so with one instrument and one endogenous variable it is ``N - rank -
    1``.
    """

    frame, result = auto_iv
    fs = result.estat_firststage()

    assert fs.df_num[0] == 1.0  # one excluded instrument
    assert fs.df_denom[0] == float(len(frame) - float(result.e["rank"]) - 1.0)


def test_a_weak_instrument_shows_a_small_partial_r2(auto_iv) -> None:
    """``displacement`` barely predicts ``length`` once ``weight`` is netted out."""

    fs = auto_iv[1].estat_firststage()

    assert fs.partial_r2[0] < 0.05
    assert fs.f[0] < 1.0


def test_a_strong_instrument_shows_a_large_partial_r2() -> None:
    rng = np.random.default_rng(7)
    n = 500
    z = rng.normal(size=n)
    x = 0.9 * z + rng.normal(scale=0.3, size=n)
    w = rng.normal(size=n)
    y = 1.0 + 0.5 * w + 2.0 * x + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"y": y, "w": w, "x": x, "z": z})

    fs = ws.IV2SLS.from_formula("y ~ w + [x ~ z]", frame).fit().estat_firststage()

    assert fs.partial_r2[0] > 0.5
    assert fs.f[0] > 100


def test_shea_is_reported_for_two_endogenous_regressors() -> None:
    rng = np.random.default_rng(21)
    n = 600
    w = rng.normal(size=n)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    common = rng.normal(size=n)
    x1 = 0.8 * z1 + 0.9 * common + rng.normal(scale=0.3, size=n)
    x2 = 0.8 * z2 + 0.9 * common + rng.normal(scale=0.3, size=n)
    y = 1.0 + 0.6 * w + 1.5 * x1 + 1.2 * x2 + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame({"y": y, "w": w, "x1": x1, "x2": x2, "z1": z1, "z2": z2})

    fs = (
        ws.IV2SLS.from_formula("y ~ w + [x1 + x2 ~ z1 + z2]", frame)
        .fit()
        .estat_firststage()
    )

    assert fs.shea is not None
    assert len(fs.shea) == 2
    assert all(0.0 <= v <= 1.0 for v in fs.shea)


def test_shea_differs_from_partial_r2_when_endogenous_are_correlated() -> None:
    """The point of Shea: with correlated endogenous regressors the two differ.

    On independent endogenous regressors they nearly coincide, so a test on such
    data would pass even if Shea were computed as the plain partial R2.
    """

    rng = np.random.default_rng(21)
    n = 600
    w = rng.normal(size=n)
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    common = rng.normal(size=n)
    x1 = 0.8 * z1 + 0.9 * common + rng.normal(scale=0.3, size=n)
    x2 = 0.8 * z2 + 0.9 * common + rng.normal(scale=0.3, size=n)
    y = 1.0 + 0.6 * w + 1.5 * x1 + 1.2 * x2 + rng.normal(scale=0.4, size=n)
    frame = pd.DataFrame({"y": y, "w": w, "x1": x1, "x2": x2, "z1": z1, "z2": z2})
    assert abs(np.corrcoef(x1, x2)[0, 1]) > 0.4, "fixture needs correlated endog"

    fs = (
        ws.IV2SLS.from_formula("y ~ w + [x1 + x2 ~ z1 + z2]", frame)
        .fit()
        .estat_firststage()
    )

    assert np.max(np.abs(fs.shea - fs.partial_r2)) > 1e-3


def test_shea_absent_for_a_single_endogenous_regressor(auto_iv) -> None:
    fs = auto_iv[1].estat_firststage()

    assert fs.shea is None
    assert fs.shea_adj is None


def test_non_iv_results_are_refused(auto_iv) -> None:
    frame, _ = auto_iv
    ols = ws.OLS.from_formula("mpg ~ weight", frame).fit()

    with pytest.raises(NotImplementedError, match="instrumental-variables"):
        ols.estat_firststage()


def test_to_frame_has_the_reported_columns(auto_iv) -> None:
    fs = auto_iv[1].estat_firststage()
    frame = fs.to_frame()

    assert "R-sq." in frame.columns
    assert "Adjusted R-sq." in frame.columns
    assert "Partial R-sq." in frame.columns


def test_output_mentions_the_endogenous_variable(auto_iv) -> None:
    text = str(auto_iv[1].estat_firststage())

    assert "First-stage regression summary statistics" in text
    assert "length" in text
