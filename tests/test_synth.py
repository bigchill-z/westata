"""Unit tests for ``ws.synth`` (B-grade synthetic control).

These test the estimator's own contracts WITHOUT Stata: data preparation, the
two solved Mata kernels, the constrained QP, the result object, the refusal
paths and the rounding order.  Numerical conformance against Stata belongs to
``tests/conformance/test_stata_synth.py``.

Data for the synthetic panels is built inline so the suite needs neither a
Stata licence nor an external ``.dta`` file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError, UnsupportedError
from westata.estimators.synth import SynthResult, _normalize, _regsval, synth


# --------------------------------------------------------------------------- #
# A small balanced panel with a known structure (K=2 predictors, J=4 controls,
# T0=5 pre-periods, 3 post-periods).  Built so no predictor is all-missing.
# --------------------------------------------------------------------------- #
UNITS = [1, 2, 3, 4, 5]          # 1 = treated
YEARS = list(range(2000, 2009))   # 2000-2004 pre, 2005 treatment, 2006-2008 post
PRE = list(range(2000, 2005))


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(20261109)
    rows = []
    for u in UNITS:
        for t in YEARS:
            rows.append(
                {
                    "unit": u,
                    "year": t,
                    "y": 10.0 + 0.7 * u + 0.3 * t + rng.normal(scale=0.01),
                    "x1": 1.0 + 0.5 * u + rng.normal(scale=0.001),
                    "x2": 2.0 - 0.25 * u + rng.normal(scale=0.001),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


@pytest.fixture(scope="module")
def fitted(panel):
    return synth(
        panel, unit="unit", time="year", depvar="y",
        predictors=["x1", "x2"], trunit=1, trperiod=2005,
    )


# --------------------------------------------------------------------------- #
# Result object
# --------------------------------------------------------------------------- #
def test_the_package_exports_synth() -> None:
    assert hasattr(ws, "synth")
    assert callable(ws.synth)


def test_result_is_a_frozen_dataclass(fitted) -> None:
    assert isinstance(fitted, SynthResult)
    with pytest.raises(Exception):
        fitted.RMSPE = 0.0  # type: ignore[misc]


def test_result_does_not_fabricate_a_coefficient_vce(fitted) -> None:
    """synth has no e(b)/e(V); these must not be invented (design G3)."""

    for forbidden in ("params", "bse", "pvalues", "cov_params", "df_resid"):
        assert not hasattr(fitted, forbidden), forbidden


def test_result_exposes_the_six_stata_matrices(fitted) -> None:
    for name in ("V_matrix", "W_weights", "X_balance", "RMSPE",
                 "Y_treated", "Y_synthetic"):
        assert hasattr(fitted, name), name
    assert fitted.gap.shape == fitted.Y_treated.shape


def test_matrix_shapes_follow_the_ado_contract(fitted) -> None:
    K, J, T = 2, 4, 9
    assert fitted.V_matrix.shape == (K, K)
    assert fitted.W_weights.shape == (J, 2)
    assert fitted.X_balance.shape == (K, 2)
    assert fitted.Y_treated.shape == (T, 1)
    assert fitted.Y_synthetic.shape == (T, 1)
    assert np.asarray(fitted.RMSPE).size == 1


def test_v_matrix_is_diagonal_nonnegative_and_sums_to_one(fitted) -> None:
    V = np.asarray(fitted.V_matrix, dtype=float)
    assert np.allclose(V, np.diag(np.diag(V)))
    assert np.all(np.diag(V) >= 0)
    assert np.diag(V).sum() == pytest.approx(1.0, abs=1e-12)


def test_unit_weights_are_nonnegative_and_in_the_simplex(fitted) -> None:
    W = np.asarray(fitted.W_weights, dtype=float)[:, 1]
    assert np.all(W >= 0)
    # roundmat() rounds to 3 dp, so the sum is only approximately 1.
    assert W.sum() == pytest.approx(1.0, abs=2e-3)


def test_stata_style_summary_is_renderable(fitted) -> None:
    text = str(fitted)
    assert "RMSPE" in text or "Root" in text
    assert "Unit Weights" in text or "W_Weight" in text or "weight" in text.lower()


# --------------------------------------------------------------------------- #
# Solved Mata kernels (pure functions, verified in P0 against live Stata)
# --------------------------------------------------------------------------- #
def test_normalize_uses_sample_sd_including_the_treated_unit() -> None:
    X = np.array([[1.0, 2.0, 4.0, 6.0, 8.0],
                  [10.0, 10.0, 20.0, 30.0, 40.0]])
    Xn, s = _normalize(X)
    np.testing.assert_allclose(s, X.std(axis=1, ddof=1))
    np.testing.assert_allclose(Xn, X / s[:, None])
    # The treated value is INCLUDED: control-only sd would be different.
    assert not np.allclose(s, X[:, 1:].std(axis=1, ddof=1))


def test_normalize_guard_for_a_constant_row() -> None:
    X = np.ones((2, 5))
    Xn, s = _normalize(X)
    assert np.all(s == 1.0)          # constant row is a no-op, not 0/0
    np.testing.assert_allclose(Xn, X)


def test_regsval_is_the_sum_of_squared_per_period_ols_coefficients() -> None:
    """Hand-computable case from the P0 report.

    With ``B = [[2,3,4,5],[0,1,2,3]]`` the per-predictor sums of squared
    coefficients are ``[54, 14]``, so V must be ``[27/34, 7/34]``.  We build a
    Z whose per-period demeaned rows are ``Xdm' B`` (which satisfies
    ``Xdm Zdm' = Sxx B`` exactly), avoiding any rank-deficient inversion.
    """

    X = np.array([[2.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                  [5.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    Xdm = X - X.mean(axis=1, keepdims=True)
    B = np.array([[2.0, 3.0, 4.0, 5.0],
                  [0.0, 1.0, 2.0, 3.0]])       # K x T0
    Zdm_T = Xdm.T @ B                          # N x T0, columns are periods
    Z = Zdm_T.T                                # T0 x N, already row-demeaned
    v = _regsval(X, Z)
    np.testing.assert_allclose(v, np.array([27 / 34, 7 / 34]), atol=1e-10)


def test_regsval_rejects_a_rank_deficient_design() -> None:
    X = np.array([[1.0, 1.0, 1.0, 1.0],
                  [2.0, 2.0, 2.0, 2.0]])
    Z = np.arange(12, dtype=float).reshape(3, 4)
    with pytest.raises(ConsistencyError, match="rank"):
        _regsval(X, Z)


def test_custom_v_is_normalised_by_absolute_value() -> None:
    res = synth(
        _panel(), unit="unit", time="year", depvar="y",
        predictors=["x1", "x2"], trunit=1, trperiod=2005,
        customV=[2.0, 2.0],
    )
    np.testing.assert_allclose(np.diag(res.V_matrix), [0.5, 0.5])


def test_custom_v_with_wrong_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="customV"):
        synth(
            _panel(), unit="unit", time="year", depvar="y",
            predictors=["x1", "x2"], trunit=1, trperiod=2005,
            customV=[1.0, 2.0, 3.0],
        )


# --------------------------------------------------------------------------- #
# Refusal paths (design §6)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kwargs",
    [
        {"nested": True},
        {"allopt": True},
        {"figure": True},
        {"keep": "out.dta"},
        {"placebo": True},
    ],
)
def test_unsupported_stata_options_raise_unsupportederror(kwargs) -> None:
    with pytest.raises(UnsupportedError):
        synth(
            _panel(), unit="unit", time="year", depvar="y",
            predictors=["x1", "x2"], trunit=1, trperiod=2005, **kwargs,
        )


def test_missing_treated_unit_is_rejected() -> None:
    with pytest.raises(ValueError):
        synth(
            _panel(), unit="unit", time="year", depvar="y",
            predictors=["x1", "x2"], trunit=99, trperiod=2005,
        )


def test_missing_treatment_period_is_rejected() -> None:
    with pytest.raises(ValueError):
        synth(
            _panel(), unit="unit", time="year", depvar="y",
            predictors=["x1", "x2"], trunit=1, trperiod=1999,
        )


def test_too_few_controls_is_rejected() -> None:
    with pytest.raises(ConsistencyError):
        synth(
            _panel(), unit="unit", time="year", depvar="y",
            predictors=["x1", "x2"], trunit=1, trperiod=2005,
            counit=[2],
        )


def test_all_missing_predictor_for_a_unit_is_rejected() -> None:
    df = _panel().copy()
    df.loc[df["unit"] == 2, "x1"] = np.nan
    with pytest.raises(ConsistencyError, match="missing"):
        synth(
            df, unit="unit", time="year", depvar="y",
            predictors=["x1", "x2"], trunit=1, trperiod=2005,
        )


def test_missing_outcome_in_the_pre_period_is_rejected() -> None:
    df = _panel().copy()
    df.loc[(df["unit"] == 3) & (df["year"] == 2001), "y"] = np.nan
    with pytest.raises(ConsistencyError, match="missing"):
        synth(
            df, unit="unit", time="year", depvar="y",
            predictors=["x1", "x2"], trunit=1, trperiod=2005,
        )


def test_non_numeric_stata_missing_sentinel_is_rejected(panel) -> None:
    df = panel.copy()
    # Stata stores missing as 8.98846567431158e307, not NaN.  Injecting it
    # must be caught before it silently corrupts V.
    df.loc[(df["unit"] == 2) & (df["year"] == 2001), "x1"] = 8.98846567431158e307
    with pytest.raises(ConsistencyError):
        synth(
            df, unit="unit", time="year", depvar="y",
            predictors=["x1", "x2"], trunit=1, trperiod=2005,
        )


# --------------------------------------------------------------------------- #
# Data-preparation contracts
# --------------------------------------------------------------------------- #
def test_predictors_are_averaged_over_the_whole_pre_period_by_default(fitted) -> None:
    # X_balance Treated must equal the mean over 2000-2004, not a single year.
    p = _panel()
    expected = [
        p.loc[(p.unit == 1) & (p.year.isin(PRE)), "x1"].mean(),
        p.loc[(p.unit == 1) & (p.year.isin(PRE)), "x2"].mean(),
    ]
    np.testing.assert_allclose(np.asarray(fitted.X_balance)[:, 0], expected, atol=1e-12)


def test_xperiod_changes_the_balance_table() -> None:
    full = synth(
        _panel(), unit="unit", time="year", depvar="y",
        predictors=["x1", "x2"], trunit=1, trperiod=2005,
    )
    short = synth(
        _panel(), unit="unit", time="year", depvar="y",
        predictors=["x1", "x2"], trunit=1, trperiod=2005,
        xperiod=[2002, 2003],
    )
    assert not np.allclose(full.V_matrix, short.V_matrix)


def test_resultsperiod_restricts_the_output_trajectories() -> None:
    res = synth(
        _panel(), unit="unit", time="year", depvar="y",
        predictors=["x1", "x2"], trunit=1, trperiod=2005,
        resultsperiod=[2003, 2004, 2005, 2006],
    )
    assert res.Y_treated.shape[0] == 4
    assert res.Y_synthetic.shape[0] == 4


def test_float32_input_is_cast_before_averaging() -> None:
    """A float32 panel must be promoted to float64 before any averaging.

    P0 finding: summing in float32 drifts V by ~3e-7 *beyond* the information
    the input actually carries.  A float32 input has already lost precision,
    so it cannot equal a float64 original -- but it MUST equal a float64
    frame holding those same (already-rounded) values.  That isolates
    accumulation error from input precision.
    """

    from westata.estimators.synth import _as_float64

    p64 = _panel()
    p32 = p64.copy()
    for c in ("y", "x1", "x2", "year", "unit"):
        p32[c] = p32[c].astype("float32")

    promoted = _as_float64(p32, ["y", "x1", "x2", "year", "unit"])
    for c in ("y", "x1", "x2", "year", "unit"):
        assert promoted[c].dtype == np.float64

    # Reference: the same values the float32 frame holds, but stored float64.
    ref = p32.copy()
    for c in ("y", "x1", "x2", "year", "unit"):
        ref[c] = ref[c].astype("float64")
    a = synth(ref, unit="unit", time="year", depvar="y",
              predictors=["x1", "x2"], trunit=1, trperiod=2005)
    b = synth(p32, unit="unit", time="year", depvar="y",
              predictors=["x1", "x2"], trunit=1, trperiod=2005)
    np.testing.assert_allclose(a.V_matrix, b.V_matrix, atol=1e-12)
