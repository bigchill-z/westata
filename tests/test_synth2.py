"""Unit tests for ``ws.synth2`` (non-graphics feature set).

These test the estimator's own contracts WITHOUT Stata: formula construction,
input validation, the placebo/LOO orchestration shape, and the refusal paths.
Numerical conformance against live Stata belongs to
``tests/conformance/test_stata_synth2.py``.

Written BEFORE the implementation (TDD RED).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import UnsupportedError
from westata.estimators.synth2 import Synth2Result, synth2

UNITS = [1, 2, 3, 4, 5, 6]
YEARS = list(range(2000, 2010))
TRUNIT, TRPERIOD = 1, 2006
PRE = list(range(2000, 2006))
POST = list(range(2006, 2010))


def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(20261111)
    rows = []
    for u in UNITS:
        for t in YEARS:
            rows.append({
                "unit": u,
                "year": t,
                "y": 10.0 + 0.7 * u + 0.3 * t + rng.normal(scale=0.02),
                "x1": 1.0 + 0.5 * u + rng.normal(scale=0.002),
                "x2": 2.0 - 0.25 * u + rng.normal(scale=0.002),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


def _base(panel, **kw):
    args = dict(data=panel, unit="unit", time="year", depvar="y",
                predictors=["x1", "x2"], trunit=TRUNIT, trperiod=TRPERIOD)
    args.update(kw)
    return synth2(**args)


# ------------------------------------------------------------------ result object
def test_the_package_exports_synth2() -> None:
    assert hasattr(ws, "synth2")
    assert callable(ws.synth2)
    assert hasattr(ws, "Synth2Result")


def test_result_is_a_frozen_dataclass(panel) -> None:
    res = _base(panel)
    assert isinstance(res, Synth2Result)
    with pytest.raises(Exception):
        res.rmse = 0.0  # type: ignore[misc]


def test_result_does_not_fabricate_a_coefficient_vce(panel) -> None:
    res = _base(panel)
    for forbidden in ("params", "bse", "pvalues", "cov_params", "df_resid"):
        assert not hasattr(res, forbidden), forbidden


def test_e_scalars_are_present(panel) -> None:
    res = _base(panel)
    for name in ("N", "T", "T0", "T1", "K", "J", "rmse", "r2", "att"):
        assert hasattr(res, name), name
        assert np.isfinite(float(getattr(res, name))), name


def test_fit_statistics_follow_synth2_ado(panel) -> None:
    """RMSE = synth.RMSPE; R2 denominator uses the SYNTHETIC outcome (ado:647)."""

    res = _base(panel)
    assert res.rmse == pytest.approx(res.synth_result.RMSPE, abs=1e-12)
    pre = res.time_index < TRPERIOD
    eff = res.gap.ravel()
    ysyn = res.Y_synthetic.ravel()
    expected_r2 = 1.0 - float(np.sum(eff[pre] ** 2)) / float(
        np.sum((ysyn[pre] - ysyn[pre].mean()) ** 2)
    )
    assert res.r2 == pytest.approx(expected_r2, abs=1e-10)
    assert res.att == pytest.approx(float(eff[~pre].mean()), abs=1e-10)


def test_balance_table_has_six_columns_with_the_stata_bias_formula(panel) -> None:
    res = _base(panel)
    bal = np.asarray(res.bal, dtype=float)
    assert bal.shape[0] == 2          # K = 2 predictors
    assert bal.shape[1] == 6
    np.testing.assert_allclose(
        bal[:, 3], (bal[:, 2] / bal[:, 1] - 1) * 100, atol=1e-9
    )
    np.testing.assert_allclose(
        bal[:, 5], (bal[:, 4] / bal[:, 1] - 1) * 100, atol=1e-9
    )


def test_U_wt_contains_only_nonzero_donors_in_descending_order(panel) -> None:
    res = _base(panel)
    u = np.asarray(res.U_wt, dtype=float).ravel()
    assert np.all(u > 0)
    assert np.all(np.diff(u) <= 1e-12)


def test_V_wt_is_the_diagonal_v_matrix(panel) -> None:
    res = _base(panel)
    np.testing.assert_allclose(res.V_wt, res.synth_result.V_matrix, atol=1e-12)


# ------------------------------------------------------------------ validation
def test_missing_treated_unit_is_rejected(panel) -> None:
    with pytest.raises(ValueError, match="trunit"):
        _base(panel, trunit=99)


def test_missing_treatment_period_is_rejected(panel) -> None:
    with pytest.raises(ValueError, match="trperiod"):
        _base(panel, trperiod=1999)


def test_counit_containing_treated_is_rejected(panel) -> None:
    with pytest.raises(ValueError):
        _base(panel, counit=[1, 2, 3])


def test_counit_with_unknown_unit_is_rejected(panel) -> None:
    with pytest.raises(ValueError):
        _base(panel, counit=[2, 3, 99])


def test_ctrlunit_and_counit_conflict(panel) -> None:
    with pytest.raises(ValueError):
        _base(panel, counit=[2, 3], ctrlunit=[2, 3])


def test_ctrlunit_is_a_deprecated_alias(panel) -> None:
    a = _base(panel, counit=[2, 3, 4])
    b = _base(panel, ctrlunit=[2, 3, 4])
    np.testing.assert_allclose(a.rmse, b.rmse, atol=1e-10)


def test_preperiod_with_a_post_period_is_rejected(panel) -> None:
    with pytest.raises(ValueError, match="preperiod"):
        _base(panel, preperiod=[2000, 2001, 2007])


def test_postperiod_must_start_at_trperiod(panel) -> None:
    with pytest.raises(ValueError, match="postperiod"):
        _base(panel, postperiod=[2007, 2008])


def test_postperiod_affects_T1(panel) -> None:
    a = _base(panel)
    b = _base(panel, postperiod=[2006, 2007, 2008])
    assert b.T1 == 3 and a.T1 == 4


# ------------------------------------------------------------------ refused options
@pytest.mark.parametrize("kw", [
    {"figure": True}, {"nofigure": True}, {"savegraph": "out"},
    {"symbol": 2}, {"frame": "myframe"}, {"show": 3},
])
def test_graphics_only_options_are_refused(panel, kw) -> None:
    with pytest.raises(UnsupportedError):
        _base(panel, **kw)


# ------------------------------------------------------------------ placebo parsing
def test_placebo_unit_runs_in_space_placebo(panel) -> None:
    res = _base(panel, placebo="unit")
    assert res.mspe is not None
    assert np.asarray(res.mspe).shape[1] == 4
    assert res.pval is not None
    assert np.asarray(res.pval).shape[1] == 4


def test_placebo_mspe_first_row_is_the_treated_unit(panel) -> None:
    res = _base(panel, placebo="unit")
    mspe = np.asarray(res.mspe, dtype=float)
    # treated row: RatioTrCtrl == 1 (ado:660)
    assert mspe[0, 3] == pytest.approx(1.0, abs=1e-12)
    # RatioPostPre = PostMSPE / PreMSPE
    assert mspe[0, 2] == pytest.approx(mspe[0, 1] / mspe[0, 0], abs=1e-10)


def test_placebo_pointwise_pvalues_include_the_treated_unit(panel) -> None:
    """Denominator is (placebos + 1); every p-value >= 1/(P+1)."""

    res = _base(panel, placebo="unit")
    pval = np.asarray(res.pval, dtype=float)
    n_units = np.asarray(res.mspe).shape[0]   # includes treated row
    assert np.all(pval[:, 1:] >= 1.0 / n_units - 1e-12)
    assert np.all(pval[:, 1:] <= 1.0 + 1e-12)


def test_placebo_pvalue_formulas(panel) -> None:
    res = _base(panel, placebo="unit")
    pval = np.asarray(res.pval, dtype=float)
    tr_eff = pval[:, 0]
    # two-sided >= right-sided is not guaranteed; but all three are in [0,1]
    # and the effect column equals the treated gap over post periods.
    post = res.time_index >= TRPERIOD
    np.testing.assert_allclose(tr_eff, res.gap.ravel()[post], atol=1e-10)


def test_placebo_cutoff_marks_removed_units(panel) -> None:
    res = _base(panel, placebo="unit cut(1.0)")
    assert res.placebo_removed is not None
    # With cutoff=1 every fake unit whose pre-MSPE exceeds the treated is removed.
    assert isinstance(res.placebo_removed, (list, tuple))


def test_placebo_period_runs_in_time_placebo_without_pvalues(panel) -> None:
    res = _base(panel, predictors=["x1", "x2"], placebo="period(2003)")
    assert res.time_placebo is not None
    assert 2003 in res.time_placebo
    assert res.pval is None          # in-time placebo has NO inference
    assert res.mspe is None


def test_placebo_period_at_or_after_trperiod_is_rejected(panel) -> None:
    with pytest.raises(ValueError):
        _base(panel, placebo="period(2006)")


def test_cutoff_below_one_is_rejected(panel) -> None:
    with pytest.raises(ValueError):
        _base(panel, placebo="unit cut(0.5)")


def test_unknown_placebo_key_is_rejected(panel) -> None:
    with pytest.raises(ValueError):
        _base(panel, placebo="bogus")


# ------------------------------------------------------------------ loo
def test_loo_produces_min_max_envelopes(panel) -> None:
    res = _base(panel, loo=True)
    assert res.loo_min is not None and res.loo_max is not None
    assert np.asarray(res.loo_min).shape == res.Y_synthetic.shape
    assert np.all(np.asarray(res.loo_min) <= np.asarray(res.loo_max) + 1e-9)


def test_loo_only_iterates_over_nonzero_weight_donors(panel) -> None:
    base = _base(panel)
    res = _base(panel, loo=True)
    n_nonzero = int(np.sum(np.asarray(base.U_wt, dtype=float).ravel() > 0))
    assert res.loo_n_runs == n_nonzero


# ------------------------------------------------------------------ regression
def test_default_path_is_deterministic(panel) -> None:
    a = _base(panel)
    b = _base(panel)
    np.testing.assert_allclose(a.rmse, b.rmse, atol=1e-12)
    np.testing.assert_allclose(a.bal, b.bal, atol=1e-12)
