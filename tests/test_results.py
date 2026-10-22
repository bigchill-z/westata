"""Contract tests for the westata results container (P0 Task 3).

The Stata summary is asserted token-by-token: researchers scan for exact labels
such as ``Number of obs`` and ``[95% conf. interval]``, and downstream report
code (``esttab``) keys off the same strings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from westata.results import EReturn, StataResults

PARAM_NAMES = ["_cons", "x1", "x2"]
PARAMS = np.array([1.0, 2.0, -0.5])
COV = np.array(
    [
        [0.04, 0.006, 0.002],
        [0.006, 0.09, -0.003],
        [0.002, -0.003, 0.01],
    ]
)
BSE = np.sqrt(np.diag(COV))
TVALUES = PARAMS / BSE
RESID = np.linspace(-1.0, 1.0, 30)
FITTED = np.linspace(0.0, 1.0, 30)

STATA_LABELS = [
    "Number of obs",
    "F(",
    "Prob > F",
    "R-squared",
    "Adj R-squared",
    "Root MSE",
    "Coef.",
    "Std. err.",
    "[95% conf. interval]",
]


def make_results(*, dist: str = "t", e=None, **overrides) -> StataResults:
    """Build a deterministic toy result covering every public property."""

    kwargs = dict(
        params=PARAMS,
        param_names=PARAM_NAMES,
        cov=COV,
        resid=RESID,
        fittedvalues=FITTED,
        nobs=30,
        df_resid=27,
        df_model=2,
        dist=dist,
        model_name="OLS",
        cov_type="nonrobust",
        e=e,
    )
    kwargs.update(overrides)
    return StataResults(**kwargs)


def test_params_bse_tvalues_pvalues_match_hand_computation() -> None:
    res = make_results()

    assert isinstance(res.params, pd.Series)
    assert res.params.index.tolist() == PARAM_NAMES
    np.testing.assert_allclose(res.params.to_numpy(), PARAMS)

    assert isinstance(res.bse, pd.Series)
    assert res.bse.index.tolist() == PARAM_NAMES
    np.testing.assert_allclose(res.bse.to_numpy(), BSE)

    np.testing.assert_allclose(res.tvalues.to_numpy(), TVALUES)

    expected_pvalues = 2.0 * stats.t.sf(np.abs(TVALUES), 27)
    np.testing.assert_allclose(res.pvalues.to_numpy(), expected_pvalues)
    assert (res.pvalues.to_numpy() >= 0).all()
    assert (res.pvalues.to_numpy() <= 1).all()

    assert res.nobs == 30
    assert res.df_resid == 27
    assert res.df_model == 2


def test_conf_int_uses_t_critical_values() -> None:
    res = make_results()

    ci = res.conf_int()
    assert list(ci.columns) == ["lower", "upper"]
    assert ci.index.tolist() == PARAM_NAMES

    crit = stats.t.ppf(0.975, res.df_resid)
    np.testing.assert_allclose(ci["lower"].to_numpy(), PARAMS - crit * BSE)
    np.testing.assert_allclose(ci["upper"].to_numpy(), PARAMS + crit * BSE)

    ci90 = res.conf_int(alpha=0.10)
    crit90 = stats.t.ppf(0.95, res.df_resid)
    np.testing.assert_allclose(ci90["lower"].to_numpy(), PARAMS - crit90 * BSE)
    np.testing.assert_allclose(ci90["upper"].to_numpy(), PARAMS + crit90 * BSE)


def test_z_distribution_uses_normal_critical_values_and_z_pvalues() -> None:
    res = make_results(dist="z")

    expected_pvalues = 2.0 * stats.norm.sf(np.abs(TVALUES))
    np.testing.assert_allclose(res.pvalues.to_numpy(), expected_pvalues)

    ci = res.conf_int()
    crit = stats.norm.ppf(0.975)
    np.testing.assert_allclose(ci["lower"].to_numpy(), PARAMS - crit * BSE)
    np.testing.assert_allclose(ci["upper"].to_numpy(), PARAMS + crit * BSE)

    np.testing.assert_allclose(res.zvalues.to_numpy(), TVALUES)
    np.testing.assert_allclose(res.tvalues.to_numpy(), res.zvalues.to_numpy())


def test_summary_stata_contains_required_labels_and_t_statistics() -> None:
    text = make_results().summary(style="stata")

    for token in STATA_LABELS:
        assert token in text, f"missing Stata label {token!r}"
    assert "P>|t|" in text
    assert "P>|z|" not in text
    assert "Number of obs" in text
    for name in PARAM_NAMES:
        assert name in text


def test_summary_stata_uses_z_labels_for_z_distribution() -> None:
    text = make_results(dist="z").summary("stata")

    for token in STATA_LABELS:
        assert token in text, f"missing Stata label {token!r}"
    assert "P>|z|" in text
    assert "P>|t|" not in text


def test_summary_statsmodels_includes_coefficients() -> None:
    res = make_results()
    text = res.summary()
    assert isinstance(text, str)
    for name in PARAM_NAMES:
        assert name in text
    assert "coef" in text.lower()
    assert "std err" in text.lower()
    assert "P>|t|" in text


def test_summary_rejects_unknown_style() -> None:
    with pytest.raises(ValueError, match="style"):
        make_results().summary(style="bogus")


def test_ereturn_supports_item_and_attribute_access() -> None:
    e = EReturn({"r2": 0.5, "df_r": 27})

    assert isinstance(e, dict)
    assert e["r2"] == 0.5
    assert e.r2 == 0.5
    assert e.df_r == 27

    with pytest.raises(KeyError):
        e["does_not_exist"]
    assert e.does_not_exist is None


def test_result_e_merges_computed_defaults_with_extra_mapping() -> None:
    res = make_results(
        e={
            "r2": 0.75,
            "r2_a": 0.70,
            "ll": -12.5,
            "aic": 100.0,
            "bic": 110.0,
            "rmse": 1.5,
            "F": 12.0,
        }
    )

    assert isinstance(res.e, EReturn)
    assert res.e["N"] == 30
    assert res.e.df_m == 2
    assert res.e.df_r == 27
    assert res.e.r2 == 0.75
    assert res.e.r2_a == 0.70

    assert res.llf == pytest.approx(-12.5)
    assert res.rsquared == pytest.approx(0.75)
    assert res.rsquared_adj == pytest.approx(0.70)
    assert res.aic == pytest.approx(100.0)
    assert res.bic == pytest.approx(110.0)

    # An explicit e() scalar wins over the computed default (Stata semantics).
    assert make_results(e={"df_r": 25.0}).e.df_r == 25.0


def test_missing_e_scalars_are_nan_and_unknown_attributes_are_none() -> None:
    res = make_results()

    assert res.e.N == 30
    assert res.e.df_m == 2
    assert res.e.df_r == 27
    assert np.isnan(res.llf)
    assert np.isnan(res.rsquared)
    assert np.isnan(res.rsquared_adj)
    assert np.isnan(res.aic)
    assert np.isnan(res.bic)
    assert res.e.unrecorded_scalar is None
    with pytest.raises(KeyError):
        res.e["unrecorded_scalar"]


def test_cov_params_returns_ndarray() -> None:
    res = make_results()
    cov = res.cov_params()
    assert isinstance(cov, np.ndarray)
    assert isinstance(res.cov, np.ndarray)
    np.testing.assert_allclose(cov, COV)
    np.testing.assert_allclose(res.cov, COV)


def test_invalid_inputs_raise_value_error() -> None:
    with pytest.raises(ValueError):
        make_results(dist="normal")
    with pytest.raises(ValueError):
        make_results(params=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        make_results(cov=np.eye(2))
    with pytest.raises(ValueError):
        make_results(resid=np.zeros(29))
