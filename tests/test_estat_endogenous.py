"""Tests for Stata's ``estat endogenous`` (Durbin-Wu endogeneity test).

Formulas from ``ivregress_estat.ado``'s ``Endog2sls`` program:

    uc    = 2SLS residuals (consistent model)
    ue    = OLS residuals on exog + endog (the efficient model, when every
            endogenous regressor is being tested)
    USSR  = sum(ue^2)
    uePue = mss of (ue ~ exog + endog + addlinst, noconstant)
    ucPuc = mss of (uc ~ exog + addlinst, noconstant)
    Durbin = (uePue - ucPuc) / (USSR/N)
    Wu     = ((uePue-ucPuc)/df) / ((USSR-(uePue-ucPuc))/dfFdenom)

    dfFdenom = N - k - df

``nocons`` is explicit in the ado because ``exog`` already carries the constant.

The Durbin-Wu form requires an unadjusted VCE; under robust/cluster the ado runs
a regression-based variant instead, which is refused here rather than reported
under the Durbin-Wu name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import westata as ws


@pytest.fixture()
def iv_fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length", "displacement", "price"]].dropna()
    return frame, ws.IV2SLS.from_formula(
        "mpg ~ weight + [length ~ displacement + price]", frame
    ).fit()


def _ols(yv: np.ndarray, design: np.ndarray):
    beta, *_ = np.linalg.lstsq(design, yv, rcond=None)
    resid = yv - design @ beta
    rss = float(resid @ resid)
    centred = yv - yv.mean()
    tss = float(centred @ centred)
    return resid, tss - rss  # (residuals, mss)


def _by_hand(frame, result):
    n = len(frame)
    y = frame["mpg"].to_numpy(float)
    const = np.ones(n)
    weight = frame["weight"].to_numpy(float)
    endog = frame["length"].to_numpy(float)
    addlinst = np.column_stack(
        [frame["displacement"].to_numpy(float), frame["price"].to_numpy(float)]
    )
    exog_full = np.column_stack([const, weight])
    exogeff = np.column_stack([const, weight, endog])

    uc = np.asarray(result.resid, dtype=float)
    ue, _ = _ols(y, exogeff)
    ussr = float(ue @ ue)

    _, ue_pue = _ols(ue, np.column_stack([exogeff, addlinst]))
    _, uc_puc = _ols(uc, np.column_stack([exog_full, addlinst]))

    difference = ue_pue - uc_puc
    durbin = difference / (ussr / n)
    k = exog_full.shape[1] + 1  # const + weight + length
    df_denom = n - k - 1.0
    wu = (difference / 1.0) / ((ussr - difference) / df_denom)
    return durbin, wu, df_denom


def test_durbin_matches_the_ado_formula(iv_fit) -> None:
    frame, result = iv_fit
    expected, _, _ = _by_hand(frame, result)

    assert result.estat_endogenous().durbin == pytest.approx(expected, rel=1e-10)


def test_wu_matches_the_ado_formula(iv_fit) -> None:
    frame, result = iv_fit
    _, expected, df_denom = _by_hand(frame, result)

    rendered = result.estat_endogenous()
    assert rendered.wu == pytest.approx(expected, rel=1e-10)
    assert rendered.wu_df == 1.0
    assert rendered.wu_df_denom == float(df_denom)


def test_pvalues_are_upper_tail(iv_fit) -> None:
    _, result = iv_fit
    rendered = result.estat_endogenous()

    assert rendered.durbin_pvalue == pytest.approx(
        float(stats.chi2.sf(rendered.durbin, rendered.wu_df)), rel=1e-12
    )
    assert rendered.wu_pvalue == pytest.approx(
        float(stats.f.sf(rendered.wu, rendered.wu_df, rendered.wu_df_denom)),
        rel=1e-12,
    )


def test_df_denom_is_n_minus_k_minus_df(iv_fit) -> None:
    frame, result = iv_fit
    _, _, expected = _by_hand(frame, result)

    assert result.estat_endogenous().wu_df_denom == pytest.approx(expected)


def test_a_genuinely_endogenous_regressor_is_detected() -> None:
    """The regressor is correlated with the structural error."""

    rng = np.random.default_rng(11)
    n = 800
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    w = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.9 * z1 + 0.4 * z2 + 0.9 * v
    y = 1.0 + 0.5 * w + 2.0 * x + 1.5 * v + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"y": y, "w": w, "x": x, "z1": z1, "z2": z2})

    rendered = ws.IV2SLS.from_formula("y ~ w + [x ~ z1 + z2]", frame).fit().estat_endogenous()

    assert rendered.durbin_pvalue < 0.01
    assert rendered.wu_pvalue < 0.01


def test_an_exogenous_regressor_is_not_flagged() -> None:
    rng = np.random.default_rng(11)
    n = 800
    z1, z2 = rng.normal(size=n), rng.normal(size=n)
    w = rng.normal(size=n)
    v = rng.normal(size=n)
    x = 0.9 * z1 + 0.4 * z2 + 0.9 * v
    y = 1.0 + 0.5 * w + 2.0 * x + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"y": y, "w": w, "x": x, "z1": z1, "z2": z2})

    rendered = ws.IV2SLS.from_formula("y ~ w + [x ~ z1 + z2]", frame).fit().estat_endogenous()

    assert rendered.durbin_pvalue > 0.05


def test_robust_vce_switches_to_the_score_test(iv_fit) -> None:
    """Round 12 refused a robust VCE because the score test was not implemented.

    It is now (see ``tests/test_estat_endogenous_score.py``), so a robust VCE
    reports Wooldridge's robust score rather than an error.
    """

    frame, _ = iv_fit
    robust = ws.IV2SLS.from_formula(
        "mpg ~ weight + [length ~ displacement + price]", frame
    ).fit(cov_type="HC0")

    rendered = robust.estat_endogenous()

    assert rendered.test == "robwool"


def test_non_iv_results_are_refused(iv_fit) -> None:
    frame, _ = iv_fit
    ols_fit = ws.OLS.from_formula("mpg ~ weight", frame).fit()

    with pytest.raises(NotImplementedError, match="instrumental-variables"):
        ols_fit.estat_endogenous()


def test_tested_names_are_reported(iv_fit) -> None:
    rendered = iv_fit[1].estat_endogenous()

    assert rendered.tested == ["length"]
    assert rendered.test_count == 1


def test_to_dict_uses_stata_r_names(iv_fit) -> None:
    payload = iv_fit[1].estat_endogenous().to_dict()

    assert set(payload) == {"durbin", "p_durbin", "wu", "p_wu", "wudf_r"}


def test_output_names_both_statistics(iv_fit) -> None:
    text = str(iv_fit[1].estat_endogenous())

    assert "Tests of endogeneity" in text
    assert "Durbin" in text
    assert "Wu-Hausman" in text
    assert "variables are exogenous" in text
