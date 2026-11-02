"""Tests for the per-estimator Stata header blocks in ``summary(style="stata")``.

The renderer used to infer "is this a likelihood model?" from whether
``e(r2_p)`` was present.  That is wrong: ``qreg`` stores its check-function
pseudo-R2 under the same key, so it was rendered as an ML model and printed an
empty ``Log likelihood = .`` line Stata never prints.  The block is now keyed on
the estimator, and these tests pin the behaviours that changed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def auto():
    df = pd.read_stata("tests/data/auto.dta")
    df["dom"] = (df["foreign"] == "Foreign").astype(int)
    return df


def _survey_frame() -> pd.DataFrame:
    rng = np.random.default_rng(301)
    rows = []
    for stratum in range(1, 9):
        stratum_effect = rng.normal(scale=0.5)
        for psu in range(1, 7):
            psu_effect = rng.normal(scale=0.4)
            weight = float(rng.uniform(0.5, 2.5))
            for _ in range(12):
                x1 = rng.normal()
                y = 1.0 + 0.8 * x1 + stratum_effect + psu_effect + rng.normal(scale=0.6)
                rows.append((stratum, psu, y, x1, weight))
    return pd.DataFrame(rows, columns=["stratum", "psu", "y", "x1", "w"])


# --------------------------------------------------------------- regress / ML
def test_regress_keeps_the_ols_block(auto) -> None:
    text = ws.OLS.from_formula("mpg ~ weight + foreign", auto).fit().summary("stata")

    assert "R-squared" in text
    assert "Adj R-squared" in text
    assert "Root MSE" in text
    assert "Log likelihood" not in text


def test_logit_keeps_the_likelihood_block(auto) -> None:
    text = ws.Logit.from_formula("dom ~ weight + mpg", auto).fit().summary("stata")

    assert "Log likelihood" in text
    assert "Pseudo R2" in text
    assert "R-squared" not in text
    assert "Root MSE" not in text


# ---------------------------------------------------------------------- qreg
def test_qreg_does_not_print_a_log_likelihood(auto) -> None:
    """qreg is not a likelihood model, so the line must be absent."""

    text = (
        ws.QuantileRegression.from_formula("mpg ~ weight", auto, quantile=0.25)
        .fit()
        .summary("stata")
    )

    assert "Log likelihood" not in text


def test_qreg_prints_the_deviation_lines(auto) -> None:
    """Stata prints both sums of deviations (p8_outputs.txt:21-22)."""

    result = (
        ws.QuantileRegression.from_formula("mpg ~ weight", auto, quantile=0.25).fit()
    )
    text = result.summary("stata")

    assert "Raw sum of deviations" in text
    assert "Min sum of deviations" in text
    assert "Pseudo R2" in text
    # The numbers come straight out of e(), not from a re-derivation.
    assert f"{result.e['sum_rdev']:.7g}" in text


def test_qreg_prints_no_model_test_or_r_squared(auto) -> None:
    text = (
        ws.QuantileRegression.from_formula("mpg ~ weight", auto, quantile=0.5)
        .fit()
        .summary("stata")
    )

    assert "F(" not in text
    assert "Root MSE" not in text
    assert "Adj R-squared" not in text


# ----------------------------------------------------------------- svy: mean
def test_svy_header_shows_the_design_block() -> None:
    frame = _survey_frame()
    design = ws.SurveyDesign(
        psu="psu", strata="stratum", weights="w", singleunit="certainty"
    )
    text = ws.SvyMean.from_frame(["y"], frame, design).fit().summary("stata")

    assert "Number of strata" in text
    assert "Number of PSUs" in text
    assert "Population size" in text
    assert "Design df" in text


def test_svy_header_has_no_bogus_root_mse() -> None:
    """``Root MSE`` on a survey mean was a semantic error, not a blank."""

    frame = _survey_frame()
    design = ws.SurveyDesign(psu="psu", strata="stratum", weights="w")
    text = ws.SvyMean.from_frame(["y"], frame, design).fit().summary("stata")

    assert "Root MSE" not in text
    assert "Adj R-squared" not in text


def test_svy_header_counts_obs_once() -> None:
    """Stata prints ``Number of obs`` inside the design block, not twice."""

    frame = _survey_frame()
    design = ws.SurveyDesign(psu="psu", strata="stratum", weights="w")
    text = ws.SvyMean.from_frame(["y"], frame, design).fit().summary("stata")

    assert text.count("Number of obs") == 1


def test_svy_mean_prints_no_model_f_test() -> None:
    frame = _survey_frame()
    design = ws.SurveyDesign(psu="psu", strata="stratum", weights="w")
    text = ws.SvyMean.from_frame(["y"], frame, design).fit().summary("stata")

    assert "F(0" not in text


# ----------------------------------------------------- models with no fit block
def test_arima_prints_no_r_squared_placeholder() -> None:
    """Stata's arima header has no R-squared and no Root MSE -- but DOES log likelihood.

    Measured from ``arima y, arima(1,0,0)``: the block is

      Sample: 1 thru 100                    Number of obs     =
                                            Wald chi2(1)      =
      Log likelihood = -145.8048            Prob > chi2       =

    so "Log likelihood" IS printed, on the left of the two-column block. The old assertion
    required it absent, which the capture contradicts; the two that genuinely do not appear
    are R-squared and Root MSE.
    """

    rng = np.random.default_rng(5)
    y = np.zeros(200)
    for t in range(1, y.size):
        y[t] = 0.7 * y[t - 1] + rng.normal(scale=0.5)
    text = ws.ARIMA(y, order=(1, 0, 0)).fit().summary("stata")

    assert "R-squared" not in text
    assert "Root MSE" not in text
    assert "Log likelihood" in text
    assert "Wald chi2" in text
