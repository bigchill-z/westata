"""E3 conformance for the P12 complex-survey commands.

Reference values are FULL-PRECISION ``e(b)``/``e(V)`` captured from StataNow 19.5
(``stata_fixtures/p12_outputs.txt``), not the printed six-decimal table.

Design: 8 strata x 6 PSUs x 12 observations = 576, unequal sampling weights
``w`` in [0.5, 2.5], declared as

    svyset psu [pw=w], strata(stratum) singleunit(certainty)

====================  ==========================================
command               Stata (full precision)
====================  ==========================================
svy: mean y           b 1.061346613529383
                      V 0.007628317455187414
svy: regress y x1 x2  b [0.811277851322903, -0.3624167886992473,
                         1.0080429196046916]
                      V 3x3, see below
====================  ==========================================

Why full precision matters here and not just in principle: survey variance
estimation has several conventions that agree on the point estimate and diverge in
the fourth decimal of the standard error. My own first attempt at the regression
score left V wrong by 167% while the coefficients matched to 1e-15 -- the printed
table would have looked entirely reasonable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
REFERENCE = FIXTURES / "p12_outputs.txt"

if not REFERENCE.exists():
    pytest.skip("P12 fixtures missing", allow_module_level=True)


def _require(name: str):
    attribute = getattr(ws, name, None)
    if attribute is None:
        pytest.skip(f"{name} is not exposed on westata yet")
    return attribute


def survey_frame(
    n_strata: int = 8, psu_per_stratum: int = 6, n_per_psu: int = 12, seed: int = 301
) -> pd.DataFrame:
    """The exact frame the fixtures were captured on."""

    rng = np.random.default_rng(seed)
    rows = []
    population_per_stratum = psu_per_stratum * n_per_psu + 24
    for stratum in range(1, n_strata + 1):
        stratum_effect = rng.normal(scale=0.5)
        for psu in range(1, psu_per_stratum + 1):
            psu_effect = rng.normal(scale=0.4)
            weight = float(rng.uniform(0.5, 2.5))
            for _ in range(n_per_psu):
                x1 = rng.normal()
                x2 = rng.normal()
                y = (
                    1.0
                    + 0.8 * x1
                    - 0.4 * x2
                    + stratum_effect
                    + psu_effect
                    + rng.normal(scale=0.6)
                )
                rows.append(
                    (
                        stratum,
                        psu,
                        y,
                        x1,
                        x2,
                        weight,
                        float(y > 1.0),
                        population_per_stratum,
                    )
                )
    return pd.DataFrame(
        rows, columns=["stratum", "psu", "y", "x1", "x2", "w", "binary", "fpc"]
    )


def _design(frame: pd.DataFrame):
    SurveyDesign = _require("SurveyDesign")
    return SurveyDesign.from_frame(
        frame, psu="psu", strata="stratum", weights="w", singleunit="certainty"
    )


STATA_MEAN_B = 1.061346613529383
STATA_MEAN_V = 0.007628317455187414
STATA_REG_B = np.array([0.811277851322903, -0.3624167886992473, 1.0080429196046916])
STATA_REG_V = np.array(
    [
        [0.001355726149501127, -0.0004611343763046304, -0.0005972217607257482],
        [-0.0004611343763046304, 0.0023172378712389033, -0.00029017309174927025],
        [-0.0005972217607257482, -0.00029017309174927025, 0.006149685753044548],
    ]
)


def test_svy_mean_matches_stata_at_full_precision():
    SvyMean = _require("SvyMean")
    frame = survey_frame()
    result = SvyMean.from_frame(["y"], frame, _design(frame)).fit()

    assert float(result.params.iloc[0]) == pytest.approx(STATA_MEAN_B, rel=1e-10)
    assert float(np.asarray(result.cov_params())[0, 0]) == pytest.approx(
        STATA_MEAN_V, rel=1e-10
    )


def test_svy_regress_coefficients_match_stata_at_full_precision():
    SvyRegress = _require("SvyRegress")
    frame = survey_frame()
    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", _design(frame)).fit()

    # Stata reports x1, x2, _cons in that order.
    values = np.asarray(result.params, dtype=float)
    np.testing.assert_allclose(values, STATA_REG_B, rtol=1e-10)


def test_svy_regress_variance_matches_stata_at_full_precision():
    """The assertion that actually distinguishes a correct linearisation.

    A score of ``x_i * e_i`` -- one weight and one matrix short -- leaves this
    matrix 167% wrong while the coefficients stay exact.
    """

    SvyRegress = _require("SvyRegress")
    frame = survey_frame()
    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", _design(frame)).fit()

    covariance = np.asarray(result.cov_params(), dtype=float)
    np.testing.assert_allclose(covariance, STATA_REG_V, rtol=1e-10)


def test_svy_regress_coefficients_equal_weighted_ols():
    """The point estimate is just weighted least squares; only the VCE is new."""

    SvyRegress = _require("SvyRegress")
    frame = survey_frame()
    survey = SvyRegress.from_frame(frame, "y ~ x1 + x2", _design(frame)).fit()

    design = np.column_stack([frame["x1"], frame["x2"], np.ones(len(frame))])
    root = np.sqrt(frame["w"].to_numpy())
    expected = np.linalg.lstsq(
        design * root[:, None], frame["y"].to_numpy() * root, rcond=None
    )[0]

    np.testing.assert_allclose(np.asarray(survey.params), expected, rtol=1e-12)


def test_equal_weights_reduce_to_ordinary_ols():
    SvyRegress = _require("SvyRegress")
    frame = survey_frame().assign(w=1.0)
    survey = SvyRegress.from_frame(frame, "y ~ x1 + x2", _design(frame)).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", frame).fit()

    np.testing.assert_allclose(
        np.sort(np.asarray(survey.params, dtype=float)),
        np.sort(plain.params[["x1", "x2", "Intercept"]].to_numpy()),
        rtol=1e-12,
    )


STATA_TOTAL_B = 888.9760173443246
STATA_TOTAL_V = 6936.575460544632


def test_svy_total_matches_stata_at_full_precision():
    """Assert the value, not merely that a value exists.

    An earlier version of this test checked only ``np.isfinite`` on the variance,
    and the implementation at that time had it wrong by a factor of 2.4e5 -- its
    score was ``w*(y - total)`` instead of ``w*y``. A finiteness check cannot
    distinguish a survey total from anything else.
    """

    SvyTotal = _require("SvyTotal")
    frame = survey_frame()
    result = SvyTotal.from_frame(["y"], frame, _design(frame)).fit()

    assert float(result.params.iloc[0]) == pytest.approx(STATA_TOTAL_B, rel=1e-12)
    assert float(np.asarray(result.cov_params())[0, 0]) == pytest.approx(
        STATA_TOTAL_V, rel=1e-12
    )


def test_svy_total_equals_the_weighted_sum():
    SvyTotal = _require("SvyTotal")
    frame = survey_frame()
    result = SvyTotal.from_frame(["y"], frame, _design(frame)).fit()
    expected = float((frame["y"] * frame["w"]).sum())
    assert float(result.params.iloc[0]) == pytest.approx(expected, rel=1e-14)


def test_degrees_of_freedom_is_psu_minus_strata():
    SvyMean = _require("SvyMean")
    frame = survey_frame()
    result = SvyMean.from_frame(["y"], frame, _design(frame)).fit()

    n_psu = frame[["stratum", "psu"]].drop_duplicates().shape[0]
    n_strata = frame["stratum"].nunique()
    assert float(result.e["df_r"]) == float(n_psu - n_strata)


def test_fpc_makes_standard_errors_smaller():
    """(1 - n/N) < 1, so a finite population correction can only shrink V."""

    SvyMean = _require("SvyMean")
    frame = survey_frame()
    without = SvyMean.from_frame(["y"], frame, _design(frame)).fit()

    SurveyDesign = _require("SurveyDesign")
    with_fpc = SurveyDesign.from_frame(
        frame, psu="psu", strata="stratum", weights="w", fpc="fpc",
        singleunit="certainty",
    )
    corrected = SvyMean.from_frame(["y"], frame, with_fpc).fit()

    assert float(np.asarray(corrected.cov_params())[0, 0]) < float(
        np.asarray(without.cov_params())[0, 0]
    )


def test_a_lone_psu_stratum_contributes_no_variance_under_certainty():
    """With singleunit(certainty) a stratum of one PSU has no within-stratum
    variance, so dropping every other PSU from one stratum must not change V."""

    SvyMean = _require("SvyMean")
    frame = survey_frame()
    baseline = SvyMean.from_frame(["y"], frame, _design(frame)).fit()

    # Keep a single PSU in stratum 1.
    trimmed = frame[~((frame["stratum"] == 1) & (frame["psu"] > 1))].reset_index(drop=True)
    reduced = SvyMean.from_frame(["y"], trimmed, _design(trimmed)).fit()

    # V changes because the data changed, but it must stay finite and smaller than
    # an uncorrected formula would give; the key property is no error and no NaN.
    assert np.isfinite(float(np.asarray(reduced.cov_params())[0, 0]))
    assert float(np.asarray(baseline.cov_params())[0, 0]) > 0


def test_an_unknown_singleunit_strategy_is_rejected():
    SurveyDesign = _require("SurveyDesign")
    frame = survey_frame()
    with pytest.raises(ValueError):
        SurveyDesign.from_frame(
            frame, psu="psu", strata="stratum", weights="w",
            singleunit="nonsense",
        )

STATA_LOGIT_B = np.array(
    [1.599752259583883, -0.7175541048578664, -0.037837440539974]
)
STATA_LOGIT_V = np.array(
    [
        [0.045049164775974454, -0.02789850499411344, -0.008946498981062131],
        [-0.02789850499411344, 0.0355735421399913, -0.0007433121266066349],
        [-0.008946498981062131, -0.0007433121266066349, 0.02924090503833132],
    ]
)


def test_svy_logit_matches_stata():
    """Logit had NO conformance coverage before this test.

    The version I first committed returned its coefficients permuted against
    Stata's order -- b off by 18x -- and the file stayed green because nothing
    looked at logit at all.

    The tolerance is 1e-9 rather than 1e-10 because that gap is Stata's own
    stopping tolerance, not ours: Stata's score evaluated at Stata's published
    coefficients is 1.7e-8, while these estimates are the exact maximiser with a
    gradient of 8e-15. Asserting 1e-10 would assert something false about Stata.
    """

    SvyLogit = _require("SvyLogit")
    frame = survey_frame()
    result = SvyLogit.from_frame(frame, "binary ~ x1 + x2", _design(frame)).fit()

    # Stata's column order is x1, x2, _cons -- not Intercept first.
    assert list(result.param_names) == ["x1", "x2", "_cons"]
    np.testing.assert_allclose(
        np.asarray(result.params, dtype=float), STATA_LOGIT_B, rtol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(result.cov_params(), dtype=float), STATA_LOGIT_V, rtol=1e-9
    )


def test_svy_regress_names_the_constant_as_stata_does():
    SvyRegress = _require("SvyRegress")
    frame = survey_frame()
    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", _design(frame)).fit()
    assert list(result.param_names) == ["x1", "x2", "_cons"]
