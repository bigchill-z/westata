"""Tests for complex-survey estimation (``svyset`` + ``svy:``).

The reference values are FULL-PRECISION ``e(b)``/``e(V)`` from StataNow 19.5
(``tests/conformance/stata_fixtures/p12_outputs.txt``), never the printed
six-decimal table.  That distinction is the whole point of this module: survey
variance conventions agree on the point estimate and separate in the fourth
decimal of a standard error.

Design under test (``design A``): 8 strata x 6 PSUs x 12 observations = 576,
unequal sampling weights ``w`` in [0.5, 2.5],

    svyset psu [pw=w], strata(stratum) singleunit(certainty)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError
from westata.estimators.survey import (
    SurveyDesign,
    SvyLogit,
    SvyMean,
    SvyProportion,
    SvyRegress,
    SvyTotal,
)

# ---------------------------------------------------------------------------
# The captured design and its full-precision targets
# ---------------------------------------------------------------------------
STATA_MEAN_B = np.array([1.061346613529383])
STATA_MEAN_V = np.array([[0.007628317455187414]])

STATA_TOTAL_B = np.array([888.9760173443246])
STATA_TOTAL_V = np.array([[6936.575460544632]])

STATA_REG_B = np.array(
    [0.811277851322903, -0.3624167886992473, 1.0080429196046916]
)
STATA_REG_V = np.array(
    [
        [0.001355726149501127, -0.0004611343763046304, -0.0005972217607257482],
        [-0.0004611343763046304, 0.0023172378712389033, -0.00029017309174927025],
        [-0.0005972217607257482, -0.00029017309174927025, 0.006149685753044548],
    ]
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
#: Printed-only target for ``svy: proportion binary`` (levels 0 and 1).
STATA_PROPORTION_SE = 0.0307108
STATA_PROPORTION_LEVEL0 = 0.4903781
STATA_PROPORTION_LEVEL1 = 0.5096219
#: Printed-only targets for the fpc design (``design B``).
STATA_FPC_MEAN_SE = 0.0845668

#: Machine-precision target: ``mean`` is pure algebra over the fixture values.
EXACT = 1e-10


def survey_frame(
    n_strata: int = 8,
    psu_per_stratum: int = 6,
    n_per_psu: int = 12,
    seed: int = 301,
) -> pd.DataFrame:
    """The exact frame the Stata fixtures were captured on."""

    rng = np.random.default_rng(seed)
    rows = []
    # Population per stratum exceeds the sample, so the fpc is below one.
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


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return survey_frame()


def design_for(frame: pd.DataFrame, **overrides) -> SurveyDesign:
    options: dict[str, object] = {
        "psu": "psu",
        "strata": "stratum",
        "weights": "w",
        "singleunit": "certainty",
    }
    options.update(overrides)
    return SurveyDesign.from_frame(frame, **options)


def one_lone_psu(frame: pd.DataFrame) -> pd.DataFrame:
    """The fixture frame with stratum 1 reduced to a single PSU."""

    mask = ~((frame["stratum"] == 1) & (frame["psu"] > 1))
    return frame[mask].reset_index(drop=True)


def largest_relative_gap(actual, expected) -> float:
    """Max elementwise ``|a - e| / |e|``; the number reported in the summary."""

    a = np.asarray(actual, dtype=float).reshape(-1)
    e = np.asarray(expected, dtype=float).reshape(-1)
    return float(np.max(np.abs(a - e) / np.abs(e)))


# ---------------------------------------------------------------------------
# svy: mean
# ---------------------------------------------------------------------------
def test_svy_mean_matches_stata_at_full_precision(frame):
    result = SvyMean.from_frame(["y"], frame, design_for(frame)).fit()

    assert largest_relative_gap(result.params, STATA_MEAN_B) < EXACT
    assert largest_relative_gap(result.cov_params(), STATA_MEAN_V) < EXACT


def test_mean_requires_the_weight_inside_the_influence(frame):
    """Guard the exact spelling of the influence function.

    ``psi_i = y_i / sum(w)`` (no weight, no centring) is 65% low, and
    ``psi_i = w_i y_i / sum(w)`` (weight, no centring) is 30% low, on this
    design -- while the point estimate is identical for all three.  Only the
    weighted, centred form reproduces Stata, so this test pins the formula
    rather than merely its output.
    """

    fitted = SvyMean.from_frame(["y"], frame, design_for(frame)).fit()
    layout = design_for(frame).layout(frame)
    y = frame["y"].to_numpy(dtype=float)
    w = layout.weights
    centre = float(w @ y / w.sum())

    unweighted = layout.linearized_variance(y / w.sum())
    weighted_only = layout.linearized_variance(w * y / w.sum())

    assert largest_relative_gap(fitted.cov_params(), STATA_MEAN_V) < EXACT
    # ... and both wrong spellings are demonstrably far off.
    assert abs(unweighted[0, 0] - STATA_MEAN_V[0, 0]) / STATA_MEAN_V[0, 0] > 0.5
    assert abs(weighted_only[0, 0] - STATA_MEAN_V[0, 0]) / STATA_MEAN_V[0, 0] > 0.1
    # The centred influence is the verified one.
    centred = layout.linearized_variance(w * (y - centre) / w.sum())
    assert largest_relative_gap(centred, STATA_MEAN_V) < EXACT


def test_svy_mean_equals_the_weighted_average(frame):
    result = SvyMean.from_frame(["y"], frame, design_for(frame)).fit()
    expected = float(frame["w"] @ frame["y"] / frame["w"].sum())

    assert float(result.params.iloc[0]) == pytest.approx(expected, rel=1e-14)


def test_mean_reports_the_population_size(frame):
    result = SvyMean.from_frame(["y"], frame, design_for(frame)).fit()

    # Stata prints "Population size = 837.592551".
    assert float(result.e["N_pop"]) == pytest.approx(837.592551, rel=1e-9)


# ---------------------------------------------------------------------------
# svy: total
# ---------------------------------------------------------------------------
def test_svy_total_matches_stata_at_full_precision(frame):
    result = SvyTotal.from_frame(["y"], frame, design_for(frame)).fit()

    assert largest_relative_gap(result.params, STATA_TOTAL_B) < EXACT
    assert largest_relative_gap(result.cov_params(), STATA_TOTAL_V) < EXACT


def test_total_influence_function_is_uncentred(frame):
    """``psi_i = w_i y_i`` -- centring it on the total wrecks the variance.

    ``w_i (y_i - T)`` changes the point estimate not at all but inflates ``V``
    by a factor of ~244,000 on this design, because ``T`` is a global constant
    interacting with unequal weights.  The gap is asserted so a future
    "simplification" of the formula fails loudly.
    """

    fitted = SvyTotal.from_frame(["y"], frame, design_for(frame)).fit()
    layout = design_for(frame).layout(frame)
    y = frame["y"].to_numpy(dtype=float)
    w = layout.weights
    total = float(w @ y)

    centred = layout.linearized_variance(w * (y - total))

    assert largest_relative_gap(fitted.cov_params(), STATA_TOTAL_V) < EXACT
    ratio = float(centred[0, 0] / STATA_TOTAL_V[0, 0])
    assert ratio > 1e4, f"centring the total only scaled V by {ratio:g}"


def test_total_equals_the_weighted_sum(frame):
    result = SvyTotal.from_frame(["y"], frame, design_for(frame)).fit()

    assert float(result.params.iloc[0]) == pytest.approx(
        float(frame["w"] @ frame["y"]), rel=1e-14
    )


def test_total_is_the_mean_times_the_weight_sum(frame):
    """An identity that does hold: ``T = ybar * sum(w)``.

    The corresponding variance identity does not hold -- the stratified cluster
    variance accumulates inside PSUs and across strata, neither of which factors
    through a constant -- so only the point estimates are related.
    """

    design = design_for(frame)
    total = SvyTotal.from_frame(["y"], frame, design).fit()
    mean = SvyMean.from_frame(["y"], frame, design).fit()

    weight_sum = float(frame["w"].sum())
    assert float(total.params.iloc[0]) == pytest.approx(
        float(mean.params.iloc[0]) * weight_sum, rel=1e-12
    )


# ---------------------------------------------------------------------------
# svy: regress
# ---------------------------------------------------------------------------
def test_svy_regress_coefficients_match_stata_at_full_precision(frame):
    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", design_for(frame)).fit()

    assert largest_relative_gap(result.params, STATA_REG_B) < EXACT


def test_svy_regress_variance_matches_stata_at_full_precision(frame):
    """The assertion that separates a correct linearisation from a plausible one.

    A score of ``x_i e_i`` -- one weight and one inverse short -- leaves this
    matrix 167% wrong while the coefficients stay exact to 1e-15.
    """

    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", design_for(frame)).fit()

    assert largest_relative_gap(result.cov_params(), STATA_REG_V) < EXACT


def test_regress_score_needs_both_the_weight_and_the_inverse(frame):
    """Both halves of ``(X'WX)^-1 w_i x_i e_i`` are load-bearing."""

    fitted = SvyRegress.from_frame(frame, "y ~ x1 + x2", design_for(frame)).fit()
    layout = design_for(frame).layout(frame)
    X = np.column_stack([frame["x1"], frame["x2"], np.ones(len(frame))])
    y = frame["y"].to_numpy(dtype=float)
    w = layout.weights
    xtwx_inv = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = xtwx_inv @ (X.T @ (w * y))
    resid = y - X @ beta

    wrong = layout.linearized_variance(X * resid[:, None])  # x_i e_i
    right = layout.linearized_variance((X * (w * resid)[:, None]) @ xtwx_inv.T)

    assert largest_relative_gap(right, STATA_REG_V) < EXACT
    assert largest_relative_gap(wrong, STATA_REG_V) > 1.0


def test_regress_coefficients_equal_weighted_least_squares(frame):
    """The point estimate is weighted OLS; only the VCE is new."""

    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", design_for(frame)).fit()
    plain = ws.OLS.from_formula(
        "y ~ x1 + x2", frame, weights=frame["w"], weight_type="pweight"
    ).fit()

    np.testing.assert_allclose(
        np.asarray(result.params, dtype=float),
        plain.params[["x1", "x2", "Intercept"]].to_numpy(),
        rtol=1e-12,
    )


def test_regress_coefficients_equal_hand_rolled_wls(frame):
    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", design_for(frame)).fit()

    design = np.column_stack([frame["x1"], frame["x2"], np.ones(len(frame))])
    root = np.sqrt(frame["w"].to_numpy(dtype=float))
    expected = np.linalg.lstsq(
        design * root[:, None], frame["y"].to_numpy(dtype=float) * root, rcond=None
    )[0]

    np.testing.assert_allclose(np.asarray(result.params, dtype=float), expected, rtol=1e-12)


def test_equal_weights_reduce_to_ordinary_ols():
    """With constant weights the design is still stratified, but the
    coefficients must collapse to plain OLS."""

    flat = survey_frame().assign(w=1.0)
    survey = SvyRegress.from_frame(flat, "y ~ x1 + x2", design_for(flat)).fit()
    plain = ws.OLS.from_formula("y ~ x1 + x2", flat).fit()

    expected = plain.params[["x1", "x2", "Intercept"]].to_numpy(dtype=float)
    np.testing.assert_allclose(
        np.asarray(survey.params, dtype=float), expected, rtol=1e-12
    )


def test_regress_reports_slopes_before_the_constant(frame):
    """Stata's coefficient order: slopes first, then ``_cons``."""

    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", design_for(frame)).fit()

    assert list(result.param_names) == ["x1", "x2", "_cons"]
    # And the reported values follow that same order.
    assert largest_relative_gap(result.params, STATA_REG_B) < EXACT


def test_regress_reports_r_squared(frame):
    """Stata prints ``R-squared = 0.5090``."""

    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", design_for(frame)).fit()

    assert float(result.e["r2"]) == pytest.approx(0.5090, abs=5e-5)


def test_regress_reports_stata_survey_f(frame):
    """Stata prints ``F(2, 39) = 238.36``; the test uses its survey formula."""

    result = SvyRegress.from_frame(frame, "y ~ x1 + x2", design_for(frame)).fit()

    assert float(result.e["df_m"]) == 2.0
    assert float(result.e["F"]) == pytest.approx(238.36, rel=2e-4)


# ---------------------------------------------------------------------------
# svy: logit
# ---------------------------------------------------------------------------
def test_svy_logit_matches_stata(frame):
    """Tolerance is ``1e-9``, not ``1e-10``: the residual is Stata's tolerance.

    Evaluating Stata's own score at Stata's published coefficients leaves a
    gradient of 1.7e-8; the estimates here are the exact maximiser (8e-15).
    Demanding 1e-10 would be asserting a property Stata does not have.
    """

    result = SvyLogit.from_frame(frame, "binary ~ x1 + x2", design_for(frame)).fit()

    assert largest_relative_gap(result.params, STATA_LOGIT_B) < 1e-9
    assert largest_relative_gap(result.cov_params(), STATA_LOGIT_V) < 1e-9


def test_logit_estimates_solve_the_weighted_score_equations(frame):
    """The returned coefficients are the exact pseudo-ML maximiser."""

    result = SvyLogit.from_frame(frame, "binary ~ x1 + x2", design_for(frame)).fit()
    layout = design_for(frame).layout(frame)
    X = np.column_stack([frame["x1"], frame["x2"], np.ones(len(frame))])
    y = frame["binary"].to_numpy(dtype=float)
    beta = np.asarray(result.params, dtype=float)

    probabilities = 1.0 / (1.0 + np.exp(-(X @ beta)))
    gradient = X.T @ (layout.weights * (y - probabilities))

    assert float(np.max(np.abs(gradient))) < 1e-10


def test_logit_reports_a_likelihood_and_ordered_names(frame):
    result = SvyLogit.from_frame(frame, "binary ~ x1 + x2", design_for(frame)).fit()

    assert np.isfinite(float(result.e["ll"]))
    assert list(result.param_names) == ["x1", "x2", "_cons"]


# ---------------------------------------------------------------------------
# svy: proportion
# ---------------------------------------------------------------------------
def test_svy_proportion_matches_stata(frame):
    result = SvyProportion.from_frame(["binary"], frame, design_for(frame)).fit()

    values = np.asarray(result.params, dtype=float)
    # Stata labels levels as printed: ``binary=0`` / ``binary=1`` for a 0/1 variable.
    assert result.param_names == ["binary=0", "binary=1"]
    assert float(values[0]) == pytest.approx(STATA_PROPORTION_LEVEL0, rel=1e-6)
    assert float(values[1]) == pytest.approx(STATA_PROPORTION_LEVEL1, rel=1e-6)
    assert float(result.bse.iloc[0]) == pytest.approx(STATA_PROPORTION_SE, rel=1e-6)


def test_proportion_levels_sum_to_one_and_are_perfectly_negatively_correlated(frame):
    result = SvyProportion.from_frame(["binary"], frame, design_for(frame)).fit()
    covariance = np.asarray(result.cov_params(), dtype=float)

    assert float(np.asarray(result.params, dtype=float).sum()) == pytest.approx(
        1.0, rel=1e-12
    )
    # A binary variable's two levels partition every observation, so their
    # estimates are perfectly negatively correlated.
    correlation = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
    assert correlation == pytest.approx(-1.0, abs=1e-10)


def test_proportion_has_one_parameter_per_level(frame):
    multi = frame.assign(three=np.where(frame["y"] > 1.0, "a", np.where(frame["y"] > 0, "b", "c")))
    result = SvyProportion.from_frame(["three"], multi, design_for(multi)).fit()

    assert len(result.param_names) == 3
    assert float(np.asarray(result.params, dtype=float).sum()) == pytest.approx(
        1.0, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Degrees of freedom
# ---------------------------------------------------------------------------
def test_degrees_of_freedom_is_psu_minus_strata(frame):
    """``df_r = #PSU - #strata`` = 48 - 8 = 40, as Stata prints ``Design df 40``."""

    result = SvyMean.from_frame(["y"], frame, design_for(frame)).fit()

    assert float(result.e["df_r"]) == 48.0 - 8.0
    assert float(result.df_resid) == 48.0 - 8.0
    assert result.dist == "t"


def test_design_dof_is_not_the_observation_count(frame):
    """Using ``N`` instead of ``#PSU - #strata`` inflates every t statistic."""

    result = SvyMean.from_frame(["y"], frame, design_for(frame)).fit()

    assert result.nobs == 576
    assert float(result.e["df_r"]) != float(result.nobs)


def test_all_five_commands_report_the_same_design_dof(frame):
    design = design_for(frame)

    for result in (
        SvyMean.from_frame(["y"], frame, design).fit(),
        SvyTotal.from_frame(["y"], frame, design).fit(),
        SvyProportion.from_frame(["binary"], frame, design).fit(),
        SvyRegress.from_frame(frame, "y ~ x1 + x2", design).fit(),
        SvyLogit.from_frame(frame, "binary ~ x1 + x2", design).fit(),
    ):
        assert float(result.e["df_r"]) == 40.0
        assert result.dist == "t"


def test_model_names_follow_stata(frame):
    design = design_for(frame)

    assert SvyMean.from_frame(["y"], frame, design).fit().model_name == "svy: mean"
    assert SvyTotal.from_frame(["y"], frame, design).fit().model_name == "svy: total"
    assert (
        SvyProportion.from_frame(["binary"], frame, design).fit().model_name
        == "svy: proportion"
    )
    assert (
        SvyRegress.from_frame(frame, "y ~ x1 + x2", design).fit().model_name
        == "svy: regress"
    )
    assert (
        SvyLogit.from_frame(frame, "binary ~ x1 + x2", design).fit().model_name
        == "svy: logit"
    )


# ---------------------------------------------------------------------------
# Finite population correction
# ---------------------------------------------------------------------------
def test_fpc_shrinks_the_variance(frame):
    """``(1 - n/N) < 1``, so a finite population correction can only shrink V."""

    without = SvyMean.from_frame(["y"], frame, design_for(frame)).fit()
    corrected = SvyMean.from_frame(
        ["y"], frame, design_for(frame, fpc="fpc")
    ).fit()

    assert float(corrected.cov_params()[0, 0]) < float(without.cov_params()[0, 0])


def test_fpc_matches_the_stata_standard_error(frame):
    """The fpc design is ``design A`` scaled by ``(1 - 6/96) = 0.9375``."""

    without = SvyMean.from_frame(["y"], frame, design_for(frame)).fit()
    corrected = SvyMean.from_frame(["y"], frame, design_for(frame, fpc="fpc")).fit()

    # 6 PSUs sampled out of a population of 96 per stratum.
    factor = 1.0 - 6.0 / 96.0
    assert float(corrected.cov_params()[0, 0]) == pytest.approx(
        float(without.cov_params()[0, 0]) * factor, rel=1e-12
    )
    assert float(corrected.bse.iloc[0]) == pytest.approx(STATA_FPC_MEAN_SE, rel=1e-6)


def test_fpc_leaves_the_point_estimate_untouched(frame):
    without = SvyMean.from_frame(["y"], frame, design_for(frame)).fit()
    corrected = SvyMean.from_frame(["y"], frame, design_for(frame, fpc="fpc")).fit()

    assert float(corrected.params.iloc[0]) == pytest.approx(
        float(without.params.iloc[0]), rel=1e-15
    )


# ---------------------------------------------------------------------------
# Single PSU strata
# ---------------------------------------------------------------------------
def test_a_lone_psu_stratum_contributes_zero_under_certainty(frame):
    """The defining property of ``singleunit(certainty)``.

    Adding a lone-PSU stratum must change ``V`` only through the totals it moves
    (the point estimate), never through a within-stratum term of its own.  The
    clean check is that its contribution is *exactly* zero.
    """

    trimmed = one_lone_psu(frame)
    layout = design_for(trimmed).layout(trimmed)
    y = trimmed["y"].to_numpy(dtype=float)
    w = layout.weights
    centre = float(w @ y / w.sum())
    influence = w * (y - centre) / w.sum()

    totals = layout.psu_totals(influence)
    lone_stratum = [s for s, psus in layout.psus_by_stratum.items() if psus.size == 1]
    assert lone_stratum, "the trimmed frame should contain a lone PSU"

    full = layout.linearized_variance(influence)
    without = 0.0
    for stratum, psus in layout.psus_by_stratum.items():
        if stratum in lone_stratum:
            continue
        n_h = int(psus.size)
        block = totals[psus]
        centred = block - block.mean(axis=0)
        without += (n_h / (n_h - 1.0)) * float((centred.T @ centred).item())

    # certainty: the lone stratum adds literally nothing.
    assert float(full[0, 0]) == pytest.approx(without, rel=1e-12)


def test_certainty_equals_dropping_the_lone_stratum_variance(frame):
    """Zero contribution means the lone stratum cannot move V beyond the mean."""

    trimmed = one_lone_psu(frame)
    layout = design_for(trimmed).layout(trimmed)

    # Keep only the lone PSU's influence: its variance contribution is 0.
    y = trimmed["y"].to_numpy(dtype=float)
    w = layout.weights
    centre = float(w @ y / w.sum())
    influence = (w * (y - centre) / w.sum()).reshape(-1, 1)

    certainty = layout.linearized_variance(influence, singleunit="certainty")
    zeros = np.zeros_like(influence)
    for stratum, psus in layout.psus_by_stratum.items():
        if psus.size == 1:
            zeros[layout.psu_codes == psus[0]] = influence[layout.psu_codes == psus[0]]

    assert float(layout.linearized_variance(zeros, singleunit="certainty")[0, 0]) == 0.0
    assert float(certainty[0, 0]) > 0.0  # the other strata still contribute


def test_the_singleunit_strategies_differ(frame):
    trimmed = one_lone_psu(frame)
    design = design_for(trimmed)

    variances = {}
    for strategy in ("certainty", "scaling", "centered"):
        result = SvyMean.from_frame(
            ["y"], trimmed, design_for(trimmed, singleunit=strategy)
        ).fit()
        variances[strategy] = float(np.asarray(result.cov_params())[0, 0])

    assert variances["certainty"] != pytest.approx(variances["scaling"], rel=1e-9)
    assert variances["certainty"] != pytest.approx(variances["centered"], rel=1e-9)
    assert variances["scaling"] != pytest.approx(variances["centered"], rel=1e-9)
    # certainty contributes least: it adds nothing for the lone stratum.
    assert variances["certainty"] < variances["scaling"]
    assert variances["certainty"] < variances["centered"]
    # ... and the design object is unchanged by the per-call override.
    assert design.singleunit == "certainty"


def test_singleunit_missing_produces_nan(frame):
    trimmed = one_lone_psu(frame)
    result = SvyMean.from_frame(
        ["y"], trimmed, design_for(trimmed, singleunit="missing")
    ).fit()

    assert np.isnan(float(np.asarray(result.cov_params())[0, 0]))


def test_an_unknown_singleunit_strategy_is_rejected(frame):
    with pytest.raises(ValueError, match="singleunit"):
        SurveyDesign.from_frame(frame, psu="psu", strata="stratum", singleunit="nonsense")


def test_an_unknown_singleunit_strategy_is_rejected_at_construction(frame):
    with pytest.raises(ValueError, match="singleunit"):
        SurveyDesign(psu="psu", singleunit="nonsense")


def test_singleunit_is_normalised():
    assert SurveyDesign(psu="psu", singleunit="  Certainty ").singleunit == "certainty"


# ---------------------------------------------------------------------------
# Design object and input validation
# ---------------------------------------------------------------------------
def test_design_rejects_a_missing_column(frame):
    with pytest.raises(ConsistencyError, match="not in the data"):
        SurveyDesign.from_frame(frame, psu="nonexistent")


def test_design_rejects_non_positive_weights(frame):
    broken = frame.copy()
    broken.loc[0, "w"] = 0.0
    with pytest.raises(ConsistencyError, match="strictly positive"):
        design_for(broken).weight_vector(broken)


def test_design_rejects_negative_weights(frame):
    broken = frame.copy()
    broken.loc[0, "w"] = -1.0
    with pytest.raises(ConsistencyError, match="strictly positive"):
        design_for(broken).weight_vector(broken)


def test_design_is_frozen(frame):
    design = design_for(frame)
    with pytest.raises(Exception):
        design.psu = "other"  # type: ignore[misc]


def test_weights_default_to_one_when_undeclared(frame):
    design = SurveyDesign.from_frame(frame, psu="psu", strata="stratum")
    weights = design.weight_vector(frame)

    np.testing.assert_array_equal(weights, np.ones(len(frame)))


def test_psu_labels_may_repeat_across_strata(frame):
    """``psu`` numbering restarts per stratum, as in Stata; codes must separate."""

    design = design_for(frame)
    layout = design.layout(frame)

    assert layout.n_psu == 48  # not 6: each (stratum, psu) pair is a PSU
    assert layout.n_strata == 8


def test_svyset_applies_to_any_frame_with_the_declared_columns(frame):
    """A design holds names, not data."""

    design = design_for(frame)
    smaller = frame[frame["stratum"].isin([1, 2, 3])].reset_index(drop=True)

    result = SvyMean.from_frame(["y"], smaller, design).fit()

    assert float(result.e["N_psu"]) == 18
    assert float(result.e["N_strata"]) == 3
    assert float(result.e["df_r"]) == 15
