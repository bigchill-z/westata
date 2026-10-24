"""Tests for the ``teffects`` family (P11 task 1).

Reference values are StataNow 19.5 output captured in
``tests/conformance/stata_fixtures/p11_outputs.txt``:

==========================  =========  ==========  ==========
command                     ATE/ATET   se          POmean(0)
==========================  =========  ==========  ==========
teffects ra                 2.034515   .0610149    .9785032 (.0619697)
teffects ipw                1.888171   .0939242    1.018208 (.1001317)
teffects aipw               2.018836   .0625084    .9801461 (.0615501)
teffects ra, atet           2.04958    .06474      1.48981  (.0771474)
teffects nnmatch, n(1)      2.051839   .0691114    --
==========================  =========  ==========  ==========

The DGP effect is 2.0.  The tests below are deliberately stricter about the
regression-adjustment numbers (1e-4 relative) than about the weighting ones
(1e-2): the RA algebra is exact and reproduces every printed digit, while IPW
and AIPW depend on a numerically-fitted logit whose last digits are not
guaranteed to agree with Stata's optimiser.

Standard errors get their own tests.  A wrong influence function leaves the
point estimate perfectly correct, so a suite that only checked ``estimate``
would pass on standard errors that are 30-70% off.  The exact errors are
asserted tightly and the one approximate error is asserted loosely *and*
checked to be labelled approximate on the result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from westata.errors import ConsistencyError
from westata.estimators.treatreg_effects import (
    AIPW,
    IPW,
    NearestNeighbourMatching,
    RegressionAdjustment,
    TreatmentEffectResult,
)

# Stata reference values (p11_outputs.txt).
STATA = {
    "ra": {"ate": 2.034515, "ate_se": 0.0610149, "po0": 0.9785032, "po0_se": 0.0619697},
    "ipw": {"ate": 1.888171, "ate_se": 0.0939242, "po0": 1.018208, "po0_se": 0.1001317},
    "aipw": {
        "ate": 2.018836,
        "ate_se": 0.0625084,
        "po0": 0.9801461,
        "po0_se": 0.0615501,
    },
    "ra_atet": {
        "atet": 2.04958,
        "atet_se": 0.06474,
        "po0": 1.48981,
        "po0_se": 0.0771474,
    },
    "nnmatch": {"ate": 2.051839, "ate_se": 0.0691114},
}

EXACT = "influence function"
APPROX = "Abadie-Imbens approximation"


def treatment_frame(n: int = 1200, seed: int = 211) -> pd.DataFrame:
    """The fixture DGP: ``x1`` drives both the outcome and the treatment.

    The confounding is the point.  ``x1`` raises ``y`` and also raises the
    probability of treatment, so the naive difference in means is biased away
    from the true effect of 2.0 and every estimator has to do real work.
    """

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1 - 0.4 * x2)))
    treated = (rng.random(size=n) < propensity).astype(float)
    y = 1.0 + 2.0 * treated + 1.5 * x1 - 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "treated": treated, "x1": x1, "x2": x2})


# --------------------------------------------------- regression adjustment


def test_regression_adjustment_ate_matches_stata_exactly():
    result = RegressionAdjustment.from_formula("y", "treated", treatment_frame()).fit()

    assert result.estimand == "ATE"
    assert result.estimate == pytest.approx(STATA["ra"]["ate"], rel=1e-4)


def test_regression_adjustment_standard_error_matches_stata():
    """Stata's ``ra`` standard error is an influence function, not an OLS formula.

    Two things go wrong if the derivation is short-cut.  Forgetting that the
    residuals are orthogonal to the design within each arm gives an error 33x too
    small (0.0018); reporting the OLS coefficient covariance instead of the
    influence-function variance gives 0.06063.  Neither is a rounding
    difference, and both leave the point estimate untouched.
    """

    result = RegressionAdjustment.from_formula("y", "treated", treatment_frame()).fit()

    assert result.std_err == pytest.approx(STATA["ra"]["ate_se"], rel=1e-3)
    assert result.se_method == EXACT


def test_regression_adjustment_potential_outcome_mean_matches_stata():
    """The counterfactual level is a separate quantity from the effect.

    An estimator can hit the right ATE while sitting at the wrong level, which is
    why Stata prints ``POmean`` with its own standard error.
    """

    result = RegressionAdjustment.from_formula("y", "treated", treatment_frame()).fit()
    mean0, se0 = result.potential_outcomes["0"]

    assert mean0 == pytest.approx(STATA["ra"]["po0"], rel=1e-4)
    assert se0 == pytest.approx(STATA["ra"]["po0_se"], rel=1e-3)


def test_regression_adjustment_reports_both_potential_outcomes():
    result = RegressionAdjustment.from_formula("y", "treated", treatment_frame()).fit()

    assert set(result.potential_outcomes) == {"0", "1"}
    for level, (mean, error) in result.potential_outcomes.items():
        assert np.isfinite(mean), level
        assert np.isfinite(error) and error > 0.0, level
    # The treated potential outcome must sit above the control one by the ATE.
    difference = result.potential_outcomes["1"][0] - result.potential_outcomes["0"][0]
    assert difference == pytest.approx(result.estimate, rel=1e-10)


def test_atet_differs_from_ate_and_matches_stata():
    """ATE and ATET coincide only without confounding; this DGP is confounded."""

    frame = treatment_frame()
    ate = RegressionAdjustment.from_formula("y", "treated", frame, estimand="ate").fit()
    atet = RegressionAdjustment.from_formula(
        "y", "treated", frame, estimand="atet"
    ).fit()

    assert ate.estimand == "ATE"
    assert atet.estimand == "ATET"
    assert atet.estimate == pytest.approx(STATA["ra_atet"]["atet"], rel=1e-4)
    assert ate.estimate != pytest.approx(atet.estimate, rel=1e-4)
    # The treated subsample is positively selected, so its effect is larger.
    assert atet.estimate > ate.estimate


def test_atet_standard_error_and_potential_outcome_match_stata():
    atet = RegressionAdjustment.from_formula(
        "y", "treated", treatment_frame(), estimand="atet"
    ).fit()
    mean0, se0 = atet.potential_outcomes["0"]

    assert atet.std_err == pytest.approx(STATA["ra_atet"]["atet_se"], rel=1e-3)
    assert mean0 == pytest.approx(STATA["ra_atet"]["po0"], rel=1e-4)
    assert se0 == pytest.approx(STATA["ra_atet"]["po0_se"], rel=1e-3)


def test_atet_potential_outcome_is_conditional_on_the_treated():
    """``POmean(0)`` under ``atet`` is the control mean *for the treated*.

    Stata reports 1.48981 for the ATET against 0.9785 for the ATE.  If the ATET
    reused the ATE's potential-outcome mean the difference would be invisible in
    the effect but wrong in the reported table.
    """

    frame = treatment_frame()
    ate = RegressionAdjustment.from_formula("y", "treated", frame).fit()
    atet = RegressionAdjustment.from_formula(
        "y", "treated", frame, estimand="atet"
    ).fit()

    assert ate.potential_outcomes["0"][0] == pytest.approx(0.9785032, rel=1e-4)
    assert atet.potential_outcomes["0"][0] == pytest.approx(1.48981, rel=1e-4)
    assert atet.potential_outcomes["0"][0] > ate.potential_outcomes["0"][0]


def test_regression_adjustment_with_no_covariates_is_the_difference_in_means():
    """With only an intercept the outcome model is the arm mean, so RA collapses.

    This pins the algebra: the ATE becomes ``mean(y|t=1) - mean(y|t=0)``.
    """

    frame = treatment_frame()
    result = RegressionAdjustment.from_formula(
        "y", "treated", frame, covariates=[]
    ).fit()

    treated = frame["treated"].to_numpy() > 0.5
    y = frame["y"].to_numpy()
    naive = y[treated].mean() - y[~treated].mean()
    assert result.estimate == pytest.approx(naive, rel=1e-10)
    assert result.potential_outcomes["0"][0] == pytest.approx(y[~treated].mean(), rel=1e-10)


# ---------------------------------------------------------------------- ipw


def test_ipw_ate_matches_stata():
    result = IPW.from_formula("y", "treated", treatment_frame()).fit()

    assert result.estimand == "ATE"
    assert result.estimate == pytest.approx(STATA["ipw"]["ate"], rel=1e-2)


def test_ipw_standard_error_matches_stata():
    """The propensity score is estimated, and its error belongs in the sandwich.

    Treating ``p`` as fixed reports 0.1216 for the ATE instead of 0.0939 -- the
    treatment model's contribution is a third of the total variance here.
    """

    result = IPW.from_formula("y", "treated", treatment_frame()).fit()

    assert result.std_err == pytest.approx(STATA["ipw"]["ate_se"], rel=1e-3)
    assert result.se_method == EXACT


def test_ipw_potential_outcome_matches_stata():
    result = IPW.from_formula("y", "treated", treatment_frame()).fit()
    mean0, se0 = result.potential_outcomes["0"]

    assert mean0 == pytest.approx(STATA["ipw"]["po0"], rel=1e-3)
    assert se0 == pytest.approx(STATA["ipw"]["po0_se"], rel=1e-3)


def test_ipw_is_normalised_not_a_plain_sum_of_weights():
    """Stata reports Hajek (normalised) weighted means, not unnormalised sums.

    The two differ materially on this sample: 1.8882 against 1.9355.  An
    implementation that summed ``t*y/p`` and ``t/p`` separately would pass a
    loose test and still be wrong.
    """

    frame = treatment_frame()
    result = IPW.from_formula("y", "treated", frame).fit()

    y = frame["y"].to_numpy()
    t = frame["treated"].to_numpy()
    design = np.column_stack([np.ones(len(frame)), frame["x1"], frame["x2"]])
    beta = np.zeros(design.shape[1])
    for _ in range(200):
        p = expit(design @ beta)
        step = np.linalg.solve(
            (design * (p * (1 - p))[:, None]).T @ design, design.T @ (t - p)
        )
        beta = beta + step
        if np.max(np.abs(step)) < 1e-14:
            break
    p = expit(design @ beta)

    unnormalised = (t * y / p).mean() - ((1 - t) * y / (1 - p)).mean()
    assert unnormalised == pytest.approx(1.9355, rel=5e-3)
    assert result.estimate == pytest.approx(1.888171, rel=1e-2)
    assert result.estimate != pytest.approx(unnormalised, rel=1e-3)


def test_ipw_atet_differs_from_ate():
    """The ATET reweights controls to the treated odds, so it is a different number."""

    frame = treatment_frame()
    ate = IPW.from_formula("y", "treated", frame, estimand="ate").fit()
    atet = IPW.from_formula("y", "treated", frame, estimand="atet").fit()

    assert atet.estimand == "ATET"
    assert atet.estimate != pytest.approx(ate.estimate, rel=1e-4)
    assert np.isfinite(atet.std_err) and atet.std_err > 0.0


def test_ipw_raises_on_a_saturated_propensity_model():
    """A perfectly predicted treatment must raise, not divide by zero.

    With ``x1`` separating the arms the logit drives ``p`` to 0 and 1, so the
    weights ``1/p`` and ``1/(1-p)`` are unbounded.  Clipping to an epsilon would
    return a confident number built from an arbitrary constant, so the estimator
    refuses the sample -- the same stance Stata takes.
    """

    rng = np.random.default_rng(5)
    n = 300
    x1 = rng.normal(size=n)
    # Treatment is a deterministic function of x1: perfect separation.
    treated = (x1 > 0).astype(float)
    frame = pd.DataFrame(
        {"y": 1.0 + 2.0 * treated + x1 + rng.normal(size=n), "treated": treated, "x1": x1}
    )

    with pytest.raises(ConsistencyError, match="perfectly predicted|propensity"):
        IPW.from_formula("y", "treated", frame).fit()


def test_aipw_raises_on_a_saturated_propensity_model_too():
    rng = np.random.default_rng(5)
    n = 300
    x1 = rng.normal(size=n)
    treated = (x1 > 0).astype(float)
    frame = pd.DataFrame(
        {"y": 1.0 + 2.0 * treated + x1 + rng.normal(size=n), "treated": treated, "x1": x1}
    )

    with pytest.raises(ConsistencyError):
        AIPW.from_formula("y", "treated", frame).fit()


# ---------------------------------------------------------------------- aipw


def test_aipw_ate_matches_stata():
    result = AIPW.from_formula("y", "treated", treatment_frame()).fit()

    assert result.estimand == "ATE"
    assert result.estimate == pytest.approx(STATA["aipw"]["ate"], rel=1e-2)


def test_aipw_standard_error_matches_stata():
    """Both the propensity model and the two outcome models are estimated.

    Omitting any of the three coefficient influence functions reports 0.0641
    against Stata's 0.0625084; the omitted terms are negatively correlated with
    the augmented outcome, so the error comes out too *large*, not too small.
    """

    result = AIPW.from_formula("y", "treated", treatment_frame()).fit()

    assert result.std_err == pytest.approx(STATA["aipw"]["ate_se"], rel=1e-3)
    assert result.se_method == EXACT


def test_aipw_potential_outcome_matches_stata():
    result = AIPW.from_formula("y", "treated", treatment_frame()).fit()
    mean0, se0 = result.potential_outcomes["0"]

    assert mean0 == pytest.approx(STATA["aipw"]["po0"], rel=1e-3)
    assert se0 == pytest.approx(STATA["aipw"]["po0_se"], rel=1e-3)


def test_aipw_sits_between_the_regression_and_the_weighting_estimate():
    """Double robustness has a visible signature on this sample.

    ``ra`` (2.0345) and ``ipw`` (1.8882) bracket ``aipw`` (2.0188), which is what
    an augmented estimator is supposed to do when both models are roughly right
    but neither is exact.
    """

    frame = treatment_frame()
    ra = RegressionAdjustment.from_formula("y", "treated", frame).fit().estimate
    ipw = IPW.from_formula("y", "treated", frame).fit().estimate
    aipw = AIPW.from_formula("y", "treated", frame).fit().estimate

    assert min(ra, ipw) < aipw < max(ra, ipw)


def test_aipw_atet_is_reported():
    atet = AIPW.from_formula("y", "treated", treatment_frame(), estimand="atet").fit()

    assert atet.estimand == "ATET"
    assert np.isfinite(atet.estimate)
    assert np.isfinite(atet.std_err) and atet.std_err > 0.0


# Stata reference values for the ATET branch, captured 2026-10-24 by
# retest/fix/systems/capture_aipw_tests.py on THIS module's sample
# (treatment_frame(), n=1200, seed=211):
#   teffects aipw (y x1 x2) (treated x1 x2), atet
#     ATET      2.046425868899699     se .06553792047575811
#     POmean(1) 3.539389746575276     se .07160377318609237
#     POmean(0) 1.4929638776755771    se .07777389087146738
STATA_AIPW_ATET = 2.046425868899699
STATA_AIPW_ATET_SE = 0.06553792047575811
STATA_AIPW_ATET_PO1 = 3.539389746575276
STATA_AIPW_ATET_SE1 = 0.07160377318609237
STATA_AIPW_ATET_PO0 = 1.4929638776755771
STATA_AIPW_ATET_SE0 = 0.07777389087146738


def test_aipw_atet_matches_stata_on_the_po_means_and_the_effect():
    """The ATET branch: an extra 1/p and a vanished augmentation used to live here.

    Stata stores only ``r1vs0.treated`` and ``0.treated`` in ``e(b)``;
    ``POmean(1)`` is their sum and its standard error is
    ``sqrt(V11 + V22 + 2 V12)``.  Before the fix westata averaged
    ``m1 + t*r1/p`` over the treated (a spurious 1/p) and ``m0`` over the treated
    (the ``(1-t)`` factor in augmented0 removed every control contribution), so
    the ATET came out 6.3e-3 off with a 2.9e-2 standard-error error.
    """

    atet = AIPW.from_formula("y", "treated", treatment_frame(),
                             estimand="atet").fit()

    assert atet.estimate == pytest.approx(STATA_AIPW_ATET, rel=1e-12)
    assert atet.std_err == pytest.approx(STATA_AIPW_ATET_SE, rel=1e-9)
    po1, se1 = atet.potential_outcomes["1"]
    po0, se0 = atet.potential_outcomes["0"]
    assert po1 == pytest.approx(STATA_AIPW_ATET_PO1, rel=1e-12)
    assert se1 == pytest.approx(STATA_AIPW_ATET_SE1, rel=1e-9)
    assert po0 == pytest.approx(STATA_AIPW_ATET_PO0, rel=1e-12)
    assert se0 == pytest.approx(STATA_AIPW_ATET_SE0, rel=1e-9)


def test_aipw_ate_is_unaffected_by_the_atet_repair():
    """The ATE branch was already exact (1e-16); pin it so a fix cannot move it.

    ``teffects aipw (y x1 x2) (treated x1 x2)``, captured 2026-10-24:
      ATE 2.0188360574679662 (se .06250843751655079),
      PO(1) 2.9989821296280583 (se .06389270615544881),
      PO(0) 0.9801460721600921 (se .061550089230236575).
    """

    ate = AIPW.from_formula("y", "treated", treatment_frame()).fit()

    assert ate.estimate == pytest.approx(2.0188360574679662, rel=1e-12)
    assert ate.std_err == pytest.approx(0.06250843751655079, rel=1e-9)
    po1, se1 = ate.potential_outcomes["1"]
    po0, se0 = ate.potential_outcomes["0"]
    assert po1 == pytest.approx(2.9989821296280583, rel=1e-12)
    assert se1 == pytest.approx(0.06389270615544881, rel=1e-9)
    assert po0 == pytest.approx(0.9801460721600921, rel=1e-12)
    assert se0 == pytest.approx(0.061550089230236575, rel=1e-9)


# ------------------------------------------------------------------ nnmatch


def test_nearest_neighbour_matching_ate_is_within_two_percent_of_stata():
    """Matching is the one quantity here that is not reproduced to the digit.

    Only because of tie-breaking and the exact covariance estimator: the point
    estimate lands on 2.0518395 against Stata's 2.051839, comfortably inside 2%.
    """

    result = NearestNeighbourMatching.from_formula(
        "y", "treated", treatment_frame(), nneighbor=1
    ).fit()

    assert result.estimand == "ATE"
    assert result.estimate == pytest.approx(STATA["nnmatch"]["ate"], rel=2e-2)


def test_nearest_neighbour_matching_uses_the_mahalanobis_metric():
    """Euclidean distance is 0.25% off; Mahalanobis reproduces Stata.

    ``teffects nnmatch`` defaults to Mahalanobis under the sample covariance, so
    the metric is part of the contract rather than an implementation detail.
    """

    result = NearestNeighbourMatching.from_formula(
        "y", "treated", treatment_frame(), nneighbor=1
    ).fit()

    # Euclidean gives 2.046771; assert we are much closer to Stata than that.
    assert abs(result.estimate - 2.051839) < 0.001
    assert abs(result.estimate - 2.046771) > abs(result.estimate - 2.051839)


def test_nearest_neighbour_matching_gives_the_atet_its_own_variance():
    """The ATET SE must not be the ATE SE under a different label.

    ``teffects nnmatch`` on the 400-observation retest sample, captured
    2026-10-24 (``retest/fix/systems/repro_S4.py``): ATE se .1199330971,
    ATET se .1359368418 -- a ratio of 1.1334.  westata used to report one shared
    ``sum((1+reuse)^2 sigma^2)/n^2`` for both, so the two numbers were bit-for-bit
    identical.  The ATET variance now uses its own weights: treated units enter
    once with 1/n1, re-used controls enter K_i times.
    """

    from westata.estimators.treatreg_effects import NearestNeighbourMatching as NN

    rng = np.random.default_rng(7)
    n = 400
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(0.6 * x1 - 0.4 * x2)))
    treated = (rng.uniform(size=n) < propensity).astype(int)
    y = 2.0 * treated + 1.0 * x1 + 0.5 * x2 + rng.normal(size=n)
    import pandas as pd

    frame = pd.DataFrame(dict(y=y, t=treated, x1=x1, x2=x2))

    ate = NN.from_formula("y", "t", frame, covariates=["x1", "x2"],
                          nneighbor=1, estimand="ate").fit()
    atet = NN.from_formula("y", "t", frame, covariates=["x1", "x2"],
                           nneighbor=1, estimand="atet").fit()

    assert ate.std_err != atet.std_err
    # Stata's own ratio is 1.133439; the weights are now the same shape, so the
    # ratio is within the sigma^2 estimator's documented residual.
    assert atet.std_err / ate.std_err == pytest.approx(1.133439, rel=5e-2)


def test_nearest_neighbour_matching_labels_its_standard_error_approximate():
    """The matching standard error does NOT reproduce Stata exactly.

    Measured 0.0706220 against Stata's 0.0691114 (+2.2%).  The gap lives in the
    estimated conditional variance, whose exact Stata estimator is not publicly
    documented to the digit.  The result must say so rather than present the
    number as conformant, and the test asserts the gap rather than hiding it.
    """

    result = NearestNeighbourMatching.from_formula(
        "y", "treated", treatment_frame(), nneighbor=1
    ).fit()

    assert result.se_method == APPROX
    assert result.se_method != EXACT
    assert result.std_err == pytest.approx(STATA["nnmatch"]["ate_se"], rel=5e-2)


def test_nearest_neighbour_matching_reports_potential_outcomes():
    result = NearestNeighbourMatching.from_formula(
        "y", "treated", treatment_frame(), nneighbor=1
    ).fit()

    assert set(result.potential_outcomes) == {"0", "1"}
    for level, (mean, error) in result.potential_outcomes.items():
        assert np.isfinite(mean), level
        assert np.isfinite(error) and error > 0.0, level


def test_nearest_neighbour_matching_accepts_more_neighbours():
    """``nneighbor`` averages several matches rather than needing an exact tie."""

    result = NearestNeighbourMatching.from_formula(
        "y", "treated", treatment_frame(), nneighbor=3
    ).fit()

    assert np.isfinite(result.estimate)
    assert result.estimate == pytest.approx(2.0, abs=0.2)


def test_nearest_neighbour_matching_rejects_a_nonpositive_neighbour_count():
    with pytest.raises(ValueError, match="nneighbor"):
        NearestNeighbourMatching.from_formula(
            "y", "treated", treatment_frame(), nneighbor=0
        )


def test_nearest_neighbour_matching_atet_is_supported():
    atet = NearestNeighbourMatching.from_formula(
        "y", "treated", treatment_frame(), nneighbor=1, estimand="atet"
    ).fit()

    assert atet.estimand == "ATET"
    assert np.isfinite(atet.estimate)


# ------------------------------------------------------------- common surface


@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (RegressionAdjustment, {}),
        (IPW, {}),
        (AIPW, {}),
        (NearestNeighbourMatching, {"nneighbor": 1}),
    ],
)
def test_every_estimator_returns_a_treatment_effect_result(estimator, kwargs):
    result = estimator.from_formula("y", "treated", treatment_frame(), **kwargs).fit()

    assert isinstance(result, TreatmentEffectResult)
    assert result.estimand in ("ATE", "ATET")
    assert np.isfinite(result.estimate)
    assert np.isfinite(result.std_err) and result.std_err > 0.0
    assert np.isfinite(result.z)
    assert 0.0 <= result.pvalue <= 1.0
    assert result.ci_lower < result.estimate < result.ci_upper
    # The interval must be the normal-based one, not an arbitrary width.
    assert result.ci_upper - result.ci_lower == pytest.approx(
        2 * 1.959963984540054 * result.std_err, rel=1e-9
    )


@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (RegressionAdjustment, {}),
        (IPW, {}),
        (AIPW, {}),
        (NearestNeighbourMatching, {"nneighbor": 1}),
    ],
)
def test_every_estimator_reports_potential_outcomes(estimator, kwargs):
    result = estimator.from_formula("y", "treated", treatment_frame(), **kwargs).fit()

    assert set(result.potential_outcomes) == {"0", "1"}
    for level, value in result.potential_outcomes.items():
        assert isinstance(value, tuple) and len(value) == 2, level
        mean, error = value
        assert np.isfinite(mean), level
        assert np.isfinite(error) and error > 0.0, level


@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (RegressionAdjustment, {}),
        (IPW, {}),
        (AIPW, {}),
        (NearestNeighbourMatching, {"nneighbor": 1}),
    ],
)
def test_standard_errors_are_positive_and_finite_everywhere(estimator, kwargs):
    """A standard error of zero or nan is a silent failure, so assert both."""

    result = estimator.from_formula("y", "treated", treatment_frame(), **kwargs).fit()

    assert np.isfinite(result.std_err)
    assert result.std_err > 0.0
    assert np.isfinite(result.z)
    assert np.isfinite(result.ci_lower) and np.isfinite(result.ci_upper)
    for mean, error in result.potential_outcomes.values():
        assert np.isfinite(error) and error > 0.0
        assert np.isfinite(mean)


@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (RegressionAdjustment, {}),
        (IPW, {}),
        (AIPW, {}),
        (NearestNeighbourMatching, {"nneighbor": 1}),
    ],
)
def test_an_unsupported_estimand_raises_value_error(estimator, kwargs):
    with pytest.raises(ValueError, match="estimand"):
        estimator.from_formula(
            "y", "treated", treatment_frame(), estimand="atex", **kwargs
        )


def test_every_estimator_is_reasonably_close_to_the_true_effect():
    """The DGP effect is 2.0.  A confounded difference in means is not.

    This catches an implementation that "converges" to the naive contrast, which
    is the failure mode all four estimators exist to avoid.
    """

    frame = treatment_frame()
    treated = frame["treated"].to_numpy() > 0.5
    y = frame["y"].to_numpy()
    naive = y[treated].mean() - y[~treated].mean()
    assert naive > 2.15  # the confounding is real and material

    for estimator, kwargs in [
        (RegressionAdjustment, {}),
        (IPW, {}),
        (AIPW, {}),
        (NearestNeighbourMatching, {"nneighbor": 1}),
    ]:
        result = estimator.from_formula("y", "treated", frame, **kwargs).fit()
        assert result.estimate == pytest.approx(2.0, abs=0.15), estimator.__name__


def test_a_single_arm_is_rejected():
    frame = treatment_frame().assign(treated=1.0)
    for estimator, kwargs in [
        (RegressionAdjustment, {}),
        (IPW, {}),
        (AIPW, {}),
        (NearestNeighbourMatching, {"nneighbor": 1}),
    ]:
        with pytest.raises(ConsistencyError, match="arm is empty|two values"):
            estimator.from_formula("y", "treated", frame, **kwargs).fit()


def test_a_three_valued_treatment_is_rejected():
    """Anything other than two levels is not a binary treatment.

    Silently taking "the top level" of a 0/1/2 column would produce a number
    that looks like a treatment effect and is not one.
    """

    frame = treatment_frame().assign(
        treated=np.where(
            treatment_frame()["x1"] > 1.0,
            2.0,
            np.where(treatment_frame()["x1"] > -1.0, 1.0, 0.0),
        )
    )
    with pytest.raises(ConsistencyError, match="two values"):
        RegressionAdjustment.from_formula("y", "treated", frame).fit()


def test_a_zero_one_coded_treatment_is_accepted():
    """Stata's indicator convention: any two distinct values map to 0/1 by order."""

    frame = treatment_frame()
    recoded = frame.assign(treated=np.where(frame["treated"] > 0.5, 7.0, 3.0))
    original = RegressionAdjustment.from_formula("y", "treated", frame).fit()
    remapped = RegressionAdjustment.from_formula("y", "treated", recoded).fit()

    assert remapped.estimate == pytest.approx(original.estimate, rel=1e-12)


def test_the_result_str_shows_the_estimand_and_the_potential_outcomes():
    result = RegressionAdjustment.from_formula("y", "treated", treatment_frame()).fit()
    text = str(result)

    assert "ATE" in text
    assert "POmean(0)" in text
    assert "POmean(1)" in text


def test_the_approximate_note_appears_only_for_matching():
    exact = RegressionAdjustment.from_formula("y", "treated", treatment_frame()).fit()
    approximate = NearestNeighbourMatching.from_formula(
        "y", "treated", treatment_frame(), nneighbor=1
    ).fit()

    assert "note:" not in str(exact)
    assert "note:" in str(approximate)


@pytest.mark.parametrize(
    "estimator, kwargs",
    [
        (RegressionAdjustment, {}),
        (IPW, {}),
        (AIPW, {}),
        (NearestNeighbourMatching, {"nneighbor": 1}),
    ],
)
def test_results_are_immutable(estimator, kwargs):
    """The dataclass is frozen so a reported interval cannot be edited in place."""

    result = estimator.from_formula("y", "treated", treatment_frame(), **kwargs).fit()
    with pytest.raises(Exception):
        result.estimate = 0.0  # type: ignore[misc]
