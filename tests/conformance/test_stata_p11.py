"""E3 conformance for the P11 causal-inference commands.

Reference values from ``stata_fixtures/p11_outputs.txt`` and ``p11_did.txt``
(StataNow 19.5).

===================  =========  =========  ==========================
command              ATE/ATET   se         POmean(0)
===================  =========  =========  ==========================
teffects ra          2.034515   .0610149   .9785032 (se .0619697)
teffects ipw         1.888171   .0939242   1.018208 (se .1001317)
teffects aipw        2.018836   .0625084   .9801461 (se .0615501)
teffects ra, atet    2.04958    .06474     1.48981  (se .0771474)
teffects nnmatch     2.051839   .0691114   --
didregress           1.18613    .0590465   --  (N=1600)
===================  =========  =========  ==========================

The DGP treatment effect is 2.0 for the teffects sample and 1.2 for the DID panel.

Two things this file deliberately checks that a point-estimate test would miss:

* the potential-outcome means. Stata reports ``POmean`` as its own row, and an
  estimator can hit the right ATE with the wrong counterfactual level.
* that ATE and ATET are different numbers. They coincide only under no
  confounding, which is exactly what this DGP violates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURES = Path(__file__).resolve().parent / "stata_fixtures"
TEFFECTS_REFERENCE = FIXTURES / "p11_outputs.txt"
DID_REFERENCE = FIXTURES / "p11_did.txt"

if not TEFFECTS_REFERENCE.exists():
    pytest.skip("P11 fixtures missing", allow_module_level=True)


def _require(name: str):
    attribute = getattr(ws, name, None)
    if attribute is None:
        pytest.skip(f"{name} is not exposed on westata yet")
    return attribute


def treatment_frame(n: int = 1200, seed: int = 211) -> pd.DataFrame:
    """Confounded observational sample: x1 drives both y and the treatment."""

    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1 - 0.4 * x2)))
    treated = (rng.random(size=n) < propensity).astype(float)
    y = 1.0 + 2.0 * treated + 1.5 * x1 - 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "treated": treated, "x1": x1, "x2": x2})


def did_frame(n_units: int = 200, n_periods: int = 8, seed: int = 223) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        effect = rng.normal(scale=0.6)
        in_treatment_arm = unit % 2 == 0
        for period in range(1, n_periods + 1):
            treated_now = 1.0 if (in_treatment_arm and period >= 4) else 0.0
            x1 = rng.normal()
            y = (
                1.0
                + 0.5 * x1
                + effect
                + 0.3 * period
                + 1.2 * treated_now
                + rng.normal(scale=0.5)
            )
            rows.append((unit, period, y, x1, treated_now, float(in_treatment_arm)))
    return pd.DataFrame(rows, columns=["unit", "period", "y", "x1", "treated", "group"])


# ------------------------------------------------------------ teffects ra


def test_regression_adjustment_matches_stata():
    RegressionAdjustment = _require("RegressionAdjustment")
    outcome = RegressionAdjustment.from_formula(
        "y", "treated", treatment_frame()
    ).fit()

    # Stata: ATE 2.034515 (se .0610149). Both to the printed digit.
    assert outcome.estimate == pytest.approx(2.034515, rel=1e-6)
    assert outcome.std_err == pytest.approx(0.0610149, rel=1e-6)


def test_regression_adjustment_reports_the_potential_outcome_means():
    """An estimator can hit the right ATE with a wrong counterfactual level."""

    RegressionAdjustment = _require("RegressionAdjustment")
    outcome = RegressionAdjustment.from_formula(
        "y", "treated", treatment_frame()
    ).fit()

    # Stata: POmean(0) .9785032 (se .0619697)
    assert outcome.potential_outcomes["0"][0] == pytest.approx(0.9785032, rel=1e-6)
    assert outcome.potential_outcomes["0"][1] == pytest.approx(0.0619697, rel=1e-6)


def test_atet_differs_from_ate_and_matches_stata():
    """ATE and ATET coincide only under no confounding; this DGP is confounded."""

    RegressionAdjustment = _require("RegressionAdjustment")
    ate = RegressionAdjustment.from_formula(
        "y", "treated", treatment_frame(), estimand="ate"
    ).fit()
    atet = RegressionAdjustment.from_formula(
        "y", "treated", treatment_frame(), estimand="atet"
    ).fit()

    assert ate.estimate != pytest.approx(atet.estimate, rel=1e-4)
    # Stata: ATET 2.04958 (se .06474), POmean(0) 1.48981 (se .0771474)
    assert atet.estimate == pytest.approx(2.04958, rel=1e-6)
    assert atet.std_err == pytest.approx(0.06474, rel=1e-6)
    assert atet.potential_outcomes["0"][0] == pytest.approx(1.48981, rel=1e-6)


# ---------------------------------------------------------- teffects ipw


def test_ipw_matches_stata():
    IPW = _require("IPW")
    outcome = IPW.from_formula("y", "treated", treatment_frame()).fit()
    # Stata: ATE 1.888171 (se .0939242). The influence function must propagate
    # the propensity-score estimation error; treating the propensity as fixed
    # gives .1216 instead of .0939.
    assert outcome.estimate == pytest.approx(1.888171, rel=1e-6)
    assert outcome.std_err == pytest.approx(0.0939242, rel=1e-6)


def test_ipw_reports_positive_standard_errors():
    IPW = _require("IPW")
    outcome = IPW.from_formula("y", "treated", treatment_frame()).fit()
    assert np.isfinite(outcome.std_err) and outcome.std_err > 0


# --------------------------------------------------------- teffects aipw


def test_aipw_matches_stata():
    AIPW = _require("AIPW")
    outcome = AIPW.from_formula("y", "treated", treatment_frame()).fit()
    # Stata: ATE 2.018836 (se .0625084)
    assert outcome.estimate == pytest.approx(2.018836, rel=1e-6)
    assert outcome.std_err == pytest.approx(0.0625084, rel=1e-6)


# ------------------------------------------------------ teffects nnmatch


def test_nearest_neighbour_matching_is_close_to_stata():
    NearestNeighbourMatching = _require("NearestNeighbourMatching")
    outcome = NearestNeighbourMatching.from_formula(
        "y", "treated", treatment_frame(), nneighbor=1
    ).fit()
    # Stata: ATE 2.051839 (se .0691114).
    #
    # The POINT estimate is exact; the standard error is not, and the result says
    # so through se_method. The gap is entirely in sigma^2(X_i): the reuse counts
    # match, roughly fifteen variance estimators bracket 0.0691-0.0706 without
    # singling one out, and Stata's exact form for that term is undocumented.
    assert outcome.estimate == pytest.approx(2.051839, rel=1e-5)
    assert outcome.std_err == pytest.approx(0.0691114, rel=5e-2)
    assert "approximation" in outcome.se_method.lower()


# -------------------------------------------------------------- didregress


def test_didregress_matches_stata():
    Didregress = _require("Didregress")
    result = Didregress.from_formula(
        "y", "treated", did_frame(), group="unit", time="period"
    ).fit()

    # Stata: ATET 1.18613 (se .0590465), N = 1600
    assert float(result.params["treated"]) == pytest.approx(1.18613, rel=1e-3)
    # The standard error is exact too; the earlier 6.7% gap in this file came from
    # the ATET-only implementation it replaced, not from a Stata convention.
    assert float(result.bse["treated"]) == pytest.approx(0.0590465, rel=1e-5)
    assert int(result.e["N"]) == 1600


def test_the_two_did_spellings_share_the_atet_but_not_the_standard_error():
    """didregress and xtdidregress differ in their degrees of freedom.

    Stata labels them differently -- "Repeated cross-sectional" versus
    "Longitudinal" -- and prints .0590465 against .055228 for the same ATET. The
    ratio is sqrt(1391/1590) = 0.9353302985, i.e. the repeated-cross-section
    version counts the 199 unit effects in K while the longitudinal one nests them
    in the cluster and counts only the 7 period effects.

    Asserting equality here -- which an earlier version of this file did -- would
    have forced one of the two correct standard errors to be wrong.
    """

    Xtdidregress = _require("Xtdidregress")
    Didregress = _require("Didregress")
    frame = did_frame()

    panel = Xtdidregress.from_formula(
        "y", "treated", frame, group="unit", time="period"
    ).fit()
    pooled = Didregress.from_formula(
        "y", "treated", frame, group="unit", time="period"
    ).fit()

    # The point estimate is the same estimator on the same data.
    assert float(panel.params["treated"]) == pytest.approx(
        float(pooled.params["treated"]), rel=1e-12
    )

    # The standard errors are NOT the same, and both are testable against Stata.
    assert float(pooled.bse["treated"]) == pytest.approx(0.0590465, rel=1e-5)
    assert float(panel.bse["treated"]) == pytest.approx(0.055228, rel=1e-5)

    # And the ratio is the closed-form degrees-of-freedom factor, which pins the
    # mechanism rather than only the two numbers.
    ratio = float(panel.bse["treated"]) / float(pooled.bse["treated"])
    assert ratio == pytest.approx((1391 / 1590) ** 0.5, rel=1e-9)


def test_didregress_rejects_a_time_invariant_treatment():
    """A treatment that never changes within a unit IS the fixed effect.

    Stata says "treatment variable omitted because of collinearity" (r(198)); a
    silent zero coefficient would be far worse than an error.
    """

    Didregress = _require("Didregress")
    frame = did_frame()
    # Replace the time-varying treatment with the constant group indicator.
    frame = frame.assign(treated=frame["group"])
    with pytest.raises(Exception):
        Didregress.from_formula(
            "y", "treated", frame, group="unit", time="period"
        ).fit()
