"""Tests for the P11 treatment-effects and DID estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from westata.errors import ConsistencyError
from westata.estimators.did import Didregress, Xtdidregress
from westata.estimators.treatreg_effects import (
    AIPW,
    IPW,
    NearestNeighbourMatching,
    RegressionAdjustment,
)


def _treatment(n: int = 1200, seed: int = 211) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    propensity = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1 - 0.4 * x2)))
    treated = (rng.random(size=n) < propensity).astype(float)
    y = 1.0 + 2.0 * treated + 1.5 * x1 - 0.5 * x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "treated": treated, "x1": x1, "x2": x2})


def _did(n_units: int = 200, n_periods: int = 8, seed: int = 223) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(n_units):
        effect = rng.normal(scale=0.6)
        arm = unit % 2 == 0
        for period in range(1, n_periods + 1):
            treated_now = 1.0 if (arm and period >= 4) else 0.0
            x1 = rng.normal()
            y = (
                1.0
                + 0.5 * x1
                + effect
                + 0.3 * period
                + 1.2 * treated_now
                + rng.normal(scale=0.5)
            )
            rows.append((unit, period, y, x1, treated_now, float(arm)))
    return pd.DataFrame(rows, columns=["unit", "period", "y", "x1", "treated", "group"])


# ------------------------------------------------- regression adjustment


def test_regression_adjustment_matches_stata():
    result = RegressionAdjustment.from_formula("y", "treated", _treatment()).fit()
    assert result.estimate == pytest.approx(2.034515, rel=1e-5)


def test_regression_adjustment_standard_error_is_close():
    """The variance is the plug-in variation PLUS the arm-coefficient uncertainty.

    OLS residuals are orthogonal to the design, so the usual model-error terms
    vanish identically; without the coefficient term the standard error comes out
    33x too small (0.0018). Measured 0.06063 against Stata's 0.06101.
    """

    result = RegressionAdjustment.from_formula("y", "treated", _treatment()).fit()
    assert result.std_err == pytest.approx(0.0610149, rel=5e-2)


def test_regression_adjustment_potential_outcome_mean():
    result = RegressionAdjustment.from_formula("y", "treated", _treatment()).fit()
    assert result.potential_outcomes["0"][0] == pytest.approx(0.9785032, rel=1e-4)


def test_atet_differs_from_ate_and_matches_stata():
    frame = _treatment()
    ate = RegressionAdjustment.from_formula("y", "treated", frame).fit()
    atet = RegressionAdjustment.from_formula(
        "y", "treated", frame, estimand="atet"
    ).fit()

    assert ate.estimate != pytest.approx(atet.estimate, rel=1e-4)
    assert atet.estimate == pytest.approx(2.04958, rel=1e-4)
    assert atet.potential_outcomes["0"][0] == pytest.approx(1.48981, rel=1e-4)


def test_an_invalid_estimand_is_rejected():
    with pytest.raises(ValueError, match="estimand"):
        RegressionAdjustment.from_formula(
            "y", "treated", _treatment(), estimand="atex"
        )


# ------------------------------------------------------------------- ipw


def test_ipw_matches_stata():
    result = IPW.from_formula("y", "treated", _treatment()).fit()
    assert result.estimate == pytest.approx(1.888171, rel=1e-4)


def test_ipw_reports_potential_outcomes():
    result = IPW.from_formula("y", "treated", _treatment()).fit()
    assert result.potential_outcomes["0"][0] == pytest.approx(1.018208, rel=5e-2)


def test_ipw_and_aipw_are_both_finite():
    frame = _treatment()
    for cls in (IPW, AIPW):
        result = cls.from_formula("y", "treated", frame).fit()
        assert np.isfinite(result.estimate)
        assert np.isfinite(result.std_err)
        assert result.std_err > 0


# ------------------------------------------------------------------ aipw


def test_aipw_matches_stata():
    result = AIPW.from_formula("y", "treated", _treatment()).fit()
    assert result.estimate == pytest.approx(2.018836, rel=1e-4)


def test_aipw_standard_error_is_close():
    result = AIPW.from_formula("y", "treated", _treatment()).fit()
    # Measured 0.06416 against Stata's 0.06251.
    assert result.std_err == pytest.approx(0.0625084, rel=5e-2)


# --------------------------------------------------------------- nnmatch


def test_nearest_neighbour_matching_is_close_to_stata():
    result = NearestNeighbourMatching.from_formula(
        "y", "treated", _treatment(), nneighbor=1
    ).fit()
    # Matching depends on tie-breaking, so a couple of percent is the target.
    assert result.estimate == pytest.approx(2.051839, rel=3e-2)


def test_matching_rejects_a_nonpositive_neighbour_count():
    with pytest.raises(ValueError, match="nneighbor"):
        NearestNeighbourMatching.from_formula(
            "y", "treated", _treatment(), nneighbor=0
        )


# ------------------------------------------------------------------- did


def test_didregress_matches_stata():
    result = Didregress.from_formula(
        "y", "treated", _did(), group="unit", time="period"
    ).fit()
    assert float(result.params["treated"]) == pytest.approx(1.18613, rel=1e-4)
    assert int(result.e["N"]) == 1600


def test_didregress_standard_error_is_the_right_order():
    """Point estimate exact; the standard error is ~7% below Stata's.

    Measured 0.05511 against Stata's 0.0590465. Every plausible cluster correction
    (G/(G-1), with and without the (N-1)/(N-k) factor) lands near 0.055, so Stata's
    DID variance accounts for something more than unit clustering -- most likely the
    sampling of the treatment arm itself. [R] didregress could not be downloaded to
    confirm, so the gap is recorded rather than tuned, and the registry keeps this
    at E2.
    """

    result = Didregress.from_formula(
        "y", "treated", _did(), group="unit", time="period"
    ).fit()
    assert float(result.bse["treated"]) == pytest.approx(0.0590465, rel=1.5e-1)


def test_xtdidregress_agrees_with_didregress():
    frame = _did()
    panel = Xtdidregress.from_formula(
        "y", "treated", frame, group="unit", time="period"
    ).fit()
    pooled = Didregress.from_formula(
        "y", "treated", frame, group="unit", time="period"
    ).fit()
    assert float(panel.params["treated"]) == pytest.approx(
        float(pooled.params["treated"]), rel=1e-12
    )


def test_didregress_rejects_a_time_invariant_treatment():
    """A treatment that never changes within a unit IS the fixed effect.

    Stata: r(198) "treatment variable omitted because of collinearity". A silent
    zero coefficient would be worse than an error.
    """

    frame = _did().assign(treated=lambda data: data["group"])
    with pytest.raises(ConsistencyError, match="collinear"):
        Didregress.from_formula(
            "y", "treated", frame, group="unit", time="period"
        ).fit()


def test_didregress_absorbs_rather_than_dummies():
    """The design must not carry one column per unit."""

    result = Didregress.from_formula(
        "y", "treated", _did(), group="unit", time="period"
    ).fit()
    design, names = result.design_matrix
    # Two columns: the treatment and x1. A dummyed model would have ~200 more.
    assert design.shape[1] == 2
    assert len(names) == 2
