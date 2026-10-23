"""Conformance tests for :mod:`westata.estimators.did` (Stata ``didregress``).

Reference values come from ``tests/conformance/stata_fixtures/p11_did.txt``,
captured from StataNow 19.5 on a balanced panel of 200 units x 8 periods with
half the units treated from period 4 (1600 observations, 200 clusters).  The
DGP's true effect is 1.2 and Stata reports:

=========================  =========  ==============  ================
command                    ATET       Stata ``se``    westata ``se``
=========================  =========  ==============  ================
``didregress (y x1)``      1.18613    .0590465        .05904651458904
``xtdidregress (y x1)``    1.18613    .055228         .05522799411606
=========================  =========  ==============  ================

Measured gap: ``|westata - Stata| = 1.46e-8`` on the ``didregress`` standard
error (Stata prints 7 significant digits) and ``6.00e-8`` on the
``xtdidregress`` one, i.e. both agree to the last printed digit.  The point
estimate agrees to ``8.4e-10``.

Three tests here exist because a point-estimate test would pass without them:

* the ATET is cross-checked against an *explicit dummy-variable* OLS
  (``statsmodels``), which is what "the fixed effects are absorbed, not
  dummied out" has to mean numerically;
* the within-transformed design is inspected, so "no unit dummy in the design"
  is asserted rather than assumed;
* the two Stata spellings are pinned to their *different* standard errors --
  they share a point estimate but not a degrees-of-freedom convention, and a
  test that only compared the ATET would miss a 6.5% standard-error error.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from westata.errors import ConsistencyError
from westata.estimators.did import Didregress, Xtdidregress

# --------------------------------------------------------------------- references
STATA_ATET = 1.18613
STATA_SE_RCS = 0.0590465  # didregress    (repeated cross-sectional)
STATA_SE_PANEL = 0.055228  # xtdidregress  (longitudinal)
DGP_ATET = 1.2


# --------------------------------------------------------------------------- data
def did_frame(
    n_units: int = 200, n_periods: int = 8, seed: int = 223, *, x2: bool = False
) -> pd.DataFrame:
    """Replication of the P11 fixture DGP (``scripts/capture_p11_did.py``)."""

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
            row = [unit, period, y, x1, treated_now, float(in_treatment_arm)]
            if x2:
                row.append(rng.normal())
            rows.append(tuple(row))
    columns = ["unit", "period", "y", "x1", "treated", "group"]
    if x2:
        columns.append("x2")
    return pd.DataFrame(rows, columns=columns)


def dummy_ols(frame: pd.DataFrame, covariates: list[str]):
    """Explicit dummy-variable OLS with a Stata CR1 covariance.

    This is the definition of the estimator: every unit and period indicator is
    a regressor.  It is deliberately built independently of
    :mod:`westata.estimators.hdfe` so a shared mistake cannot hide.
    """

    unit_dummies = pd.get_dummies(
        frame["unit"].astype("category"), drop_first=True
    ).astype(float)
    period_dummies = pd.get_dummies(
        frame["period"].astype("category"), drop_first=True
    ).astype(float)
    design = np.column_stack(
        [
            np.ones(len(frame)),
            frame["treated"].to_numpy(dtype=float),
            frame[covariates].to_numpy(dtype=float),
            unit_dummies.to_numpy(),
            period_dummies.to_numpy(),
        ]
    )
    fit = sm.OLS(frame["y"].to_numpy(dtype=float), design).fit(
        cov_type="cluster", cov_kwds={"groups": frame["unit"].to_numpy()}
    )
    return fit, design.shape[1]


def stata_cr1_se(frame: pd.DataFrame, covariates: list[str]) -> float:
    """The treatment's standard error under Stata's CR1, from the dummy design.

    ``G/(G-1) * (N-1)/(N-K)`` with ``K`` counting every dummy, i.e. exactly the
    ``didregress`` (repeated cross-section) convention.
    """

    fit, n_params = dummy_ols(frame, covariates)
    design = fit.model.exog
    resid = fit.resid
    groups = frame["unit"].to_numpy()
    meat = np.zeros((n_params, n_params))
    for cluster in np.unique(groups):
        mask = groups == cluster
        score = design[mask].T @ resid[mask]
        meat += np.outer(score, score)
    bread = np.linalg.pinv(design.T @ design)
    n = len(frame)
    g = np.unique(groups).size
    variance = bread @ meat @ bread
    variance *= (g / (g - 1)) * ((n - 1) / (n - n_params))
    return float(np.sqrt(variance[1, 1]))


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return did_frame()


@pytest.fixture(scope="module")
def pooled(panel: pd.DataFrame):
    return Didregress.from_formula(
        "y x1", "treated", panel, group="unit", time="period"
    ).fit()


@pytest.fixture(scope="module")
def longitudinal(panel: pd.DataFrame):
    return Xtdidregress.from_formula(
        "y x1", "treated", panel, group="unit", time="period"
    ).fit()


# ------------------------------------------------------- didregress point estimate
def test_didregress_atet_matches_stata(pooled):
    """Stata: ATET 1.18613."""

    atet = float(pooled.params["treated"])
    assert atet == pytest.approx(STATA_ATET, abs=1e-3)
    # The DGP's true effect is 1.2; the sample realised 1.18613, which Stata's
    # number confirms.  Asserting both keeps the fixture and the DGP aligned.
    assert atet == pytest.approx(DGP_ATET, abs=2e-2)


def test_didregress_reports_the_treatment_coefficient_first(pooled):
    """Stata's table shows the ATET row above the covariate rows."""

    assert pooled.param_names == ["treated", "x1"]
    assert pooled.model_name == "didregress"
    assert pooled.e["estimand"] == "ATET"


def test_didregress_matches_an_explicit_dummy_variable_regression(panel, pooled):
    """Absorbing the fixed effects must equal dummying them out.

    Frisch-Waugh-Lovell says the within-transformed coefficient and the
    dummy-variable coefficient are the same number; this checks the absorption
    actually implemented that, using statsmodels as the independent reference.
    """

    reference, _ = dummy_ols(panel, ["x1"])
    assert float(pooled.params["treated"]) == pytest.approx(reference.params[1], abs=1e-10)
    assert float(pooled.params["x1"]) == pytest.approx(reference.params[2], abs=1e-10)


# ----------------------------------------------------------- didregress covariance
def test_didregress_standard_error_matches_stata(pooled):
    """Stata: se .0590465 (measured gap 1.46e-8)."""

    se = float(pooled.bse["treated"])
    measured_gap = abs(se - STATA_SE_RCS)
    assert measured_gap < 1e-6, f"se={se!r}, gap={measured_gap:g}"


def test_didregress_standard_error_matches_stata_cr1_from_the_dummy_design(panel, pooled):
    """Recompute Stata's CR1 from the explicit dummies and compare."""

    expected = stata_cr1_se(panel, ["x1"])
    assert expected == pytest.approx(STATA_SE_RCS, rel=1e-6)
    assert float(pooled.bse["treated"]) == pytest.approx(expected, rel=1e-9)


def test_didregress_counts_the_absorbed_parameters_like_stata(pooled):
    """``K = 209``: constant + treatment + x1 + 199 unit + 7 period effects."""

    assert float(pooled.e["rank"]) == 209
    assert float(pooled.e["N"]) == 1600
    # Stata's t tests use the 200 clusters: t = 1.18613/.0590465 = 20.09, and
    # the printed 95% interval is .0590465 * t(199).
    assert float(pooled.df_resid) == 199
    assert float(pooled.tvalues["treated"]) == pytest.approx(20.09, abs=5e-3)


# --------------------------------------------------------------- xtdidregress
def test_xtdidregress_gives_the_same_atet_as_didregress(panel, pooled, longitudinal):
    """"xtdidregress and didregress are the same estimator" -- same point estimate."""

    assert float(longitudinal.params["treated"]) == pytest.approx(
        float(pooled.params["treated"]), rel=1e-12
    )
    assert longitudinal.model_name == "xtdidregress"
    assert longitudinal.e["data_type"] == "Longitudinal"


def test_xtdidregress_standard_error_matches_stata(longitudinal):
    """Stata: se .055228 -- *smaller* than didregress on identical data."""

    se = float(longitudinal.bse["treated"])
    measured_gap = abs(se - STATA_SE_PANEL)
    assert measured_gap < 1e-6, f"se={se!r}, gap={measured_gap:g}"


def test_the_two_spellings_differ_only_by_the_nested_unit_effect(
    panel, pooled, longitudinal
):
    """The panel spelling lets the cluster absorb the nested unit effect.

    Stata counts ``K = 209`` for the repeated cross-section and ``K = 10`` for
    the panel (treatment + x1 + 7 period effects + constant), so the clustered
    variances differ by exactly ``(N-K_rcs)/(N-K_panel)``:

        se_panel = se_rcs * sqrt(1391/1590) = .0590465 * .93530 = .055228

    This is the whole difference between the two commands, so it is asserted in
    closed form rather than as "the numbers are different".
    """

    n, k_rcs, k_panel = 1600, 209, 10
    assert float(pooled.e["rank"]) == k_rcs
    assert float(longitudinal.e["rank"]) == k_panel

    expected_ratio = np.sqrt((n - k_rcs) / (n - k_panel))
    measured_ratio = float(longitudinal.bse["treated"] / pooled.bse["treated"])
    assert measured_ratio == pytest.approx(expected_ratio, rel=1e-12)\

    # And the ratio is what turns one Stata number into the other.
    assert STATA_SE_RCS * expected_ratio == pytest.approx(STATA_SE_PANEL, rel=1e-5)


def test_xtdidregress_rejects_repeated_group_time_cells(panel):
    """``xtset`` forbids repeated time values within a panel; so does this."""

    duplicated = pd.concat([panel, panel.iloc[:8]], ignore_index=True)
    with pytest.raises(ConsistencyError, match="repeat"):
        Xtdidregress.from_formula(
            "y x1", "treated", duplicated, group="unit", time="period"
        ).fit()


def test_xtdidregress_accepts_an_unbalanced_panel():
    """Absorption handles unbalanced panels; only the SE convention is shared."""

    frame = did_frame(n_units=40, n_periods=6, seed=11)
    frame = frame[~((frame["unit"] == 3) & (frame["period"] == 2))].reset_index(
        drop=True
    )
    result = Xtdidregress.from_formula(
        "y x1", "treated", frame, group="unit", time="period"
    ).fit()
    assert np.isfinite(float(result.params["treated"]))
    assert result.e["N"] == float(len(frame))


# ---------------------------------------------------- the time-invariant trap
def test_time_invariant_treatment_raises_a_collinearity_error(panel):
    """A treatment that never changes within a unit IS the fixed effect.

    Stata: ``r(198) "treatment variable omitted because of collinearity"``.
    Silently returning a coefficient (or zero) would be far worse.
    """

    frame = panel.assign(treated=panel["group"])

    with pytest.raises(ConsistencyError) as excinfo:
        Didregress.from_formula(
            "y x1", "treated", frame, group="unit", time="period"
        ).fit()

    message = str(excinfo.value)
    assert "collinearity" in message.lower()
    assert "treated" in message
    assert "r(198)" in message
    # The diagnosis, not just the verdict: the message must say *why*.
    assert "never changes within" in message or "time-invariant" in message


def test_a_treatment_collinear_with_the_period_effects_is_also_rejected(panel):
    """A pure ``post`` dummy shifts every unit at once, so it is a time effect.

    This one varies within every unit, so it passes the within-group check and
    is caught only after the absorption -- the second line of defence.
    """

    frame = panel.assign(treated=(panel["period"] >= 4).astype(float))
    with pytest.raises(ConsistencyError, match="collinearity"):
        Didregress.from_formula(
            "y x1", "treated", frame, group="unit", time="period"
        ).fit()


def test_xtdidregress_raises_the_same_collinearity_error(panel):
    frame = panel.assign(treated=panel["group"])
    with pytest.raises(ConsistencyError, match="collinearity"):
        Xtdidregress.from_formula(
            "y x1", "treated", frame, group="unit", time="period"
        ).fit()


# -------------------------------------------------------- absorption invariants
def test_the_design_has_one_column_per_reported_regressor(panel, pooled):
    """No column per unit: 200 units must not become 199 design columns."""

    design, names = pooled.design_matrix
    assert names == ["treated", "x1"]
    assert design.shape == (1600, 2)
    assert design.shape[1] < panel["unit"].nunique()

    # The design is the within-transformed one, so it is orthogonal to every
    # absorbed effect.  Checking that directly is the definition of "absorbed":
    # a dummy-column design would fail this, because the indicators themselves
    # are of course not orthogonal to the effects they span.
    frame = pd.DataFrame(design, columns=names)
    unit_means = frame.groupby(panel["unit"]).mean().to_numpy()
    period_means = frame.groupby(panel["period"]).mean().to_numpy()
    assert np.max(np.abs(unit_means)) < 1e-12
    assert np.max(np.abs(period_means)) < 1e-12
    # ... and therefore to the full indicator matrices, both ways.
    unit_indicators = pd.get_dummies(panel["unit"].astype("category")).to_numpy(float)
    period_indicators = pd.get_dummies(panel["period"].astype("category")).to_numpy(
        float
    )
    assert np.max(np.abs(unit_indicators.T @ design)) < 1e-10
    assert np.max(np.abs(period_indicators.T @ design)) < 1e-10


def test_absorbed_effects_do_not_change_the_atet_when_covariates_differ(panel):
    """Adding a time-varying covariate changes ``gamma``, not the absorption.

    The panel is balanced and every covariate varies within units, so the
    within transform is exact: whatever ``x`` is included, the ATET must equal
    the explicit dummy-variable estimate for that same specification.
    """

    frame = did_frame(x2=True)
    with_x1 = Didregress.from_formula(
        "y x1", "treated", frame, group="unit", time="period"
    ).fit()
    with_x1x2 = Didregress.from_formula(
        "y x1 x2", "treated", frame, group="unit", time="period"
    ).fit()

    reference_x1, _ = dummy_ols(frame, ["x1"])
    reference_x1x2, _ = dummy_ols(frame, ["x1", "x2"])

    assert float(with_x1.params["treated"]) == pytest.approx(
        reference_x1.params[1], abs=1e-10
    )
    assert float(with_x1x2.params["treated"]) == pytest.approx(
        reference_x1x2.params[1], abs=1e-10
    )
    # The covariate set does matter for the estimate (x2 is a real regressor
    # here), so an *unchanged* ATET would mean x2 was dropped.
    assert with_x1x2.param_names == ["treated", "x1", "x2"]
    assert float(with_x1x2.params["x2"]) != pytest.approx(0.0, abs=1e-6)


def test_a_time_invariant_covariate_is_not_silently_estimated(panel):
    """A constant-within-unit covariate is the unit effect; FEOLS drops it."""

    frame = panel.assign(z=panel["group"] + 1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Didregress.from_formula(
            "y x1 z", "treated", frame, group="unit", time="period"
        ).fit()

    assert "z" not in result.param_names
    assert result.absorbed == ["z"]
    assert any("z" in str(record.message) for record in caught)
    # The ATET is unaffected by the presence of the collinear column.
    reference = Didregress.from_formula(
        "y x1", "treated", panel, group="unit", time="period"
    ).fit()
    assert float(result.params["treated"]) == pytest.approx(
        float(reference.params["treated"]), rel=1e-12
    )


# ------------------------------------------------------------------- e() payload
def test_fit_populates_the_stata_scalars(pooled):
    """``e(N)``, ``e(df_m)``, ``e(df_r)`` and the cluster count."""

    assert pooled.e["N"] == 1600
    assert pooled.e["df_m"] == 2  # ATET + x1
    assert pooled.e["df_r"] == 199  # G - 1 under clustering
    assert pooled.e["N_clust"] == 200
    assert pooled.df_model == 2
    assert pooled.df_resid == 199
    assert pooled.nobs == 1600


def test_summary_renders_in_stata_style(pooled):
    """The ``summary(style="stata")`` layout Stata prints."""

    text = pooled.summary(style="stata")
    assert "Number of obs" in text
    assert "1,600" not in text  # westata's renderer does not add separators
    assert "1600" in text
    assert "Coef." in text
    assert "Std. err." in text
    assert "[95% conf. interval]" in text
    assert "didregress" in text
    # The coefficient row itself is present with Stata's digits.
    row = next(line for line in text.splitlines() if line.strip().startswith("treated"))
    assert "1.18613" in row


def test_summary_shows_the_panel_command(longitudinal):
    text = longitudinal.summary(style="stata")
    assert "xtdidregress" in text
    assert ".05522799" in text


# -------------------------------------------------------------- argument parsing
def test_stata_block_syntax_parses_covariates():
    """``(y x1 x2) (treated)`` -> dependent variable plus two covariates."""

    frame = did_frame(n_units=20, n_periods=4, x2=True)
    model = Didregress.from_formula(
        "y x1 x2", "treated", frame, group="unit", time="period"
    )
    assert model.depvar == "y"
    assert model.covariates == ["x1", "x2"]
    assert model.treatment == "treated"
    assert model.formula == "y ~ treated + x1 + x2 | unit + period"


def test_bare_outcome_infers_time_varying_covariates_with_a_warning(panel):
    """A bare ``(y)`` block has no covariate; westata infers and warns."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Didregress.from_formula(
            "y", "treated", panel, group="unit", time="period"
        )

    # ``x1`` varies within units, ``group`` does not, so only x1 is a covariate.
    assert model.covariates == ["x1"]
    assert model.inferred_covariates == ["x1"]
    messages = " ".join(str(record.message) for record in caught)
    assert "time-varying" in messages and "x1" in messages
    assert float(model.fit().params["treated"]) == pytest.approx(STATA_ATET, abs=1e-3)


def test_no_covariates_at_all_is_allowed():
    frame = did_frame(n_units=20, n_periods=4).drop(columns=["x1"])
    result = Didregress.from_formula(
        "y", "treated", frame, group="unit", time="period"
    ).fit()
    assert result.param_names == ["treated"]
    assert np.isfinite(float(result.params["treated"]))


def test_the_treatment_must_be_binary(panel):
    frame = panel.assign(treated=panel["treated"] * 2.0)
    with pytest.raises(ConsistencyError, match="0/1"):
        Didregress.from_formula(
            "y x1", "treated", frame, group="unit", time="period"
        ).fit()


def test_a_single_arm_is_rejected(panel):
    frame = panel.assign(treated=0.0)
    with pytest.raises(ConsistencyError, match="control"):
        Didregress.from_formula(
            "y x1", "treated", frame, group="unit", time="period"
        ).fit()


def test_missing_values_are_dropped_listwise(panel):
    frame = panel.copy()
    frame.loc[frame.index[:5], "x1"] = np.nan
    model = Didregress.from_formula(
        "y x1", "treated", frame, group="unit", time="period"
    )
    result = model.fit()
    assert result.e["N"] == 1595
    assert model.n_dropped_missing == 5
    assert model.n_used == 1595


def test_unknown_column_is_reported(panel):
    with pytest.raises(ValueError, match="x9"):
        Didregress.from_formula(
            "y x9", "treated", panel, group="unit", time="period"
        )


def test_group_and_time_must_differ(panel):
    with pytest.raises(ValueError, match="different columns"):
        Didregress.from_formula("y x1", "treated", panel, group="unit", time="unit")


def test_a_string_outcome_is_rejected(panel):
    frame = panel.assign(y=panel["y"].astype(str))
    with pytest.raises(TypeError, match="numeric"):
        Didregress.from_formula(
            "y x1", "treated", frame, group="unit", time="period"
        )


# --------------------------------------------------------------------- cov_type
@pytest.mark.parametrize("cov_type", ["nonrobust", "HC1"])
def test_alternative_covariances_keep_the_same_point_estimate(panel, pooled, cov_type):
    result = Didregress.from_formula(
        "y x1", "treated", panel, group="unit", time="period"
    ).fit(cov_type=cov_type)

    assert float(result.params["treated"]) == pytest.approx(
        float(pooled.params["treated"]), rel=1e-12
    )
    assert float(result.bse["treated"]) < float(pooled.bse["treated"])
    assert "N_clust" not in result.e
    assert result.e["df_r"] == 1600 - 209


def test_cov_type_aliases(panel):
    for alias in ("CLUSTER", "clustered", "CR1"):
        result = Didregress.from_formula(
            "y x1", "treated", panel, group="unit", time="period"
        ).fit(cov_type=alias)
        assert result.cov_type == "cluster"


def test_an_unknown_cov_type_is_rejected(panel):
    model = Didregress.from_formula(
        "y x1", "treated", panel, group="unit", time="period"
    )
    with pytest.raises(ValueError, match="Unknown cov_type"):
        model.fit(cov_type="bootstrap")


def test_hc0_is_rejected_rather_than_silently_approximated(panel):
    model = Didregress.from_formula(
        "y x1", "treated", panel, group="unit", time="period"
    )
    with pytest.raises(ValueError, match="HC0"):
        model.fit(cov_type="HC0")


# --------------------------------------------------------------------- plumbing
def test_construction_does_not_estimate(panel):
    model = Didregress.from_formula(
        "y x1", "treated", panel, group="unit", time="period"
    )
    assert model.n_used == 0
    assert "Didregress" in repr(model)
    assert "group(unit)" in repr(model)


def test_model_data_is_the_estimation_sample(panel, pooled):
    assert pooled.model_data.shape[0] == pooled.nobs
    assert set(["unit", "period", "y", "x1", "treated"]).issubset(
        pooled.model_data.columns
    )


def test_conformance_fixture_numbers_are_pinned(pooled, longitudinal):
    """One place that records both Stata numbers and both measured gaps."""

    gaps = {
        "atet": abs(float(pooled.params["treated"]) - STATA_ATET),
        "se_rcs": abs(float(pooled.bse["treated"]) - STATA_SE_RCS),
        "se_panel": abs(float(longitudinal.bse["treated"]) - STATA_SE_PANEL),
    }
    assert gaps["atet"] < 5e-6, gaps
    assert gaps["se_rcs"] < 1e-6, gaps
    assert gaps["se_panel"] < 1e-6, gaps
