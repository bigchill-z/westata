"""``estat mundlak`` -- the panel specification test, verified against a manual refit.

``xtreg_postestimation.sthlp:366-370``:

    estat mundlak performs a Mundlak specification test, a test that nu is uncorrelated
    with the regressors in x.  estat mundlak is for use after xtreg with the re, cre, and
    fe options but not with the absorb() option or the vce(hc2) option.

The test adds the WITHIN-GROUP MEANS of the regressors to the equation and tests them
jointly to zero. Those means are exactly what carries any correlation between the
unobserved effect and ``x``, so the joint test is a test of that correlation. A rejection
means the random-effects estimate is inconsistent.

Two API facts surfaced while building this, and the tests record them:

* ``model_name`` is Stata's spelling (``"xtreg, re"``), not a descriptive word;
* the result keeps ``panel_entity`` -- the coded GROUP LABELS -- but not the panel COLUMN
  names, so the augmented refit needs them supplied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

import westata as ws


def _panel(*, correlated: bool, seed: int, entities: int = 60, periods: int = 6):
    """``correlated`` decides whether the unobserved effect moves with ``x``.

    When it does, the Mundlak test must reject; when it does not, it must not.
    """

    rng = np.random.default_rng(seed)
    n = entities * periods
    entity = np.repeat(np.arange(entities), periods)
    effect = rng.normal(size=entities) * 2.0
    covariate = effect[entity] * (1.0 if correlated else 0.0) + rng.normal(size=n)
    outcome = 1.0 + 0.8 * covariate + effect[entity] + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": outcome,
            "x1": covariate,
            "firm": entity,
            "year": np.tile(np.arange(periods), entities),
        }
    )


@pytest.fixture(scope="module")
def correlated():
    frame = _panel(correlated=True, seed=950)
    return frame, ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()


@pytest.fixture(scope="module")
def uncorrelated():
    frame = _panel(correlated=False, seed=952)
    return frame, ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="firm", time="year"
    ).fit()


def _manual_mundlak(frame: pd.DataFrame):
    """Rebuild the test: add group means, refit, Wald-test the added terms."""

    augmented = frame.copy()
    augmented["x1bar"] = augmented.groupby("firm")["x1"].transform("mean")
    refit = ws.RandomEffects.from_formula(
        "y ~ x1 + x1bar", augmented, entity="firm", time="year"
    ).fit()
    names = list(refit.param_names)
    position = names.index("x1bar")
    coefficient = float(np.asarray(refit.params, dtype=float)[position])
    covariance = np.asarray(refit.cov_params(), dtype=float)
    variance = float(covariance[position, position])
    statistic = coefficient * coefficient / variance
    return statistic, 1.0, refit


def test_the_statistic_matches_the_manual_refit(correlated) -> None:
    frame, result = correlated
    expected, df, _ = _manual_mundlak(frame)

    reported = result.estat_mundlak(entity="firm", time="year")

    assert reported.statistic == pytest.approx(expected, rel=1e-10)
    assert reported.df == df


def test_the_p_value_is_the_upper_chi2_tail(correlated) -> None:
    reported = correlated[1].estat_mundlak(entity="firm", time="year")

    assert reported.pvalue == pytest.approx(
        float(scipy_stats.chi2.sf(reported.statistic, reported.df)), rel=1e-10
    )


def test_correlation_is_detected(correlated) -> None:
    """Data built WITH correlation must reject, or the test proves nothing."""

    reported = correlated[1].estat_mundlak(entity="firm", time="year")

    assert reported.pvalue < 0.001


def test_no_correlation_is_not_detected(uncorrelated) -> None:
    """The other direction: an uncorrelated effect must not be rejected."""

    reported = uncorrelated[1].estat_mundlak(entity="firm", time="year")

    assert reported.pvalue > 0.05


def test_the_two_panels_differ(correlated, uncorrelated) -> None:
    """Guard: if both behaved alike the direction tests are vacuous."""

    loud = correlated[1].estat_mundlak(entity="firm", time="year").pvalue
    quiet = uncorrelated[1].estat_mundlak(entity="firm", time="year").pvalue

    assert loud < 0.001 < 0.05 < quiet


def test_to_dict_reports_the_r_names(correlated) -> None:
    table = correlated[1].estat_mundlak(entity="firm", time="year").to_dict()

    assert set(table) == {"chi2", "df", "p"}


def test_the_tested_terms_are_named(correlated) -> None:
    reported = correlated[1].estat_mundlak(entity="firm", time="year")

    assert reported.tested == ["x1"]


def test_the_rendered_text_states_the_null(correlated) -> None:
    rendered = str(correlated[1].estat_mundlak(entity="firm", time="year"))

    assert "Mundlak" in rendered
    assert "uncorrelated" in rendered
    assert "chi2(" in rendered


def test_two_regressors_give_two_degrees_of_freedom() -> None:
    """One restriction per slope, not per observation."""

    rng = np.random.default_rng(954)
    entities, periods = 50, 5
    entity = np.repeat(np.arange(entities), periods)
    n = entity.size
    first = rng.normal(size=n)
    second = rng.normal(size=n)
    effect = rng.normal(size=entities)
    frame = pd.DataFrame(
        {
            "y": 1.0 + 0.5 * first - 0.3 * second + effect[entity] + rng.normal(size=n),
            "x1": first,
            "x2": second,
            "firm": entity,
            "year": np.tile(np.arange(periods), entities),
        }
    )
    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", frame, entity="firm", time="year"
    ).fit()

    reported = result.estat_mundlak(entity="firm", time="year")

    assert reported.df == 2.0
    assert set(reported.tested) == {"x1", "x2"}


# ------------------------------------------------------------------ refusals
def test_a_missing_entity_is_refused(correlated) -> None:
    """The refit needs the panel COLUMN names, which the result does not keep.

    Refusing with that reason beats a bare ``TypeError`` from the estimator, which says
    nothing about what to pass.
    """

    with pytest.raises(ValueError) as caught:
        correlated[1].estat_mundlak()

    message = str(caught.value)
    assert "entity" in message
    assert "storer" in message


def test_a_non_panel_result_is_refused() -> None:
    rng = np.random.default_rng(956)
    frame = pd.DataFrame({"y": rng.normal(size=100), "x": rng.normal(size=100)})

    with pytest.raises(NotImplementedError, match="xtreg"):
        ws.OLS.from_formula("y ~ x", frame).fit().estat_mundlak(entity="x")


def test_a_storer_callback_bypasses_the_entity_requirement(correlated) -> None:
    """The escape hatch for callers who want to control the refit."""

    frame, result = correlated
    seen: dict[str, pd.DataFrame] = {}

    def storer(augmented: pd.DataFrame):
        seen["frame"] = augmented
        return ws.RandomEffects.from_formula(
            "y ~ x1 + __mundlak_x1", augmented, entity="firm", time="year"
        ).fit()

    reported = result.estat_mundlak(storer=storer)

    assert "frame" in seen
    assert "__mundlak_x1" in seen["frame"].columns
    assert np.isfinite(reported.statistic)