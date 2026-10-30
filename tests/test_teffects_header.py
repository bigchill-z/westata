"""Tests for the ``teffects`` header block and ``.e`` mapping.

``TreatmentEffectResult`` is a frozen dataclass rather than a ``StataResults``, so
it had no ``e`` at all and no header: Stata prints four lines above the table
(``p11_outputs.txt:8-11``) that westata simply did not carry.

    Treatment-effects estimation    Number of obs     =      1,200
    Estimator      : regression adjustment
    Outcome model  : linear
    Treatment model: none

The ``Estimator``/``Outcome model``/``Treatment model`` labels are per estimator
and are pinned here against the fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture()
def frame():
    rng = np.random.default_rng(211)
    n = 1200
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    treated = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.5 * z)))).astype(float)
    y = 1.0 + 0.8 * x + 1.2 * treated + rng.normal(scale=0.8, size=n)
    return pd.DataFrame({"y": y, "treated": treated, "x": x, "z": z})


def _fit(estimator, frame, estimand="ate"):
    return estimator(
        y=frame["y"],
        treated=frame["treated"],
        x=frame[["x", "z"]],
        estimand=estimand,
    ).fit()


# ------------------------------------------------------------- header block
def test_regression_adjustment_header(frame) -> None:
    result = _fit(ws.RegressionAdjustment, frame)

    assert result.estimator == "regression adjustment"
    assert result.outcome_model == "linear"
    assert result.treatment_model == "none"
    assert result.nobs == len(frame)


def test_ipw_header(frame) -> None:
    result = _fit(ws.IPW, frame)

    assert result.estimator == "inverse probability weighting"
    # IPW models treatment only, not the outcome.
    assert result.outcome_model == "none"
    assert result.treatment_model == "logit"


def test_aipw_header(frame) -> None:
    result = _fit(ws.AIPW, frame)

    assert result.estimator == "augmented inverse probability weighting"
    assert result.outcome_model == "linear"
    assert result.treatment_model == "logit"


def test_nnmatch_header(frame) -> None:
    result = _fit(ws.NearestNeighbourMatching, frame)

    assert result.estimator == "nearest-neighbor matching"
    assert result.outcome_model == "none"
    assert result.treatment_model == "logit"
    # This estimator reports a precomputed standard error, so nobs is supplied
    # explicitly rather than inferred from an influence array.
    assert result.nobs == len(frame)


# ---------------------------------------------------------------------- e()
def test_e_carries_the_reported_table(frame) -> None:
    result = _fit(ws.RegressionAdjustment, frame)
    e = result.e

    assert e["N"] == float(len(frame))
    assert e["ate"] == result.estimate
    assert e["se_ate"] == result.std_err
    assert e["z_ate"] == result.z
    assert e["p_ate"] == result.pvalue
    assert e["ci_lower"] == result.ci_lower
    assert e["ci_upper"] == result.ci_upper


def test_e_carries_both_potential_outcome_means(frame) -> None:
    result = _fit(ws.RegressionAdjustment, frame)

    for level in ("0", "1"):
        mean, error = result.potential_outcomes[level]
        assert result.e[f"po{level}"] == mean
        assert result.e[f"se_po{level}"] == error


def test_e_carries_the_header_strings(frame) -> None:
    result = _fit(ws.IPW, frame)

    assert result.e["estimator"] == "inverse probability weighting"
    assert result.e["outcome_model"] == "none"
    assert result.e["treatment_model"] == "logit"


def test_e_keys_the_effect_by_estimand(frame) -> None:
    """ATE and ATET must not share a key, or a mixed report is ambiguous."""

    ate = _fit(ws.RegressionAdjustment, frame, estimand="ate")
    atet = _fit(ws.RegressionAdjustment, frame, estimand="atet")

    assert "ate" in ate.e and "atet" not in ate.e
    assert "atet" in atet.e and "ate" not in atet.e
    assert atet.e["atet"] == atet.estimate


# ------------------------------------------------------------ presentation
def test_str_prints_statas_header(frame) -> None:
    text = str(_fit(ws.RegressionAdjustment, frame))

    assert "Treatment-effects estimation" in text
    assert "Number of obs" in text
    assert "Estimator" in text
    assert "Outcome model" in text
    assert "Treatment model" in text
    assert str(len(frame)) in text


def test_with_header_returns_an_independent_copy(frame) -> None:
    """The dataclass is frozen, so the header is attached by replacement."""

    result = _fit(ws.RegressionAdjustment, frame)

    copy = result.with_header(
        nobs=1, estimator="x", outcome_model="y", treatment_model="z"
    )

    assert copy.nobs == 1
    assert result.nobs == len(frame)
    assert copy is not result
