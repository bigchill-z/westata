"""Tests for Stata's ``estat vif`` (variance inflation factors).

Every value is asserted against an independent least-squares recomputation of
``VIF_j = 1/(1-R2_j)``, so the implementation cannot pass by agreeing with
itself.

The definition comes from ``vif.ado`` v1.5.4, which regresses each regressor on
the others with ``_regress`` (a constant included unless the ``uncentered``
branch sets ``nocons``) and then reports ``1/(1-e(r2))``, its reciprocal, and the
mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.vif import estat_vif

_CONSTANT = {"intercept", "const", "_cons"}


@pytest.fixture()
def linear_fit():
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length", "displacement"]].dropna()
    return frame, ws.OLS.from_formula(
        "mpg ~ weight + length + displacement", frame
    ).fit()


def _hand_vif(matrix: np.ndarray, target: int, *, uncentered: bool) -> float:
    """Independent VIF for one column, by explicit least squares."""

    y = matrix[:, target]
    others = np.delete(matrix, target, axis=1)
    keep = [j for j in range(others.shape[1]) if np.ptp(others[:, j]) > 0.0]
    others = others[:, keep] if keep else np.empty((y.size, 0))

    if uncentered:
        design = others if others.size else np.empty((y.size, 0))
        total = float(y @ y)
    else:
        design = (
            np.column_stack([np.ones_like(y), others])
            if others.size
            else np.ones((y.size, 1))
        )
        centred = y - y.mean()
        total = float(centred @ centred)

    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    r2 = 1.0 - float(resid @ resid) / total
    return 1.0 / (1.0 - r2)


def test_vif_matches_an_independent_computation(linear_fit) -> None:
    frame, result = linear_fit
    matrix, names = result.design_matrix
    matrix = np.asarray(matrix, dtype=float)

    rendered = result.estat_vif()

    for j, name in enumerate(names):
        if name.lower() in _CONSTANT:
            continue
        expected = _hand_vif(matrix, j, uncentered=False)
        index = rendered.names.index(name)
        assert rendered.vif[index] == pytest.approx(expected, rel=1e-10), name


def test_tolerance_is_the_reciprocal_of_vif(linear_fit) -> None:
    _, result = linear_fit
    rendered = result.estat_vif()

    np.testing.assert_allclose(
        rendered.tolerance, 1.0 / rendered.vif, rtol=1e-14
    )


def test_mean_vif_is_the_arithmetic_mean(linear_fit) -> None:
    _, result = linear_fit
    rendered = result.estat_vif()

    assert rendered.mean_vif == pytest.approx(float(np.mean(rendered.vif)), rel=1e-12)


def test_results_are_sorted_by_descending_vif(linear_fit) -> None:
    """``vif.ado`` does ``gsort -vvv nms`` before printing."""

    _, result = linear_fit
    rendered = result.estat_vif()

    assert list(rendered.vif) == sorted(rendered.vif, reverse=True)


def test_nosort_keeps_the_model_order(linear_fit) -> None:
    _, result = linear_fit
    matrix, names = result.design_matrix
    expected = [n for n in names if n.lower() not in _CONSTANT]

    assert result.estat_vif(nosort=True).names == expected


def test_the_constant_is_not_a_vif_row_by_default(linear_fit) -> None:
    _, result = linear_fit
    rendered = result.estat_vif()

    assert "_cons" not in rendered.names
    assert "Intercept" not in rendered.names


def test_uncentered_matches_an_independent_computation(linear_fit) -> None:
    _, result = linear_fit
    matrix, names = result.design_matrix
    matrix = np.asarray(matrix, dtype=float)

    rendered = result.estat_vif(uncentered=True)

    for j, name in enumerate(names):
        if name.lower() in _CONSTANT:
            continue
        expected = _hand_vif(matrix, j, uncentered=True)
        index = rendered.names.index(name)
        assert rendered.vif[index] == pytest.approx(expected, rel=1e-10), name


def test_uncentered_appends_the_constant_row(linear_fit) -> None:
    """``vif.ado`` appends ``_cons`` when ``uncentered`` is requested."""

    _, result = linear_fit
    assert "_cons" in result.estat_vif(uncentered=True).names


def test_uncentered_vif_exceeds_centred_vif(linear_fit) -> None:
    """Direction matters and is easy to invert.

    The uncentered R-squared divides by the *uncentered* total sum of squares,
    which is the larger quantity, so the R-squared is smaller and the VIF
    correspondingly larger.
    """

    _, result = linear_fit
    centred = result.estat_vif()
    uncentered = result.estat_vif(uncentered=True)

    for name in centred.names:
        assert (
            uncentered.vif[uncentered.names.index(name)]
            >= centred.vif[centred.names.index(name)] - 1e-9
        ), name


def test_a_collinear_regressor_gets_a_huge_vif() -> None:
    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight"]].dropna().copy()
    rng = np.random.default_rng(0)
    frame["weight_copy"] = frame["weight"] + rng.normal(scale=1e-6, size=len(frame))

    rendered = ws.OLS.from_formula("mpg ~ weight + weight_copy", frame).fit().estat_vif()

    assert rendered.vif.max() > 1000.0
    # The mean must stay computable: an infinity would poison it.
    assert np.isfinite(rendered.mean_vif)


def test_model_without_a_constant_refuses_centred_vif() -> None:
    """``vif.ado`` errors with "not appropriate after regress, nocons"."""

    df = pd.read_stata("tests/data/auto.dta")
    frame = df[["mpg", "weight", "length"]].dropna()
    fit = ws.OLS.from_formula("mpg ~ weight + length - 1", frame).fit()

    with pytest.raises(ValueError, match="uncentered"):
        fit.estat_vif()

    # ... but accepts the uncentered request.
    rendered = fit.estat_vif(uncentered=True)
    assert rendered.names
    assert all(np.isfinite(v) for v in rendered.vif)


def test_nonlinear_models_are_refused() -> None:
    df = pd.read_stata("tests/data/auto.dta")
    frame = df.assign(dom=(df["foreign"] == "Foreign").astype(int))
    fit = ws.Logit.from_formula("dom ~ weight + mpg", frame).fit()

    with pytest.raises(NotImplementedError, match="linear"):
        fit.estat_vif()


def test_module_function_matches_the_method(linear_fit) -> None:
    _, result = linear_fit

    assert estat_vif(result).to_dict() == result.estat_vif().to_dict()


def test_output_has_statas_four_columns(linear_fit) -> None:
    _, result = linear_fit
    text = str(result.estat_vif())

    assert "Variable" in text
    assert "VIF" in text
    assert "1/VIF" in text
    assert "Mean VIF" in text


def test_to_frame_exposes_both_columns(linear_fit) -> None:
    _, result = linear_fit
    frame = result.estat_vif().to_frame()

    assert list(frame.columns) == ["VIF", "1/VIF"]
    assert list(frame.index) == result.estat_vif().names
