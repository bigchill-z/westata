"""The three elasticity forms compose with ``at()`` and ``over()``.

Round 155 found a real bug in exactly this area: ``at()`` was applied AFTER the derivative
form was chosen, so ``margins(dyex="x1", at={"x1": 2})`` returned the unadjusted number.
That ordering is fixed, and the same risk applies to every form -- so each is checked
with an adjustment and with groups.

The expected values are rebuilt from ``predict()`` on the ADJUSTED design rather than
from the implementation's own intermediate quantities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    """A model whose response is positive, so ``1/E[y]`` is always finite."""

    rng = np.random.default_rng(839)
    n = 200
    covariate = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": np.abs(1.0 + 0.8 * covariate + rng.normal(size=n)) + 0.5,
            "x1": covariate,
        }
    )
    return frame, ws.OLS.from_formula("y ~ x1", frame).fit()


def _expected(form: str, slope: float, covariate: np.ndarray, prediction: np.ndarray):
    """Rebuild the elasticity factor from the definition."""

    if form == "dyex":
        return slope * float(np.mean(covariate))
    if form == "eyex":
        return slope * float(np.mean(covariate / prediction))
    return slope * float(np.mean(1.0 / prediction))


@pytest.mark.parametrize("form", ["dyex", "eyex", "eydx"])
def test_the_forms_agree_at_the_estimation_sample(fitted, form) -> None:
    """Baseline: every form matches its definition on the unadjusted design."""

    frame, result = fitted
    slope = float(result.params["x1"])
    covariate = np.asarray(frame["x1"], dtype=float)
    prediction = np.asarray(result.predict(), dtype=float)

    expected = _expected(form, slope, covariate, prediction)

    assert float(result.margins(**{form: "x1"}).estimate) == pytest.approx(
        expected, rel=1e-10
    )


@pytest.mark.parametrize("form", ["dyex", "eyex", "eydx"])
def test_the_forms_agree_after_an_at_adjustment(fitted, form) -> None:
    """``at()`` replaces the covariates, so BOTH the factor and the prediction move.

    Applying the adjustment late would leave the factor computed from the original
    data -- the bug round 155 fixed for ``dyex``. This covers the other two.
    """

    frame, result = fitted
    slope = float(result.params["x1"])
    matrix = np.asarray(result.design_matrix[0], dtype=float).copy()
    matrix[:, 1] = 2.0  # every x set to 2

    covariate = matrix[:, 1]
    prediction = np.asarray(result.predict(matrix), dtype=float)
    expected = _expected(form, slope, covariate, prediction)

    assert float(
        result.margins(**{form: "x1"}, at={"x1": 2.0}).estimate
    ) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("form", ["dyex", "eyex", "eydx"])
def test_the_adjustment_actually_changes_the_answer(fitted, form) -> None:
    """Guard: if ``at()`` were ignored, the two numbers would coincide.

    A weaker test that only checked the adjusted value against its formula could pass
    with the adjustment silently dropped, if the formula were also built from the
    unadjusted design.
    """

    _, result = fitted

    plain = float(result.margins(**{form: "x1"}).estimate)
    adjusted = float(result.margins(**{form: "x1"}, at={"x1": 2.0}).estimate)

    assert adjusted != pytest.approx(plain, rel=1e-6)


@pytest.mark.parametrize("form", ["dyex", "eyex", "eydx"])
def test_each_group_uses_its_own_factor(form) -> None:
    """``over`` reports one row per group, and the factor is the GROUP's.

    The two groups are given covariate levels three standard deviations apart, so a
    pooled factor would produce two identical numbers whenever the derivative is
    constant -- which it is here.
    """

    rng = np.random.default_rng(841)
    covariate = np.concatenate([rng.normal(0, 1, 100), rng.normal(3, 1, 100)])
    # ``+ 2.0`` keeps the response comfortably positive: round 159 made
    # ``eyex``/``eydx`` refuse when any fitted value is negative, because they are
    # derivatives of ln(E[y]).  Without the shift one fitted value landed below zero and
    # the call raised -- correct behaviour, wrong fixture.
    frame = pd.DataFrame(
        {
            "y": np.abs(2.0 + 0.8 * covariate + rng.normal(size=200)) + 0.5,
            "x1": covariate,
        }
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()
    groups = np.array(["a"] * 100 + ["b"] * 100)
    slope = float(result.params["x1"])
    matrix = np.asarray(result.design_matrix[0], dtype=float)

    rows = result.margins(**{form: "x1"}, over=groups)

    assert len(rows) == 2
    for row in rows:
        mask = groups == row.label
        prediction = np.asarray(result.predict(matrix[mask]), dtype=float)
        expected = _expected(form, slope, matrix[mask][:, 1], prediction)
        assert float(row.estimate) == pytest.approx(expected, rel=1e-10), row.label


def test_the_two_group_values_differ() -> None:
    """Guard for the previous test: equal group factors would make it vacuous."""

    rng = np.random.default_rng(853)
    covariate = np.concatenate([rng.normal(0, 1, 100), rng.normal(4, 1, 100)])
    # Shifted positive for the same reason as above: ``eydx`` takes a logarithm of the
    # fitted response, so a negative prediction is undefined (round 159).
    frame = pd.DataFrame(
        {
            "y": np.abs(2.0 + 0.8 * covariate + rng.normal(size=200)) + 0.5,
            "x1": covariate,
        }
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()
    groups = np.array(["a"] * 100 + ["b"] * 100)

    rows = result.margins(eydx="x1", over=groups)
    values = sorted(float(row.estimate) for row in rows)

    assert values[0] != pytest.approx(values[1], rel=1e-6)