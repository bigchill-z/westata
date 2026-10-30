"""``eyex``/``eydx`` are log derivatives, so a negative fitted value is undefined.

Until round 159 both returned a finite number when ``E[y] < 0``: the implementation
divided by ``prediction``, and ``1/x`` is perfectly well defined for negative ``x`` — the
formula "worked" while the quantity it stands for, a derivative of ``ln E[y]``, does not
exist.

Stata's behaviour here cannot be settled from ``ado/base``: the factors are computed
inside a Mata routine with no source. Returning a plausible number for an undefined
expression is the failure shape this session keeps removing, so the call refuses and names
how many fitted values are negative.

``dyex`` is unaffected: its factor is ``x_i`` alone, with no logarithm.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def signed_frame():
    """An unconstrained OLS response, so the fitted values span zero."""

    rng = np.random.default_rng(859)
    n = 300
    covariate = rng.normal(size=n)
    frame = pd.DataFrame(
        {"y": 1.0 + 0.8 * covariate + rng.normal(size=n), "x1": covariate}
    )
    return frame, ws.OLS.from_formula("y ~ x1", frame).fit()


def test_the_fixture_really_has_negative_predictions(signed_frame) -> None:
    """Guard: without negatives the refusals below would never fire."""

    _, result = signed_frame
    prediction = np.asarray(result.predict(), dtype=float)

    assert np.any(prediction < 0.0)


@pytest.mark.parametrize("form", ["eyex", "eydx"])
def test_a_negative_prediction_is_refused(signed_frame, form) -> None:
    """``ln(E[y])`` does not exist for a negative argument."""

    _, result = signed_frame

    with pytest.raises(ValueError) as caught:
        result.margins(**{form: "x1"})

    message = str(caught.value)
    assert "ln(E[y])" in message or "ln" in message
    assert "negative" in message


def test_the_message_counts_the_offending_values(signed_frame) -> None:
    """A bare refusal is less useful than one that says how many rows are affected."""

    _, result = signed_frame
    negative = int(np.sum(np.asarray(result.predict(), dtype=float) < 0.0))

    with pytest.raises(ValueError) as caught:
        result.margins(eydx="x1")

    assert str(negative) in str(caught.value)


def test_the_message_points_at_a_working_alternative(signed_frame) -> None:
    """``dydx`` is the level derivative and needs no logarithm."""

    _, result = signed_frame

    with pytest.raises(ValueError) as caught:
        result.margins(eyex="x1")

    assert "dydx" in str(caught.value)


def test_dyex_still_works_on_the_same_data(signed_frame) -> None:
    """Its factor is ``x_i``; no logarithm is involved, so negatives are harmless."""

    _, result = signed_frame

    rendered = result.margins(dyex="x1")

    assert np.isfinite(float(rendered.estimate))


def test_dydx_still_works_on_the_same_data(signed_frame) -> None:
    _, result = signed_frame

    rendered = result.margins(dydx="x1")

    assert np.isfinite(float(rendered.estimate))


def test_a_strictly_positive_response_is_unaffected() -> None:
    """The guard must not fire for a model that never predicts a negative value."""

    rng = np.random.default_rng(863)
    n = 200
    covariate = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": np.abs(2.0 + 0.8 * covariate + rng.normal(size=n)) + 0.5,
            "x1": covariate,
        }
    )
    result = ws.OLS.from_formula("y ~ x1", frame).fit()

    assert np.all(np.asarray(result.predict(), dtype=float) > 0.0)
    for form in ("dyex", "eyex", "eydx"):
        assert np.isfinite(float(result.margins(**{form: "x1"}).estimate)), form


def test_a_logit_fit_is_never_refused_for_sign() -> None:
    """A probability scale is strictly positive, so the guard cannot trigger."""

    rng = np.random.default_rng(877)
    n = 200
    latent = rng.normal(size=n)
    frame = pd.DataFrame(
        {"yb": (latent > 0).astype(int), "x1": rng.normal(size=n)}
    )
    logit = ws.Logit.from_formula("yb ~ x1", frame).fit()

    for form in ("eyex", "eydx"):
        assert np.isfinite(float(logit.margins(**{form: "x1"}).estimate)), form