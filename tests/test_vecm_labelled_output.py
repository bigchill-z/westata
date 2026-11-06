"""Labelled VECM output as an ADDITION, not a replacement (v2 report section 6 class 4).

The report wanted ``VECM.beta`` and ``.alpha`` to be named rather than bare ndarrays. Round 19
tried making them DataFrames and an existing test failed within seconds with
``KeyError: (0, 0)`` -- because a DataFrame's ``[]`` indexes by label while an ndarray's
indexes by position, and ``result.beta[0, 0]`` is how downstream code reads it.

So the labels moved to new properties instead:

    result.beta            ndarray   (unchanged; positional access keeps working)
    result.beta_labelled   DataFrame (rows = variables, columns = ec1..ecR)

That is the measured cost of each choice, not a guess: both forms now exist and neither caller
has to change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(7)
    n = 200
    values = np.cumsum(rng.normal(size=(n, 2)), axis=0)
    return ws.VECM(values, coint_rank=1, lags=2, names=["y1", "y2"]).fit()


def test_beta_is_still_a_positional_ndarray(fitted) -> None:
    """The contract round 19 broke: ``[0, 0]`` must keep working."""

    assert isinstance(fitted.beta, np.ndarray)
    assert float(fitted.beta[0, 0]) == pytest.approx(1.0)


def test_alpha_is_still_a_positional_ndarray(fitted) -> None:
    assert isinstance(fitted.alpha, np.ndarray)
    assert fitted.alpha.shape == (2, 1)


def test_beta_labelled_carries_the_variable_names(fitted) -> None:
    labelled = fitted.beta_labelled
    assert isinstance(labelled, pd.DataFrame)
    assert list(labelled.index) == ["y1", "y2"]
    assert list(labelled.columns) == ["ec1"]


def test_alpha_labelled_carries_the_variable_names(fitted) -> None:
    labelled = fitted.alpha_labelled
    assert isinstance(labelled, pd.DataFrame)
    assert list(labelled.index) == ["y1", "y2"]


def test_the_labelled_values_equal_the_plain_ones(fitted) -> None:
    """Two forms, one set of numbers -- a mismatch would be the worst outcome."""

    assert np.allclose(np.asarray(fitted.beta_labelled), fitted.beta)
    assert np.allclose(np.asarray(fitted.alpha_labelled), fitted.alpha)


def test_both_forms_are_reachable_at_once(fitted) -> None:
    """Neither shadows the other."""

    assert fitted.beta is not None
    assert fitted.beta_labelled is not None
