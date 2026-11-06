"""``predict`` under fweight returns the CALLER's rows, as Stata does.

v2 report section 5.2 lists this as a silent trap: "fweight + predict returns 2201 rows (after
expansion), where Stata has 24". The point estimates were right and only the SHAPE was wrong,
which is the worst kind -- code that zips the prediction against the frame either raises on a
length mismatch far away, or silently pairs row i of a 2201-row vector with row i of a 24-row
frame.

The model must be fitted on the expanded sample (that is what fweight means), so the arithmetic
still uses the expansion. Only the OUTPUT is collapsed back, taking one value per original row,
which is valid because rows with the same covariates share a prediction.

Reference, ``logit survive child female class1 [fweight=freq]`` then ``predict pr, pr`` on
titanic::

    pr[1] = .3088922    pr[2] = .7920259    pr[3] = .4267619
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\titanic.dta"


@pytest.fixture(scope="module")
def titanic():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("titanic.dta is not available")


@pytest.fixture(scope="module")
def fitted(titanic):
    return ws.Logit.from_formula(
        "survive ~ child + female + class1", titanic, freq_weights="freq"
    ).fit()


def test_predict_returns_the_caller_row_count(fitted, titanic) -> None:
    """The trap: 2201 expanded rows instead of the frame's 24."""

    assert len(np.asarray(fitted.predict())) == len(titanic)


def test_the_collapsed_values_match_stata(fitted) -> None:
    predicted = np.asarray(fitted.predict(), dtype=float)
    assert predicted[0] == pytest.approx(0.3088922, abs=1e-7)
    assert predicted[1] == pytest.approx(0.7920259, abs=1e-7)
    assert predicted[2] == pytest.approx(0.4267619, abs=1e-7)


def test_the_weighted_expansion_still_drives_the_fit(fitted, titanic) -> None:
    """Guard against "fixing" the shape by dropping the expansion from the arithmetic.

    The weighted probabilities must NOT equal the unweighted ones; if someone removes the
    repeat, the fit silently becomes unweighted and the golden values above stop being Stata's.
    """

    unweighted = ws.Logit.from_formula(
        "survive ~ child + female + class1", titanic
    ).fit()
    weighted = np.asarray(fitted.predict(), dtype=float)
    plain = np.asarray(unweighted.predict(), dtype=float)
    assert not np.allclose(weighted, plain)


def test_predictions_are_bounded(fitted) -> None:
    predicted = np.asarray(fitted.predict(), dtype=float)
    assert np.all(predicted >= 0.0) and np.all(predicted <= 1.0)
