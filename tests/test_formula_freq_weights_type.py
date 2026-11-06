"""from_formula's freq_weights takes a column NAME, and says so when it does not get one.

Round 116 concluded that the formula interface "silently accepted an array and dropped it". That
was WRONG -- re-checking this round shows it has always raised. What actually happened is that the
round-116 measurement called ``from_formula(formula, data)`` with NO freq_weights argument at all,
got 24 rows, and the story about a dropped argument was fitted to that afterwards.

The retraction matters because the wrong explanation would have sent a later round looking for a
silent-drop bug that does not exist. What DID need fixing is smaller and real: the error for a
wrong-typed argument came from a dict lookup as

    TypeError: unhashable type: 'numpy.ndarray'

which names neither the argument nor the expected type. It now says what to pass:

    ConsistencyError: logit: freq_weights must be the NAME of a column, not a ndarray.
    The formula interface looks the column up in the data it was given; pass e.g.
    freq_weights='freq'.

Correct usage, for the record:

    ws.Logit.from_formula(f, d)                       N = 24   (no weights)
    ws.Logit.from_formula(f, d, freq_weights="freq")  N = 2201 (weights expanded)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\titanic.dta"
FORMULA = "survive ~ child + female + class1"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("titanic.dta is not available")


def test_a_column_name_is_accepted(frame) -> None:
    fitted = ws.Logit.from_formula(FORMULA, frame, freq_weights="freq").fit()
    assert int(fitted.nobs) == 2201


def test_no_weights_leaves_the_row_count_at_the_source_rows(frame) -> None:
    """This is the state round 116 misread as "the array was dropped"."""

    fitted = ws.Logit.from_formula(FORMULA, frame).fit()
    assert int(fitted.nobs) == 24


def test_an_array_is_refused_with_an_explanatory_message(frame) -> None:
    with pytest.raises(ws.ConsistencyError) as caught:
        ws.Logit.from_formula(FORMULA, frame, freq_weights=frame["freq"].to_numpy()).fit()
    message = str(caught.value)
    assert "NAME of a column" in message
    assert "ndarray" in message


def test_a_series_is_refused_the_same_way(frame) -> None:
    with pytest.raises(ws.ConsistencyError) as caught:
        ws.Logit.from_formula(FORMULA, frame, freq_weights=frame["freq"]).fit()
    assert "NAME of a column" in str(caught.value)


def test_an_unknown_column_name_is_still_refused(frame) -> None:
    with pytest.raises(ws.ConsistencyError):
        ws.Logit.from_formula(FORMULA, frame, freq_weights="nosuchcolumn").fit()
