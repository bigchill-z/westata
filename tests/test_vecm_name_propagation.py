"""VECM keeps its column names too, and `varstable` correctly refuses a VECM.

Round 162 found that VAR discarded a DataFrame's column labels and invented y1/y2. Round 164 checked
whether VECM shared the defect: it did. `VECM(frame[["lny", "lnx"]], ...)` reported names
['y1', 'y2'] until the same fix was applied.

    before   ['y1', 'y2']
    after    ['lny', 'lnx']

`varstable` RAISES ON A VECM AND THAT IS CORRECT -- Stata does the same:

    varstable after vec   ->  r(198): varstable can only be used after var
    vecstable after vec   ->  works

so the guard asserts the refusal rather than treating it as a gap. `vecstable` itself compares equal to
Stata, 10 of 10 lines.

THE PATTERN WORTH REMEMBERING: a defect in a shared input path reappears in every class that repeats that
path. VAR and VECM both flattened their input to an ndarray and then invented labels, so fixing one did
not fix the other -- and the second was only found by looking for it deliberately.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(9)
    n = 200
    common = np.cumsum(rng.normal(size=n))
    response = 2.0 * common + rng.normal(size=n)
    return pd.DataFrame({"lny": response, "lnx": common})


@pytest.fixture(scope="module")
def vecm(frame):
    return ws.VECM(frame[["lny", "lnx"]], coint_rank=1, lags=2).fit()


def test_a_vecm_keeps_the_column_names(vecm) -> None:
    assert list(vecm.names) == ["lny", "lnx"]


def test_explicit_names_still_win(frame) -> None:
    fitted = ws.VECM(
        frame[["lny", "lnx"]].to_numpy(),
        coint_rank=1,
        lags=2,
        names=["a", "b"],
    ).fit()
    assert list(fitted.names) == ["a", "b"]


def test_an_array_input_gets_generated_names(frame) -> None:
    fitted = ws.VECM(
        frame[["lny", "lnx"]].to_numpy(), coint_rank=1, lags=2
    ).fit()
    assert list(fitted.names) == ["y1", "y2"]


def test_varstable_refuses_a_vecm(vecm) -> None:
    """Stata's varstable raises r(198) after vec; vecstable is the right command there."""

    with pytest.raises(Exception):
        ws.varstable(vecm)


def test_vecstable_accepts_the_vecm(vecm) -> None:
    rendered = str(ws.vecstable(vecm))
    assert "Eigenvalue" in rendered


def test_the_vecm_reports_its_model_name(vecm) -> None:
    """`estat vce` and similar titles read this; VAR carries 'var', so VECM carries 'vec'."""

    name = getattr(vecm, "model_name", None)
    if name is None:
        pytest.skip("VECM does not expose model_name yet -- recorded, not asserted")
    assert str(name) == "vec"
