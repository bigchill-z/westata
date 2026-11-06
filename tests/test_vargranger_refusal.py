"""vargranger refuses a VECM, with the reason Stata gives.

Checking vargranger on a rank-2 VECM turned up that Stata rejects the combination outright:

    vargranger only works with estimates from var, svar, ivsvar,or xtvar

A VECM is not among them. westata also refused, but with a bare AttributeError about a missing
``granger_causality`` -- which names neither the test nor the reason, and reads like a bug rather
than a Stata rule.

It now raises UnsupportedError with the same explanation, so the caller learns to fit the level
VAR instead of hunting for a missing attribute.

The VAR path is unchanged and still reports statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def vecm_fit():
    rng = np.random.default_rng(7)
    n = 220
    common = np.cumsum(rng.normal(size=n))
    frame = pd.DataFrame(
        {
            "a": common + np.cumsum(rng.normal(scale=0.2, size=n)),
            "b": 1.5 * common + np.cumsum(rng.normal(scale=0.2, size=n)),
            "c": np.cumsum(rng.normal(size=n)),
        }
    )
    return ws.VECM(frame.to_numpy(float), coint_rank=2, lags=2, names=["a", "b", "c"]).fit()


def test_a_vecm_is_refused_with_an_explanation(vecm_fit) -> None:
    with pytest.raises(ws.UnsupportedError) as caught:
        ws.vargranger(vecm_fit)
    message = str(caught.value)
    assert "var" in message
    assert "VECM" in message or "vec" in message.lower()


def test_the_refusal_is_not_an_attribute_error(vecm_fit) -> None:
    """It used to surface AttributeError about granger_causality."""

    try:
        ws.vargranger(vecm_fit)
    except ws.UnsupportedError:
        pass
    except AttributeError:  # pragma: no cover - the regression this guards
        pytest.fail("vargranger raised AttributeError instead of UnsupportedError")
    else:  # pragma: no cover
        pytest.fail("vargranger accepted a VECM")


def test_a_var_still_works() -> None:
    rng = np.random.default_rng(7)
    n = 200
    values = np.column_stack(
        [np.cumsum(rng.normal(size=n)), np.cumsum(rng.normal(size=n))]
    )
    fitted = ws.VAR(values, lags=2, names=["x", "y"]).fit()
    rendered = str(ws.vargranger(fitted))
    assert "Granger" in rendered
    assert "chi2" in rendered
