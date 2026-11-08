"""`response_name` resolves for every estimator, including the ones that never set it.

Rounds 162 and 164 found that VAR and VECM discarded column names. This round asked the same question of
every other estimator and found a different, quieter gap: OLS and the formula-based fits assign
`result.response_name`, while ARIMA and GARCH never did -- so postestimation code reading it saw None.

The base class now falls back to `e(depvar)`, which every fit carries:

    ARIMA    alpha
    GARCH    alpha
    OLS      y
    Logit    binary

THE SETTER MATTERS AS MUCH AS THE GETTER. The formula-based estimators do
`results.response_name = ...` during fit, so making the attribute a read-only property broke them
immediately with "can't set attribute". A guard exercises both directions.

`VAR` and `VECM` keep their own property, which returns the space-separated series list rather than a
single name -- a distinct meaning for the same attribute, and worth pinning so a later unification does
not collapse them.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(23)
    return pd.Series(np.cumsum(rng.normal(size=100)), name="alpha")


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(23)
    return pd.DataFrame(
        {"y": rng.normal(size=60), "x": rng.normal(size=60)}
    )


def test_arima_reports_its_response_name(series) -> None:
    fitted = ws.ARIMA(series, order=(1, 0, 0)).fit()
    assert fitted.response_name == "alpha"


def test_garch_reports_its_response_name(series) -> None:
    fitted = ws.GARCH(series, p=1, q=1).fit()
    assert fitted.response_name == "alpha"


def test_the_formula_based_estimators_still_report_theirs(frame) -> None:
    """They assign the attribute directly, which is what the setter keeps working."""

    assert ws.OLS.from_formula("y ~ x", frame).fit().response_name == "y"

    binary = frame.assign(binary=(frame["y"] > 0).astype(float))
    assert ws.Logit.from_formula("binary ~ x", binary).fit().response_name == "binary"


def test_a_var_reports_all_series_not_one_name(frame) -> None:
    """A different meaning for the same attribute: every equation's series, space separated."""

    wide = pd.DataFrame({"alpha": frame["y"], "beta": frame["x"]})
    fitted = ws.VAR(wide, lags=1).fit()
    assert fitted.response_name == "alpha beta"


def test_an_empty_response_name_does_not_raise(frame) -> None:
    """A fit with no dependent variable should give an empty string rather than an error."""

    fitted = ws.OLS.from_formula("y ~ x", frame).fit()
    fitted.response_name = None
    assert fitted.response_name == ""
