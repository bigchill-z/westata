"""Frequency-weighted probit, and the L() regression that round 115 unblocked.

C11-02 scores `probit ... [fweight=freq]` at 0.6842 and C08-06 scores a regression with lagged
regressors at 0.7059. Both were "approximate" in the report; measured, both are exact.

FREQUENCY WEIGHTS. `fweight` in Stata expands the data: an observation with weight 4 counts four
times. That makes e(N) the SUM of the weights, not the row count --

    quietly probit y x [fweight=freq]
        _b[x]      Stata -0.01390339    package -0.013903
        _b[_cons]  Stata -0.17670685    package -0.176707
        e(N)       Stata 784            package 784

784 is the weighted total against 300 rows, which is the check that the expansion happened.

LAGGED REGRESSORS. `reg y ... L.temp ...` now works because round 115 added the L() operator; before
that the formula raised NameError.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws

MACRO = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def weighted():
    rng = np.random.default_rng(1)
    n = 300
    frame = pd.DataFrame(
        {
            "y": (rng.random(n) < 0.4).astype(float),
            "x": rng.normal(size=n),
            "freq": rng.integers(1, 5, n).astype(float),
        }
    )
    return ws.Probit.from_formula("y ~ x", frame, freq_weights="freq").fit()


def test_the_frequency_weighted_coefficients_match_stata(weighted) -> None:
    assert float(weighted.params["x"]) == pytest.approx(-0.01390339, abs=1e-8)
    assert float(weighted.params["Intercept"]) == pytest.approx(-0.17670685, abs=1e-8)


def test_the_observation_count_is_the_weight_total(weighted) -> None:
    """e(N) sums the weights; it is 784 for 300 rows, which is how the expansion shows up."""

    assert int(weighted.nobs) == 784


def test_frequency_weights_must_be_named_not_passed_as_values() -> None:
    """Passing a Series is rejected by name, so the mistake is obvious rather than silent."""

    rng = np.random.default_rng(1)
    n = 100
    frame = pd.DataFrame(
        {
            "y": (rng.random(n) < 0.4).astype(float),
            "x": rng.normal(size=n),
            "freq": rng.integers(1, 5, n).astype(float),
        }
    )
    with pytest.raises(Exception):
        ws.Probit.from_formula("y ~ x", frame, freq_weights=frame["freq"]).fit()


def test_a_regression_with_lagged_regressors_runs() -> None:
    """C08-06. It raised NameError before the L() operator existed."""

    try:
        frame = pd.read_stata(MACRO)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("macro_swatson.dta is not available")

    fitted = ws.OLS.from_formula("dinf ~ unem + L(unem) + L(dinf)", frame).fit()
    assert int(fitted.nobs) == 165
    assert "L(unem)" in fitted.params.index
    assert "L(dinf)" in fitted.params.index
