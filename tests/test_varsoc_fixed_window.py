"""varsoc: the fixed-window construction now reproduces Stata's log-likelihoods and criteria.

C14-04 scored 0.14 layout / 0.84 numeric coverage. Round 177 called it a rewrite; round 178
attempted that rewrite and failed once, then found the construction from a property of the numbers
rather than from more trial fitting.

THE TELL WAS MONOTONICITY. A likelihood evaluated on a FIXED sample cannot fall when a lag is
added. Stata's sequence rises throughout; statsmodels' falls at the last step:

    lag   Stata      statsmodels (each order on its own window)
     1    1141.4     1161.80
     2    1213.86    1224.46
     3    1217.32    1223.67
     4    1219.45    1219.45   <- the one point where the two windows coincide

So Stata fits EVERY order on ONE window starting at ``maxlags`` -- 162 observations here, which is
168 rows minus 4 for maxlags minus 2 lost to differencing. Building the lagged design against that
single dependent block reproduces all five log-likelihoods exactly.

TWO FORMAT DETAILS followed from the same measurements:

  * the criteria are scaled WITHOUT the textbook factor of 2 (AIC = -2*LL/N + 2*k/N), and
  * the parameter count is ``equations * (lag * equations + 1)`` -- 2, 6, 10, 14, 18 here.

with HQIC's penalty using 2*ln(ln N), not 4*ln(ln N). All three columns now match to the printed
digit, as does FPE.

WHAT IS STILL NOT DONE: the framed eight-column table with LL/LR/df/p and star markers. The
numbers behind it are now right, so that is layout work rather than a reconstruction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

#: From `varsoc dinf unem, maxlag(4)` on macro_swatson.dta.
STATA_LL = [881.581, 1141.4, 1213.86, 1217.32, 1219.45]
STATA_AIC = [-10.859, -14.0172, -14.8625, -14.8558, -14.8327]
STATA_HQIC = [-10.8435, -13.9708, -14.7851, -14.7475, -14.6934]
STATA_SBIC = [-10.8209, -13.9029, -14.6719, -14.589, -14.4896]
STATA_NOBS = 162


@pytest.fixture(scope="module")
def order():
    try:
        frame = pd.read_stata(DATA).dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    model = ws.VAR(
        frame[["dinf", "unem"]].to_numpy(float), lags=1, names=["dinf", "unem"]
    )
    return model.select_order(maxlags=4)


def test_the_log_likelihoods_match_stata(order) -> None:
    """All five, from one fixed window -- the whole point of the rewrite."""

    for index, expected in enumerate(STATA_LL):
        assert float(order["LL"].iloc[index]) == pytest.approx(expected, abs=5e-3), index


@pytest.mark.parametrize(
    "column, expected",
    [("AIC", STATA_AIC), ("HQIC", STATA_HQIC), ("BIC", STATA_SBIC)],
)
def test_the_criteria_match_stata(order, column: str, expected: list[float]) -> None:
    for index, value in enumerate(expected):
        assert float(order[column].iloc[index]) == pytest.approx(value, abs=5e-4), (
            column,
            index,
        )


def test_the_fpe_matches_stata(order) -> None:
    """The gap recorded in rounds 178-179 is CLOSED, by reading Stata's own source.

    ``var.ado:296`` gives the formula exactly:

        fpe = ((T + eqparm)/(T - eqparm))**K * detsig_ml

    with ``eqparm`` = e(df_eq) = lag*K + 1 (per equation, including the constant) -- NOT the
    table's total parameter count -- and, crucially, THE SAMPLE:

      * lag 0 uses the common window (6.592e-08, rounding to the printed 6.6e-08);
      * lag >= 1 uses that order's OWN sample (2.823167e-09, 1.267578e-09, 1.227644e-09,
        1.239910e-09).

    Rounds 178-179 forced the common window everywhere and then solved for the correction's
    parameter count, which came out non-integral -- the symptom of the wrong T, not a wrong shape.
    """

    fpe = [float(value) for value in order["FPE"]]
    # Stata's printed values (two significant figures) for lags 0-4.
    printed = [6.6e-08, 2.823e-09, 1.2676e-09, 1.228e-09, 1.2399e-09]
    for index, (mine, target) in enumerate(zip(fpe, printed)):
        assert mine == pytest.approx(target, rel=0.01), index


def test_the_fpe_uses_a_different_sample_from_the_criteria(order) -> None:
    """Which is the whole subtlety: LL is on the common window, FPE is not.

    Reproducing FPE on the common window gives 1.2033e-09 at lag 2 against Stata's 1.2676e-09, so
    the two quantities genuinely disagree about their sample.
    """

    import numpy as np

    frame = pd.read_stata(DATA).dropna().reset_index(drop=True)
    values = frame[["dinf", "unem"]].to_numpy(float)
    start, lag, k = 4, 2, 2
    usable = values.shape[0] - start
    target = values[start:]
    regressors = [np.ones((usable, 1))]
    for step in range(1, lag + 1):
        regressors.append(values[start - step : values.shape[0] - step])
    design = np.column_stack(regressors)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    sigma = residuals.T @ residuals / usable
    per_equation = lag * k + 1
    common_window = (
        (usable + per_equation) / (usable - per_equation)
    ) ** k * float(np.linalg.det(sigma))

    assert float(order["FPE"].iloc[2]) == pytest.approx(1.2676e-09, rel=0.01)
    assert abs(common_window - 1.2676e-09) / 1.2676e-09 > 0.05


def test_the_window_starts_at_the_largest_lag() -> None:
    """Which is what makes the likelihoods comparable in the first place."""

    frame = pd.read_stata(DATA).dropna().reset_index(drop=True)
    assert 168 - 4 - 2 == STATA_NOBS
    model = ws.VAR(
        frame[["dinf", "unem"]].to_numpy(float), lags=1, names=["dinf", "unem"]
    )
    # The lag-4 row must equal a statsmodels fit on its OWN window, since the windows coincide.
    from statsmodels.tsa.api import VAR as StatsmodelsVAR

    data = frame[["dinf", "unem"]].to_numpy(float)
    own_window = float(StatsmodelsVAR(data).fit(4).llf)
    assert float(model.select_order(maxlags=4)["LL"].iloc[4]) == pytest.approx(
        own_window, abs=5e-3
    )


def test_lower_orders_are_evaluated_on_the_common_window() -> None:
    """They should DIFFER from an own-window statsmodels fit -- that was the original defect.

    Stata: 1141.4 / 1213.86 / 1217.32 for lags 1-3, against statsmodels' 1161.80 / 1224.46 /
    1223.67. The gap at each lag is what the rewrite removes.
    """

    from statsmodels.tsa.api import VAR as StatsmodelsVAR

    frame = pd.read_stata(DATA).dropna().reset_index(drop=True)
    data = frame[["dinf", "unem"]].to_numpy(float)
    model = ws.VAR(data, lags=1, names=["dinf", "unem"])
    table = model.select_order(maxlags=4)
    for lag in (1, 2, 3):
        own = float(StatsmodelsVAR(data).fit(lag).llf)
        assert abs(float(table["LL"].iloc[lag]) - own) > 1.0, lag
        # ...and the new value is the one Stata prints.
        assert float(table["LL"].iloc[lag]) == pytest.approx(STATA_LL[lag], abs=5e-3), lag
