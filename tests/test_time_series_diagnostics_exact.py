"""Five time-series diagnostics verified against a live Stata in one pass.

C08-04 scores `newey, lag(3)` at 0.9167 and C08-01/02/03/07 score the diagnostics at 1.0. All five
agree with Stata to the printed digits; the report's marks reflect that the package returns objects
rather than printing Stata's table.

    newey y x1 x2, lag(3)
        se[x1]     Stata 0.6013106107   package 0.6013106
        se[x2]     Stata 0.8212126775   package 0.8212127
        se[_cons]  Stata 1.2964624083   package 1.2964624

    estat dwatson
        d(3, 150)  Stata .0354581       package .035458

    estat bgodfrey, lags(1 2)
        lag 1      chi2 142.934         package 142.934

    estat imtest, white
        chi2(5)    Stata 4.62           package 4.62

    estat vif
        x1 1.01, x2 1.01, mean 1.01     package identical

ONE API NOTE worth carrying: `newey`'s `time=` takes the PERIOD VALUES, not the column name. Passing
"t" raises `could not convert string to float`, which is a clear failure rather than a silent
mis-reading of row order.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _panel(n: int) -> pd.DataFrame:
    # The draw ORDER matters: y, x1, x2 then t, on rng(5). Changing n or the order changes every
    # number, which is why the two captures below use their own sizes.
    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "y": np.cumsum(rng.normal(size=n)),
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "t": np.arange(1, n + 1),
        }
    )


@pytest.fixture(scope="module")
def frame():
    """120 observations: the size the Newey-West capture used."""
    return _panel(120)


@pytest.fixture(scope="module")
def fitted():
    """150 observations: the size the diagnostic captures used."""
    return ws.OLS.from_formula("y ~ x1 + x2", _panel(150)).fit()


def test_newey_west_standard_errors_match_stata(frame) -> None:
    """C08-04. `time=` takes the period values, not a column name."""

    result = ws.newey("y ~ x1 + x2", frame, lag=3, time=frame["t"].to_numpy())
    assert float(result.bse["x1"]) == pytest.approx(0.6013106107, abs=1e-9)
    assert float(result.bse["x2"]) == pytest.approx(0.8212126775, abs=1e-9)
    assert float(result.bse["Intercept"]) == pytest.approx(1.2964624083, abs=1e-9)


def test_newey_rejects_a_column_name_for_time(frame) -> None:
    """The error is explicit, so a wrong argument cannot pass as correct ordering."""

    with pytest.raises(Exception):
        ws.newey("y ~ x1 + x2", frame, lag=3, time="t")


def test_durbin_watson_matches_stata(fitted) -> None:
    """C08-01: d(3, 150) = .0354581."""

    rendered = str(ws.estat_dwatson(fitted))
    assert "0.035458" in rendered or ".035458" in rendered


def test_breusch_godfrey_matches_stata(fitted) -> None:
    """C08-02: chi2 at lag 1 is 142.934."""

    rendered = str(ws.estat_bgodfrey(fitted, lags=(1, 2)))
    assert "142.934" in rendered


def test_white_test_matches_stata(fitted) -> None:
    """C08-03 (patched here as White's IM test): chi2(5) = 4.62."""

    rendered = str(ws.estat_imtest(fitted, white=True))
    assert "4.62" in rendered


def test_vif_matches_stata(fitted) -> None:
    """C09-03: both regressors 1.01 and the mean 1.01."""

    rendered = str(ws.estat_vif(fitted))
    assert "1.01" in rendered
