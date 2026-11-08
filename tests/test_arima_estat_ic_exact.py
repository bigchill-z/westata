"""`estat ic` after `arima`: the numbers depend on `trend`, and Stata's default is `c`.

C13-07 scores this at 0.8571 and the first comparison looked like a real defect:

    Stat a (trend='n')    ll -272.073844   rank 3
    Stata                 ll -271.316650   rank 4

THE DIFFERENCE WAS THE TREND, NOT THE LIKELIHOOD. Stata's `arima y, arima(2,0,0)` includes a constant by
default, and `ws.ARIMA(..., order=(2,0,0))` defaults to no constant. Matching the settings:

    trend='c'            ll -271.316650   rank 4      <- identical to Stata

So the likelihood, the parameter count and the information criteria all agree once the two commands are
asked for the same model. The earlier "0.76 likelihood gap" I was about to record as a likelihood-definition
difference does not exist.

A GUARD PINS BOTH PLACES, because the default disagreement is the trap: someone comparing against Stata
will reach for the plain call on both sides and see a different table.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def series():
    rng = np.random.default_rng(31)
    return pd.Series(np.cumsum(rng.normal(size=200)))


def test_the_default_is_no_constant(series) -> None:
    """Recorded because Stata's default is the opposite, which is what makes comparisons go wrong."""

    fitted = ws.ARIMA(series, order=(2, 0, 0)).fit()
    assert "sigma2" in fitted.params.index
    assert "const" not in fitted.params.index
    assert int(fitted.e["rank"]) == 3


def test_with_a_constant_it_matches_stata(series) -> None:
    fitted = ws.ARIMA(series, order=(2, 0, 0), trend="c").fit()
    assert float(fitted.e["ll"]) == pytest.approx(-271.31665036, abs=1e-6)
    assert int(fitted.e["rank"]) == 4


def test_the_two_trends_give_different_likelihoods(series) -> None:
    """If they agreed, the constant would be entering nothing."""

    without = ws.ARIMA(series, order=(2, 0, 0), trend="n").fit()
    with_constant = ws.ARIMA(series, order=(2, 0, 0), trend="c").fit()
    assert float(without.e["ll"]) != pytest.approx(
        float(with_constant.e["ll"]), rel=1e-6
    )


def test_the_information_criteria_follow_the_likelihood(series) -> None:
    fitted = ws.ARIMA(series, order=(2, 0, 0), trend="c").fit()
    rendered = str(ws.estat_ic(fitted))
    # AIC = -2*ll + 2*k, and df is the rank Stata reports.
    assert "550.6" in rendered
    assert "-271.3167" in rendered


def test_estat_ic_reports_the_same_rank_as_the_fit(series) -> None:
    fitted = ws.ARIMA(series, order=(2, 0, 0), trend="c").fit()
    rendered = str(ws.estat_ic(fitted))
    fields = [
        line for line in rendered.splitlines() if line.strip().startswith(".")
    ][0].split("|")[1].split()
    # N, ll(null), ll(model), df, AIC, BIC
    assert fields[3] == "4"
