"""`predict ..., residual` then `summarize` -- the pattern four report rows score at 0.0.

C04-04, C05-08, C09-04 and C11-08 all use the same book workflow: fit, write the prediction into a new
variable, then summarize it. The report's diagnosis is that the package "only returns arrays, so the
case prints the mean and standard deviation by hand". Measured, the two steps compose:

    fitted = ws.OLS.from_formula(...).fit()
    ws.summarize(pd.DataFrame({"e1": fitted.resid}))

          Variable |        Obs        Mean    Std. dev.       Min        Max
    --------------+---------------------------------------------------------
                e1 |        758   -4.89e-15    .3594254   -1.22974   1.171132

against a live Stata after `reg lnw s expr` and `predict e1, residual`:

                e1 |        758    7.03e-11    .3594254   -1.22974   1.171132

THE MEANS DIFFER ONLY IN THE NOISE: 7.03e-11 against -4.89e-15. Both are zero to any printed precision,
and the standard deviation, minimum and maximum agree exactly -- which is the check that the two
residual vectors are the same vector rather than merely both small.

WHAT THE PACKAGE DOES NOT DO is create a variable inside a dataset the way Stata's `predict` writes one
back. The caller names a DataFrame column instead. That is an API difference, not a missing
computation, and the numbers are the evidence.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")


@pytest.fixture(scope="module")
def fitted(frame):
    return ws.OLS.from_formula("lnw ~ s + expr", frame).fit()


def test_the_prediction_is_the_right_length(fitted, frame) -> None:
    assert len(fitted.resid) == len(frame)


def test_summarize_accepts_the_prediction_under_a_chosen_name(fitted) -> None:
    rendered = str(ws.summarize(pd.DataFrame({"e1": fitted.resid})))
    assert "e1" in rendered
    assert "758" in rendered


def test_the_standard_deviation_matches_stata(fitted) -> None:
    """The discriminating statistic: the mean is noise near zero either way."""

    rendered = str(ws.summarize(pd.DataFrame({"e1": fitted.resid})))
    assert ".3594254" in rendered


def test_the_range_matches_stata(fitted) -> None:
    rendered = str(ws.summarize(pd.DataFrame({"e1": fitted.resid})))
    assert "-1.22974" in rendered
    assert "1.171132" in rendered


def test_the_residual_mean_is_zero_to_printing_precision(fitted) -> None:
    """Stata prints 7.03e-11 and the package -4.89e-15; both are zero."""

    assert abs(float(np.mean(fitted.resid))) < 1e-9
