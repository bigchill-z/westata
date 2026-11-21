"""A datetime `time=` axis on `varbasic`.

Stata's `varbasic` runs on a `tsset` axis that is often a calendar variable. Three defects were
measured on `macro_swatson.dta` (quarterly, 168 rows, 1960q2 to 2002q1) against StataNow 19.5,
and each is pinned here:

* passing `time="quarter"` where `quarter` is `datetime64` raised
  "varbasic: no usable observations", because the axis was converted to nanoseconds and the
  lag lookup `current - delta * lag` lost the low bits in float64;
* the `Sample:` line printed a spurious ", but with a gap", because quarters are 90, 91 or 92
  days apart and the gap test compared elapsed days;
* `Sample:` tried `float()` on a Timestamp.

Stata reference, `varbasic dinf unem, lags(1/2)`:

    Sample: 1961q2 thru 2002q1                      Number of obs     =        164
    Log likelihood =   1224.456                     AIC               =  -14.81044
    FPE            =   1.27e-09                     HQIC              =  -14.73371
    Det(Sigma_ml)  =   1.12e-09                     SBIC              =  -14.62143

and with `if quarter<tq(1999q1)`:

    Sample: 1961q2 thru 1998q4                      Number of obs     =        151
    Log likelihood =   1118.534                     AIC               =  -14.68256
    FPE            =   1.44e-09                     HQIC              =  -14.60139
    Det(Sigma_ml)  =   1.26e-09                     SBIC              =  -14.48274
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(), reason="macro_swatson.dta is not available"
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    data = pd.read_stata(DATA, convert_categoricals=False)
    data["quarter"] = pd.to_datetime(data["quarter"])
    return data


def test_a_datetime_axis_is_usable(frame) -> None:
    """It used to raise 'no usable observations' on a frame that fits fine without `time`."""

    result = ws.varbasic(frame, ["dinf", "unem"], lags=(1, 2), time="quarter")
    assert result.nobs == 164


def test_the_datetime_axis_changes_no_number(frame) -> None:
    """`time` is load-bearing for the SAMPLE, not for the estimates: both must agree here."""

    with_time = ws.varbasic(frame, ["dinf", "unem"], lags=(1, 2), time="quarter")
    without = ws.varbasic(frame, ["dinf", "unem"], lags=(1, 2))
    assert with_time.nobs == without.nobs


def test_the_sample_line_labels_calendar_periods(frame) -> None:
    result = ws.varbasic(frame, ["dinf", "unem"], lags=(1, 2), time="quarter")
    text = str(result)
    assert "Sample: 1961q2 thru 2002q1" in text


def test_a_quarterly_axis_is_not_reported_as_gappy(frame) -> None:
    """Quarters are 90 to 92 days apart; the gap test must not read that as missing periods."""

    result = ws.varbasic(frame, ["dinf", "unem"], lags=(1, 2), time="quarter")
    assert result.sample_has_gap is False
    assert "but with a gap" not in str(result)


def test_the_estimation_numbers_match_stata(frame) -> None:
    result = ws.varbasic(frame, ["dinf", "unem"], lags=(1, 2), time="quarter")
    assert round(float(result.ll), 3) == 1224.456
    assert round(float(result.aic), 5) == -14.81044
    assert round(float(result.hqic), 5) == -14.73371
    assert round(float(result.sbic), 5) == -14.62143
    assert round(float(result.fpe) * 1e9, 2) == 1.27
    assert round(float(result.det_sigma) * 1e9, 2) == 1.12


def test_a_tiny_value_prints_in_scientific_notation(frame) -> None:
    """`_coef` rendered 1.27e-09 as '.0' -- a number that reads as zero."""

    text = str(ws.varbasic(frame, ["dinf", "unem"], lags=(1, 2), time="quarter"))
    assert "1.27e-09" in text
    assert "1.12e-09" in text
    # The broken form must not appear on either line.
    for line in text.splitlines():
        if line.startswith("FPE") or line.startswith("Det(Sigma_ml)"):
            assert "=        .0" not in line


def test_the_head_cut_sample_matches_stata(frame) -> None:
    """Stata: `varbasic dinf unem if quarter<tq(1999q1), lags(1/2)` -> N = 151."""

    result = ws.varbasic(
        frame,
        ["dinf", "unem"],
        lags=(1, 2),
        time="quarter",
        subset=frame["quarter"] < pd.Timestamp("1999-01-01"),
    )
    assert result.nobs == 151
    assert "Sample: 1961q2 thru 1998q4" in str(result)
    assert round(float(result.ll), 3) == 1118.534
    assert round(float(result.aic), 5) == -14.68256
    assert round(float(result.fpe) * 1e9, 2) == 1.44
    assert round(float(result.det_sigma) * 1e9, 2) == 1.26
