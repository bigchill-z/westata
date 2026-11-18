"""`varbasic`'s ``time`` parameter: what it must refuse, and what it must align to.

Two defects, both measured against StataNow 19.5 on gdp_china.dta, and both invisible unless the
frame is compared to Stata's own output:

1. THE LAG WAS TAKEN ON ROW POSITION, NOT ON THE TIME AXIS. Stata's ``tsset`` makes ``L1.y`` mean
   "the previous PERIOD", so with an interior gap a row whose predecessor period is missing cannot
   enter the sample. Shifting by row position instead keeps one row too many AND pairs it with the
   wrong observation. Measured with ``drop if year == 1989``:

       Stata:    Sample: 1980 thru 2013, but with a gap    Number of obs = 32
                 _cons .049534   L1 .5016988
       before:   nobs 33, and 1990's lag taken from 1988

   Stata's own `e(sample)` confirms the mechanism: ``levelsof year if e(sample)`` is
   1980..1988 then 1991..2013 -- 1990 is ABSENT because its L1 is 1989, which no longer exists.

2. ``time`` NAMING A MISSING COLUMN WAS SILENTLY IGNORED. ``varbasic(..., time="nope")`` fell back
   to ``Sample: (not recorded)``, which is indistinguishable from deliberately omitting the argument
   and, worse, claims nothing about the axis the output was never checked against.

The ``, but with a gap`` suffix is Stata's, printed at the END of the Sample line. It describes the
AXIS, so it appears even when no row was dropped because of the gap.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import ConsistencyError

DATA = r"D:\myCodeProject\test_westata\data\gdp_china.dta"

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(DATA).exists(),
    reason="gdp_china.dta is not available on this machine",
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    data = pd.read_stata(DATA)
    data["lny"] = np.log(data["y"])
    data["dlny"] = data["lny"].diff()
    return data


@pytest.fixture(scope="module")
def gapped(frame) -> pd.DataFrame:
    """`drop if year == 1989`, the capture that exposes the row-position bug."""

    return frame[frame["year"] != 1989].reset_index(drop=True)


# ---------------------------------------------------------------- the lag follows the axis
def test_a_gap_drops_the_row_whose_predecessor_period_is_missing(gapped) -> None:
    """Stata: 32 observations, with 1990 absent because its L1 is the removed 1989."""

    result = ws.varbasic(gapped, "dlny", lags=1, time="year")
    assert result.nobs == 32, "Stata reports 32; a row-position shift keeps 33"

    # The mechanism, stated as a fact about the sample rather than the count. Two independent ways a
    # row can fail, and Stata's `e(sample)` shows BOTH are in play here:
    #   * 1979 leaves because its L1 is 1978's `dlny`, which `diff()` made missing (the usual trim);
    #   * 1990 leaves because its L1 period -- 1989 -- is not on the axis at all.
    # A row-position shift sees only the first, keeps 1990, and pairs it with 1988.
    years = gapped["year"].to_numpy(dtype=float)
    value_at = dict(zip(years.tolist(), gapped["dlny"].to_numpy(dtype=float)))
    usable = {
        year for year in years
        if not np.isnan(value_at[year]) and not np.isnan(value_at.get(year - 1.0, np.nan))
    }
    assert 1979 not in usable, "its L1 (1978) is missing, the ordinary head trim"
    assert 1990 not in usable, "its L1 (1989) is off the axis -- this is the defect"
    assert 1991 in usable, "its L1 (1990) survives, so it enters"
    assert len(usable) == 32, sorted(usable)


def test_the_coefficients_match_stata_on_the_gapped_frame(gapped) -> None:
    """Measured: `_cons .049534`, `L1 .5016988` -- the row-position version agrees on neither."""

    result = ws.varbasic(gapped, "dlny", lags=1, time="year")
    assert result.coefficients[0, 0] == pytest.approx(0.049534, abs=1e-6)
    assert result.coefficients[1, 0] == pytest.approx(0.5016988, abs=1e-6)


def test_the_information_criteria_match_stata_on_the_gapped_frame(gapped) -> None:
    """One extra row does not just add a data point: it moves every criterion."""

    result = ws.varbasic(gapped, "dlny", lags=1, time="year")
    assert result.ll == pytest.approx(84.2783, abs=1e-4)
    assert result.aic == pytest.approx(-5.142394, abs=1e-6)
    assert result.sbic == pytest.approx(-5.050785, abs=1e-6)
    assert result.fpe == pytest.approx(0.0003422, abs=1e-7)


def test_the_gap_suffix_is_printed_at_the_end_of_the_sample_line(gapped) -> None:
    """Byte-identical to Stata, including the 48-character field before `Number of obs`."""

    rendered = str(ws.varbasic(gapped, "dlny", lags=1, time="year")).splitlines()[3]
    assert rendered == (
        "Sample: 1980 thru 2013, but with a gap"
        "          Number of obs     =         32"
    ), rendered


def test_a_gapless_frame_carries_no_suffix(frame) -> None:
    """Guard against the suffix leaking onto ordinary data."""

    result = ws.varbasic(frame, "dlny", lags=1, time="year")
    assert result.sample_has_gap is False
    assert "but with a gap" not in str(result)
    assert result.nobs == 34


def test_the_suffix_describes_the_axis_not_the_retained_rows(frame) -> None:
    """A gap in the AXIS is reported even when the estimation sample is not short of a row.

    This is the distinction the two mechanisms would otherwise blur: `tsset` calls the axis gapped
    from the data alone, and Stata prints that regardless of which rows a lag happens to drop.
    """

    holed = frame.drop(index=frame.index[5]).reset_index(drop=True)
    result = ws.varbasic(holed, "dlny", lags=1, time="year")
    assert result.sample_has_gap is True


# ---------------------------------------------------------------- time must be honoured
def test_an_unknown_time_column_is_refused(frame) -> None:
    """Silently degrading to `(not recorded)` hides a typo and claims an unverified sample."""

    with pytest.raises(ConsistencyError, match="not a column"):
        ws.varbasic(frame, "dlny", lags=1, time="nope")


def test_the_refusal_lists_the_available_columns(frame) -> None:
    with pytest.raises(ConsistencyError) as caught:
        ws.varbasic(frame, "dlny", lags=1, time="nope")
    message = str(caught.value)
    for column in ("year", "y", "lny", "dlny"):
        assert column in message, (column, message)


def test_the_column_name_is_case_sensitive(frame) -> None:
    """Stata's variable names are case-sensitive, so a near-miss must not be accepted."""

    with pytest.raises(ConsistencyError, match="not a column"):
        ws.varbasic(frame, "dlny", lags=1, time="YEAR")


def test_a_time_axis_with_missing_values_is_refused(frame) -> None:
    """A hole in the axis makes the lag target unknowable; refusing beats guessing."""

    holed = frame.copy()
    holed.loc[holed.index[0], "year"] = np.nan
    with pytest.raises(ConsistencyError, match="missing values"):
        ws.varbasic(holed, "dlny", lags=1, time="year")


def test_duplicate_periods_are_refused(frame) -> None:
    """`tsset` refuses duplicates; with two rows claiming one period the lag is ambiguous."""

    duplicate = frame.copy()
    duplicate.loc[duplicate.index[1], "year"] = duplicate.loc[duplicate.index[0], "year"]
    with pytest.raises(ConsistencyError, match="duplicate periods"):
        ws.varbasic(duplicate, "dlny", lags=1, time="year")


def test_omitting_time_still_works(frame) -> None:
    """`time` is optional: without it the sample line says so and every number is unchanged."""

    with_time = ws.varbasic(frame, "dlny", lags=1, time="year")
    without = ws.varbasic(frame, "dlny", lags=1)
    assert np.allclose(with_time.coefficients, without.coefficients)
    assert with_time.nobs == without.nobs
    assert "Sample: (not recorded)" in str(without)


def test_the_class_form_forwards_the_same_validations(frame) -> None:
    """`Varbasic(...).fit()` must not be a second, laxer path."""

    with pytest.raises(ConsistencyError, match="not a column"):
        ws.Varbasic(frame, "dlny", lags=1, time="nope").fit()
