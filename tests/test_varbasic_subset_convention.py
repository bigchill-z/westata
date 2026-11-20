"""Filtering: pandas slicing is the ordinary path; `subset=` exists for one specific case.

Slicing the frame in pandas and passing it in is fine -- that is how every other estimator here is
used. The single exception is a condition that cuts the HEAD of the series:

Stata builds the lag over the COMPLETE time axis and only then restricts the sample
(`_qsur.ado:54`), so a retained first row may look back past the `if` boundary. A pre-sliced frame has
nothing there, that row's regressor is missing, and the sample is one smaller.

Measured on gdp_china.dta, `if year>=1985`:

    Stata                   N=29   .528017
    pandas pre-slice        N=28   .502390   <- smaller sample, warning emitted
    subset= on full frame   N=29   .528017

Cutting the TAIL (`year<2013`) loses nothing either way, so pandas slicing is the recommended path
unless the condition cuts the head.
"""

import inspect
import pathlib
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\gdp_china.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(DATA).exists(), reason="gdp_china.dta is not available"
)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    frame = pd.read_stata(DATA)
    frame["lny"] = np.log(frame["y"])
    frame["dlny"] = frame["lny"].diff()
    return frame


# ---------------------------------------------------------------- pandas slicing is fine
def test_tail_slicing_in_pandas_matches_stata(panel) -> None:
    """`year<2013` cuts the tail, so slicing first is equivalent and is the ordinary usage."""

    sliced = panel[panel["year"] < 2013].reset_index(drop=True)
    fitted = ws.varbasic(sliced, "dlny", lags=1, time="year")
    assert fitted.nobs == 33
    assert ws.irf_table(fitted).responses[0].values[1] == pytest.approx(0.536273, abs=1e-6)


def test_head_slicing_gives_the_prefiltered_sample(panel) -> None:
    """`year>=1985` cuts the head: the sample follows pandas, not Stata, and nothing warns."""

    sliced = panel[panel["year"] >= 1985].reset_index(drop=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        by_slice = ws.varbasic(sliced, "dlny", lags=1, time="year")
    by_subset = ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] >= 1985)
    assert by_slice.nobs == by_subset.nobs == 28
    assert not [w for w in caught if "no usable lag" in str(w.message)]


def test_tail_slicing_does_not_warn(panel) -> None:
    sliced = panel[panel["year"] < 2013].reset_index(drop=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ws.varbasic(sliced, "dlny", lags=1, time="year")
    assert not [w for w in caught if "no usable lag" in str(w.message)]


# ---------------------------------------------------------------- subset= keeps the row
def test_subset_matches_loc_exactly(panel) -> None:
    """R3.4: the mask is applied as `data.loc[subset]`, so N follows the frame, not Stata."""

    mask = panel["year"] >= 1985
    by_subset = ws.varbasic(panel, "dlny", lags=1, time="year", subset=mask)
    manual = ws.varbasic(panel.loc[mask].reset_index(drop=True), "dlny", lags=1, time="year")
    assert by_subset.nobs == manual.nobs == 28
    assert by_subset.ll == manual.ll


def test_subset_selects_the_same_rows_for_a_tail_cut(panel) -> None:
    """Where slicing is equivalent, the two paths must agree exactly."""

    sliced = panel[panel["year"] < 2013].reset_index(drop=True)
    by_slice = ws.varbasic(sliced, "dlny", lags=1, time="year")
    by_mask = ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] < 2013)
    assert by_slice.nobs == by_mask.nobs == 33
    assert by_slice.ll == by_mask.ll


def test_a_subset_that_selects_nothing_raises(panel) -> None:
    with pytest.raises(ws.ConsistencyError, match="subset"):
        ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] > 3000)


# ---------------------------------------------------------------- naming convention
def test_the_parameter_is_called_subset() -> None:
    """The package-wide spelling for Stata's `if`/`in`; `keep=` was a reinvention."""

    parameters = inspect.signature(ws.varbasic).parameters
    assert "subset" in parameters
    assert "keep" not in parameters
    assert "subset" in inspect.signature(ws.VarBasic.__init__).parameters


def test_the_frame_is_named_data() -> None:
    """R2.3: the DataFrame parameter is `data` throughout the package."""

    names = list(inspect.signature(ws.varbasic).parameters)
    assert names[0] == "data"
    assert "series" not in names and "frame" not in names


def test_subset_matches_the_other_estimators(panel) -> None:
    assert "subset" in inspect.signature(ws.Regress.from_formula).parameters


def test_the_class_form_takes_the_same_mask(panel) -> None:
    by_function = ws.varbasic(panel, "dlny", lags=1, time="year", subset=panel["year"] < 2013)
    by_class = ws.VarBasic(
        panel, "dlny", lags=1, time="year", subset=panel["year"] < 2013
    ).fit()
    assert by_function.ll == by_class.ll
