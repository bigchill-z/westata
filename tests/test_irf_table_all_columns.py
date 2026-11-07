"""`irf_table()` shows every (impulse, response) column by default.

C13-06 of the v3 report rates `irf table oirf` as showing one column where Stata shows four. The
method existed and was correct; it DEFAULTED `impulse`/`response` to the first variable, so a
bivariate VAR produced a single column. Calling it with neither argument now prints the full table,
while naming either one still gives the narrow form unchanged.

THE COLUMN ORDER IS NOT THE ARRAY'S ORDER. For a bivariate VAR(2) whose `orth_irfs[0]` is

    [[.994097, 0       ],   response=y1, impulse=y1 then y2
     [.071477, .966327]]   response=y2, impulse=y1 then y2

Stata's four columns read

    step 0 |  .994097     .071477           0     .966327

so they are (i=y1,r=y1) (i=y1,r=y2) (i=y2,r=y1) (i=y2,r=y2) -- IMPULSE OUTER, response inner,
while the array is indexed [step, response, impulse]. Neither `.ravel()` nor a response-outer loop
reproduces Stata's order; the first attempt here got `.994097  0  .071477  .966327` and a guard
below pins the correct sequence so it cannot drift again.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame({"y1": rng.normal(size=n), "y2": rng.normal(size=n)})
    return ws.VAR(frame[["y1", "y2"]], lags=2).fit()


def test_the_default_shows_four_columns(fitted) -> None:
    """A bivariate VAR gives k*k = 4 columns, which is what Stata prints."""

    lines = fitted.irf_table(periods=4).splitlines()
    header = next(line for line in lines if "(1)" in line)
    for marker in ("(1)", "(2)", "(3)", "(4)"):
        assert marker in header, marker


def test_no_fifth_column(fitted) -> None:
    lines = fitted.irf_table(periods=4).splitlines()
    header = next(line for line in lines if "(1)" in line)
    assert "(5)" not in header


def test_the_step_zero_row_matches_stata(fitted) -> None:
    """Captured: `       0 |  .994097     .071477           0     .966327`."""

    lines = fitted.irf_table(periods=4).splitlines()
    row = next(line for line in lines if "|" in line and " 0 |" in line)
    values = [token for token in row.split("|")[1].split() if token]
    assert values == [".994097", ".071477", "0", ".966327"]


def test_the_column_order_is_impulse_outer(fitted) -> None:
    """The array is [step, response, impulse]; Stata's columns are impulse-outer.

    Getting this backwards swaps columns 2 and 3 and still produces plausible numbers, so the
    order is asserted directly rather than only through the step-0 row.
    """

    responses = np.asarray(fitted.irf(periods=4).orth_irfs, dtype=float)
    lines = fitted.irf_table(periods=4).splitlines()
    row = next(line for line in lines if "|" in line and " 1 |" in line)
    values = [token for token in row.split("|")[1].split() if token]

    expected = [
        responses[1, 0, 0],  # response y1, impulse y1
        responses[1, 1, 0],  # response y2, impulse y1
        responses[1, 0, 1],  # response y1, impulse y2
        responses[1, 1, 1],  # response y2, impulse y2
    ]
    for token, value in zip(values, expected):
        assert float(token) == pytest.approx(float(value), abs=5e-7)


def test_naming_an_impulse_keeps_the_narrow_form(fitted) -> None:
    """The single-column table must not have moved."""

    lines = fitted.irf_table(impulse="y1", response="y1", periods=2).splitlines()
    header = next(line for line in lines if "(1)" in line)
    assert "(2)" not in header
    assert any("impulse = y1" in line for line in lines)


def test_the_narrow_numbers_are_unchanged(fitted) -> None:
    lines = fitted.irf_table(impulse="y1", response="y1", periods=2).splitlines()
    row = next(line for line in lines if " 0 |" in line)
    assert ".994097" in row


def test_the_footnote_names_every_pair(fitted) -> None:
    text = fitted.irf_table(periods=1)
    for pair in ("y1->y1", "y1->y2", "y2->y1", "y2->y2"):
        assert pair in text, pair
