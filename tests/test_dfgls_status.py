"""dfgls: P0-4 FIXED in round 151, after five rounds of chasing the wrong default.

THE ROOT CAUSE, found by copying the OFFICIAL dfgls.ado and instrumenting it (not a hand-written
simplified copy, which is what rounds 119-123 used):

    dfgls.ado:39-46     if "`trend'"=="notrend" { local trend 0 }
                        else { local trend 1 }        <- DEFAULT IS A TREND

Stata's bare `dfgls lrgnp` uses cbar=-13.5 and subtracts `b_iota + b_qdtrd*t` from the series.
The `notrend` option is what uses cbar=-7 and subtracts only `b_iota`. Live Stata on
nelson_plosser.dta's lrgnp:

    dfgls lrgnp             lag 10 tau = -1.284   (default, trend)
    dfgls lrgnp, notrend    lag 10 tau = +0.806   (the no-trend case)

The package defaulted `trend=False`, so it implemented the SECOND case and was compared against
the FIRST. Every number in rounds 119-123 was self-consistent and exactly right -- for the wrong
specification. `cbar=-13.5` was even listed as "eliminated" in round 122's write-up, because that
round tested it as a variant of the no-trend case rather than as Stata's default.

WHAT ROUND 151 CHANGED: `dfgls(series)` now defaults to `trend=True`; `notrend=True` or
`trend=False` selects the no-trend case. `_gls_detrend` already supported both -- the defect was
purely the default value and its docstring, which described `trend` as an optional flag.

WHAT THE FIVE-ROUND DETOUR TAUGHT, kept as a guard rather than deleted:
  * a hand-rebuilt "equivalent" of an ado is not the ado. Instrument the real file;
  * when a reconstruction agrees with itself and disagrees with the reference, suspect that the
    two are answering DIFFERENT questions before suspecting hidden internals;
  * "candidate eliminated" lists are only as good as the specification they were tested under.

All ten tau values now match Stata to the printed digit in both specifications.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nelson_plosser.dta"

#: From `dfgls lrgnp` (default) and `dfgls lrgnp, notrend` on nelson_plosser.dta.
STATA_TREND = [-1.2843, -1.4057, -1.7480, -2.0974, -2.0185, -1.7582, -1.9293, -2.1342, -2.4361, -2.6371]
STATA_NOTREND_FIRST = 0.806


@pytest.fixture(scope="module")
def series():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nelson_plosser.dta is not available")
    values = np.asarray(frame["lrgnp"], dtype=float)
    return values[np.isfinite(values)]


def test_the_default_is_statas_default_and_matches_every_tau(series) -> None:
    """Bare dfgls() is the TREND case -- the inverted default was the whole five-round defect."""

    result = ws.dfgls(series)
    assert [cell.lags for cell in result.cells] == list(range(10, 0, -1))
    assert all(cell.nobs == 51 for cell in result.cells)
    assert [round(float(cell.statistic), 4) for cell in result.cells] == pytest.approx(
        STATA_TREND, abs=5e-4
    )


def test_notrend_reproduces_the_other_stata_case(series) -> None:
    """The +0.806 that rounds 119-123 thought was wrong IS Stata -- for `notrend`."""

    for kwargs in ({"notrend": True}, {"trend": False}):
        result = ws.dfgls(series, **kwargs)
        assert float(result.cells[0].statistic) == pytest.approx(
            STATA_NOTREND_FIRST, abs=5e-4
        ), kwargs


def test_the_two_specifications_are_different_estimands(series) -> None:
    """Pinned so a future 'simplification' cannot quietly merge them again."""

    trended = float(ws.dfgls(series).cells[0].statistic)
    detrended = float(ws.dfgls(series, notrend=True).cells[0].statistic)
    assert trended < 0 < detrended
    assert abs(trended - detrended) > 2.0


def test_the_full_table_matches_stata_line_for_line(series) -> None:
    """Round 154 completed the table: three critical-value columns plus the SIC/MAIC footers.

    All 20 non-blank lines of `dfgls lrgnp` now match a live Stata capture exactly, and all 20 of
    `dfgls lrgnp, notrend` too. This replaces the earlier guard, which asserted that the critical
    columns were ABSENT -- true then, wrong now.
    """

    rendered = str(ws.dfgls(series, name="lrgnp"))
    lines = [line for line in rendered.splitlines() if line.strip()]
    assert len(lines) == 20
    # Every line between the two rules is 62 characters, header included.
    first_rule = lines.index("-" * 62)
    last_rule = len(lines) - 1 - lines[::-1].index("-" * 62)
    assert all(len(line) == 62 for line in lines[first_rule : last_rule + 1])
    assert "     10          -1.284       -3.724       -2.721       -2.440" in lines
    assert "      1          -2.637       -3.724       -3.160       -2.860" in lines
    assert lines[-1] == "Min MAIC = -5.376693 at lag 1 with RMSE = .0573553"


def test_the_critical_columns_use_two_different_sample_sizes(series) -> None:
    """1% is interpolated on the ORIGINAL length (constant down the table); 5%/10% move.

    Getting this wrong is subtle: using one N for both columns reproduces neither.
    """

    result = ws.dfgls(series)
    ones = {round(cell.critical_1, 3) for cell in result.cells}
    assert ones == {-3.724}, "the 1% column must be constant"
    fives = [round(cell.critical_5, 3) for cell in result.cells]
    assert len(set(fives)) == len(fives), "the 5% column must vary with the lag"
    assert fives[0] == pytest.approx(-2.721, abs=5e-4)
    assert fives[-1] == pytest.approx(-3.160, abs=5e-4)


def test_the_footer_selects_on_the_sequential_t_rule_not_the_smallest_rmse(series) -> None:
    """Opt lag is the first lag (down from maxlag) with a highest-lag p-value below 0.10.

    The old property returned the minimum-RMSE lag, which is 10 here, while Stata prints 1.
    """

    result = ws.dfgls(series)
    assert result.best_lag == 1
    smallest_rmse = min(result.cells, key=lambda cell: cell.rmse).lags
    assert smallest_rmse != result.best_lag
