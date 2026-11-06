"""dfgls: round 120's key finding -- a faithful reproduction of the visible ado steps also fails.

Round 119 concluded the divergence was in the GLS fit. This round read the ado directly and found
that conclusion WRONG: b_iota matches exactly once the data is aligned.

FIRST, A DATA TRAP that produced a false lead. `lrgnp` in nelson_plosser.dta has 31 leading NaNs
and 62 real observations. My first check pushed the raw frame to Stata, so every `lrgnp` value
arrived as missing and the GLS fit ran on garbage, giving b_iota = 5.8068915787. Dropping the NaNs
first:

    Stata   b_iota = 5.2182086367   N = 62
    westata b_iota = 5.2182086367   N = 62      <- exact

So the GLS detrending, the alpha, the quasi-difference and the `dty = lrgnp - b_iota` step are all
CORRECT. Round 119's diagnosis was an artefact of the missing-value misalignment.

THE REAL DIVERGENCE is one level further down, and it is not explicable from the ado text. Running
the ado's own regression specification on the correctly aligned data:

    gen double dty = lrgnp - b_iota
    reg D.dty L.dty DL(1/10).dty, noc        /// verbatim from dfgls.ado:207
        e(N) = 51
        tau  = 0.806085

    dfgls lrgnp                              /// Stata's own command
        lag 10 tau = -1.284

Same data, same visible steps, two different answers. So `dfgls.ado` does something in its full
control flow that neither its comments nor a line-for-line reconstruction reveals -- likely in how
`touse` is built across the two `markout` calls at lines 106-107 and 141-142, which I could not
reconcile by reading.

WHAT THIS BUYS: the next attempt should stop re-deriving the formula (it is right) and instead run
the ado's own steps inside Stata with `set trace on`, capturing which sample each `reg` actually
uses. The single number to compare is `e(N)` in the D.`dty` regression -- if the ado's own run
reports something other than 51, the discrepancy is in `touse`, not in the arithmetic.

Not fixed. A unit-root statistic with the wrong sign remains worse than a refusal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nelson_plosser.dta"

#: Stata's `dfgls lrgnp`, first row (lag 10).
STATA_TAU_LAG10 = -1.284

#: A line-for-line reproduction of the ado's visible steps gives this instead.
ADO_REPRODUCTION = 0.806085

#: Stata's `_b[iota]` from the GLS detrending, on the filtered sample.
STATA_B_IOTA = 5.2182086367


@pytest.fixture(scope="module")
def series():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nelson_plosser.dta is not available")
    values = np.asarray(frame["lrgnp"], dtype=float)
    return values[np.isfinite(values)]


def test_the_source_series_needs_the_nan_rows_dropped() -> None:
    """31 leading NaNs; using the raw column misaligns every downstream quantity."""

    frame = pd.read_stata(DATA)
    assert len(frame) == 111
    assert int(frame["lrgnp"].notna().sum()) == 62


def test_the_gls_detrending_constant_matches_stata(series) -> None:
    """b_iota is CORRECT; round 119 blamed it wrongly because of the NaN misalignment."""

    n = series.size
    alpha = 1.0 + (-7.0) / n
    quasi_y = np.empty(n)
    quasi_y[0] = series[0]
    for position in range(1, n):
        quasi_y[position] = series[position] - alpha * series[position - 1]
    quasi_iota = np.empty(n)
    quasi_iota[0] = 1.0
    quasi_iota[1:] = 1.0 - alpha
    b_iota = float(np.linalg.lstsq(quasi_iota[:, None], quasi_y, rcond=None)[0][0])
    assert b_iota == pytest.approx(STATA_B_IOTA, abs=5e-9)


def test_the_package_matches_the_ado_reproduction_not_stata(series) -> None:
    """The defect, stated precisely: it agrees with the visible steps and not with Stata.

    That is why this could not be fixed by re-reading dfgls.ado -- the arithmetic the code
    performs is right; Stata's own command does something else as well.
    """

    result = ws.dfgls(series)
    assert result.cells[0].statistic == pytest.approx(ADO_REPRODUCTION, abs=5e-4)
    assert abs(result.cells[0].statistic - STATA_TAU_LAG10) > 1.0
