"""dfgls: an exhausted investigation, recorded so it is not restarted blindly.

Three rounds have gone into this. The state of knowledge, after reading dfgls.ado, tracing a live
run, and reconstructing every visible step:

CONFIRMED CORRECT in the package:
  * the NaN handling -- lrgnp has 31 leading NaNs, 62 real values, and the code keeps 62;
  * alpha = 1 + (-7.0)/62, cbar -7 without trend;
  * maxlag = int(12*(62/100)^0.25) = 10, matching Schwert and Stata's "Maximum lag = 10";
  * the GLS quasi-difference and its fit: b_iota = 5.2182086367, exactly Stata's;
  * `dty = lrgnp - b_iota` -- the OLS residual at the original scale, as dfgls.ado:117 does;
  * the ADF regression form `reg D.dty L.dty DL(1/i).dty, noc`, verbatim from dfgls.ado:207;
  * the statistic `_b[L.dty]/_se[L.dty]`, matching `return scalar dfti=...`;
  * e(N) = 51 in that regression, matching Stata's reported "Number of obs = 51".

AND YET, on the same data:

    Stata's own dfgls lrgnp, lag 10       -1.284
    a line-for-line reconstruction         0.806085
    this package                           0.8061

`set trace on` confirms the ado executes exactly the lines reconstructed -- the same `gen double
dty = lrgnp - _b[iota]`, the same `reg D.dty L.dty DL(1/i).dty if touse, noc`. So the two results
differ while the commands print identically, which points at something not visible in the trace:
most likely a value of `_b[iota]` captured at a moment I cannot observe from outside, since `_b`
is re-set by each `reg` and the ado reads it after several intermediate commands.

WHAT I AM NOT DOING: guessing a correction. A unit-root statistic with the wrong sign is unusable,
and three rounds of reconstruction have not located the discrepancy; a fourth guess would more
likely introduce a wrong answer that looks deliberate than find the right one.

NEXT STEP IS A DIFFERENT KIND OF MOVE, recorded here: copy dfgls.ado to a scratch program, insert
`di "b_iota=" _b[iota]` immediately before the dty generation, and run it. Reading the number the
ado itself sees is the only remaining way to split "the value differs" from "the value is the same
and something later differs". I did not attempt it this round because editing a copy of a base ado
in place risks polluting later runs, and it wants a clean session.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nelson_plosser.dta"

STATA_TAU_LAG10 = -1.284
RECONSTRUCTION = 0.806085
STATA_B_IOTA = 5.2182086367


@pytest.fixture(scope="module")
def series():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nelson_plosser.dta is not available")
    values = np.asarray(frame["lrgnp"], dtype=float)
    return values[np.isfinite(values)]


def test_the_source_series_is_sixty_two_observations_after_dropping_nans() -> None:
    frame = pd.read_stata(DATA)
    assert len(frame) == 111
    assert int(frame["lrgnp"].notna().sum()) == 62


def test_the_detrending_constant_is_correct(series) -> None:
    """Three rounds agreed this part is right; it is pinned so it stays out of the search."""

    n = series.size
    alpha = 1.0 - 7.0 / n
    quasi_y = np.empty(n)
    quasi_y[0] = series[0]
    for position in range(1, n):
        quasi_y[position] = series[position] - alpha * series[position - 1]
    quasi_iota = np.r_[1.0, np.full(n - 1, 1.0 - alpha)]
    b_iota = float(np.linalg.lstsq(quasi_iota[:, None], quasi_y, rcond=None)[0][0])
    assert b_iota == pytest.approx(STATA_B_IOTA, abs=5e-9)


def test_the_package_equals_a_faithful_reconstruction(series) -> None:
    """The defect stated precisely: it agrees with the ado's visible steps, not with Stata."""

    result = ws.dfgls(series)
    assert result.cells[0].statistic == pytest.approx(RECONSTRUCTION, abs=5e-4)


def test_the_gap_against_stata_is_still_open(series) -> None:
    """Pinned so a future fix must change this on purpose rather than drift into it."""

    result = ws.dfgls(series)
    assert abs(result.cells[0].statistic - STATA_TAU_LAG10) > 1.0


def test_the_table_structure_and_sample_are_correct(series) -> None:
    result = ws.dfgls(series)
    assert [cell.lags for cell in result.cells] == list(range(10, 0, -1))
    assert all(cell.nobs == 51 for cell in result.cells)
