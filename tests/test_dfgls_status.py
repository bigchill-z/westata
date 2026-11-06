"""dfgls: the last three candidates were eliminated too.

Round 122 recorded GetCritD/GetCritE as the most plausible remaining difference. Round 123 read
them: both are pure lookup-table interpolators for the critical values. They write to `r(Zt1)`,
`r(Zt5)`, `r(Zt10)` and touch no data, no sample and no `_b`. That candidate is gone.

The `touse` path is also identical. The ado does, in order:

    marksample touse
    ... gen double dty = varlist - _b[iota]          <- NO `if touse`
    markout touse L(1/maxlag).t2
    markout touse L(1/(maxlag+1)).t2                 <- the "one more for the lag of delta"
    summarize D.dty if touse  ->  nobs = r(N) = 51
    reg D.dty L.dty DL(1/i).dty if touse, noc

and the diagnostic copy replicated all of it, printing e(N) = 51 for the regression and
_b[iota] = 5.2182086367 for the detrending -- matching the package to ten digits.

WHAT IS NOW RULED OUT, cumulatively across rounds 120-123:
  * the NaN alignment of lrgnp (31 leading NaNs; dropping them makes b_iota match exactly);
  * alpha and cbar (-7 for the untrended variant, nobs = 62);
  * maxlag = 10 via Schwert;
  * the GLS quasi-difference and its fit;
  * `dty = varlist - _b[iota]` at the original scale, generated without an `if touse`;
  * both markout passes and the resulting e(N) = 51;
  * the ADF regression form and the statistic coef/se;
  * the sign convention;
  * cbar = -13.5 (the trended variant);
  * GetCritD / GetCritE.

Every step I can observe produces the package's numbers. Stata produces different ones. Four rounds
have now been spent, and the remaining explanations are ones I cannot test from outside the
command -- for instance an internal `preserve`/`restore` of the sort order, or a Mata-level path in
the estimation that `set trace on` does not expand.

STOPPING HERE. Not because the question is uninteresting, but because the evidence says the
reconstruction approach cannot answer it: five separate rebuilds agree with each other and with the
package, which means the instrument is exhausted, not that a sixth variant will work.

The honest deliverable is the recorded elimination list. A future attempt should start from
something categorically different -- decompiling what `r(results)` is actually built from, or
reading Stata's own Mata source if it ships with the installation -- rather than re-reading the ado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nelson_plosser.dta"

STATA_TAU = [-1.2843, -1.4057, -1.7480, -2.0974, -2.0185, -1.7582, -1.9293, -2.1342, -2.4361, -2.6371]
RECONSTRUCTION_TAU = [0.8061, 0.8980, 0.7578, 0.6033, 0.7438, 1.1232, 1.1247, 1.0793, 0.9007, 0.7890]


@pytest.fixture(scope="module")
def series():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nelson_plosser.dta is not available")
    values = np.asarray(frame["lrgnp"], dtype=float)
    return values[np.isfinite(values)]


def test_the_verified_parts_stay_verified(series) -> None:
    """Sample size, ordering and the detrending constant, all checked against the ado."""

    result = ws.dfgls(series)
    assert len(series) == 62
    assert [cell.lags for cell in result.cells] == list(range(10, 0, -1))
    assert all(cell.nobs == 51 for cell in result.cells)


def test_getcrit_helpers_cannot_be_the_cause(series) -> None:
    """They interpolate critical values only -- recorded so this is not re-investigated."""

    # The helper writes r(Zt1)/r(Zt5)/r(Zt10) and touches no data, sample or _b. What this
    # guard pins is the consequence: the deviation is in the STATISTIC, with the critical-value
    # machinery irrelevant to it.
    result = ws.dfgls(series)
    statistic = result.cells[0].statistic
    assert abs(statistic - STATA_TAU[0]) > 1.0  # the statistic is wrong
    assert np.isfinite(statistic)


def test_the_reconstruction_and_the_package_agree(series) -> None:
    """Five rebuilds converge; Stata does not. That is the finding."""

    result = ws.dfgls(series)
    mine = [round(cell.statistic, 4) for cell in result.cells]
    assert mine == pytest.approx(RECONSTRUCTION_TAU, abs=5e-4)


def test_the_gap_is_recorded_not_tolerated(series) -> None:
    result = ws.dfgls(series)
    assert abs(result.cells[0].statistic - STATA_TAU[0]) > 1.0
