"""dfgls: the implementation matches its own formula, and the formula does not match Stata.

P0-4 in the report says dfgls "returns the coefficient, not tau". That is NOT what happens -- the
code computes `coefficient / standard_error`, which is tau, and a hand reproduction of the same
arithmetic gives the same number the package does:

    lag 10   westata tau = 0.8061      Stata tau = -1.284

So the defect is not "coefficient instead of tau". It is that the tau VALUE is wrong.

WHAT IS CONFIRMED CORRECT against dfgls.ado:

  * cbar = -7.0 without trend, alpha = 1 + cbar/nobs with nobs = 62 (the ORIGINAL length) --
    matches `scalar alpha = 1+cbar/nobs`;
  * maxlag = int(12*(62/100)^0.25) = 10 -- matches Schwert's rule and Stata's "Maximum lag = 10";
  * the ADF sample is n - (maxlag+1) = 51 -- matches Stata's "Number of obs = 51";
  * the detrending is the OLS residual at the ORIGINAL scale, `y - b_iota`, not the GLS residual --
    matches the ado's own comment at line 109: "generate OLS residual, not GLS residual";
  * the statistic is `_b[L.dty] / sqrt(V[1,1])` -- matches `return scalar dft0=...`.

WHAT IS STILL WRONG: the fitted `_b[iota]`. Reproducing the whole chain by hand yields
b_iota = 5.21820864, dty[0:3] = [-0.457745, -0.429884, -0.4044], coef 0.017490, se 0.021697,
tau 0.8061 -- identical to the package, so the divergence enters at the GLS fit itself and not in
the ADF stage.

NEXT STEP, recorded rather than guessed: instrument the ado directly to read the value it gets for
_b[iota] on this data. Reading that one number separates "our quasi-difference is built wrong" from
"our regression sample for the detrending differs", which is the only remaining fork.

Not fixed. A wrong sign and magnitude on a unit-root statistic is worse than a refusal, and I have
no verified replacement to put in its place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nelson_plosser.dta"

#: From `dfgls lrgnp` on nelson_plosser.dta. Stata's first row is lag 10 -> -1.284.
STATA_TAU_LAG10 = -1.284


@pytest.fixture(scope="module")
def series():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nelson_plosser.dta is not available")
    values = np.asarray(frame["lrgnp"], dtype=float)
    return values[np.isfinite(values)]


def test_the_table_structure_matches_stata(series) -> None:
    """Structure is right even though the statistic is not: 10 down to 1, N = 51."""

    result = ws.dfgls(series)
    lags = [cell.lags for cell in result.cells]
    assert lags == list(range(10, 0, -1))
    assert all(cell.nobs == 51 for cell in result.cells)


def test_the_statistic_is_not_the_coefficient(series) -> None:
    """The report says it returns the coefficient. It does not -- it returns coef/se."""

    result = ws.dfgls(series)
    tau = result.cells[0].statistic
    # The raw coefficient for lag 10 is ~0.0175; the statistic is ~0.806. If the package were
    # returning the coefficient the value would be two orders of magnitude smaller.
    assert abs(tau) > 0.1


def test_the_statistic_is_still_wrong(series) -> None:
    """Pinned as a KNOWN DEFECT so a future fix must deliberately change this test.

    Stata: -1.284. Ours: +0.806. Sign and magnitude both wrong, which makes the statistic
    unusable for a unit-root decision -- hence this stays a recorded defect rather than a
    tolerance.
    """

    result = ws.dfgls(series)
    assert result.cells[0].statistic == pytest.approx(0.8061, abs=5e-4)
    assert abs(result.cells[0].statistic - STATA_TAU_LAG10) > 1.0
