"""ARIMA: two gaps measured in round 181, recorded rather than claimed as fixed.

The round-180 success came from reading Stata's own source (``var.ado:296`` settled ``varsoc``'s
FPE after two rounds of failed formula-fitting). Round 181 applied that systematically and checked
``arima``, which has never been compared against Stata.

TWO FINDINGS:

1. THE ESTIMATOR DIFFERS. On airpassengers.dta with ``arima airpassengers, arima(1,1,0)``:

       Stata      ar = .30378675   ll = -698.73643
       westata    ar =  0.306550   ll = -698.92606

   Causes ruled OUT by measurement, each with the number that rules it out:

     * ``concentrate_scale`` True and False: 0.30655172 and 0.30655049 -- both far from Stata.
     * the conditional-likelihood branch: ``arima.ado:62`` runs ``arch`` only when the user passes
       ``, condition``, so it is not the default path.
     * a separate ``, mle`` option: Stata 1x9 rejects it (``invalid 'mle' r(198)``), so there is
       only one estimation method to choose from.

   What remains is a difference in parameterisation or likelihood definition between the two
   state-space implementations. Twenty-odd probes produced no direction, so it is recorded with
   the evidence rather than fitted further.

2. ``arimasoc`` IS NOT IMPLEMENTED. ``ws.ARIMA`` exposes ``fit`` only; Stata's lag-order selection
   table has no counterpart. This is a missing feature, not a mismatch, and is listed as such.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\airpassengers.dta"

#: From `arima airpassengers, arima(1,1,0) nolog` on a numeric-indexed tsset.
STATA_AR = 0.30378675
STATA_LL = -698.73643


@pytest.fixture(scope="module")
def series():
    try:
        return pd.read_stata(DATA)["airpassengers"].to_numpy(float)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("airpassengers.dta is not available")


def test_the_estimator_still_differs_from_stata(series) -> None:
    """Pinned as a KNOWN gap. If a future round closes it, replace this with equality."""

    fitted = ws.ARIMA(series, order=(1, 1, 0)).fit()
    ar = float(fitted.params.iloc[0])
    likelihood = float(fitted.llf)
    assert abs(ar - STATA_AR) > 1e-3
    assert abs(likelihood - STATA_LL) > 1e-2
    # ...but it is in the right neighbourhood: this is a definition difference, not a broken fit.
    assert ar == pytest.approx(STATA_AR, abs=0.01)
    assert likelihood == pytest.approx(STATA_LL, abs=1.0)


def test_arimasoc_has_no_counterpart() -> None:
    """`ARIMA` exposes fit only -- the lag-order table is a genuinely missing feature."""

    optional = {name for name in dir(ws.ARIMA) if not name.startswith("_")}
    assert optional == {"fit"}
    assert "select_order" not in optional


def test_the_ar_coefficient_is_stationary(series) -> None:
    """A sanity property that holds regardless of which estimator is used."""

    fitted = ws.ARIMA(series, order=(1, 1, 0)).fit()
    assert abs(float(fitted.params.iloc[0])) < 1.0
