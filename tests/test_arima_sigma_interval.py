"""Pin arima's ``/sigma`` interval to Stata's symmetric-normal rule.

Reference from a live ``arima dinf, arima(1,0,0)`` on macro_swatson.dta::

        dinf | Coefficient  std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
          _cons |     .00012    .001168     0.10   0.918    -.0021692    .0024092
    -------------+----------------------------------------------------------------
    -------------+----------------------------------------------------------------
         /sigma |   .0164621   .0007197    22.87   0.000     .0150515    .0178727

THE RULE. Recovering the implied z from Stata's two bounds gives

    (lo - sigma)/se = -1.959983      (hi - sigma)/se = +1.959983

i.e. a SYMMETRIC band on the SIGMA scale with the normal's 1.96. This is NOT tobit's rule,
which lives on the log-variance scale with a t(df_r) -- the two commands' scale rows follow
different conventions, and the previous code applied yet a third (square root of a variance
interval), which matched neither.

RESIDUAL, reported rather than claimed. The level of sigma still differs in its fifth
significant figure (.0164569 against .0164621) because the optimiser stops at a slightly
different point. Measured directly: statsmodels' own loglikelihood AT Stata's parameters is
446.146874240 against Stata's e(ll) 446.146874230, i.e. 1e-8 apart, so both maximise the same
function and the difference is where they stopped, not what they optimised. The interval's
WIDTH is exactly proportional to the estimate, so this test asserts the rule (symmetry,
1.96, proportionality) rather than the absolute digits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    frame = frame.copy()
    frame["t"] = np.arange(1, len(frame) + 1)
    sample = frame[["dinf", "t"]].dropna().reset_index(drop=True)
    return ws.ARIMA(sample["dinf"], order=(1, 0, 0), trend="c").fit()


def test_the_interval_is_symmetric_on_the_sigma_scale(fitted) -> None:
    """The midpoint of the printed bounds must equal the printed estimate."""

    rendered = [
        line for line in str(fitted.summary(style="stata")).splitlines()
        if "/sigma" in line
    ]
    assert rendered, "the /sigma row is missing"

    from westata.results.summary import _stata_number

    line = rendered[0]
    body = line.split("|", 1)[1]
    # Fixed columns: estimate, se, z, p, lower, upper.
    estimate = float(body[0:11].replace(" ", "") or 0)
    lower = float(body[41:54].replace(" ", ""))
    upper = float(body[54:66].replace(" ", ""))
    assert abs((lower + upper) / 2.0 - estimate) < 5e-7
    _ = _stata_number  # imported only to assert the formatter exists


def test_the_rule_is_the_normal_1_96(fitted) -> None:
    """Recover z from the printed bounds; it must be 1.96 on both sides."""

    line = [
        row for row in str(fitted.summary(style="stata")).splitlines() if "/sigma" in row
    ][0]
    body = line.split("|", 1)[1]
    estimate = float(body[0:11].replace(" ", "") or 0)
    se = float(body[11:22].replace(" ", "") or 0)
    lower = float(body[41:54].replace(" ", ""))
    upper = float(body[54:66].replace(" ", ""))
    z_low = (lower - estimate) / se
    z_high = (upper - estimate) / se
    assert z_low == pytest.approx(-1.959963984540054, abs=5e-3)
    assert z_high == pytest.approx(1.959963984540054, abs=5e-3)


def test_the_width_is_proportional_to_the_estimate(fitted) -> None:
    """Stata's own numbers satisfy width/sigma = 2*1.96*se/sigma; ours must too.

    This is the property that survives the optimiser's stopping-point difference: the ratio
    is governed by the rule, not by where the optimiser landed.
    """

    line = [
        row for row in str(fitted.summary(style="stata")).splitlines() if "/sigma" in row
    ][0]
    body = line.split("|", 1)[1]
    estimate = float(body[0:11].replace(" ", "") or 0)
    se = float(body[11:22].replace(" ", "") or 0)
    lower = float(body[41:54].replace(" ", ""))
    upper = float(body[54:66].replace(" ", ""))
    assert (upper - lower) == pytest.approx(2.0 * 1.959963984540054 * se, rel=5e-3)
    assert estimate == pytest.approx(0.0164621, rel=1e-3)


def test_the_loglikelihood_matches_stata_at_its_parameters() -> None:
    """The evidence that both packages maximise the SAME likelihood."""

    from statsmodels.tsa.arima.model import ARIMA as _SMARIMA

    frame = pd.read_stata(DATA, convert_categoricals=False).copy()
    frame["t"] = np.arange(1, len(frame) + 1)
    sample = frame[["dinf", "t"]].dropna().reset_index(drop=True)
    model = _SMARIMA(sample["dinf"].to_numpy(), order=(1, 0, 0), trend="c")
    stata_params = np.array([0.00012001, -0.24098776, 0.01646209**2])
    assert float(model.loglike(stata_params)) == pytest.approx(446.146874230, abs=1e-7)
