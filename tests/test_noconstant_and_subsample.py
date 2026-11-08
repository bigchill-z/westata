"""No-constant regressions and subsample regressions both match Stata.

C04-02 scores `reg lnw s, noc` at 0.9091 and C05-03 scores an `if`-restricted regression at 0.9333.

NO CONSTANT is expressed through the FORMULA, not a keyword argument -- `OLS.from_formula` takes no
`noconstant=`, which raises TypeError. The two patsy spellings both work and agree:

    lnw ~ s - 1        ['s']
    lnw ~ 0 + s        ['s']

    b[s] 0.4154001027   se 0.0021675667   R2 0.9798048435

against `reg lnw s, noconstant`.

SUBSAMPLES are handled by filtering the frame before the fit, which is the equivalent of Stata's `if`:

    ws.OLS.from_formula("lnw ~ s + expr", frame[frame["s"] > 12]).fit()
        349 observations, b[s] 0.0934136310

against `reg lnw s expr if s > 12`. The OBSERVATION COUNT is part of the guard, because a filter that
silently did nothing would still return a plausible coefficient on the full sample.

NEITHER ROUTE IS A KEYWORD. That is worth stating plainly: Stata users reach for `, noconstant` and `if`,
and the package expresses both in the formula or the input frame instead.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")


def test_from_formula_takes_no_noconstant_keyword(frame) -> None:
    """Recorded because it is the first thing a Stata user tries."""

    with pytest.raises(TypeError):
        ws.OLS.from_formula("lnw ~ s", frame, noconstant=True)


@pytest.mark.parametrize("formula", ["lnw ~ s - 1", "lnw ~ 0 + s"])
def test_the_no_constant_fit_matches_stata(frame, formula) -> None:
    fitted = ws.OLS.from_formula(formula, frame).fit()
    assert list(fitted.params.index) == ["s"]
    assert float(fitted.params["s"]) == pytest.approx(0.4154001027, abs=1e-9)
    assert float(fitted.bse["s"]) == pytest.approx(0.0021675667, abs=1e-9)
    assert float(fitted.rsquared) == pytest.approx(0.9798048435, abs=1e-9)


def test_the_two_spellings_agree(frame) -> None:
    minus = ws.OLS.from_formula("lnw ~ s - 1", frame).fit()
    plus = ws.OLS.from_formula("lnw ~ 0 + s", frame).fit()
    assert float(minus.params["s"]) == pytest.approx(
        float(plus.params["s"]), abs=1e-12
    )


def test_a_subsample_regression_matches_stata(frame) -> None:
    subset = frame[frame["s"] > 12]
    fitted = ws.OLS.from_formula("lnw ~ s + expr", subset).fit()
    assert int(fitted.nobs) == 349
    assert float(fitted.params["s"]) == pytest.approx(0.0934136310, abs=1e-9)


def test_the_subsample_is_actually_smaller(frame) -> None:
    """A filter that did nothing would still give a plausible coefficient."""

    subset = frame[frame["s"] > 12]
    assert len(subset) < len(frame)
    full = ws.OLS.from_formula("lnw ~ s + expr", frame).fit()
    part = ws.OLS.from_formula("lnw ~ s + expr", subset).fit()
    assert float(full.params["s"]) != pytest.approx(float(part.params["s"]), rel=1e-3)
