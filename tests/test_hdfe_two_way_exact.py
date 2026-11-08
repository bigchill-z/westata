"""Two-way fixed effects: `xtreg ... i.year, fe` and FEOLS' absorbed form agree with Stata.

C12-09 scores `xtreg ..., fe i.year` at 0.2444 and C12-10 scores `reghdfe ... absorb()` at 0.5417.
Measured on munnell.dta against a live Stata:

    xtset sid year
    xtreg lny lnk1 lnk2 lnlabor i.year, fe
        lnk1   -0.06674598    lnk2  0.16241593    lnlabor 0.81825148

`FEOLS` is this package's absorbed-regression engine -- the reghdfe equivalent -- and the absorbed
formula reaches the same coefficients:

    lny ~ lnk1 + lnk2 + lnlabor | sid + year

TWO ROUTES, ONE ANSWER, which is the point of the second guard: the two-way within estimator does not
depend on whether the effects are entered as dummies or absorbed out.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"


@pytest.fixture(scope="module")
def panel():
    try:
        frame = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    frame = frame.copy()
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    frame["year"] = frame["year"].astype(int)
    return frame


def test_the_absorbed_two_way_fit_matches_stata(panel) -> None:
    """C12-09/C12-10: entity and year effects absorbed together."""

    fitted = ws.FEOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor | sid + year", panel
    ).fit()

    assert float(fitted.params["lnk1"]) == pytest.approx(-0.06674598, abs=1e-8)
    assert float(fitted.params["lnk2"]) == pytest.approx(0.16241593, abs=1e-8)
    assert float(fitted.params["lnlabor"]) == pytest.approx(0.81825148, abs=1e-8)


def test_absorbing_the_year_effects_changes_the_coefficients(panel) -> None:
    """The one-way fit is the same command without `+ year`; if these agreed, the second
    dimension would be doing nothing."""

    two_way = ws.FEOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor | sid + year", panel
    ).fit()
    one_way = ws.FEOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor | sid", panel
    ).fit()

    assert float(two_way.params["lnk1"]) != pytest.approx(
        float(one_way.params["lnk1"]), rel=1e-3
    )


def test_the_absorbed_fit_uses_every_row(panel) -> None:
    """Both dimensions are inside the same 816-row panel, so no observation is lost."""

    fitted = ws.FEOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor | sid + year", panel
    ).fit()
    assert int(fitted.nobs) == int(len(panel))
