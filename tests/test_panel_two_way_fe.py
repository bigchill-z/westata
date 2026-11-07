"""Two-way fixed effects (`xtreg ..., fe i.year`): exact, via PanelOLS(time_effects=True).

C12-09 scored 0.00 / 0.01 -- the lowest numeric coverage in chapter 12 after `xtserial`. Round 176
checked it against a live

    encode st_abb, gen(sid)
    xtset sid year
    xtreg lny lnk1 lnk2 lnlabor i.year, fe

on munnell.dta (48 states x 17 years, 816 observations):

                westata                     Stata
    lnk1       -0.066746  (0.025277)      -.066746   (.0252767)
    lnk2        0.162416  (0.027836)       .1624159  (.0278359)
    lnlabor     0.818251  (0.025062)       .8182515  (.0250616)

Both the point estimates and the standard errors agree. The entity-only fit gives lnk1 -.0880288,
so the year effects are genuinely in play -- this is not the same number twice.

WHY THE SCORE WAS ZERO: as with C12-02 next to it, there is no top-level `xtreg`. `i.year` is not a
syntax the package parses either; the time effects are a keyword on `fit()`. A checker typing the
Stata command finds nothing while the estimator is present and correct.

NOT PINNED HERE: the per-year coefficient rows. Stata lists all sixteen year dummies as explicit
coefficients (F(19, 749)); `PanelOLS` absorbs them, so they do not appear in `params`. The absorbed
fit is the same estimate -- the guards below assert that on the common coefficients rather than
inventing per-year values.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\munnell.dta"
FORMULA = "lny ~ lnk1 + lnk2 + lnlabor"

#: From `xtreg lny lnk1 lnk2 lnlabor i.year, fe`.
STATA_COEF = {"lnk1": -0.066746, "lnk2": 0.1624159, "lnlabor": 0.8182515}
STATA_SE = {"lnk1": 0.0252767, "lnk2": 0.0278359, "lnlabor": 0.0250616}

#: The entity-only fit, for contrast: the year effects change the estimates materially.
ENTITY_ONLY_LNK1 = -0.0880288


@pytest.fixture(scope="module")
def panel():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("munnell.dta is not available")


@pytest.fixture(scope="module")
def two_way(panel):
    model = ws.PanelOLS.from_formula(FORMULA, panel, entity="st_abb", time="year")
    return model.fit(time_effects=True)


def test_the_two_way_coefficients_match_stata(two_way) -> None:
    params = dict(two_way.params)
    for name, expected in STATA_COEF.items():
        assert float(params[name]) == pytest.approx(expected, abs=1e-6), name


def test_the_two_way_standard_errors_match_stata(two_way) -> None:
    errors = dict(two_way.bse)
    for name, expected in STATA_SE.items():
        assert float(errors[name]) == pytest.approx(expected, abs=1e-7), name


def test_the_year_effects_actually_change_the_estimate(panel) -> None:
    """Otherwise this guard would pass on the entity-only fit by accident."""

    model = ws.PanelOLS.from_formula(FORMULA, panel, entity="st_abb", time="year")
    entity_only = model.fit()
    two_way = model.fit(time_effects=True)
    assert float(entity_only.params["lnk1"]) == pytest.approx(ENTITY_ONLY_LNK1, abs=1e-7)
    assert abs(float(two_way.params["lnk1"]) - ENTITY_ONLY_LNK1) > 0.02


def test_time_effects_are_a_fit_keyword_not_i_dot_syntax() -> None:
    """Recorded so the zero score is not read as a missing estimator, and so the gap is explicit."""

    import inspect

    from westata.estimators.panel import PanelOLS

    signature = inspect.signature(PanelOLS.fit)
    assert "time_effects" in signature.parameters
    assert not hasattr(ws, "xtreg")
