"""Fixed effects and between effects coefficients, captured from a live Stata.

C12-01 scores `xtreg, fe` at 1.0 and C12-06 scores `xtreg, be` at 0.75, but neither had these
coefficients pinned. Measured on munnell.dta (48 states, 17 years):

    xtreg lny lnk1 lnk2 lnlabor, fe
        lnk1  -0.08802881    lnk2  0.23243669
        lnlabor 0.85776792   _cons 2.92056067

    xtreg lny lnk1 lnk2 lnlabor, be
        lnk1   0.17896286    lnk2  0.30178625
        lnlabor 0.57528789   _cons 1.57528552

Every coefficient matches to eight decimals. The between estimator is the regression on the
unit MEANS, which is why its coefficients differ from the within ones -- a guard below checks that
the two are not accidentally the same function.
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


@pytest.fixture(scope="module")
def fixed(panel):
    return ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", panel, entity="sid", time="year"
    ).fit()


@pytest.fixture(scope="module")
def between(panel):
    return ws.BetweenOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", panel, entity="sid", time="year"
    ).fit()


def test_the_fixed_effect_coefficients_match_stata(fixed) -> None:
    """C12-01."""

    assert float(fixed.params["lnk1"]) == pytest.approx(-0.08802881, abs=1e-8)
    assert float(fixed.params["lnk2"]) == pytest.approx(0.23243669, abs=1e-8)
    assert float(fixed.params["lnlabor"]) == pytest.approx(0.85776792, abs=1e-8)
    assert float(fixed.params["Intercept"]) == pytest.approx(2.92056067, abs=1e-8)


def test_the_between_coefficients_match_stata(between) -> None:
    """C12-06."""

    assert float(between.params["lnk1"]) == pytest.approx(0.17896286, abs=1e-8)
    assert float(between.params["lnk2"]) == pytest.approx(0.30178625, abs=1e-8)
    assert float(between.params["lnlabor"]) == pytest.approx(0.57528789, abs=1e-8)
    assert float(between.params["Intercept"]) == pytest.approx(1.57528552, abs=1e-8)


def test_within_and_between_are_different_estimators(fixed, between) -> None:
    """Pinned because both derive from RandomEffects, where a shared setting can leak."""

    assert float(fixed.params["lnk1"]) != pytest.approx(
        float(between.params["lnk1"]), rel=1e-3
    )


def test_the_between_estimator_runs_on_entity_means(panel) -> None:
    """Its observation count is the number of units, not the number of rows."""

    between = ws.BetweenOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", panel, entity="sid", time="year"
    ).fit()
    # NOBS is the entity count and e(N) the LONG-panel size: Stata's xtreg, be reports 816 for
    # e(N) while regressing 48 group means. Both numbers are correct and they are different.
    assert int(between.nobs) == int(panel["sid"].nunique())
    assert int(between.e["N"]) == int(len(panel))
