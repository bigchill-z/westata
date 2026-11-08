"""Four report items scored 0.0-0.4 that are in fact exact. Verified against a live Stata.

The pattern these share is worth naming: the report scores OUTPUT LAYOUT, and several entries read
"0.0" for a capability that works. Round 75 checked four of them individually.

C12-02 `xtreg, fe vce(robust)` -- scored 0.0, standard errors identical to ten decimals:

    lnk1     .0538468 vs .0538467869
    lnk2     .0506720 vs .0506720321
    lnlabor  .0607880 vs .0607879585
    _cons    .2507401 vs .2507401122

C12-03 `xtreg, re` theta -- scored 0.4, exact at 0.8857362899.
C10-01 `ivregress ..., first` -- `first_stage()` returns the full table.
C10-02 `estat firststage` -- scored 0.2788, identical: R-sq 0.1049, Adjusted 0.1013,
        Partial 0.0493, F 19.5592, Prob 0.0000.

A NOTE ON THE REPORT'S CONVENTION: `first_stage()` (coefficient table) and `estat_firststage()`
(summary statistics plus the minimum eigenvalue) are two different entry points. Both exist; the
report's item refers to the second.

THE BROADER POINT, recorded as a lesson about reporting rather than about code: after verifying
about 24 of the report's 108 numbered items I described the remainder as "all blocked on external
information". That was extrapolating a sample to a population, and four counterexamples turned up
immediately on checking.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"
GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def munnell():
    try:
        frame = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    frame = frame.copy()
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    frame["year"] = frame["year"].astype(int)
    return frame


@pytest.fixture(scope="module")
def grilic():
    try:
        return pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "rns", "mrt"]
        )
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")


def test_c12_02_fe_robust_standard_errors_are_exact(munnell) -> None:
    """Scored 0.0 in the report; identical to `xtreg ..., fe vce(robust)` to ten decimals."""

    fitted = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", munnell, entity="sid", time="year"
    ).fit(cov_type="robust")

    expected = {
        "lnk1": 0.0538467869,
        "lnk2": 0.0506720321,
        "lnlabor": 0.0607879585,
        "Intercept": 0.2507401122,
    }
    for name, value in expected.items():
        assert float(fitted.bse[name]) == pytest.approx(value, abs=1e-9), name


def test_c12_03_theta_is_exact(munnell) -> None:
    fitted = ws.RandomEffects.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", munnell, entity="sid", time="year"
    ).fit()
    assert float(fitted.e["theta"]) == pytest.approx(0.8857362899, abs=1e-9)


def test_c10_02_estat_firststage_matches_stata(grilic) -> None:
    """The summary table, not the coefficient table -- the report's item refers to this one."""

    fitted = ws.IV2SLS.from_formula("lnw ~ expr + [s ~ rns + mrt]", grilic).fit()
    rendered = str(fitted.estat_firststage())

    assert "0.1049" in rendered  # R-squared
    assert "0.1013" in rendered  # Adjusted R-squared
    assert "0.0493" in rendered  # Partial R-squared
    assert "19.5592" in rendered  # F(2,754)


def test_c10_01_the_coefficient_table_is_a_separate_entry_point(grilic) -> None:
    """`first_stage()` and `estat_firststage()` are both implemented and are different tables."""

    fitted = ws.IV2SLS.from_formula("lnw ~ expr + [s ~ rns + mrt]", grilic).fit()

    coefficients = str(fitted.first_stage())
    summary = str(fitted.estat_firststage())

    assert "Coefficient" in coefficients
    assert "Partial" in summary
    assert coefficients != summary
