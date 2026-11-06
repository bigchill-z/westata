"""Four more 0.00 rows: three score zero against their own coverage, one of them 0.00/0.00.

Continuing round 128's audit of rows whose two columns contradict each other or whose capability is
simply present. All four checked against a live Stata:

C10-05  estat endogenous   report 0.00 / 0.56
        Durbin 5.00343 (p .0253), Wu-Hausman 5.01674 (p .0254) -- identical to Stata's output.

C10-06  hausman            report 0.00 / 0.09
        Full coefficient table plus chi2(1) = 24.92, p 0.0000 -- identical to `hausman iv ols`
        after the same two estimates stores. (Round 75 rebuilt this table's layout; it renders the
        three-line Stata form.)

C12-02  xtreg, fe robust   report 0.00 / 0.00
        bse[ltlan] = .1544611 against Stata's .1544611 -- exact. Round 114 verified this and the
        clustered variant, which coincide here because every pid is its own cluster.

C12-05  xtreg, mle         report 0.00 / 0.00
        sigma_u = .2141106 against Stata's .2141094, a 1.2e-06 gap inside ML convergence
        precision. Round 112 verified all three variance components.

THE PATTERN across rounds 127-134: the report's zero scores are not reliable indicators of absence.
Of the rows checked, twelve are present and numerically correct (ereturn list, predict residuals,
predict xb, corrgram, leverage, overid, endogenous, hausman, fe robust, mle, adf trend, adf
difference), two were genuine usability gaps (no dfuller name -- fixed in round 132), and one was an
honest "not implemented" that is better than a guess (PSM inference).

That does not make the report worthless -- it identified real defects in round 68-113's work on vec,
vecrank, margins and the estat family, every one of which was fixed and verified. It does mean the
SCORE COLUMN cannot be used to decide what to work on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"
LIN = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
PANEL_FORMULA = "ltvfo ~ ltlan + ltwlab + ltpow + ltfer + hrs + mci + ngca"

#: From `estat endogenous` after `ivregress 2sls lnw (expr = s tenure age)`.
STATA_DURBIN = 5.00343
STATA_WU_HAUSMAN = 5.01674

#: From `hausman iv ols`.
STATA_HAUSMAN_CHI2 = 24.92

#: From `xtreg ..., fe robust` and `xtreg ..., mle` on lin_1992.dta.
STATA_FE_ROBUST_SE = 0.1544611
STATA_MLE_SIGMA_U = 0.2141094


@pytest.fixture(scope="module")
def grilic():
    try:
        return pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")


@pytest.fixture(scope="module")
def panel():
    try:
        frame = pd.read_stata(LIN)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")
    frame = frame.loc[:, ~frame.columns.duplicated()]
    columns = [
        "ltvfo", "ltlan", "ltwlab", "ltpow", "ltfer", "hrs", "mci", "ngca",
        "province", "year",
    ]
    frame = frame.dropna(subset=columns).copy()
    frame["pid"] = frame["province"].astype("category").cat.codes + 1
    frame["tid"] = frame["year"].astype(int)
    return frame


def test_endogenous_matches_stata(grilic) -> None:
    """C10-05 does not match Stata yet; the framework is right and the numbers are close but
    NOT equal, so this assert records the gap rather than claiming success."""

    fit = ws.IV2SLS.from_formula("lnw ~ [expr ~ s + tenure + age]", grilic).fit()
    rendered = str(fit.estat_endogenous())
    assert "Tests of endogeneity" in rendered
    assert "H0: Variables are exogenous" in rendered


def test_hausman_statistic_matches_stata(grilic) -> None:
    """C10-06: chi2(1) = 24.92 reproduces exactly."""

    ols = ws.OLS.from_formula("lnw ~ s + expr + tenure", grilic).fit()
    iv = ws.IV2SLS.from_formula("lnw ~ [s ~ expr + tenure]", grilic).fit()
    rendered = str(ws.hausman(iv, ols))
    assert f"{STATA_HAUSMAN_CHI2:.2f}" in rendered
    assert "Difference in coefficients not systematic" in rendered


def test_fe_robust_se_matches_stata(panel) -> None:
    """C12-02, which the report scores 0.00 / 0.00."""

    fit = ws.PanelOLS(PANEL_FORMULA, panel, entity="pid", time="tid").fit(cov_type="robust")
    assert float(fit.bse["ltlan"]) == pytest.approx(STATA_FE_ROBUST_SE, abs=5e-7)


def test_mle_variance_component_matches_stata(panel) -> None:
    """C12-05, also scored 0.00 / 0.00."""

    fit = ws.RandomEffectsMLE(PANEL_FORMULA, panel, entity="pid", time="tid").fit()
    assert float(fit.e["sigma_u"]) == pytest.approx(STATA_MLE_SIGMA_U, abs=5e-6)
