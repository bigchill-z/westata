"""The four P0 items from the report's §7: two verified fixed, and a trap worth recording.

P0-1  xtreg, fe robust / cluster SE
    VERIFIED WORKING. Against a live `xtreg ..., fe` on lin_1992.dta:

        coefficient  .6399658    westata .6399658
        default SE   .0644132    westata .06441319
        robust SE    .1544611    westata .15446111
        cluster SE   .1544611    westata .15446111

    Stata's robust and clustered SEs are IDENTICAL here because every pid is its own cluster, so
    the clustered sandwich collapses to the heteroskedasticity-robust one.

    THE TRAP: twice this round I read `bse` by POSITION and got .72596946, concluded the robust SE
    was 4.7x wrong, and started diagnosing a non-existent defect. .72596946 is the CONSTANT's
    standard error. Selecting by label -- `bse["ltlan"]` -- gives the right number immediately.
    A guard below asserts the two are different so the mistake is at least visible.

P0-2  estat ovtest, rhs
    VERIFIED FIXED.

        default   Stata F(3, 751) = 1.33, p 0.2620    westata 1.3344 / 3 / 751 / 0.2620
        rhs       Stata F(9, 745) = 1.70, p 0.0842    westata 1.7042 / 9 / 745 / 0.0842
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

LIN = r"D:\myCodeProject\test_westata\data\lin_1992.dta"
GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"
FORMULA = "ltvfo ~ ltlan + ltwlab + ltpow + ltfer + hrs + mci + ngca"

STATA_FE = {"coef": 0.6399658, "default_se": 0.0644132, "robust_se": 0.1544611}


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


def test_fe_coefficient_and_default_se(panel) -> None:
    fit = ws.PanelOLS(FORMULA, panel, entity="pid", time="tid").fit()
    assert float(fit.params["ltlan"]) == pytest.approx(STATA_FE["coef"], abs=5e-7)
    assert float(fit.bse["ltlan"]) == pytest.approx(STATA_FE["default_se"], abs=5e-7)


def test_fe_robust_se(panel) -> None:
    fit = ws.PanelOLS(FORMULA, panel, entity="pid", time="tid").fit(cov_type="robust")
    assert float(fit.bse["ltlan"]) == pytest.approx(STATA_FE["robust_se"], abs=5e-7)


def test_fe_clustered_se_equals_robust_here(panel) -> None:
    """Every pid is its own cluster on this data, so the two coincide -- as in Stata."""

    fit = ws.PanelOLS(FORMULA, panel, entity="pid", time="tid").fit(
        cov_type="clustered", cov_kwds={"groups": panel["pid"]}
    )
    assert float(fit.bse["ltlan"]) == pytest.approx(STATA_FE["robust_se"], abs=5e-7)


def test_the_constant_se_is_not_the_ltlan_se(panel) -> None:
    """The trap: .72596946 is the constant's SE, not ltlan's."""

    fit = ws.PanelOLS(FORMULA, panel, entity="pid", time="tid").fit(cov_type="robust")
    by_position = float(np.asarray(fit.bse).ravel()[0])
    by_label = float(fit.bse["ltlan"])
    assert by_position == pytest.approx(0.72596946, abs=5e-6)
    assert by_label == pytest.approx(0.1544611, abs=5e-7)
    assert abs(by_position - by_label) > 0.5


@pytest.fixture(scope="module")
def ols_fit():
    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


def test_ovtest_default_matches_stata(ols_fit) -> None:
    values = ols_fit.estat_ovtest().to_dict()
    assert values["F"] == pytest.approx(1.3343880671, rel=1e-6)
    assert values["df"] == 3
    assert values["df_r"] == 751
    assert values["p"] == pytest.approx(0.2620, abs=5e-5)


def test_ovtest_rhs_matches_stata(ols_fit) -> None:
    """Stata: F(9, 745) = 1.70, p 0.0842. The report called this P0-broken."""

    values = ols_fit.estat_ovtest(rhs=True).to_dict()
    assert values["F"] == pytest.approx(1.7042, abs=5e-4)
    assert values["df"] == 9
    assert values["df_r"] == 745
    assert values["p"] == pytest.approx(0.0842, abs=5e-5)
