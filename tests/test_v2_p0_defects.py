"""Guards for the three P0 defects the v2 gap report identified and this round fixed.

Each reference below came from a live StataNow run in the same session, not from the report.

    P0-1  xtreg ltvfo ..., fe vce(cluster province) on lin_1992
    P0-2  estat ovtest, rhs on grilic
    P0-3  margins, dydx(child female) atmeans after a fweight logit on titanic
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data"


def _dedupe(frame: pd.DataFrame) -> pd.DataFrame:
    seen: dict[str, int] = {}
    names = []
    for column in frame.columns:
        count = seen.get(column, 0)
        seen[column] = count + 1
        names.append(column if count == 0 else f"{column}_{count + 1}")
    frame = frame.copy()
    frame.columns = names
    return frame


@pytest.fixture(scope="module")
def lin():
    try:
        return _dedupe(pd.read_stata(f"{DATA}/lin_1992.dta").dropna())
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("lin_1992.dta is not available")


@pytest.fixture(scope="module")
def grilic():
    try:
        return pd.read_stata(f"{DATA}/grilic.dta", convert_categoricals=False).dropna()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("grilic.dta is not available")


@pytest.fixture(scope="module")
def titanic():
    try:
        return pd.read_stata(f"{DATA}/titanic.dta", convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("titanic.dta is not available")


# ------------------------------------------------------------------ P0-1 panel clustered SE
PANEL_FORMULA = "ltvfo ~ ltlan + ltwlab + ltpow + ltfer + hrs + mci + ngca"
STATA_CLUSTERED_SE = {
    "ltlan": 0.154461106203,
    "ltwlab": 0.060334025167,
    "ltpow": 0.058925780613,
}


@pytest.mark.parametrize(
    "as_frame", [False, True], ids=["1d-array", "2d-single-column"]
)
def test_panel_clustered_se_matches_stata(lin, as_frame: bool) -> None:
    """The (n,1) form is what linearmodels accepts; it used to be refused outright."""

    clusters = lin[["province"]].values if as_frame else lin["province"].values
    model = ws.PanelOLS.from_formula(PANEL_FORMULA, lin, entity="province", time="year")
    fitted = model.fit(
        entity_effects=True, cov_type="cluster", cov_kwds={"clusters": clusters}
    )
    names = [str(name) for name in fitted.params.index]
    for variable, expected in STATA_CLUSTERED_SE.items():
        got = float(fitted.bse[names.index(variable)])
        assert got == pytest.approx(expected, rel=1e-10), variable


def test_panel_robust_se_also_matches_stata(lin) -> None:
    """``cov_type="robust"`` was reported as broken too; it was not, and still works."""

    model = ws.PanelOLS.from_formula(PANEL_FORMULA, lin, entity="province", time="year")
    fitted = model.fit(entity_effects=True, cov_type="robust")
    names = [str(name) for name in fitted.params.index]
    got = float(fitted.bse[names.index("ltlan")])
    assert got == pytest.approx(STATA_CLUSTERED_SE["ltlan"], rel=1e-10)


# ------------------------------------------------------------------------ P0-2 ovtest, rhs
def test_ovtest_skips_indicators(grilic) -> None:
    """Each dummy used to count as two test columns; Stata leaves indicators out entirely."""

    with_dummies = ws.OLS.from_formula(
        "lnw ~ s + expr + tenure + rns + smsa", grilic
    ).fit()
    rendered = str(with_dummies.estat_ovtest(rhs=True))
    assert "F(9, 743) =   2.03" in rendered, rendered


def test_ovtest_all_continuous_is_unchanged(grilic) -> None:
    """The guard for the fix not over-reaching: continuous-only models must not move."""

    continuous = ws.OLS.from_formula("lnw ~ s + expr + tenure", grilic).fit()
    rendered = str(continuous.estat_ovtest(rhs=True))
    assert "F(9, 745) =   1.70" in rendered, rendered


# ------------------------------------------------------- P0-3 margins atmeans under fweight
STATA_AT_MEANS = {
    "child": ".0495229",
    "female": ".2135393",
    "class1": ".1476602",
    "class2": ".1294866",
    "class3": ".3207633",
}


def test_atmeans_reports_weighted_means(titanic) -> None:
    """The means were taken on the 24 collapsed rows, not the 2201 weighted ones."""

    fitted = ws.Logit.from_formula(
        "survive ~ child + female + class1 + class2 + class3",
        titanic,
        freq_weights="freq",
    ).fit()
    rendered = str(fitted.margins(dydx=["child", "female"], atmeans=True))
    for variable, expected in STATA_AT_MEANS.items():
        assert expected in rendered, f"{variable}: expected {expected} in At: block"
    # The unweighted collapsed means must NOT appear.
    assert ".3333333 (mean)" not in rendered
    assert ".5 (mean)" not in rendered


def test_atmeans_marginal_effect_carries_the_weights(titanic) -> None:
    """With the means wrong, the MEM was 18% off (.2628119 against .2223422)."""

    fitted = ws.Logit.from_formula(
        "survive ~ child + female + class1 + class2 + class3",
        titanic,
        freq_weights="freq",
    ).fit()
    rendered = str(fitted.margins(dydx=["child", "female"], atmeans=True))
    child_row = next(line for line in rendered.splitlines() if "child |" in line)
    assert ".2223422" in child_row
    assert ".0510772" in child_row
