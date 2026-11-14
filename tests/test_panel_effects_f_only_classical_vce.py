"""The effects F test is printed only under a classical VCE.

The user noticed that `PanelOLS` output always ended with

    F test that all u_i=0: F(47, 766) = 104.52                   Prob > F = 0.0000

while the corresponding Stata run sometimes has no such line. Measured on munnell.dta:

    xtreg lny lnk1 lnk2, fe                     -> has the row
    xtreg lny lnk1 lnk2, fe r                   -> NO row
    xtreg lny lnk1 lnk2, fe vce(cluster sid)    -> NO row
    xtreg lny lnk1 lnk2, re                     -> NO row (never has one)

The row is the CLASSICAL F test of the individual effects, which assumes the homoskedastic model.
Under a robust or clustered VCE Stata drops it. The fit still computes `e(F_f)` -- 110.94 under the
robust VCE -- so testing `F_f` for finiteness was not enough, and the extra row appeared on every
robust panel fit.

Verified: with the VCE spelled the same way on both sides the outputs match byte for byte,
24 lines for `fe` and 25 for `fe r` (the extra line being the cluster note).
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"

pytestmark = pytest.mark.skipif(
    not pathlib.Path(MUNNELL).exists(), reason="munnell.dta is not available"
)


@pytest.fixture(scope="module")
def panel():
    frame = pd.read_stata(MUNNELL).copy()
    frame["year"] = frame["year"].astype(int)
    return frame


def _lines(model):
    return [line for line in str(model.summary()).splitlines() if line.strip()]


def _has_effects_f(model):
    return any("F test that all u_i=0" in line for line in _lines(model))


# ---------------------------------------------------------------- fixed effects
def test_fixed_effects_prints_the_row(panel) -> None:
    model = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2", panel, entity="st_abb", time="year"
    ).fit()
    assert _has_effects_f(model)
    assert "F test that all u_i=0: F(47, 766) = 104.52" in str(model.summary())


def test_robust_fixed_effects_omits_the_row(panel) -> None:
    """Stata's `xtreg ..., fe r` has no effects F test."""

    model = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2", panel, entity="st_abb", time="year"
    ).fit(cov_type="robust")
    assert not _has_effects_f(model)


def test_clustered_fixed_effects_omits_the_row(panel) -> None:
    """Stata's `xtreg ..., fe vce(cluster sid)` has no effects F test either."""

    model = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2", panel, entity="st_abb", time="year"
    ).fit(cov_type="cluster", cov_kwds={"groups": panel["st_abb"]})
    assert not _has_effects_f(model)


def test_the_fit_still_computes_the_statistic(panel) -> None:
    """The e() scalar is present even when the row is suppressed -- that is why the row leaked."""

    model = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2", panel, entity="st_abb", time="year"
    ).fit(cov_type="robust")
    stored = model.e or {}
    assert stored.get("F_f") is not None
    assert float(stored["F_f"]) > 0


# ---------------------------------------------------------------- other estimators
def test_random_effects_has_no_row(panel) -> None:
    model = ws.RandomEffects.from_formula(
        "lny ~ lnk1 + lnk2", panel, entity="st_abb", time="year"
    ).fit()
    assert not _has_effects_f(model)


def test_between_effects_has_no_row(panel) -> None:
    model = ws.BetweenOLS.from_formula(
        "lny ~ lnk1 + lnk2", panel, entity="st_abb", time="year"
    ).fit()
    assert not _has_effects_f(model)


# ---------------------------------------------------------------- the closing rule
def test_the_classical_form_ends_with_the_f_row(panel) -> None:
    """xtreg, fe ends with the F row; the rule sits ABOVE it."""

    model = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2", panel, entity="st_abb", time="year"
    ).fit()
    lines = _lines(model)
    assert "F test that all u_i=0" in lines[-1], lines[-2:]
    assert set(lines[-2].strip()) == {"-"}, lines[-2]


def test_the_robust_form_ends_with_the_rule(panel) -> None:
    """xtreg, fe r drops the F row, so the rule becomes the last line."""

    model = ws.PanelOLS.from_formula(
        "lny ~ lnk1 + lnk2", panel, entity="st_abb", time="year"
    ).fit(cov_type="robust")
    lines = _lines(model)
    assert set(lines[-1].strip()) == {"-"}, lines[-1]
    assert len(lines[-1]) == 78
