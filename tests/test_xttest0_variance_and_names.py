"""`xttest0` after `xtreg ..., re`: four measured defects, all fixed this round.

The user's script was

    RE = ws.RandomEffects.from_formula(formula, d, entity="province", time="year").fit(cov_type="robust")
    ws.xttest0(RE)

against Stata's `xtreg ..., re` then `xttest0`. Verified against a live session on munnell.dta
(48 entities x 17 years); with the panel variable named the same way on both sides the two outputs
are byte-identical, 15 lines, zero differences:

    Breusch and Pagan Lagrangian multiplier test for random effects

            lny[sid,t] = Xb + u[sid] + e[sid,t]

            Estimated results:
                             |       Var     SD = sqrt(Var)
                    ---------+-----------------------------
                         lny |    1.04271       1.021132
                           e |   .0037374       .0611346
                           u |    .022464         .14988

            Test: Var(u) = 0
                                 chibar2(01) =  4278.48
                              Prob > chibar2 =   0.0000

THE FOUR DEFECTS:

1. THE `e` ROW WAS 3.0% HIGH. It came from within-residuals/(N-n) = .003851073; Stata's value is
   e(sigma_e)^2 = .0037374, which the fit already stores. A code comment had claimed
   e(sigma_e)^2 was "2.5% too high" -- that comment was simply wrong.
2. THE VARIANCE CELLS WERE ONE DIGIT TOO LONG. The block is right-aligned with fixed column ends,
   so Stata picks the most digits that fit 8 characters; a shared significant-figure count gave
   '.003737434' where Stata prints '.0037374'.
3. THE EQUATION LINE PRINTED PLACEHOLDERS (`y[id,t]`). `StataResults` has no entity_name attribute;
   the names live in e(ivar)/e(tvar), which `xtreg` did not record at all on this path.
4. `xttest0(fit)` REFUSED WITHOUT entity=. Label discovery tried a hard-coded name list
   ("id"/"entity"/"group"/"county"/"firm"/"prov"), so munnell's `st_abb` -- or the user's
   `province` -- was never found even though the fit knew the name.
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")
import numpy as np
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


@pytest.fixture(scope="module")
def fitted(panel):
    return ws.RandomEffects.from_formula(
        "lny ~ lnk1 + lnk2", panel, entity="st_abb", time="year"
    ).fit(cov_type="robust")


# ---------------------------------------------------------------- 1. the e row
def test_the_e_row_is_the_fits_own_sigma_e_squared(fitted) -> None:
    """Stata's `e` row is e(sigma_e)^2 = .0037374, not within-residuals/(N-n) = .003851."""

    expected = float((fitted.e or {})["sigma_e"]) ** 2
    result = ws.xttest0(fitted)
    assert abs(result.var_e - expected) < 1e-12, result.var_e
    assert "   .0037374 " in str(result), str(result)


def test_the_u_row_is_sigma_u_squared(fitted) -> None:
    expected = float((fitted.e or {})["sigma_u"]) ** 2
    assert abs(ws.xttest0(fitted).var_u - expected) < 1e-12


def test_the_lm_statistic_is_unchanged(fitted) -> None:
    """The statistic never depended on the variance block: Stata reports 4278.48."""

    result = ws.xttest0(fitted)
    assert round(result.chibar2, 2) == 4278.48
    assert result.pvalue < 1e-4


# ---------------------------------------------------------------- 2. the cell precision
def test_variance_cells_fit_eight_characters(fitted) -> None:
    rendered = str(ws.xttest0(fitted))
    assert ".0037374 " in rendered, rendered
    assert ".003737434" not in rendered, rendered
    # The SD column keeps its own precision.
    assert ".0611346" in rendered, rendered


# ---------------------------------------------------------------- 3. the equation line
def test_the_fit_records_the_panel_names(fitted) -> None:
    assert (fitted.e or {}).get("ivar") == "st_abb"
    assert (fitted.e or {}).get("tvar") == "year"


def test_the_equation_line_names_the_panel_variable(fitted) -> None:
    rendered = str(ws.xttest0(fitted))
    assert "lny[st_abb,t] = Xb + u[st_abb] + e[st_abb,t]" in rendered, rendered
    assert "[id,t]" not in rendered, rendered


def test_an_arbitrary_entity_name_also_works(panel) -> None:
    """The user's own figure used `province`; any name must reach the equation line."""

    frame = panel.rename(columns={"st_abb": "province"})
    model = ws.RandomEffects.from_formula(
        "lny ~ lnk1 + lnk2", frame, entity="province", time="year"
    ).fit(cov_type="robust")
    rendered = str(ws.xttest0(model))
    assert "lny[province,t]" in rendered, rendered


# ---------------------------------------------------------------- 4. no entity= needed
def test_no_entity_argument_is_required(fitted) -> None:
    result = ws.xttest0(fitted)
    assert result.nobs == 816
    assert result.entities == 48
    assert result.periods == 17


def test_the_leading_blank_line_is_present(fitted) -> None:
    assert str(ws.xttest0(fitted)).splitlines()[0] == ""
