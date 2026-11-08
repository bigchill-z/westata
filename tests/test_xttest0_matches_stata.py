"""The Breusch-Pagan LM test for random effects matches Stata's chibar2.

C12-04 scores `xttest0` at 0.7273. Measured on munnell.dta:

    quietly xtreg lny lnk1 lnk2 lnlabor, re
    xttest0
        chibar2(01) =  3921.14
        Prob > chibar2 =   0.0000

The package reports 3921.14.

TWO API POINTS the guards pin, both of which are deliberate rather than incidental:

  * `xttest0` needs the entity LABELS passed explicitly. A random-effects result does not carry the
    grouping vector, and the test is a function of it -- so the package asks rather than silently
    reconstructing it from row order;
  * the statistic is a MIXED chi-bar-squared, not a plain chi2. Stata labels it `chibar2(01)` because
    the null sits on the boundary of the parameter space, which is why `r(chi2)` is empty after
    `xttest0` and a caller who reads that scalar gets nothing.
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
def random_effects(panel):
    return ws.RandomEffects.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", panel, entity="sid", time="year"
    ).fit()


def test_the_lm_statistic_matches_stata(panel, random_effects) -> None:
    result = ws.xttest0(random_effects, entity=panel["sid"].to_numpy())
    rendered = str(result)
    assert "3921.14" in rendered


def test_the_test_reports_a_chi_bar_squared_not_a_chi2(panel, random_effects) -> None:
    """Stata labels it `chibar2(01)`; the boundary null is why the label differs."""

    rendered = str(ws.xttest0(random_effects, entity=panel["sid"].to_numpy()))
    assert "chibar2" in rendered


def test_it_refuses_without_the_entity_labels(random_effects) -> None:
    """The grouping vector is not on the result, so guessing it would be a silent error."""

    with pytest.raises(Exception):
        ws.xttest0(random_effects)
