"""C12-09 CLOSED: `xtreg y x i.year, fe` now matches Stata byte for byte, 42 lines to 42.

Rounds 215-220 closed this in six steps, each of which was a separate defect:

  215  the panel header had no leading blank line -- every header row compared one line off
  216  C(year)/i.year could not even be written: _panel_frame hid the time column in the index
  217  factor levels were right-aligned; the indent is name_width - 1 - len(text)
  218  patsy put the expanded year columns BEFORE k1; Stata keeps the formula order,
       and the factor block needed a blank pipe line before _cons
  219  the corr(u_i, Xb) row is 78 wide, not 77
  220  the coefficient table is NOT closed by a full border before sigma_u -- only a split rule,
       with the full border after rho

The final comparison against a live run on munnell (48 states x 17 years):

    xtreg y k1 i.year, fe       42 lines
    PanelOLS.from_formula(...)  42 lines
    differing                   0

The estimates were right from the beginning -- round 215 measured k1 = 2.230512 and every year
coefficient to the printed digit before any of this work -- which is why the report's 0.01 numeric
score for C12-09 was misleading rather than merely stale.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\munnell.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("munnell.dta is not available")
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    return ws.PanelOLS.from_formula("y ~ k1 + C(year)", frame, entity="sid", time="year").fit()


def test_the_block_is_forty_two_lines(fitted) -> None:
    rendered = str(fitted.summary(style="stata")).splitlines()
    assert len(rendered) == 42


def test_it_opens_with_a_blank_line_then_the_banner(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    assert lines[0] == ""
    assert lines[1].startswith("Fixed-effects (within) regression")


def test_k1_comes_before_the_year_block(fitted) -> None:
    """The formula order, not patsy's column order."""

    lines = str(fitted.summary(style="stata")).splitlines()
    k1_index = next(i for i, line in enumerate(lines) if line.strip().startswith("k1 |"))
    year_index = next(i for i, line in enumerate(lines) if line.strip() == "year |")
    assert k1_index < year_index


def test_the_year_levels_are_indented_and_the_pipe_aligns(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    level = next(line for line in lines if line.strip().startswith("1971"))
    assert level.startswith("       1971  |"), repr(level[:16])


def test_a_blank_pipe_line_closes_the_factor_block(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    cons_index = next(i for i, line in enumerate(lines) if line.strip().startswith("_cons"))
    assert lines[cons_index - 1].strip() == "|"


def test_a_split_rule_precedes_sigma_u_and_the_border_follows_rho(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    sigma_index = next(i for i, line in enumerate(lines) if line.strip().startswith("sigma_u"))
    rho_index = next(i for i, line in enumerate(lines) if line.strip().startswith("rho |"))
    assert lines[sigma_index - 1].startswith("-------------+")
    assert lines[rho_index + 1].startswith("---") and "+" not in lines[rho_index + 1]


def test_the_corr_row_is_seventy_eight_wide(fitted) -> None:
    lines = str(fitted.summary(style="stata")).splitlines()
    corr = next(line for line in lines if line.startswith("corr(u_i, Xb)"))
    assert len(corr) == 78
