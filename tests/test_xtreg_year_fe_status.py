"""C12-09 `xtreg ..., fe i.year`: coefficients are exact; the year BLOCK is not grouped.

Round 215 audited the last item the progress report listed as "doable". On munnell (48 states x
17 years), `xtreg y k1 i.year, fe`:

    WHAT MATCHES
        the whole two-column header, after round 215 added its leading blank line
        R-squared within/between/overall (0.4827 / 0.9586 / 0.9426)
        every year coefficient to the printed digit: 1971 -713.0255, 1972 310.3079,
        1984 7834.845, 1986 12005.98, ...
        sigma_u, sigma_e, rho, and the F test of the u_i

    WHAT DOES NOT
        Stata prints a `year` group header followed by a blank pipe row:

                        |
                   year |
                  1971  |  -713.0255   1957.38 ...
                  1972  |   310.3079  1968.997 ...

        This package lists the same rows as ordinary coefficients (`yr1971`, `yr1972`, ...) in line
        with k1. Same numbers, no grouping.

So C12-09's 0.01 numeric score was understating it -- the estimates are right -- while its
structural gap is real. Closing it needs the factor metadata to survive into the renderer so a
`C(year)`/`i.year` block can be recognised as a group, which is the same machinery round 183 built
for `C(var)[T.level]` in ordinary models; the panel path does not route through it yet.

FIXED THIS ROUND: the panel header's missing leading blank line. Stata's capture starts with
chr(10) before 'Fixed-effects (within) regression', and without it every row of the header compared
one line off against Stata's -- the same defect the 189-196 prefix survey found ten times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\munnell.dta"


@pytest.fixture(scope="module")
def panel():
    try:
        frame = pd.read_stata(DATA).reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("munnell.dta is not available")
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    years = sorted(frame["year"].unique())[1:]
    for year in years:
        frame[f"yr{year}"] = (frame["year"] == year).astype(float)
    formula = "y ~ k1 + " + " + ".join(f"yr{year}" for year in years)
    fitted = ws.PanelOLS.from_formula(formula, frame, entity="sid", time="year").fit()
    return frame, fitted


def test_the_panel_header_opens_with_a_blank_line(panel) -> None:
    _, fitted = panel
    rendered = str(fitted.summary(style="stata"))
    assert rendered.startswith("\n")
    assert "Fixed-effects (within) regression" in rendered.splitlines()[1]


def test_the_r_squared_decomposition_matches_stata(panel) -> None:
    _, fitted = panel
    rendered = str(fitted.summary(style="stata"))
    assert "0.4827" in rendered      # within
    assert "0.9586" in rendered      # between
    assert "0.9426" in rendered      # overall


def test_the_year_coefficients_match_stata(panel) -> None:
    _, fitted = panel
    params = fitted.params
    # A few from the middle and both ends of the year range.
    assert float(params["yr1971"]) == pytest.approx(-713.0255, abs=5e-4)
    assert float(params["yr1972"]) == pytest.approx(310.3079, abs=5e-4)
    assert float(params["yr1986"]) == pytest.approx(12005.98, abs=5e-3)


def test_the_year_block_is_not_grouped_yet(panel) -> None:
    """Recorded as the OPEN half of C12-09.

    Stata prints a `year` header row and a blank pipe row before the levels. Replace this with a
    structural assertion when the factor metadata reaches the panel renderer.
    """

    _, fitted = panel
    lines = str(fitted.summary(style="stata")).splitlines()
    assert any("yr1971" in line for line in lines), "the rows exist"
    assert not any(line.strip() == "year |" for line in lines), "Stata's group header is absent"
