"""estat firststage: the relative-bias row was hard-coded away, and the name column was 1 left.

C10-02 scored 0.28 layout / 0.86 numeric. Round 188 found two real defects in the weak-instrument
block, both of which had been hiding correct data:

1. THE RELATIVE-BIAS ROW said "(not available)" unconditionally. The Stock-Yogo critical values
   were already computed and stored in ``matrix[0]`` -- grilic.dta's own ``estat firststage``
   prints 13.91 9.08 6.46 5.39 there -- but the renderer never read them. The phrase IS right for
   a fit whose row is entirely ``nan`` (the synthetic panel in test_firststage_layout has no
   relative-bias cell), so the row must branch on the data rather than pick one spelling.

2. THE NAME COLUMN was a fixed ten-space prefix, which put the pipe at 12 for a one-character
   endogenous name and at 15 for a longer one. Stata puts it at 15 in BOTH cases.

3. THE EIGENVALUE LINE'S trailing padding is a fixed TOTAL width (45), not a fixed number of
   spaces: '190.149' needs five and '44.963' needs six. Hard-coding either broke the other fit,
   which is how the two reference fits in the suite disagreed by exactly one space.

Both reference fits now match Stata 18 of 18 lines.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def grilic_fit():
    try:
        frame = pd.read_stata(DATA).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.IV2SLS.from_formula("lnw ~ [s ~ expr + tenure + age]", frame).fit()


def test_the_relative_bias_row_reports_the_values(grilic_fit) -> None:
    """It said '(not available)' while Stata prints four numbers."""

    rendered = str(ws.estat_firststage(grilic_fit))
    assert "(not available)" not in rendered
    assert "13.91" in rendered and "5.39" in rendered


def test_the_relative_bias_row_still_degrades_honestly() -> None:
    """The synthetic fit in test_firststage_layout has no relative-bias cells at all."""

    import sys

    sys.path.insert(0, "tests")
    from test_firststage_layout import _frame

    fit = ws.IV2SLS.from_formula("y ~ x + [endog ~ z1 + z2]", _frame()).fit()
    rendered = str(ws.estat_firststage(fit))
    assert "(not available)" in rendered


def test_the_pipe_sits_at_fifteen_for_any_name_length(grilic_fit) -> None:
    """A one-character name used to push it to 12, a five-character one to 15."""

    rendered = str(ws.estat_firststage(grilic_fit))
    rows = [
        line
        for line in rendered.splitlines()
        if "|" in line and "R-sq" not in line and line.strip().startswith(("s", "endog"))
    ]
    assert rows
    for line in rows:
        assert line.index("|") == 15, repr(line)


def test_the_eigenvalue_line_has_a_fixed_total_width(grilic_fit) -> None:
    """45 characters, whatever the statistic's digit count."""

    rendered = str(ws.estat_firststage(grilic_fit))
    eigenvalue = next(line for line in rendered.splitlines() if "Minimum eigenvalue" in line)
    assert len(eigenvalue) == 45
