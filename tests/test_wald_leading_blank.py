"""`test s` opens with a blank line -- the sibling of lincom, fixed one round later.

Round 195 fixed `lincom`; round 196 walked the remaining block renderers and found that `test`
carried the identical omission. The raw capture of `test s` after `regress lnw s expr tenure` is:

    (blank)
     ( 1)  s = 0
    (blank)
           F(  1,   754) =  320.66
                Prob > F =    0.0000

while the package printed four lines starting at the constraint. Every number and every column
already matched.

THE FULL SURVEY RESULT: across rounds 189-196 the "walk every renderer's raw prefix" check found
TEN blocks with the same class of defect -- estat endogenous, estat ic, estat bgodfrey, estat
dwatson, estat vif, estat imtest, margins, lincom, contrast and now test. All ten are fixed and
guarded. The check itself is one line (`str(out)[0]`), which is why it was worth repeating.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


@pytest.mark.parametrize("spec", ["s", "s expr", "s = expr"])
def test_the_block_opens_with_a_blank_line(fitted, spec: str) -> None:
    rendered = str(fitted.test(spec))
    assert rendered.startswith("\n")
    assert rendered.splitlines()[0] == ""


def test_the_single_hypothesis_has_five_lines(fitted) -> None:
    lines = str(fitted.test("s")).splitlines()
    assert len(lines) == 5
    assert lines[1] == " ( 1)  s = 0"
    assert lines[2] == ""
    assert lines[3].startswith("       F(  1,   754) =  320.66")
    assert lines[4].startswith("            Prob > F =    0.0000")


def test_the_joint_hypothesis_still_lists_both(fitted) -> None:
    lines = str(fitted.test("s expr")).splitlines()
    assert lines[1] == " ( 1)  s = 0"
    assert lines[2] == " ( 2)  expr = 0"


def test_every_block_renderer_starts_with_a_newline(grilic_module_fit=None) -> None:
    """The survey as a standing guard: the postestimation blocks all open with a blank line."""

    try:
        frame = pd.read_stata(GRILIC).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    ols = ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()
    blocks = {
        "estat_ovtest": ws.estat_ovtest(ols),
        "estat_hettest": ws.estat_hettest(ols, method="iid"),
        "estat_ic": ws.estat_ic(ols),
        "estat_vce": ws.estat_vce(ols),
        "estat_summarize": ws.estat_summarize(ols),
        "lincom": ols.lincom("s"),
        "test": ols.test("s"),
    }
    for name, block in blocks.items():
        assert str(block).startswith("\n"), name
