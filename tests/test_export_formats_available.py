"""C20-02 (Markdown/LaTeX export, scored 0.00) is implemented and works; the score was stale.

Auditing this row found all three exporters present and correct:

    to_markdown(table)      -> Markdown pipe table
    to_latex(table)         -> booktabs-style LaTeX float content
    render_esttab(results)  -> Stata's esttab layout

ONE THING THAT LOOKED LIKE A BUG AND WAS NOT. Calling

    render_esttab([r1, r2], model_names=["(1)", "(2)"])

printed the model row twice:

                          (1)             (2)
                          (1)             (2)

That is correct behaviour misused: esttab's FIRST row is always the column number, and the SECOND
is the `mtitle` row. Passing "(1)"/"(2)" as the titles duplicates the numbers. With real titles --
or with the m1/m2 default -- the two rows differ as they should. I nearly filed this as a rendering
defect; reading the two-row construction in `esttab.py` first is what caught it, which is the lesson
round 209 taught the hard way.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def models():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return (
        ws.OLS.from_formula("lnw ~ s", frame).fit(),
        ws.OLS.from_formula("lnw ~ s + expr", frame).fit(),
    )


def test_markdown_export_produces_a_pipe_table(models) -> None:
    table = ws.etable(list(models), model_names=["M1", "M2"])
    rendered = ws.to_markdown(table, caption="Regression results")
    assert "|" in rendered
    assert "---" in rendered
    assert "Regression results" in rendered
    assert "0.097" in rendered          # a coefficient from the first model


def test_latex_export_produces_float_content(models) -> None:
    table = ws.etable(list(models), model_names=["M1", "M2"])
    rendered = ws.to_latex(table, caption="Test", label="tab:t1")
    assert "hline" in rendered
    assert "&" in rendered
    assert "0.097" in rendered


def test_esttab_prints_the_column_numbers_then_the_titles(models) -> None:
    """Row 1 is always "(1) (2)"; row 2 is the mtitle row, which is why passing "(1)" duplicates."""

    rendered = ws.render_esttab(list(models), model_names=["Model A", "Model B"])
    lines = rendered.splitlines()
    assert "(1)" in lines[0] and "(2)" in lines[0]
    assert "Model A" in lines[1] and "Model B" in lines[1]
    assert lines[0] != lines[1]


def test_the_default_titles_are_m1_and_m2(models) -> None:
    rendered = ws.render_esttab(list(models), model_names=None)
    lines = rendered.splitlines()
    assert "(1)" in lines[0]
    assert "m1" in lines[1] and "m2" in lines[1]
