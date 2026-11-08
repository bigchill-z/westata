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
    # A live `esttab` starts with a blank line; the numbers follow it.
    lines = rendered.splitlines()
    assert lines[0] == ""
    assert "(1)" in lines[2] and "(2)" in lines[2]
    assert "Model A" in lines[3] and "Model B" in lines[3]
    assert lines[0] != lines[1]


def test_the_default_output_has_no_model_title_row(models) -> None:
    """Stata prints the model-name row only with `mtitle`, so untouched output has ONE title row.

    A live `esttab m1 m2, se` -- note the stored names, still without `mtitle` -- prints:

        ''
        '--------------------------------------------'
        '                      (1)             (2)   '
        '                        y               y   '
        '--------------------------------------------'

    This test previously asserted that m1/m2 were always present, which is what kept the default
    table one line longer than Stata's.
    """

    rendered = ws.render_esttab(list(models), model_names=None)
    lines = rendered.splitlines()
    assert lines[0] == ""
    assert "(1)" in lines[2]
    # The dependent variable, not a model name. This fixture's response is lnw, so the row
    # must be checked against the models rather than against a hard-coded name.
    assert "lnw" in lines[3]
    assert "m1" not in lines[3] and "m2" not in lines[3]

    # And naming them brings the row back.
    named = ws.render_esttab(list(models), model_names=["m1", "m2"]).splitlines()
    assert "m1" in named[3] and "m2" in named[3]
