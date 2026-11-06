"""C20-02 scored 0.00 in the report, but Markdown and LaTeX export both work.

The report's C20-02 row reads:

    | C20-02 | 报告 | Markdown/LaTeX 导出 | **B** | 0.00 | — |

Zero, with the numeric-coverage column left blank -- i.e. the item was treated as absent without
being checked. Both functions exist, are exported at top level, and produce complete output:

    ws.to_markdown(table, caption="Table 1")
    ws.to_latex(table, caption="Table 1", label="tab:1", notes="SE in parentheses")

The LaTeX path emits \\caption, \\label, \\begin{tabular}, the column spec and \\hline rules; the
Markdown path emits a pipe table with a right-aligned numeric column per model.

WHY THE SCORE IS DEFENSIBLE ANYWAY, and the reason this module does not simply say "the report is
wrong": the exports take a DataFrame from `etable`, whose precision is fixed DECIMALS
(`digits=`, default 3), whereas Stata's `esttab` uses significant figures. So a reader comparing a
LaTeX table against Stata's printed one sees "(0.0060)" where Stata prints "(0.00604)". That is a
real, if narrow, difference -- it is the same distinction `etable_as_stata` exists to bridge, and
`ws.etable_as_stata(...)` followed by `to_latex` of its own rendering is not a supported chain.

So: present and working, not Stata-formatted. Recorded as such rather than as a zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def table():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.etable(
        [
            ws.OLS.from_formula("lnw ~ s", frame).fit(),
            ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit(),
        ],
        model_names=["m1", "m2"],
    )


def test_both_exporters_are_exported() -> None:
    assert hasattr(ws, "to_markdown")
    assert hasattr(ws, "to_latex")


def test_the_markdown_export_produces_a_table(table) -> None:
    rendered = str(ws.to_markdown(table, caption="Table 1"))
    assert rendered.count("\n") > 5
    assert "|" in rendered
    assert "Table 1" in rendered


def test_the_latex_export_produces_a_complete_environment(table) -> None:
    rendered = str(ws.to_latex(table, caption="Table 1", label="tab:1", notes="SE in parentheses"))
    for piece in ("\\caption", "\\label", "\\begin{tabular}", "\\hline"):
        assert piece in rendered, piece


def test_the_exports_use_fixed_decimals_not_significant_figures(table) -> None:
    """The narrow, real difference that keeps C20-02 from being a full pass.

    etable renders with `digits` DECIMALS; esttab prints significant figures. Both are correct
    for their own purpose, but a LaTeX table copied straight across will show "(0.0060)" where
    Stata shows "(0.00604)".
    """

    rendered = str(ws.to_markdown(table))
    assert "0.006" in rendered  # three decimals from the default digits=3
    assert "0.00604" not in rendered
