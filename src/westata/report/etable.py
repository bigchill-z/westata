"""Regression tables in the style of Stata's ``esttab`` / ``estout``.

The goal is a table you can paste into a paper: coefficients with significance
stars, standard errors underneath, and a block of fit statistics.  Output is a
pandas DataFrame by default (easy to ``to_latex()`` / ``to_csv()``), with
Markdown and LaTeX renderers for convenience.

Design choices that differ from ``esttab`` on purpose:

* Column labels default to ``(1)``, ``(2)``, ... like Stata, but a
  ``model_names`` argument lets you name them, which is what you actually want
  in a paper.
* Significance stars follow Stata's convention: ``*`` p<0.05, ``**`` p<0.01,
  ``***`` p<0.001.
* Statistics whose ``e()`` value is missing render as an empty cell rather than
  a misleading number.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = ["etable", "to_latex", "to_markdown"]

_DEFAULT_STATS: tuple[tuple[str, str], ...] = (
    ("N", "Observations"),
    ("r2", "R-squared"),
)


def _stars(pvalue: float) -> str:
    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def _format(value: float, digits: int) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def etable(
    results: Sequence[Any],
    *,
    model_names: Sequence[str] | None = None,
    digits: int = 3,
    se: bool = True,
    stars: bool = True,
    stats: Sequence[tuple[str, str]] | None = None,
    stats_digits: int = 3,
    keep: Sequence[str] | None = None,
    drop: Sequence[str] | None = None,
    show_param_names: bool = True,
) -> pd.DataFrame:
    """Build a coefficient table from one or more fitted models.

    Parameters
    ----------
    results:
        Fitted westata results (any estimator).
    model_names:
        Column headers; defaults to ``(1)``, ``(2)``, ...
    se:
        Put standard errors in parentheses beneath each coefficient.
    stars:
        Append significance stars to coefficients.
    stats:
        ``(e_key, label)`` pairs for the bottom block.  Defaults to
        ``Observations`` and ``R-squared``.
    keep, drop:
        Restrict which coefficients appear, by name.
    """

    results = list(results)
    if not results:
        raise ValueError("etable requires at least one result")

    names = (
        list(model_names)
        if model_names is not None
        else [f"({i + 1})" for i in range(len(results))]
    )
    if len(names) != len(results):
        raise ValueError(
            f"model_names has {len(names)} entries but there are "
            f"{len(results)} results"
        )

    # Row order: first appearance across models (Stata keeps model order).
    row_order: list[str] = []
    for res in results:
        for name in res.param_names:
            if name not in row_order:
                row_order.append(name)

    if keep is not None:
        keep_set = set(keep)
        missing = keep_set - set(row_order)
        if missing:
            raise KeyError(f"keep names not present in any model: {sorted(missing)}")
        row_order = [name for name in row_order if name in keep_set]
    if drop is not None:
        drop_set = set(drop)
        row_order = [name for name in row_order if name not in drop_set]

    blocks: list[dict[str, str]] = []
    index: list[str] = []

    for name in row_order:
        label = name if show_param_names else ""
        coefficient_row: dict[str, str] = {}
        se_row: dict[str, str] = {}
        for column, res in zip(names, results):
            if name in res.param_names and name in res.params.index:
                coef = float(res.params[name])
                pvalue = float(res.pvalues[name])
                text = _format(coef, digits)
                if stars:
                    text += _stars(pvalue)
                coefficient_row[column] = text
                if se:
                    se_row[column] = f"({_format(float(res.bse[name]), digits)})"
                else:
                    se_row[column] = ""
            else:
                coefficient_row[column] = ""
                se_row[column] = ""
        blocks.append(coefficient_row)
        index.append(label)
        if se:
            blocks.append(se_row)
            index.append("")

    for key, label in (stats or _DEFAULT_STATS):
        stat_row: dict[str, str] = {}
        for column, res in zip(names, results):
            value = res.e.get(key)
            if value is None:
                stat_row[column] = ""
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                stat_row[column] = str(value)
                continue
            stat_row[column] = (
                f"{int(round(numeric)):,}" if key == "N" else _format(numeric, stats_digits)
            )
        blocks.append(stat_row)
        index.append(label)

    table = pd.DataFrame(blocks, index=pd.Index(index), columns=pd.Index(names))
    table.index.name = None
    return table


def to_markdown(table: pd.DataFrame, *, caption: str = "") -> str:
    """Render an :func:`etable` result as a Markdown table."""

    lines: list[str] = []
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    header = "| " + " | ".join([""] + [str(c) for c in table.columns]) + " |"
    separator = "|" + "|".join([":---"] + ["---:"] * len(table.columns)) + "|"
    lines.append(header)
    lines.append(separator)
    for label, row in table.iterrows():
        cells = [str(label)] + [str(row[column]) for column in table.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_latex(
    table: pd.DataFrame,
    *,
    caption: str = "",
    label: str = "",
    notes: str = "",
) -> str:
    """Render an :func:`etable` result as a LaTeX ``tabular``."""

    n_columns = len(table.columns)
    body: list[str] = []
    for row_label, row in table.iterrows():
        cells = " & ".join(str(row[column]) for column in table.columns)
        body.append(f"{row_label} & {cells} \\\\")

    lines = [
        "\\begin{table}[htbp]",
        "  \\centering",
    ]
    if caption:
        lines.append(f"  \\caption{{{caption}}}")
    if label:
        lines.append(f"  \\label{{{label}}}")
    lines.append("  \\begin{tabular}{l" + "r" * n_columns + "}")
    lines.append("    \\hline")
    lines.append(
        "     & " + " & ".join(f"({i + 1})" for i in range(n_columns)) + " \\\\"
    )
    lines.append("    \\hline")
    lines.extend(f"    {line}" for line in body)
    lines.append("    \\hline")
    lines.append("  \\end{tabular}")
    if notes:
        lines.append(f"  \\parbox{{\\linewidth}}{{\\footnotesize {notes}}}")
    lines.append("\\end{table}")
    return "\n".join(lines)
