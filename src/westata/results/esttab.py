"""``esttab``-style text output for ``etable``.

The v2 report (C20-01) noted that ``etable`` returns a DataFrame while Stata's ``esttab``
prints a table with a ``(1) (2)`` numbering row, a ``mtitle`` row of model names, an ``N`` /
``R-sq`` footer block AND the footnote lines. ``model_names`` already existed; what was absent
was the rendered table.

Reference, ``regress y k`` stored as m1 and ``regress y k l`` as m2, then
``esttab m1 m2, se r2 mtitle`` on cobb_douglas::

                          (1)             (2)
                           m1              m2
    --------------------------------------------
    k                   0.377***        0.159**
                     (0.0352)        (0.0417)

    l                                   0.917***
                                      (0.150)

    _cons               77.58***       -4.855
                      (9.011)         (14.54)
    --------------------------------------------
    N                      24              24
    R-sq                0.839           0.942
    --------------------------------------------
    Standard errors in parentheses
    * p<0.05, ** p<0.01, *** p<0.001

Every line is 44 characters. Measured with re.finditer: the two model columns END at 25 and
41, so each numeric field is right-aligned there; the name column is 12 wide and the blank
line between coefficient blocks carries no content.

Significance uses the three-threshold scheme the footnote describes -- ``*`` at 5%, ``**`` at
1%, ``***`` at 0.1% -- which is ``esttab``'s default, NOT the one/two/three-star 10/5/1
convention some other writers use.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

__all__ = ["render_esttab", "etable_as_stata"]

#: Column END positions measured off Stata's output. The second is the line width.
_COLUMN_ENDS = (25, 41)
_LINE_WIDTH = 44
_NAME_WIDTH = 12


def _stars(pvalue: float) -> str:
    """esttab's default: * 5%, ** 1%, *** 0.1%."""

    if not np.isfinite(pvalue):
        return ""
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return ""


def _cell(text: str, end: int, pieces: list[str]) -> None:
    used = sum(len(piece) for piece in pieces)
    pieces.append(" " * max(end - used - len(text), 0) + text)


def render_esttab(
    results: Sequence[Any],
    *,
    model_names: Sequence[str] | None,
    digits: int = 3,
    se: bool = True,
    stars: bool = True,
    stats: Sequence[tuple[str, Any]] | None = None,
    stats_digits: int = 3,
) -> str:
    """Render ``results`` as Stata's ``esttab`` prints them."""

    models = list(results)
    if not models:
        raise ValueError("esttab: at least one model is required")
    names = list(model_names) if model_names is not None else [
        f"m{index + 1}" for index in range(len(models))
    ]
    if len(names) != len(models):
        raise ValueError(
            f"esttab: {len(names)} model names for {len(models)} models"
        )
    if len(models) > len(_COLUMN_ENDS):
        raise ValueError(
            f"esttab: this renderer lays out at most {len(_COLUMN_ENDS)} models; got "
            f"{len(models)}"
        )

    def _number(value: float) -> str:
        """esttab drops one decimal place once the integer part needs two columns.

        Measured: 0.377 -> "0.377" (three decimals) beside 77.58 -> "77.58" (two), so the
        cells stay the same width. A fixed precision would print "77.579" and push the
        row one column right.
        """
        integer = len(f"{abs(value):.0f}")
        places = max(digits - max(integer - 1, 0), 0)
        return f"{value:.{places}f}"

    # The union of parameter names, first model first, so the row order is the reader's.
    order: list[str] = []
    for model in models:
        for name in model.params.index:
            label = "_cons" if str(name) in ("Intercept", "_cons", "const") else str(name)
            if label not in order:
                order.append(label)
    # esttab prints the constant LAST, after every other regressor, regardless of the
    # order the design matrix uses (patsy puts Intercept first).
    if "_cons" in order:
        order.remove("_cons")
        order.append("_cons")

    def _lookup(model: Any, label: str) -> str:
        index = [
            name
            for name in model.params.index
            if ("_cons" if str(name) in ("Intercept", "_cons", "const") else str(name))
            == label
        ]
        return str(index[0]) if index else ""

    widths = list(_COLUMN_ENDS)
    rule = "-" * _LINE_WIDTH
    lines: list[str] = []

    # "      (1)         (2)" then the mtitle row.
    number_row: list[str] = [" " * _NAME_WIDTH]
    name_row: list[str] = [" " * _NAME_WIDTH]
    for position, _ in enumerate(models):
        _cell(f"({position + 1})", widths[position], number_row)
        _cell(names[position], widths[position], name_row)
    lines.append("".join(number_row))
    lines.append("".join(name_row))
    lines.append(rule)

    for label in order:
        cells: list[str] = [f"{label:<{_NAME_WIDTH}}"]
        printed = False
        for position, model in enumerate(models):
            raw = _lookup(model, label)
            if not raw:
                _cell("", widths[position], cells)
                continue
            printed = True
            coefficient = float(model.params[raw])
            suffix = _stars(float(model.pvalues[raw])) if stars else ""
            # The stars hang PAST the column end: esttab aligns the NUMBER and then
            # appends the markers, so "0.377***" extends two columns beyond "0.377".
            number_text = _number(coefficient)
            _cell(number_text, widths[position], cells)
            if suffix:
                cells.append(suffix)
        if not printed:
            continue
        lines.append("".join(cells))
        if se and any(_lookup(model, label) for model in models):
            # The SE cells START at fixed columns (17 and 33, measured) and are NOT
            # right-aligned to the coefficient columns' ends: "(9.011)" ends at 24 while
            # "(0.0352)" ends at 25. Padding them to the column end pushed the wide ones one
            # column right.
            se_pieces: list[str] = [" " * 17]
            for position, model in enumerate(models):
                if position:
                    used = sum(len(piece) for piece in se_pieces)
                    se_pieces.append(" " * max(33 - used, 0))
                raw = _lookup(model, label)
                if not raw:
                    continue
                se_value = float(model.bse[raw])
                coefficient_text = _number(float(model.params[raw]))
                coefficient_places = (
                    len(coefficient_text.split(".")[1]) if "." in coefficient_text else 0
                )
                extra_integer = max(len(f"{abs(se_value):.0f}") - 1, 0)
                se_places = max(coefficient_places + 1 - extra_integer, 0)
                se_text = f"{se_value:.{se_places}f}"
                if "." in se_text:
                    head, _, tail = se_text.partition(".")
                    trimmed = tail.rstrip("0")
                    if len(trimmed) >= 2:
                        se_text = f"{head}.{trimmed}"
                se_pieces.append(f"({se_text})")
            lines.append("".join(se_pieces))
        lines.append("")

    if lines[-1] == "":
        lines.pop()
    if stats is not None:
        lines.append(rule)
        for shown, source in stats:
            stat_cells: list[str] = [f"{shown:<{_NAME_WIDTH}}"]
            def_target = str(source).lower()
            for position, model in enumerate(models):
                value = _statistic(model, str(source))
                if value is None:
                    _cell("", widths[position], stat_cells)
                elif def_target == "nobs":
                    _cell(f"{value:,.0f}", widths[position], stat_cells)
                else:
                    _cell(f"{value:.{stats_digits}f}", widths[position], stat_cells)
            lines.append("".join(stat_cells))
    lines.append(rule)
    if se:
        lines.append("Standard errors in parentheses")
    if stars:
        lines.append("* p<0.05, ** p<0.01, *** p<0.001")
    return "\n".join(lines)


def _with_one_more_place(value: float, digits: int) -> str:
    """``digits`` places, minus one for every integer digit beyond the first.

    Measured from Stata's standard-error rows: (0.0352) and (9.011) both carry THREE
    decimals because their integer parts are one digit, while (14.54) carries two because
    14 needs two columns. A blanket ``digits + 1`` gave (9.0112) and (14.540), which
    overflow the field.
    """

    integer = len(f"{abs(value):.0f}")
    places = max(digits + 1 - max(integer - 1, 0), 0)
    return f"{value:.{places}f}"


def _statistic(model: Any, source: str) -> float | None:
    """Resolve an ``esttab`` statistic name against the result object."""

    if source == "nobs":
        return float(model.nobs)
    if source in ("r2", "rsquared", "R-sq"):
        return float(model.rsquared)
    if source in ("r2_a", "rsquared_adj"):
        return float(model.rsquared_adj)
    candidate = getattr(model, source, None)
    if candidate is None:
        return None
    return float(candidate)


def etable_as_stata(
    results: Sequence[Any],
    *,
    model_names: Sequence[str] | None = None,
    digits: int = 3,
    se: bool = True,
    stars: bool = True,
    stats: Sequence[tuple[str, str]] | None = None,
) -> str:
    """``etable`` rendered as ``esttab`` prints it.

    ``etable`` returns a DataFrame because that is more useful to a program; this returns the
    TEXT Stata prints, which is what a comparison needs. Both are kept: neither is a
    substitute for the other.
    """

    return render_esttab(
        results,
        model_names=model_names,
        digits=digits,
        se=se,
        stars=stars,
        stats=stats,
    )
