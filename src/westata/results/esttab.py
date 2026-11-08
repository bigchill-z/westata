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
        """esttab prints three significant figures, four once the value reaches 1.

        Measured from live ``esttab m1 m2, se r2 mtitle`` output:

            0.0966245 -> "0.0966"      0.0352 -> "0.0352"      4.391 -> "4.391"
            77.5787   -> "77.58"       9.011188 -> "9.011"     -4.855 -> "-4.855"

        A fixed decimal count cannot produce both "0.107" and "4.391": three decimals gives
        "0.107" but "4.391"; four gives "0.1066" but "4.3912".
        """

        # THREE SIGNIFICANT FIGURES WITH THE TRAILING ZERO KEPT, four at or above one.
        #
        # "%.3g" gets the digits right and drops a trailing zero, which is the only difference:
        #
        #     0.3772      -> 0.377    (no trailing zero, identical)
        #     0.1586      -> 0.159
        #     0.9166      -> 0.917
        #     0.0430441   -> 0.0430   "%.3g" gives 0.043; Stata keeps the zero
        #
        # Verified against a live esttab m0 m1, se where the coefficient is 0.0430441398 and Stata
        # prints 0.0430. The earlier version differed ONLY on values whose third significant figure
        # is zero, which is why a single captured table never showed it.
        number = float(value)
        if abs(number) >= 1.0:
            return f"{number:.4g}"
        digits = f"{abs(number):.3g}"
        if "e" in digits or "E" in digits:
            return ("-" if number < 0 else "") + digits
        whole, _, fraction = digits.partition(".")
        significant = (fraction.lstrip("0") + "000")[:3]
        leading_zeros = len(fraction) - len(fraction.lstrip("0"))
        text_value = f"{whole}.{'0' * leading_zeros}{significant}"
        return ("-" if number < 0 else "") + text_value

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
    # Stata opens the table with a BLANK LINE, then the column numbers.
    lines: list[str] = [""]

    # "      (1)         (2)" then the mtitle row.
    number_row: list[str] = [" " * _NAME_WIDTH]
    name_row: list[str] = [" " * _NAME_WIDTH]
    for position, _ in enumerate(models):
        _cell(f"({position + 1})", widths[position], number_row)
        _cell(names[position], widths[position], name_row)
    # Stata prints a rule ABOVE the column numbers too:
    #     ''
    #     '--------------------------------------------'
    #     '                      (1)             (2)   '
    lines.append(rule)
    lines.append("".join(number_row))
    if model_names is not None:
        # The `mtitle` row: Stata prints it only when the caller names the models. Default output
        # has the dependent-variable row alone, which is one line shorter.
        lines.append("".join(name_row))

    # THE DEPENDENT-VARIABLE ROW IS ALWAYS PRINTED, mtitle or not:
    #
    #     '                      (1)             (2)   '
    #     '                       m1              m2   '     <- only with mtitle
    #     '                        y               y   '
    dep_cells: list[str] = [" " * _NAME_WIDTH]
    for position, model in enumerate(models):
        depvar = ""
        e_block = getattr(model, "e", None)
        if isinstance(e_block, dict):
            depvar = str(e_block.get("depvar") or "")
        _cell(depvar, widths[position], dep_cells)
    lines.append("".join(dep_cells))
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
            # Each SE cell is RIGHT-ALIGNED to its column's end (25 and 41), like the coefficient
            # above it. The old comment here claimed otherwise -- "the SE cells are NOT
            # right-aligned, (9.011) ends at 24 while (0.0352) ends at 25" -- and that claim sent
            # rounds 96 and 97 hunting a positional rule that does not exist. Capturing all six
            # cells from two reference tables shows they all end at the column edge:
            #
            #   grilic        "(0.00604)" 16-25    "(0.00595)" 32-41
            #   grilic        "(0.0821)"  17-25    "(0.0849)"  33-41
            #   cobb_douglas  "(0.0352)"  17-25    "(0.0417)"  33-41
            #   cobb_douglas  "(9.011)"   18-25    "(14.54)"   34-41
            #   cobb_douglas  "(0.150)"   --       "(0.150)"   34-41
            se_pieces: list[str] = [" " * _NAME_WIDTH]
            for position, model in enumerate(models):
                raw = _lookup(model, label)
                if not raw:
                    _cell("", widths[position], se_pieces)
                    continue
                se_value = float(model.bse[raw])
                # Three significant figures below 1, four at or above it -- the same rule the
                # coefficients use. Measured: 9.011188 -> "(9.011)", 0.0060425 -> "(0.00604)".
                # "%.3g" drops a trailing zero, so 0.150 would print "(0.15)"; Stata keeps it,
                # hence the pad to a minimum of three significant digits.
                se_text = f"{se_value:.4g}" if abs(se_value) >= 1.0 else f"{se_value:.3g}"
                # Stata's parenthesis rows carry FOUR decimals below 1 (0.0660, 0.0733) and four
                # significant figures at or above it (9.011). `%.3g` gives "0.066", so the tail is
                # padded to four places, not three.
                if abs(se_value) < 1.0 and "." in se_text:
                    head, _, tail = se_text.partition(".")
                    while len(tail.lstrip("0")) < 3:
                        tail += "0"
                    se_text = f"{head}.{tail}"
                _cell(f"({se_text})", widths[position], se_pieces)
            lines.append("".join(se_pieces))
            # A blank row separates one coefficient's block from the next; Stata prints it after
            # the standard-error line and drops it before the closing rule.
            lines.append("")
        continue

    if lines[-1] == "":
        lines.pop()
    # STATA'S esttab PRINTS N BY DEFAULT -- beneath every model, between two rules. This renderer
    # printed no statistics at all unless the caller supplied them, which is why the comparison
    # showed a table three lines short.
    if stats is None:
        stats = (("N", "nobs"),)
    if stats:
        lines.append(rule)
        for shown, source in stats:
            stat_cells: list[str] = [f"{shown:<{_NAME_WIDTH}}"]
            def_target = str(source).lower()
            for position, model in enumerate(models):
                value = _statistic(model, str(source))
                if value is None:
                    _cell("", widths[position], stat_cells)
                elif def_target == "nobs":
                    _cell(f"{value:.0f}", widths[position], stat_cells)
                else:
                    _cell(f"{value:.{stats_digits}f}", widths[position], stat_cells)
            lines.append("".join(stat_cells))
    lines.append(rule)
    if se:
        lines.append("Standard errors in parentheses")
    if stars:
        lines.append("* p<0.05, ** p<0.01, *** p<0.001")
    # esttab pads every TABLE row out to the full 44 columns; the two legend lines below the
    # closing rule are not padded. Rows built from the last column end at 41, so without this they
    # read three characters short in a line-wise comparison.
    closing_rule = len(lines) - (1 if se else 0) - (1 if stars else 0) - 1
    padded: list[str] = []
    for position, line in enumerate(lines):
        if line and position <= closing_rule and len(line) < _LINE_WIDTH:
            line = line + " " * (_LINE_WIDTH - len(line))
        padded.append(line)
    return "\n".join(padded)


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
