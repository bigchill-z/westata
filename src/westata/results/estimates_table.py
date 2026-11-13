"""Stata's `estimates table`, which is a DIFFERENT command from `esttab`.

`esttab` and `estimates table` both stack models side by side, but their layouts differ and the
package only had the first:

    esttab                    estimates table
    (1) (2) column numbers    no numbers; names sit on the header line
    dependent-variable row    none
    mtitle row                none
    stats block               a Legend line at the bottom

Measured on `estimates table ols_no_iq ols_with_iq tsls liml, b se` over grilic:

          Variable | ols_no_iq    ols_with~q      tsls         liml
    -------------+----------------------------------------------------
               s |  .10264304    .09278735    .06078035    .06063623
                 |  .00620988    .00697626    .01895051    .01903397

The geometry is FIXED-WIDTH and derived, not tabulated: a 14-character gutter
("    Variable | "), then 13 columns per model, and one more rule character per model beyond the
first (40 / 53 / 66 for two / three / four). Names longer than ten characters are abbreviated to
eight characters plus "~" plus the last character ("ols_with_iq" -> "ols_with~q").
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

import numpy as np
from westata.results.stars import stars

__all__ = ["render_estimates_table"]

#: Width of "    Variable | " -- the label gutter every row starts with.
_GUTTER = 14
#: Characters per model column.
_COLUMN_WIDTH = 13
#: Names strictly longer than this are abbreviated.
_NAME_LIMIT = 10
#: Width of the NUMBER inside a starred column; its stars sit immediately after it.
_STAR_NUMBER_WIDTH = 11


def _abbreviate(name: str, *, limit: int = _NAME_LIMIT) -> str:
    """Stata's column-name abbreviation: eight characters, "~", then the last one."""

    text = str(name)
    if len(text) <= limit:
        return text
    return text[:8] + "~" + text[-1]


def _label(raw: Any) -> str:
    """Stata writes the intercept as ``_cons``."""

    text = str(raw)
    return "_cons" if text.strip().lower() in ("intercept", "_cons", "const", "constant") else text


def _number(value: float) -> str:
    """Eight decimals, trailing zeros trimmed, leading zero dropped.

    Measured from the capture: .10264304 .09278735 .06078035 .06063623 .00620988 .01895051
    .10264304 is eight places; the constant appears as 4.103675 / 3.8951718, so the rule is the same
    "drop the padding" trimming esttab uses, not a significant-figure count.
    """

    # THE RULE HAS TWO BRANCHES, split at 1.0, and it was measured value by value:
    #
    #     |v| >= 1   ->  EIGHT SIGNIFICANT FIGURES   4.10367504 -> 4.103675
    #                                                 3.89517179 -> 3.8951718
    #     |v| <  1   ->  EIGHT DECIMALS, then trim   .10264304 -> .10264304
    #                                                 .0062098752 -> .00620988
    #
    # A single rule cannot produce both: eight decimals would render the constant as 4.10367504
    # (too wide for the column) and eight significant figures would render .0062098752 with nine
    # characters after the point. Trailing zeros are trimmed in both branches.
    if abs(float(value)) >= 1.0:
        text = format(Decimal(str(float(value))), ".8g")
    else:
        text = f"{float(value):.8f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _threshold(level: float) -> str:
    """A p-value threshold the way Stata prints it: .1, .05, .01 (leading zero dropped)."""

    text = f"{float(level):g}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _stars(pvalue: float, thresholds: Sequence[float]) -> str:
    """Delegate to the one star rule (R6.1); see results.stars.stars."""

    return stars(pvalue, thresholds)


def _nobs(model: Any) -> float | None:
    value = getattr(model, "nobs", None)
    return float(value) if value is not None else None


def _first_float(source: Any) -> float | None:
    if source is None:
        return None
    try:
        return float(source)
    except (TypeError, ValueError):
        return None


def _r2(model: Any) -> float | None:
    e = getattr(model, "e", None) or {}
    return _first_float(getattr(model, "rsquared", None)) or _first_float(e.get("r2"))


def _r2_adj(model: Any) -> float | None:
    e = getattr(model, "e", None) or {}
    return _first_float(getattr(model, "rsquared_adj", None)) or _first_float(e.get("r2_a"))


def _rmse(model: Any) -> float | None:
    e = getattr(model, "e", None) or {}
    return _first_float(e.get("rmse"))


#: Footer statistics, keyed by the names `esttab` accepts, mapping to (printed label, accessor).
_STAT_LOOKUP: dict[str, tuple[str, Any]] = {
    "n": ("N", _nobs),
    "r2": ("R-sq", _r2),
    "r2_a": ("Adj R-sq", _r2_adj),
    "rmse": ("RMSE", _rmse),
}


def render_estimates_table(
    results: Sequence[Any],
    *,
    model_names: Sequence[str] | None = None,
    se: bool = True,
    star: Sequence[float] | None = None,
    mtitle: bool = False,
    stats: Sequence[str] | None = None,
    pvalues: bool = False,
) -> str:
    """Reproduce ``estimates table m1 m2 ..., b se``.

    ``model_names`` is the equivalent of the ``estimates store`` names: Stata prints whatever the
    models were stored as, abbreviated to ten characters where needed.

    ``star`` is Stata's ``star(...)``: a descending triple of p-value thresholds. Passing it changes
    the LAYOUT, not just the cells -- see the measurement notes in the body.

    ``pvalues`` adds Stata's ``p`` row under each coefficient -- four decimal places WITH the
    leading zero, unlike the coefficient itself. ``estimates table ..., b p`` and ``..., p`` are the
    same thing (``b`` is always printed), and ``p`` cannot be combined with ``star(...)``: Stata
    rejects that pair.

    ``mtitle`` selects the **``esttab`` shape** instead of the ``estimates table`` one: a header with
    ``(1) (2) ...`` numbers and the model names, no ``Variable |`` gutter, a ``12 + 16*n`` rule and
    three-significant-figure coefficients. ``stats`` adds the footer rows that ``esttab`` prints for
    names such as ``"N"``, ``"r2"``, ``"r2_a"`` and ``"rmse"``.
    """

    models = list(results)
    if not models:
        raise ValueError("estimates table: at least one model is required")

    # THE ABBREVIATION THRESHOLD IS A FUNCTION OF THE COLUMN WIDTH, not a constant. The 13-wide
    # unstarred layout truncates past ten characters (`ols_with_iq` -> `ols_with~q`) while the
    # 16-wide starred one shows the eleven-character name in full. Measured on the same four models.
    name_limit = 12 if star is not None else _NAME_LIMIT
    names = (
        [_abbreviate(name, limit=name_limit) for name in model_names]
        if model_names is not None
        else [f"m{index + 1}" for index in range(len(models))]
    )
    if len(names) != len(models):
        raise ValueError(
            f"estimates table: {len(names)} names for {len(models)} models"
        )

    # star(...) AND p ARE MUTUALLY EXCLUSIVE in Stata: estimates table ..., b se p
    # star(...) is an error. Reject it at the entry point so the mtitle branch cannot slip past.
    if star is not None and pvalues:
        raise ValueError(
            "estimates table: star(...) and p cannot be combined; "
            "Stata's estimates table rejects that pair"
        )

    if mtitle:
        # `mtitle` selects the `esttab` layout, which is a DIFFERENT table from `estimates table`:
        # no "Variable |" gutter, a 12 + width*n rule, a statistics footer, and only ONE second row
        # per coefficient. The full esttab renderer lives in `esttab_render`.
        from westata.results.esttab_options import EsttabOptions
        from westata.results.esttab_render import render_esttab_table

        options = EsttabOptions(
            # esttab accepts exactly one second-row cell; `pvalues` maps to `p`.
            second_row="p" if pvalues else ("se" if se else "t"),
            stars=star is not None,
            star_levels=tuple(star) if star is not None else EsttabOptions().star_levels,
            stats=tuple(stats) if stats else (),
            model_titles=tuple(names),
        )
        return render_esttab_table(models, names=names, options=options)

    # Labels in first-model order, constant last -- Stata's sequence.
    order: list[str] = []
    for model in models:
        for raw in model.params.index:
            label = _label(raw)
            if label not in order:
                order.append(label)
    if "_cons" in order:
        order.remove("_cons")
        order.append("_cons")

    # GEOMETRY, measured char by char: a 14-character gutter ("    Variable |"), then 13 per model.
    # Inside each column the value occupies 11 characters RIGHT-aligned with one space either side,
    # and the name is CENTRED in those same 11:
    #
    #     '    Variable | ols_no_iq    ols_with~q      tsls         liml     '
    #     '           s |  .10264308    .09278735    .06078035    .06063623  '
    #
    # The rule is 66 characters for four models -- 14 + 13*4 - 0, i.e. _GUTTER + 13*n -- not n-1.
    # `star()` CHANGES THE COLUMN WIDTH, it is not a cosmetic cell suffix. Measured on four models:
    #
    #     without star   14 + 13*n = 66 columns of 13
    #     with star      14 + 16*n = 78 columns of 16
    #
    # because the stars ride on the coefficient and the standard-error row disappears:
    #
    #     '    Variable |   ols_no_iq      ols_with_iq        tsls            liml       '
    #     '           s |  .10264308***    .09278735***    .06078035***    .06063623***  '
    #
    # Inside a starred column the value (with its stars) right-aligns in the first 14 characters and
    # the last 2 are padding.
    starred = star is not None
    column_width = 16 if starred else _COLUMN_WIDTH
    rule_length = _GUTTER + column_width * len(models)
    lines: list[str] = [""]
    lines.append("-" * rule_length)

    # COLUMN GEOMETRY, derived from the measured index of every value's end: the values end at
    # 25, 38, 51, 64 for four models, i.e. 14 + 11 + 13k. So each column is 13 wide, the value is
    # RIGHT-aligned in its first 11 characters, and the trailing 2 are padding.
    def _centre(text_value: str, width: int) -> str:
        """Centre in ``width``, extra space on the LEFT, as Stata does for an odd remainder."""

        total = max(width - len(text_value), 0)
        left = (total + 1) // 2
        return " " * left + text_value + " " * (total - left)

    header = "    Variable |"
    for name in names:
        # MEASURED for the starred layout: the name FIELD is 14 wide inside a 16-wide column, and
        # the row advances 16 per model. One shared field width reproduces all four measured leading
        # offsets (3 for ols_no_iq, 2 for ols_with_iq, 5 for tsls/liml).
        header += _centre(name, column_width - 2) + " " * 2
    lines.append(header)
    # MEASURED: thirteen dashes, "+", then the rest. _GUTTER - 1 dashes, so "+" lands at index 13
    # where Stata puts it ('-------------+----...').
    lines.append("-" * (_GUTTER - 1) + "+" + "-" * (rule_length - _GUTTER))

    for label in order:
        row = f"{label:>12} |"
        present = False
        for model in models:
            match = [
                raw
                for raw in model.params.index
                if _label(raw) == label
            ]
            if not match:
                row += " " * column_width
                continue
            present = True
            rendered = _number(float(model.params[match[0]]))
            if starred:
                # MEASURED, every column and every row: the number's right edge sits at offset 11
                # inside the 16-wide field, its stars follow immediately, and the rest is blank.
                #
                #     '  .10264304***  '   number ends at 11, 3 stars, 2 blanks
                #     ' -.07453249**   '   number ends at 11, 2 stars, 3 blanks
                #     ' -.04352713     '   number ends at 11, no stars, 5 blanks
                #
                # So the cell is a right-aligned 11-character number plus the star run, padded to
                # the column width -- NOT a right-aligned number+stars unit.
                cell = (
                    f"{rendered:>{_STAR_NUMBER_WIDTH}}{_stars(float(model.pvalues[match[0]]), star)}"
                )
                row += cell.ljust(column_width)
                continue
            row += f"{rendered:>{column_width - 2}}" + " " * 2
        if not present:
            continue
        lines.append(row)
        if se and not starred:
            error_row = " " * 12 + " |"
            for model in models:
                match = [
                    raw
                    for raw in model.params.index
                    if _label(raw) == label
                ]
                if not match:
                    error_row += " " * column_width
                    continue
                error_row += f"{_number(float(model.bse[match[0]])):>{column_width - 2}}" + " " * 2
            lines.append(error_row)
        if pvalues and not starred:
            # STATSPACE: the p row comes AFTER se and uses a DIFFERENT format -- four decimal
            # places WITH the leading zero ('0.0000'), not the trimmed, leading-zero-dropped style
            # of the coefficient and standard-error cells. Measured on `..., b se p`:
            #
            #     '           s |  .10264304    .09278735    .06078035    .06063623  '
            #     '             |  .00620988    .00697626    .01895051    .01903397  '
            #     '             |     0.0000       0.0000       0.0013       0.0014  '
            p_row = " " * 12 + " |"
            for model in models:
                match = [raw for raw in model.params.index if _label(raw) == label]
                if not match:
                    p_row += " " * column_width
                    continue
                p_row += f"{float(model.pvalues[match[0]]):.4f}".rjust(column_width - 2) + " " * 2
            lines.append(p_row)

    lines.append("-" * rule_length)
    if starred:
        # Right-aligned to the rule, like `Legend: b/se`. The thresholds print as ".1", ".05", ".01"
        # with the leading zero dropped, and each is paired with its star run.
        # THE LEGEND READS LARGEST THRESHOLD FIRST, which is the opposite order from the argument:
        #   'Legend: * p<.1; ** p<.05; *** p<.01'
        ordered = sorted((float(level) for level in star), reverse=True)
        pairs = [
            f"{'*' * (index + 1)} p<{_threshold(level)}"
            for index, level in enumerate(ordered)
        ]
        lines.append(f"{'Legend: ' + '; '.join(pairs):>{rule_length}}")
    elif se or pvalues:
        # 'b/se', 'b/p' or 'b/se/p' -- the rows actually printed, in order (`b` is always there).
        parts = ["b"]
        if se:
            parts.append("se")
        if pvalues:
            parts.append("p")
        lines.append(f"{'Legend: ' + '/'.join(parts):>{rule_length}}")
    # Stata right-pads every line, including the last, to the rule width.
    # The leading blank stays empty; every OTHER line is padded to the rule width.
    return "\n".join(
        line if line == "" else line.ljust(rule_length) for line in lines
    )

# ---------------------------------------------------------------- the `esttab` shape
#
# `mtitle` asks for `esttab`, which is a different table from `estimates table`. Measured on four
# models with `esttab ..., se r2 mtitle star(* 0.1 ** 0.05 *** 0.01)` over grilic:
#
#     ----------------------------------------------------------------------------
#                           (1)             (2)             (3)             (4)
#                     ols_no_iq     ols_with_iq            tsls            liml
#     ----------------------------------------------------------------------------
#     s                   0.103***       0.0928***       0.0608***       0.0606***
#                     (0.00621)       (0.00698)        (0.0190)        (0.0190)
#
# THE GEOMETRY: the label occupies the first 12 columns, then 16 per model, so the rule is
# 12 + 16*n (76 for four). Every value is RIGHT-aligned to its column's end; the star run is appended
# after the value, which pushes a starred coefficient cell wider than the error row beneath it.
#
# THE NUMBER FORMAT is `%.3g` -- three significant figures, leading zero KEPT -- promoted to four
# figures at or above 10, which is what makes 4.104*** coexist with 0.00621.


def _esttab_number(value: float) -> str:
    """`esttab`'s number format: THREE significant figures, trailing zeros KEPT.

    Measured value by value off a live capture -- the trailing zeros are the part that is easy to
    lose, because `%.3g` strips them:

        .10264304 -> 0.103      .09278735 -> 0.0928     .01903397 -> 0.0190
        .00800000 -> 0.00800    .03000000 -> 0.0300     .00620988 -> 0.00621
        4.1036750 -> 4.104      3.8951718 -> 3.895

    So the rule is three significant figures, then pad back to the width those three figures need
    for this exponent. Values below 0.01 keep four or five decimals; the leading zero stays.
    """

    number = float(value)
    if number == 0.0:
        return "0"
    # AT OR ABOVE 1 THE COUNT IS FOUR, below it three. Measured: 4.103675 -> 4.104 and
    # 3.8951718 -> 3.895, against .10264304 -> 0.103 and .00620988 -> 0.00621.
    significant = 4 if abs(number) >= 1.0 else 3
    text = f"{number:.{significant}g}"
    if "e" in text or "E" in text:
        text = f"{number:.10f}".rstrip("0").rstrip(".")
        return text
    # Restore the zeros `%.3g` removed, using the exponent of the value.
    from math import floor, log10

    exponent = floor(log10(abs(number)))
    places = max(2 - exponent, 0)
    padded = f"{number:.{places}f}"
    # Keep whichever is longer: the significant-figure form or the zero-padded form.
    return padded if len(padded) > len(text) else text


def _stat_accessor(name: str):
    """The getter for a footer statistic, or ``None`` when the name is unknown."""

    key = str(name).strip().lower()
    return _STAT_LOOKUP.get(key)


def _render_esttab_shape(
    models: Sequence[Any],
    *,
    names: Sequence[str],
    starred: bool,
    star: Sequence[float] | None,
    se: bool,
    stats: Sequence[str] | None,
) -> str:
    column = 16
    rule_text = "-" * (12 + column * len(models))
    lines: list[str] = [""]

    lines.append(rule_text)
    lines.append(
        " " * 12 + "".join(f"({index + 1})".rjust(column - 3) + " " * 3 for index in range(len(models)))
    )
    lines.append(" " * 12 + "".join(str(name).rjust(column - 3) + " " * 3 for name in names))
    lines.append(rule_text)

    order: list[str] = []
    for model in models:
        for raw in model.params.index:
            label = _label(raw)
            if label not in order:
                order.append(label)
    if "_cons" in order:
        order.remove("_cons")
        order.append("_cons")

    for label in order:
        cells: list[str] = []
        present = False
        for model in models:
            match = [raw for raw in model.params.index if _label(raw) == label]
            if not match:
                cells.append("")
                continue
            present = True
            rendered = _esttab_number(float(model.params[match[0]]))
            if starred:
                rendered += _stars(float(model.pvalues[match[0]]), star)
            cells.append(rendered)
        if not present:
            continue
        # THE NUMBER right-aligns to the 13-wide field and the stars hang AFTER it, so a value with
        # more stars ends further right: .103*** reaches 28 while -.0745** reaches 27. Splitting the
        # number from its stars is what makes the two rows line up.
        rendered_cells: list[str] = []
        for cell, model in zip(cells, models):
            if not cell:
                rendered_cells.append(" " * column)
                continue
            stars_text = cell[len(cell.rstrip("*")):] if cell.endswith("*") else ""
            number_text = cell[: len(cell) - len(stars_text)] if stars_text else cell
            # number right-aligned in 13, stars appended, then pad to the 16-wide column
            unit = number_text.rjust(column - 3) + stars_text
            rendered_cells.append(unit + " " * (column - len(unit)))
        lines.append(f"{label:<12}" + "".join(rendered_cells))
        if se:
            error_cells: list[str] = []
            for model in models:
                match = [raw for raw in model.params.index if _label(raw) == label]
                if not match:
                    error_cells.append("")
                    continue
                error_cells.append(
                    f"({_esttab_number(float(model.bse[match[0]]))})"
                )
            lines.append(" " * 12 + "".join(cell.rjust(column - 3) + " " * 3 for cell in error_cells))
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    lines.append(rule_text)

    if stats:
        for requested in stats:
            accessor = _stat_accessor(requested)
            label = accessor[0] if accessor else str(requested)
            row = f"{label:<12}"
            for model in models:
                value = accessor[1](model) if accessor else None
                row += " " * column if value is None else _esttab_number(value).rjust(column - 3) + " " * 3
            lines.append(row)
        lines.append(rule_text)

    if se:
        lines.append("Standard errors in parentheses")
    if starred:
        ordered = sorted((float(level) for level in star), reverse=True)
        parts = [f"{'*' * (index + 1)} p<{level:g}" for index, level in enumerate(ordered)]
        lines.append(", ".join(parts))

    # THE TWO NOTE LINES ARE NOT PADDED -- they are the only lines shorter than the rule.
    note_lines = set()
    if se:
        note_lines.add("Standard errors in parentheses")
    if starred:
        note_lines.add(lines[-1])
    return "\n".join(
        line
        if line == "" or line in note_lines
        else line.ljust(len(rule_text))
        for line in lines
    )
