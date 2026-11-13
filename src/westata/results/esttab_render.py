"""esttab 的完整屏幕渲染器。

布局（实测，2 个模型）：

    --------------------------------------------
                          (1)             (2)   
                          lnw             lnw   
    --------------------------------------------
    s                   0.107***        0.107***
                    (0.00638)       (0.00624)   
    expr               0.0446***       0.0376*** 
                    (0.00641)       (0.00669)   
    --------------------------------------------
    N                     758             758   
    --------------------------------------------
    * p<0.05, ** p<0.01, *** p<0.001

几何：标签列 12 宽，之后每个模型 16 宽；系数行右端 28（含星号），第二行右端 25。
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Sequence

import numpy as np

from westata.results.esttab_options import ESTTAB_DEFAULT_STARS, EsttabOptions

#: Width of the label column, and of one model column.
LABEL_WIDTH = 12
COLUMN_WIDTH = 16

#: Footer statistics: esttab's name -> (printed label, accessor).
STATS: dict[str, tuple[str, Any]] = {}


def _nobs(model: Any) -> float | None:
    value = getattr(model, "nobs", None)
    return float(value) if value is not None else None


def _float_or_none(source: Any) -> float | None:
    if source is None:
        return None
    try:
        return float(source)
    except (TypeError, ValueError):
        return None


def _r2(model: Any) -> float | None:
    e = getattr(model, "e", None) or {}
    return _float_or_none(getattr(model, "rsquared", None)) or _float_or_none(e.get("r2"))


def _r2_adj(model: Any) -> float | None:
    e = getattr(model, "e", None) or {}
    return _float_or_none(getattr(model, "rsquared_adj", None)) or _float_or_none(e.get("r2_a"))


def _rmse(model: Any) -> float | None:
    e = getattr(model, "e", None) or {}
    return _float_or_none(e.get("rmse"))


STATS.update(
    {
        "n": ("N", _nobs),
        "r2": ("R-sq", _r2),
        "r2_a": ("Adj R-sq", _r2_adj),
        "rmse": ("RMSE", _rmse),
    }
)

CONSTANT_NAMES = frozenset({"intercept", "_cons", "const", "constant"})


def canonical_label(raw: Any) -> str:
    """Stata writes the intercept as ``_cons``."""

    text = str(raw)
    return "_cons" if text.strip().lower() in CONSTANT_NAMES else text


def esttab_number(value: float) -> str:
    """esttab's number: three significant figures below 1, four at or above, zeros kept.

    Measured: .10264304 -> 0.103, 4.103675 -> 4.104, .01903397 -> 0.0190, .00620988 -> 0.00621.
    """

    number = float(value)
    if number == 0.0:
        return "0"
    significant = 4 if abs(number) >= 1.0 else 3
    text = f"{number:.{significant}g}"
    if "e" in text or "E" in text:
        return f"{number:.10f}".rstrip("0").rstrip(".")
    from math import floor, log10

    places = max(2 - floor(log10(abs(number))), 0)
    padded = f"{number:.{places}f}"
    return padded if len(padded) > len(text) else text


def star_run(pvalue: float, levels: Sequence[float]) -> str:
    """The star run for one p-value, given ASCENDING thresholds."""

    if pvalue != pvalue:  # NaN
        return ""
    ascending = sorted(float(level) for level in levels)
    for index, level in enumerate(ascending):
        if pvalue < level:
            return "*" * (len(ascending) - index)
    return ""


def _ci(model: Any, name: Any, level: float) -> tuple[float, float] | None:
    params = getattr(model, "params", None)
    bse = getattr(model, "bse", None)
    if params is None or bse is None:
        return None
    try:
        estimate = float(params[name])
        error = float(bse[name])
    except (KeyError, TypeError, ValueError):
        return None
    # STATSPACE: Stata's `ci` in esttab uses the t quantile with the residual degrees of freedom,
    # not the normal one. Using the normal quantile put the second OLS column's lower bound at
    # .0245 where Stata prints .0244.
    from scipy import stats as scipy_stats

    df_resid = getattr(model, "df_resid", None)
    tail = 0.5 + level / 200.0
    if df_resid is not None and float(df_resid) > 0:
        critical = float(scipy_stats.t.ppf(tail, float(df_resid)))
    else:
        critical = float(scipy_stats.norm.ppf(tail))
    return estimate - critical * error, estimate + critical * error


def _second_row_text(
    model: Any, name: Any, options: EsttabOptions
) -> str:
    """The text of the row beneath a coefficient, with its delimiters."""

    left, right = options.second_row_brackets
    kind = options.second_row
    if kind == "se":
        return f"{left}{esttab_number(float(model.bse[name]))}{right}"
    if kind == "t":
        return f"{left}{float(model.tvalues[name]):.2f}{right}"
    if kind == "p":
        return f"{left}{float(model.pvalues[name]):.3f}{right}"
    if kind == "ci":
        interval = _ci(model, name, options.level)
        if interval is None:
            return ""
        return f"[{esttab_number(interval[0])},{esttab_number(interval[1])}]"
    return ""


def _selected_rows(models: Sequence[Any], options: EsttabOptions) -> list[str]:
    """Labels in first-model order, constant last, filtered by keep/drop and reordered."""

    order: list[str] = []
    for model in models:
        for raw in model.params.index:
            label = canonical_label(raw)
            if label not in order:
                order.append(label)
    if "_cons" in order:
        order.remove("_cons")
        order.append("_cons")

    if options.keep:
        wanted = {str(item) for item in options.keep}
        order = [label for label in order if label in wanted]
    if options.drop:
        unwanted = {str(item) for item in options.drop}
        order = [label for label in order if label not in unwanted]
    if options.order:
        preferred = [str(item) for item in options.order]
        ordered = [label for label in preferred if label in order]
        ordered += [label for label in order if label not in ordered]
        order = ordered
    return order


def _row_label(label: str, options: EsttabOptions) -> str:
    if label in options.varlabels:
        return str(options.varlabels[label])
    if label in options.rename:
        return str(options.rename[label])
    return label


def _hidden(label: str, options: EsttabOptions) -> bool:
    """Only ocons\ hides a row.

    MEASURED: \arlabels(tenure "")\ does NOT drop the row -- it blanks the LABEL and keeps the
    coefficient, which is why that capture still shows \	enure ... 0.0381***\.
    """

    return bool(options.no_constant and label == "_cons")


def render_esttab_table(
    models: Sequence[Any],
    *,
    names: Sequence[str],
    options: EsttabOptions,
) -> str:
    """Render the screen table for `esttab` with the given options."""

    if not models:
        raise ValueError("esttab: at least one model is required")

    label_width = LABEL_WIDTH
    # THE COLUMN WIDTH IS DERIVED, not constant. Measured on the same two models:
    #   b se    -> 16 per model (rule 44)      stars on, second row is se
    #   nostar  -> 13 per model (rule 38)      no star run to accommodate
    #   b ci    -> 26 per model (rule 64)      '[0.0943,0.119]' is longer than 13
    if options.second_row == "ci":
        column = 26
    elif options.stars:
        column = 16
    else:
        column = 13
    #: The 3-column slack exists for the STAR RUN only; without stars the value fills its column.
    _allowance = 3 if options.stars else 0

    rule_text = "-" * (label_width + column * len(models))
    lines: list[str] = [] if options.no_margin else [""]

    lines.append(rule_text)
    if options.numbers:
        # MEASURED: '(1)' ends at 25, i.e. column - 3 within a 16-wide slot.
        lines.append(
            " " * label_width
            + "".join(
                f"({index + 1})".rjust(column - _allowance) + " " * _allowance
                for index in range(len(models))
            )
        )

    if not options.no_type:
        if options.model_titles:
            titles = list(options.model_titles)
            while len(titles) < len(models):
                titles.append("")
            lines.append(
                " " * label_width
                + "".join(
                    str(title).rjust(column - _allowance) + " " * _allowance
                    for title in titles[: len(models)]
                )
            )
        else:
            # DEFAULT: the dependent variable's name, not the model name.
            dependents = []
            for model in models:
                e = getattr(model, "e", None) or {}
                dependents.append(str(e.get("depvar") or ""))
            lines.append(
                " " * label_width
                + "".join(name.rjust(column - _allowance) + " " * _allowance for name in dependents)
            )

    if options.column_labels:
        extra = list(options.column_labels)
        while len(extra) < len(models):
            extra.append("")
        lines.append(
            " " * label_width
            + "".join(
                str(label).rjust(column - _allowance) + " " * _allowance
                for label in extra[: len(models)]
            )
        )

    lines.append(rule_text)

    for label in _selected_rows(models, options):
        if _hidden(label, options):
            continue
        displayed = _row_label(label, options)
        coef_parts: list[str] = []
        second_parts: list[str] = []
        present = False
        for model in models:
            match = [raw for raw in model.params.index if canonical_label(raw) == label]
            if not match:
                coef_parts.append("")
                second_parts.append("")
                continue
            present = True
            raw = match[0]
            text = esttab_number(float(model.params[raw]))
            if options.stars:
                text += star_run(float(model.pvalues[raw]), options.star_levels)
            coef_parts.append(text)
            second_parts.append(_second_row_text(model, raw, options))
        if not present:
            continue

        # THE NUMBER right-aligns in a 13-wide field and the stars hang AFTER it, which is why a
        # three-star cell reaches further right than a two-star one.
        rendered: list[str] = []
        # THE `- 3` RESERVES ROOM FOR THE STAR RUN. Without stars the value fills the whole column,
        # which is why the unstarred layout measures 13 per model with no trailing slack.
        allowance = 3 if options.stars else 0
        for cell in coef_parts:
            if not cell:
                rendered.append(" " * column)
                continue
            stars_text = cell[len(cell.rstrip("*")) :] if cell.endswith("*") else ""
            number_text = cell[: len(cell) - len(stars_text)] if stars_text else cell
            unit = number_text.rjust(column - allowance) + stars_text
            rendered.append(unit + " " * (column - len(unit)))
        lines.append(f"{displayed:<{label_width}}" + "".join(rendered))

        if options.second_row is not None:
            second_cells: list[str] = []
            for cell in second_parts:
                if not cell:
                    second_cells.append(" " * column)
                    continue
                unit = cell.rjust(column - allowance)
                second_cells.append(unit + " " * (column - len(unit)))
            lines.append(" " * label_width + "".join(second_cells))
        # MEASURED: a BLANK LINE closes every coefficient block -- EXCEPT when arlabels is in
        # use, which suppresses them all:
        #     esttab m1 m2, se                     21 lines, blanks at 0,7,10,13
        #     esttab m1 m2, se varlabels(...)      18 lines, blank at 0 only
        if not options.varlabels:
            lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    lines.append(rule_text)

    # `esttab` ALWAYS prints N; `stats(...)` ADDS to it. Passing stats=["r2"] yields N and R-sq.
    # N is printed by DEFAULT, but an explicit stats=["N"] must not print it twice.
    requested_stats = [str(item).lower() for item in options.stats] or ["n"]
    if requested_stats:
        for requested in requested_stats:
            entry = STATS.get(str(requested).strip().lower())
            label = entry[0] if entry else str(requested)
            row = f"{label:<{label_width}}"
            for model in models:
                value = entry[1](model) if entry else None
                row += " " * column if value is None else (
                    esttab_number(value).rjust(column - _allowance) + " " * _allowance
                )
            lines.append(row)
        lines.append(rule_text)

    if options.second_row == "ci":
        # MEASURED: the ci form prints its own note in place of the parentheses one, BEFORE the
        # star legend.
        lines.append(f"{int(round(options.level))}% confidence intervals in brackets")
    elif options.second_row == "se":
        # THE NOTE NAMES THE DELIMITER: `brac` turns `(0.00638)` into `[0.00638]` and the note
        # follows -- 'Standard errors in brackets'.
        left, _right = options.second_row_brackets
        kind = "brackets" if left == "[" else "parentheses"
        lines.append(f"Standard errors in {kind}")
    elif options.second_row == "t":
        lines.append("t statistics in parentheses")
    elif options.second_row == "p":
        lines.append("p-values in parentheses")

    if options.stars:
        # THE MARK COUNT IS INVERSELY RELATED TO THE THRESHOLD: the SMALLEST threshold earns the
        # most stars. esttab prints the legend from the largest threshold down:
        #   '* p<0.05, ** p<0.01, *** p<0.001'
        descending = sorted((float(level) for level in options.star_levels), reverse=True)
        lines.append(
            ", ".join(
                f"{'*' * (index + 1)} p<{Decimal(str(level)).normalize():f}"
                for index, level in enumerate(descending)
            )
        )

    note_lines = set(lines[-2:]) if options.stars else set(lines[-1:])
    return "\n".join(
        line if line == "" or line in note_lines else line.ljust(len(rule_text))
        for line in lines
    )
