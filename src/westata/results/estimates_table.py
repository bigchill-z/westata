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

__all__ = ["render_estimates_table"]

#: Width of "    Variable | " -- the label gutter every row starts with.
_GUTTER = 14
#: Characters per model column.
_COLUMN_WIDTH = 13
#: Names strictly longer than this are abbreviated.
_NAME_LIMIT = 10


def _abbreviate(name: str) -> str:
    """Stata's column-name abbreviation: eight characters, "~", then the last one."""

    text = str(name)
    if len(text) <= _NAME_LIMIT:
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


def render_estimates_table(
    results: Sequence[Any],
    *,
    model_names: Sequence[str] | None = None,
    se: bool = True,
) -> str:
    """Reproduce ``estimates table m1 m2 ..., b se``.

    ``model_names`` is the equivalent of the ``estimates store`` names: Stata prints whatever the
    models were stored as, abbreviated to ten characters where needed.
    """

    models = list(results)
    if not models:
        raise ValueError("estimates table: at least one model is required")

    names = (
        [_abbreviate(name) for name in model_names]
        if model_names is not None
        else [f"m{index + 1}" for index in range(len(models))]
    )
    if len(names) != len(models):
        raise ValueError(
            f"estimates table: {len(names)} names for {len(models)} models"
        )

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
    rule_length = _GUTTER + _COLUMN_WIDTH * len(models)
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
        header += _centre(name, _COLUMN_WIDTH - 2) + " " * 2
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
                row += " " * _COLUMN_WIDTH
                continue
            present = True
            row += f"{_number(float(model.params[match[0]])):>{_COLUMN_WIDTH - 2}}" + " " * 2
        if not present:
            continue
        lines.append(row)
        if se:
            error_row = " " * 12 + " |"
            for model in models:
                match = [
                    raw
                    for raw in model.params.index
                    if _label(raw) == label
                ]
                if not match:
                    error_row += " " * _COLUMN_WIDTH
                    continue
                error_row += f"{_number(float(model.bse[match[0]])):>{_COLUMN_WIDTH - 2}}" + " " * 2
            lines.append(error_row)

    lines.append("-" * rule_length)
    if se:
        lines.append(f"{'Legend: b/se':>{rule_length}}")
    # Stata right-pads every line, including the last, to the rule width.
    # The leading blank stays empty; every OTHER line is padded to the rule width.
    return "\n".join(
        line if line == "" else line.ljust(rule_length) for line in lines
    )
