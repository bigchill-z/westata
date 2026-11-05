"""Stata's ``summarize``: the six-column descriptive table.

``estat_summarize`` already existed, but it renders the ESTIMATION-SAMPLE variant (no ``Obs``
column, headed "Estimation sample regress"). The plain ``summarize`` command -- one of Stata's
most-used -- had no entry point at all, so ``summarize resid`` after ``predict`` could not be
expressed (report cases C05-08 and C09-04 both say "you have to run summarize yourself").

Reference from a live ``summarize k l`` on cobb_douglas.dta::

        Variable |        Obs        Mean    Std. dev.       Min        Max
    -------------+---------------------------------------------------------
               k |         24    234.1667    106.2666        100        431
               l |         24    145.7917    29.61636        100        200

Every row is 71 characters with the pipe at column 13, and the columns end at 25 (Obs),
39 (Mean), 51 (Std. dev.), 62 (Min) and 71 (Max). Values use Stata's ``%9.0g`` (seven
significant digits, no leading zero), so ``106.2666`` and ``29.61636`` sit side by side with
different decimal counts -- a fixed precision would get one of them wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

__all__ = ["SummarizeTable", "summarize"]

#: Column ends, measured by re.finditer on Stata's own DATA row: Obs 25, Mean 37,
#: Std. dev. 49, Min 60, Max 71. The header's words end on the same columns.
_COLUMN_ENDS = (25, 37, 49, 60, 71)


def _stata_number(value: float) -> str:
    """Stata's ``%9.0g``: widest representation that fits, no leading zero."""

    if value is None or not np.isfinite(value):
        return "."
    for precision in range(9, 0, -1):
        text = f"%.{precision}g" % value
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
        body = text[1:] if text.startswith("-") else text
        if len(body) <= 8:
            break
    return text


@dataclass(frozen=True)
class SummarizeTable:
    """The ``summarize`` table."""

    rows: list[tuple[str, float, float, float, float, float]]

    def __str__(self) -> str:
        # The header's words and the data's values end on DIFFERENT columns for "Std. dev."
        # -- 50 in the header against 49 in the data -- while the other four coincide at
        # 25 / 37 / 60 / 71. Measured with re.finditer on each of Stata's own lines. Using
        # one tuple for both put "Std. dev." one column left of Stata's.
        _header_ends = (25, 37, 50, 60, 71)
        header_pieces: list[str] = [f"{'Variable':>12} |"]
        for word, end in zip(("Obs", "Mean", "Std. dev.", "Min", "Max"), _header_ends):
            used = sum(len(piece) for piece in header_pieces)
            header_pieces.append(" " * max(end - used - len(word), 0) + word)
        lines = [
            "".join(header_pieces),
            "-" * 13 + "+" + "-" * 57,
        ]
        for name, count, mean, deviation, minimum, maximum in self.rows:
            # Each field ends on Stata's column; the numbers themselves use %9.0g, so the
            # decimal counts differ from row to row and column to column.
            pieces = [f"{str(name):>12} |"]
            for text, end in zip(
                (
                    f"{count:.0f}",
                    _stata_number(mean),
                    _stata_number(deviation),
                    _stata_number(minimum),
                    _stata_number(maximum),
                ),
                _COLUMN_ENDS,
            ):
                used = sum(len(piece) for piece in pieces)
                pieces.append(" " * max(end - used - len(text), 0) + text)
            lines.append("".join(pieces))
        return chr(10).join(lines)


def summarize(
    data: Any,
    variables: Sequence[str] | None = None,
    *,
    weights: Any = None,
) -> SummarizeTable:
    """Stata's ``summarize`` over ``data`` (a DataFrame, Series or array).

    ``variables`` names the columns to report; omitted, every column is included. Missing
    values are excluded per variable, exactly as Stata's ``Obs`` column reports the number of
    non-missing observations.
    """

    if isinstance(data, pd.Series):
        frame = data.to_frame()
    elif isinstance(data, pd.DataFrame):
        frame = data
    else:
        values = np.asarray(data, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        frame = pd.DataFrame(values, columns=[f"x{index + 1}" for index in range(values.shape[1])])

    names = list(variables) if variables is not None else list(frame.columns)
    weight_values = None
    if weights is not None:
        weight_values = np.asarray(weights, dtype=float).reshape(-1)
        if weight_values.size != len(frame):
            raise ValueError(
                f"summarize: {weight_values.size} weights for {len(frame)} rows"
            )

    rows: list[tuple[str, float, float, float, float, float]] = []
    for name in names:
        if name not in frame.columns:
            raise KeyError(f"summarize: column {name!r} not in the data")
        column = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(column)
        if weight_values is not None:
            keep &= np.isfinite(weight_values) & (weight_values > 0.0)
        values = column[keep]
        if values.size == 0:
            rows.append((str(name), 0.0, float("nan"), float("nan"), float("nan"), float("nan")))
            continue
        if weight_values is None:
            count = float(values.size)
            mean = float(values.mean())
            deviation = float(values.std(ddof=1)) if values.size > 1 else 0.0
        else:
            weights_kept = weight_values[keep]
            total = float(weights_kept.sum())
            count = total
            mean = float((weights_kept * values).sum() / total)
            if total > 1.0:
                deviation = float(
                    np.sqrt((weights_kept * (values - mean) ** 2).sum() / (total - 1.0))
                )
            else:
                deviation = 0.0
        rows.append(
            (
                str(name),
                count,
                mean,
                deviation,
                float(values.min()),
                float(values.max()),
            )
        )
    return SummarizeTable(rows=rows)
