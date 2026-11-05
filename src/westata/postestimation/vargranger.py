"""Stata's ``vargranger``: Granger causality Wald tests as a boxed table.

Reference from a live ``var dinf unem, lags(1/2)`` then ``vargranger`` on macro_swatson.dta::

       Granger causality Wald tests
      +------------------------------------------------------------------+
      |          Equation           Excluded |   chi2     df Prob > chi2 |
      |--------------------------------------+---------------------------|
      |              dinf               unem |  48.123     2    0.000    |
      |              dinf                ALL |  48.123     2    0.000    |
      |--------------------------------------+---------------------------|
      |              unem               dinf |   9.009     2    0.011    |
      |              unem                ALL |   9.009     2    0.011    |
      +------------------------------------------------------------------+

GEOMETRY, measured rather than derived: every table row is 70 characters, the outer bars sit
at columns 2 and 67, and the inner separator bar at 40.

STRUCTURE: for each equation, one row per excluded variable in the VAR's own order, then an
"ALL" row re-testing them jointly. With two variables there is one excluded column plus ALL,
which is why the block appears twice.

PARAMETER SEMANTICS, measured against Stata: ``granger_causality(a, b)`` reports the test in
equation ``a`` excluding ``b``. Stata's row "Equation=dinf / Excluded=unem -> 48.123" is
reproduced by ``granger_causality("dinf", "unem")``, so the first argument names the EQUATION
even though the parameter is called ``causing``. Swapping them silently exchanges 48.123 and
9.009 -- both are plausible magnitudes, so the mistake is invisible without the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["VarGrangerResult", "vargranger"]


@dataclass(frozen=True)
class _Row:
    equation: str
    excluded: str
    chi2: float
    df: int
    pvalue: float


@dataclass(frozen=True)
class VarGrangerResult:
    """The ``vargranger`` table."""

    rows: list[_Row]
    equations: tuple[str, ...]

    def __str__(self) -> str:
        """Stata's boxed table, to the character."""

        # Every row is 70 characters. Positions measured with repr() on Stata's own output:
        #   '  |          Equation           Excluded |   chi2     df Prob > chi2 |'
        #   '  |              dinf               unem |  48.123     2    0.000    |'
        #
        #   column 2  = '|'        column 41 = '|'        column 69 = '|'
        #   left  block [3:41]   = 38 chars: Equation ends at 21, Excluded at 40
        #   right block [42:69]  = 27 chars: chi2 ends at 50, df at 56, p at 68
        #
        # Fields are placed by ABSOLUTE column rather than by accumulating widths; the latter
        # kept drifting because each block's own trailing space belongs to a different field.

        # Layout taken from Stata's own lines, character by character:
        #   '  |          Equation           Excluded |   chi2     df Prob > chi2 |'
        #   '  |              dinf               unem |  48.123     2    0.000    |'
        #   '  |--------------------------------------+---------------------------|'
        #   '  +-----------------------------------------------------------------+'
        # Bars at 2, 41 and 69; the separator bar "|" at 41 splits 38 + 27.
        _LEFT, _RIGHT = 38, 27

        # Stata's own template, kept verbatim so the alignment cannot drift again:
        #   '  |          Equation           Excluded |   chi2     df Prob > chi2 |'
        # Fields are placed by ABSOLUTE column, computed once from that line:
        _TEMPLATE = "  |          Equation           Excluded |   chi2     df Prob > chi2 |"
        _left_end = 41   # the bar at column 41
        _right_end = 70  # rows are 70 characters; the closing bar sits at index 69

        def _place(equation: str, excluded: str, chi2: str, df: str, p: str) -> str:
            # Column ends taken from the template: equation 20, excluded 40,
            # chi2 50, df 56, p 68.
            row = [" "] * _right_end
            # Column ends read directly off Stata's lines (index+len from the template):
            # Equation 21, Excluded 40, chi2 50, df 56, p 65.
            for text, end in (
                (equation, 21),
                (excluded, 40),
                (chi2, 50 if len(chi2) > 4 else 49),
                (df, 56),
                (p, 65 if len(p) <= 5 else 68),
            ):
                start_at = end - len(text)
                for offset, character in enumerate(text):
                    row[start_at + offset] = character
            row[2] = "|"
            row[_left_end] = "|"
            row[_right_end - 1] = "|"  # column 69
            return "".join(row)

        lines = [
            "   Granger causality Wald tests",
            "  +" + "-" * 65 + "+",
            _place("Equation", "Excluded", "chi2", "df", "Prob > chi2"),
            "  |" + "-" * 38 + "+" + "-" * 27 + "|",
        ]
        previous: str | None = None
        for row in self.rows:
            if previous is not None and row.equation != previous:
                lines.append("  |" + "-" * 38 + "+" + "-" * 27 + "|")
            lines.append(
                _place(
                    row.equation,
                    row.excluded,
                    f"{row.chi2:.3f}",
                    f"{row.df:.0f}",
                    f"{row.pvalue:.3f}",
                )
            )
            previous = row.equation
        lines.append("  +" + "-" * 65 + "+")
        return chr(10).join(lines)


def vargranger(fitted: Any) -> VarGrangerResult:
    """Build the ``vargranger`` table for a fitted ``VARResults``."""

    names = [str(name) for name in fitted.names]
    rows: list[_Row] = []
    for equation in names:
        others = [name for name in names if name != equation]
        for excluded in others:
            outcome = fitted.granger_causality(equation, excluded)
            rows.append(
                _Row(
                    equation=equation,
                    excluded=excluded,
                    chi2=float(outcome.test_statistic),
                    df=int(outcome.df),
                    pvalue=float(outcome.pvalue),
                )
            )
        if len(others) > 1:
            outcome = fitted.granger_causality(equation, others)
            rows.append(
                _Row(
                    equation=equation,
                    excluded="ALL",
                    chi2=float(outcome.test_statistic),
                    df=int(outcome.df),
                    pvalue=float(outcome.pvalue),
                )
            )
        else:
            # With one excluded variable the ALL row repeats it exactly, as Stata prints.
            last = rows[-1]
            rows.append(
                _Row(
                    equation=equation,
                    excluded="ALL",
                    chi2=last.chi2,
                    df=last.df,
                    pvalue=last.pvalue,
                )
            )
    return VarGrangerResult(rows=rows, equations=tuple(names))
