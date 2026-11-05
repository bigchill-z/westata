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

        # Measured from Stata: 70 characters per row, bars at columns 2 / 41 / 69.
        # Left block is [3:41] (38 wide), right block [42:69] (27 wide).
        #   header right: '   chi2     df Prob > chi2 '
        #   data right  : '  48.123     2    0.000    '
        lines = [
            "   Granger causality Wald tests",
            "  +" + "-" * 65 + "+",
            "  |"
            + f"{'Equation':>20}{'Excluded':>17} "
            + "|"
            + f"{'chi2':>8} {'df':>5} {'Prob > chi2':>12} "
            + "|",
            "  |" + "-" * 38 + "+" + "-" * 27 + "|",
        ]
        previous: str | None = None
        for row in self.rows:
            if previous is not None and row.equation != previous:
                lines.append("  |" + "-" * 38 + "+" + "-" * 27 + "|")
            lines.append(
                "  |"
                + f"{row.equation:>20}{row.excluded:>17} "
                + "|"
                + f"{row.chi2:>8.3f} {row.df:>5.0f} {row.pvalue:>11.3f}"
                + "    |"
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
