"""Wald lag-exclusion test for a VAR: Stata's ``varwle``.

Transcribed from ``varwle.ado:70-100`` (StataNow 19.5). For each lag ``i`` the ado builds
``test [eq]L`i'.var1 [eq]L`i'.var2 ...`` per equation and then over all equations together,
so each lag yields one row per equation plus an ``All`` row::

    foreach i in `lags' {
        foreach v1 of local eqlist {
            local ltest ""
            foreach v2 of local vlist {
                local ltest "`ltest' [`v1']L`i'.`v2' "
                local btest "`btest' [`v1']L`i'.`v2' "
            }
            qui test `ltest'
            ...
        }
        qui test `btest'
    }

Because it is a Stata ``test`` on a ``var``, the Wald statistic is built from ``e(V)`` -- the
ML-scale covariance -- not from the backend's unbiased one. That single choice is the whole
difference: on macro_swatson.dta the unbiased covariance gives 764.97 / 73.35 for lags 1 / 2
while Stata reports 796.2418 / 187.9991.

Reference values from a live ``varwle`` after ``var dinf unem, lags(1/2)``::

    Equation dinf : lag 1 =  72.72164 (df 2), lag 2 =  72.9294 (df 2)
    Equation unem : lag 1 = 741.8916  (df 2), lag 2 = 123.1287 (df 2)
    Equation All  : lag 1 = 796.2418  (df 4), lag 2 = 187.9991 (df 4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from westata.errors import ConsistencyError
from westata.results.serialisation import ResultSerialisation

__all__ = ["VarWleResult", "varwle"]


@dataclass(frozen=True)
class VarWleRow:
    """One (equation, lag) cell of the ``varwle`` table."""

    equation: str
    lag: int
    chi2: float
    df: int
    pvalue: float


@dataclass(frozen=True)
class VarWleResult(ResultSerialisation):
    """``varwle``: Wald tests that each lag's coefficients are jointly zero."""

    rows: tuple[VarWleRow, ...]
    equations: tuple[str, ...]

    def by_equation(self) -> dict[str, list[VarWleRow]]:
        """The table grouped the way ``varwle`` prints it: one block per equation."""

        grouped: dict[str, list[VarWleRow]] = {name: [] for name in self.equations}
        for row in self.rows:
            grouped.setdefault(row.equation, []).append(row)
        return grouped

    def __str__(self) -> str:
        lines: list[str] = []
        for equation, rows in self.by_equation().items():
            label = "All" if equation == "All" else equation
            lines.append(f"   Equation: {label}")
            lines.append("  +------------------------------------+")
            lines.append("  | lag |    chi2      df  Prob > chi2 |")
            lines.append("  |-----+------------------------------|")
            for row in rows:
                lines.append(
                    f"  |{row.lag:>4} |{_chi2_cell(row.chi2)}{row.df:>6}"
                    f"{row.pvalue:>10.3f}    |"
                )
            lines.append("  +------------------------------------+")
            lines.append("")
        return "\n".join(lines).rstrip()


def _chi2_cell(value: float) -> str:
    """A Wald chi-square at five decimals with trailing zeros dropped, in a 10-wide field.

    Measured from a live `varwle`:

        '  |   1 |  72.72164     2     0.000    |'
        '  |   2 |   72.9294     2     0.000    |'

    72.72164 keeps all five decimals and 72.9294 keeps four -- Stata trims the trailing zero rather than
    padding to a fixed count.
    """

    # SEVEN SIGNIFICANT FIGURES, not a fixed decimal count. Measured across the whole table:
    #
    #     72.72164    7 significant figures, five decimals
    #     72.9294     7 significant figures, four decimals
    #     741.8916    7 significant figures, four decimals
    #     796.2418    7 significant figures, four decimals
    #
    # A fixed five decimals gave 741.89158 and 123.12873, both wrong.
    #
    # THE LEADING ZERO IS DROPPED, as everywhere else in Stata's tables. Measured on a 300-point sample
    # where a lag-2 statistic is 0.1923409:
    #
    #     Stata    '  |   2 |  .1923409     2     0.908    |'
    #     package  '  |   2 | 0.1923409     2     0.908    |'
    #
    # Small chi-squares are common here -- a Wald test on a lassoed lag often lands below 1 -- which is why
    # the earlier samples, all with statistics in the tens, never showed it.
    text = f"{float(value):.7g}"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text.rjust(10)


def varwle(result: Any) -> VarWleResult:
    """Wald lag-exclusion tests, as ``varwle``.

    ``result`` must be a fitted VAR exposing the Stata-ordered parameter vector and its
    covariance (``stata_params`` / ``stata_cov_params``). Using the backend's own ordering
    would silently test the wrong coefficients.
    """

    names = [str(name) for name in getattr(result, "stata_param_names", [])]
    parameters = getattr(result, "stata_params", None)
    covariance = getattr(result, "stata_cov_params", None)
    if not names or parameters is None or covariance is None:
        raise ConsistencyError(
            "varwle: the result does not expose the Stata-ordered parameters and covariance"
        )

    vector = np.asarray(parameters, dtype=float).reshape(-1)
    matrix = np.asarray(covariance, dtype=float)
    if matrix.shape != (vector.size, vector.size):
        raise ConsistencyError(
            f"varwle: covariance is {matrix.shape} for {vector.size} parameters"
        )

    lags = int(getattr(result, "lags", 0))
    if lags < 1:
        raise ConsistencyError("varwle: the VAR has no lags")
    equation_names = [str(name) for name in getattr(result, "names", [])]
    if not equation_names:
        raise ConsistencyError("varwle: the result does not expose the equation names")
    per_equation = vector.size // len(equation_names)
    if per_equation * len(equation_names) != vector.size:
        raise ConsistencyError(
            f"varwle: {vector.size} parameters do not divide evenly across "
            f"{len(equation_names)} equations"
        )

    def positions(equation_index: int, lag: int) -> list[int]:
        """Parameter positions of one lag within one equation's block."""

        # Every lag carries its digit -- including the first. This matched the old
        # ``stata_param_names``, which spelled lag 1 as "L." ; Stata prints "L1." and
        # ``var dinf unem, lags(1/2)`` shows it plainly. Fixing the names without fixing
        # this lookup would have silently matched NOTHING for lag 1.
        prefix = f"L{lag}."
        offset = equation_index * per_equation
        block = names[offset : offset + per_equation]
        return [
            offset + index
            for index, name in enumerate(block)
            if name.startswith(prefix)
        ]

    def wald(indices: list[int]) -> float:
        restricted = matrix[np.ix_(indices, indices)]
        selected = vector[indices]
        return float(selected @ np.linalg.solve(restricted, selected))

    rows: list[VarWleRow] = []
    for equation_index, equation_name in enumerate(equation_names):
        for lag in range(1, lags + 1):
            indices = positions(equation_index, lag)
            if not indices:
                continue
            statistic = wald(indices)
            rows.append(
                VarWleRow(
                    equation=equation_name,
                    lag=lag,
                    chi2=statistic,
                    df=len(indices),
                    pvalue=float(stats.chi2.sf(statistic, len(indices))),
                )
            )

    # The ``All`` row per lag pools that lag across every equation.
    for lag in range(1, lags + 1):
        combined: list[int] = []
        for equation_index in range(len(equation_names)):
            combined.extend(positions(equation_index, lag))
        if not combined:
            continue
        statistic = wald(combined)
        rows.append(
            VarWleRow(
                equation="All",
                lag=lag,
                chi2=statistic,
                df=len(combined),
                pvalue=float(stats.chi2.sf(statistic, len(combined))),
            )
        )

    return VarWleResult(rows=tuple(rows), equations=tuple(equation_names))
