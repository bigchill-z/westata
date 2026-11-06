"""KPSS test for stationarity: Stata's ``kpss``.

Stata has no readable ``kpss.ado`` (it is builtin), but its output makes the definition
unambiguous -- and differs from statsmodels' in two ways that change every number.

Reference on macro_swatson.dta's ``inf`` column (n = 167)::

    KPSS test for inf
    Maxlag = 4 chosen by Schwert criterion
    Autocovariances weighted by Bartlett kernel
    Critical values for H0: inf is trend stationary
    Lag order    Test statistic
        0           2.14
        1           1.16
        2           .811
        3           .625
        4           .514

Two differences from the obvious implementation:

1. **The default is TREND stationary, not level stationary.** A bare ``kpss varname`` prints
   "H0: inf is trend stationary" and uses the ``ct`` critical values; ``notrend`` asks for the
   level case. statsmodels defaults to ``'c'``, so a bare wrapper answers a different question
   and reports 0.3197 where Stata's lag-0 value is 2.14.

2. **Stata reports a ROW PER LAG ORDER**, not a single number. The table runs from 0 to the
   KPSS bandwidth ``int(4*(n/100)**0.25)`` maximum (4 here), and each row re-weights the
   autocovariances with its own Bartlett bandwidth. The default statistic is the maximum-lag
   row, but printing only that hides which lag was used.

The statistic itself is the standard KPSS:

.. math::

    \\eta = \\frac{1}{n^2} \\sum_t S_t^2 \\Big/ \\hat\\sigma^2,
    \\qquad S_t = \\sum_{s \\le t} \\hat e_s

with ``\\hat e`` the residuals of a regression on a constant (level) or on a constant and a
trend, and ``\\hat\\sigma^2`` the Bartlett-weighted long-run variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from westata.errors import ConsistencyError

__all__ = ["KpssResult", "KpssRow", "kpss"]

#: Critical values for the level-stationary (``notrend``) case, Stata's own table.
_CRITICAL_LEVEL = {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739}
#: Critical values for the trend-stationary (default) case.
_CRITICAL_TREND = {0.10: 0.119, 0.05: 0.146, 0.025: 0.176, 0.01: 0.216}


@dataclass(frozen=True)
class KpssRow:
    """One lag order of the ``kpss`` table."""

    lag: int
    statistic: float


@dataclass(frozen=True)
class KpssResult:
    """``kpss``: KPSS test for stationarity, one row per lag order."""

    rows: tuple[KpssRow, ...]
    trend: bool
    nobs: int
    critical: dict[float, float]
    #: The series name Stata prints in the title and H0 line. Empty means a bare array was passed,
    #: in which case the generic wording is kept rather than inventing a variable name.
    name: str = ""
    #: Whether ``maxlag`` was supplied by the caller. Stata prints "Maxlag = 4 chosen by Schwert
    #: criterion" only for the automatic choice; an explicit ``maxlag(4)`` prints just "Maxlag = 4".
    maxlag_explicit: bool = False

    @property
    def statistic(self) -> float:
        """The reported statistic, which Stata takes at the MAXIMUM lag."""

        return self.rows[-1].statistic

    @property
    def maxlag(self) -> int:
        return self.rows[-1].lag

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            [{"lag": row.lag, "statistic": row.statistic} for row in self.rows]
        ).set_index("lag")

    def __str__(self) -> str:
        """Stata's ``kpss`` output, captured byte-for-byte.

        Two details that were wrong before round 146, both measured from live runs on
        macro_swatson.dta (``kpss inf`` and ``kpss inf, maxlag(4)``):

        * the Maxlag line says "chosen by Schwert criterion" ONLY when the lag count was chosen
          automatically; an explicit ``maxlag(4)`` prints just ``Maxlag = 4``;
        * the statistics are THREE significant figures with no leading zero (``2.14``, ``1.16``,
          ``.811``), not four decimals (``0.0146`` was three sig figs too, not four dp).

        The critical values also sit on ONE compact line, not four indented ones.
        """

        kind = "trend stationary" if self.trend else "level stationary"
        subject = self.name or "series"
        maxlag_line = f"Maxlag = {self.maxlag}"
        if not self.maxlag_explicit:
            maxlag_line += " chosen by Schwert criterion"
        labels = ((0.10, "10%"), (0.05, "5% "), (0.025, "2.5%"), (0.01, "1% "))
        critical_line = "  ".join(
            f"{label}: {self.critical[level]:.3f}" for level, label in labels
        )
        # Stata's blank lines are a SINGLE SPACE, not an empty string -- captured as len 1.
        blank = " "
        lines = [
            blank,
            f"KPSS test for {self.name}" if self.name else f"KPSS test for {kind}",
            blank,
            maxlag_line,
            "Autocovariances weighted by Bartlett kernel",
            blank,
            f"Critical values for H0: {subject} is {kind}",
            blank,
            critical_line,
            blank,
            "Lag order    Test statistic",
        ]
        for row in self.rows:
            # Three significant figures, leading zero dropped: 2.14 -> "2.14", .811 -> ".811".
            text = f"{row.statistic:.3g}"
            if text.startswith("0."):
                text = text[1:]
            elif text.startswith("-0."):
                text = "-" + text[2:]
            lines.append(f"{row.lag:>5}{text:>15}")
        return "\n".join(lines)


def _residuals(values: np.ndarray, trend: bool) -> np.ndarray:
    """Residuals of the regression on a constant, or on a constant and a trend."""

    size = values.size
    columns = [np.ones(size)]
    if trend:
        columns.append(np.arange(1, size + 1, dtype=float))
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def _long_run_variance(residual: np.ndarray, lags: int) -> float:
    """Bartlett-weighted long-run variance at ``lags``."""

    size = residual.size
    total = float(residual @ residual) / size
    for depth in range(1, lags + 1):
        weight = 1.0 - depth / (lags + 1.0)
        total += 2.0 * weight * float(residual[depth:] @ residual[: size - depth]) / size
    return total


def kpss(
    values: Any,
    *,
    trend: bool = True,
    maxlag: int | None = None,
    name: str = "",
) -> KpssResult:
    """KPSS test for stationarity, as ``kpss varname``.

    ``trend=True`` is Stata's DEFAULT (``trend stationary``); pass ``trend=False`` for its
    ``notrend`` option. ``maxlag=None`` uses KPSS's rule, ``int(4*(n/100)**0.25)``. Supplying
    ``maxlag`` explicitly also changes the printed Maxlag line, because Stata only says "chosen by
    Schwert criterion" for the automatic choice.

    ``name`` is optional: Stata prints the variable name in the title and H0 line, but a bare
    ndarray has no name, so the generic wording is kept when it is omitted.
    """

    series = np.asarray(values, dtype=float).reshape(-1)
    series = series[np.isfinite(series)]
    if series.size < 10:
        raise ValueError(f"kpss: at least 10 observations are required; got {series.size}")
    nobs = int(series.size)
    maxlag_explicit = maxlag is not None
    if maxlag is None:
        # KPSS uses a coefficient of FOUR, not the twelve ADF and PP use:
        # int(4*(n/100)**0.25) = 4 for n = 167, where twelve gives 13. Using 12 printed
        # thirteen rows where Stata prints five.
        maxlag = int(4 * (nobs / 100.0) ** 0.25)
    maxlag = int(maxlag)
    if maxlag < 0:
        raise ValueError(f"kpss: maxlag must be non-negative; got {maxlag}")

    residual = _residuals(series, trend)
    cumulative = np.cumsum(residual)
    numerator = float(cumulative @ cumulative) / (nobs * nobs)

    rows: list[KpssRow] = []
    for lags in range(0, maxlag + 1):
        variance = _long_run_variance(residual, lags)
        if variance <= 0.0:
            raise ConsistencyError(
                f"kpss: the long-run variance at lag {lags} is not positive"
            )
        rows.append(KpssRow(lag=lags, statistic=numerator / variance))

    critical = dict(_CRITICAL_TREND if trend else _CRITICAL_LEVEL)
    return KpssResult(
        rows=tuple(rows),
        trend=bool(trend),
        nobs=nobs,
        critical=critical,
        name=str(name),
        maxlag_explicit=maxlag_explicit,
    )
