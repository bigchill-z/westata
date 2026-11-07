"""DF-GLS unit-root test: Stata's ``dfgls``.

Transcribed from ``dfgls.ado:60-215`` (StataNow 19.5), which is readable source -- unlike
``dfuller``/``pperron``/``kpstest``, whose cores are builtin.

The algorithm, in the order the ado does it:

1. ``cbar`` is ``-13.5`` with a trend and ``-7.0`` without one, and the quasi-difference
   parameter is ``alpha = 1 + cbar / nobs`` (``dfgls.ado:74-83``).
2. The series is quasi-differenced -- the first observation uses the level, later ones
   ``y_t - alpha y_{t-1}`` -- and regressed on a likewise transformed constant (and trend)
   WITHOUT a constant of its own. That gives ``dty``, the GLS-demeaned/detrended series.
3. The ADF regression is run on ``dty`` with **no constant and no trend**:
   ``reg D.dty L.dty DL(1/i).dty`` (``dfgls.ado:207-209``).
4. The statistic is the t ratio on ``L.dty``.

Reference from a live ``dfgls inf, maxlag(4) notrend`` on macro_swatson.dta (nobs 162)::

    [lags]   DF-GLS mu
        4      -1.715
        3      -1.744
        2      -1.434
        1      -2.067

with ``inf`` on the latest-lag sample of 162 observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from westata.errors import ConsistencyError

__all__ = ["DFGLSResult", "dfgls"]


@dataclass(frozen=True)
class DFGLSCell:
    """One lag order of the ``dfgls`` table."""

    lags: int
    statistic: float
    nobs: int
    rmse: float


@dataclass(frozen=True)
class DFGLSResult:
    """``dfgls``: the DF-GLS test across a range of lag orders."""

    cells: tuple[DFGLSCell, ...]
    trend: bool

    @property
    def best_lag(self) -> int:
        """The lag with the smallest residual mean square, as the ado's ``Min SIC`` does."""

        return min(self.cells, key=lambda cell: cell.rmse).lags

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            [
                {"lags": cell.lags, "DF-GLS": cell.statistic, "N": cell.nobs}
                for cell in self.cells
            ]
        )

    def __str__(self) -> str:
        title = "DF-GLS tau" if self.trend else "DF-GLS mu"
        lines = [
            "DF-GLS test for unit root",
            "",
            f"{'[lags]':>9}{title:>15}",
            "-" * 30,
        ]
        for cell in self.cells:
            lines.append(f"{cell.lags:>9}{cell.statistic:>15.3f}")
        lines.append("-" * 30)
        lines.append(
            f"Opt lag (Ng\u2013Perron seq t) = {self.best_lag} with RMSE = {min(c.rmse for c in self.cells):.7g}"
        )
        return "\n".join(lines)


def _gls_detrend(series: np.ndarray, nobs: int, trend: bool) -> np.ndarray:
    """GLS-demeaned (or detrended) series, as ``dfgls.ado:83-118``.

    The quasi-difference uses ``alpha = 1 + cbar/nobs`` with ``cbar`` -7 (no trend) or -13.5
    (trend); the transformed series is regressed on the transformed deterministic terms with
    ``noconstant``, and the residual at the ORIGINAL scale (``y - b_iota - b_trd*t``) becomes
    the object of the ADF regression -- note it is the OLS residual of the fitted
    coefficients, not the GLS residual, which ``dfgls.ado:109-117`` says explicitly.
    """

    cbar = -13.5 if trend else -7.0
    alpha = 1.0 + cbar / nobs
    size = series.size

    quasi_y = np.empty(size)
    quasi_iota = np.empty(size)
    quasi_trd = np.empty(size) if trend else None

    quasi_y[0] = series[0]
    quasi_iota[0] = 1.0
    if trend:
        quasi_trd[0] = 1.0
    for position in range(1, size):
        quasi_y[position] = series[position] - alpha * series[position - 1]
        quasi_iota[position] = 1.0 - alpha
        if trend:
            step = float(position + 1)
            quasi_trd[position] = step - alpha * (step - 1.0)

    columns = [quasi_iota]
    if trend:
        columns.append(quasi_trd)
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, quasi_y, rcond=None)

    steps = np.arange(1, size + 1, dtype=float)
    if trend:
        return series - coefficients[0] - coefficients[1] * steps
    return series - coefficients[0]


def _adf_no_deterministic(
    detrended: np.ndarray, lags: int, start: int
) -> tuple[float, int, float]:
    """ADF regression without constant or trend; return (t on L1, nobs, rmse).

    ``start`` is the FIRST usable position in ``detrended`` and is shared by every lag
    order, because ``dfgls.ado:137`` marks the sample out with ``L(1/maxlag+1)`` before
    looping. Letting each order use its own start shifts the smaller-lag rows: it gave
    -1.751 and -2.090 at lags 3 and 1 against Stata's -1.744 and -2.067.
    """

    if lags < 0:
        raise ValueError(f"lags must be non-negative; got {lags}")
    size = detrended.size
    if start >= size:
        raise ValueError(f"dfgls: lag {lags} leaves no observations")
    differences = np.diff(detrended)
    response = differences[start - 1 :]
    level = detrended[start - 1 : size - 1]
    columns = [level]
    for depth in range(1, lags + 1):
        low = start - 1 - depth
        columns.append(differences[low : low + response.size])
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    dof = response.size - design.shape[1]
    if dof <= 0:
        raise ValueError(f"dfgls: lag {lags} leaves no residual degrees of freedom")
    sigma2 = float(residual @ residual) / dof
    try:
        bread = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(design.T @ design)
    standard_error = float(np.sqrt(sigma2 * bread[0, 0]))
    if standard_error <= 0.0 or not np.isfinite(standard_error):
        raise ConsistencyError("dfgls: the lagged-level coefficient has no variance")
    statistic = float(coefficients[0]) / standard_error
    return statistic, int(response.size), float(np.sqrt(sigma2))


def dfgls(
    series: Any,
    *,
    maxlag: int | None = None,
    trend: bool = True,
    notrend: bool = False,
) -> DFGLSResult:
    """DF-GLS unit-root test, as ``dfgls varname [, maxlag(#) notrend]``.

    **Stata defaults to a TREND, not to no trend.** ``dfgls.ado:39-46`` sets ``trend=1`` unless
    the user writes ``notrend``; the earlier version of this function defaulted to ``False`` and
    so implemented Stata's ``notrend`` case while being compared against Stata's default output.
    That single inverted default is the whole reason rounds 119-123 spent five rounds seeing
    tau +0.806 against Stata's -1.284 and concluding the algorithm was unexplainable.

    Measured on nelson_plosser.dta's lrgnp (both from live Stata):

        dfgls lrgnp             lag 10  tau = -1.284   (default, trend)
        dfgls lrgnp, notrend    lag 10  tau = +0.806   (the no-trend case)

    ``trend=False`` or ``notrend=True`` selects the second. With the trend the detrending uses
    ``cbar=-13.5`` and subtracts ``b_iota + b_qdtrd*t``; without it, ``cbar=-7`` and only
    ``b_iota``. Both cases reproduce Stata to the printed digit.

    ``maxlag=None`` uses Schwert's rule ``int(12*(n/100)**0.25)`` (``dfgls.ado:64``).
    """

    if notrend:
        trend = False
    elif trend is None:
        trend = True

    values = np.asarray(series, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 10:
        raise ValueError(f"dfgls: at least 10 observations are required; got {values.size}")
    nobs = int(values.size)
    if maxlag is None:
        maxlag = int(12 * (nobs / 100.0) ** 0.25)
    maxlag = int(maxlag)
    if maxlag < 1:
        raise ValueError(f"dfgls: maxlag must be at least 1; got {maxlag}")

    # ``dfgls.ado:84`` sets alpha from e(N) of the ORIGINAL series, before any lag-related
    # sample loss, so the detrending uses the full length.
    detrended = _gls_detrend(values, nobs, trend)

    # ``dfgls.ado:137``: one shared sample, set by the LONGEST lag (+1 for the differenced
    # left-hand side), so every row of the table is estimated on the same observations.
    shared_start = maxlag + 1
    cells: list[DFGLSCell] = []
    for lags in range(maxlag, 0, -1):
        statistic, used, rmse = _adf_no_deterministic(detrended, lags, shared_start)
        cells.append(DFGLSCell(lags=lags, statistic=statistic, nobs=used, rmse=rmse))
    return DFGLSResult(cells=tuple(cells), trend=bool(trend))
