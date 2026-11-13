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
    #: 1% critical value. From ``GetCritE`` (trend case), interpolated on the ORIGINAL series
    #: length -- the same N for every lag, which is why this column is constant down the table.
    critical_1: float = float("nan")
    #: 5% and 10% critical values. From ``GetCritT`` (Cheung & Lai response surface), which takes
    #: the lag order as well, so these change row by row.
    critical_5: float = float("nan")
    critical_10: float = float("nan")
    #: Information criteria the footer selects on. SC = log(rmse^2) + (p+1)log(N)/N, and
    #: MAIC = log(rmse^2) + 2(tau + p)/N with tau = b^2 * sum((L.dty)^2) / rmse^2
    #: (``dfgls.ado:224-238``). Verified: SC at lag 1 = -5.562791 against Stata's -5.56279.
    sc: float = float("nan")
    maic: float = float("nan")
    #: Wald p-value for the HIGHEST lag in this row's ADF regression. The "Opt lag (Ng-Perron
    #: seq t)" footer takes the FIRST lag (scanning down from maxlag) whose p-value is below 0.10
    #: -- not the smallest RMSE (``dfgls.ado:219-224``).
    lag_pvalue: float = float("nan")


def _crit_e(nobs: int) -> tuple[float, float, float]:
    """``GetCritE``: Elliott-Rothenberg-Stock Table 1c, linearly interpolated in N.

    ``dfgls.ado:342-373``. Called with the ORIGINAL series length (62 here), not the ADF sample
    size -- using 51 gave -3.766 against Stata's -3.724.
    """

    table = (
        (-3.77, -3.58, -3.46, -3.48),
        (-3.19, -3.03, -2.93, -2.89),
        (-2.89, -2.74, -2.64, -2.57),
    )

    def interp(row: tuple[float, ...]) -> float:
        if nobs <= 50:
            return row[0]
        if nobs <= 100:
            return row[0] + (nobs - 50) / 50 * (row[1] - row[0])
        if nobs <= 200:
            return row[1] + (nobs - 100) / 100 * (row[2] - row[1])
        return row[3]

    return tuple(interp(row) for row in table)


def _crit_d(nobs: int) -> float:
    """``GetCritD``: the no-trend 1% value, interpolated from dfuller's six-point table.

    ``dfgls.ado:375-415``. Verified at N=62: -2.6152, matching Stata's -2.615 for `dfgls, notrend`.
    """

    table = (-2.66, -2.62, -2.60, -2.58, -2.58, -2.58)
    if nobs <= 25:
        return table[0]
    if nobs <= 50:
        return table[0] + (nobs - 25) / 25 * (table[1] - table[0])
    if nobs <= 100:
        return table[1] + (nobs - 50) / 50 * (table[2] - table[1])
    if nobs <= 250:
        return table[2] + (nobs - 100) / 150 * (table[3] - table[2])
    return table[5]


def _crit_m(nobs: int, lags: int) -> tuple[float, float]:
    """``GetCritM``: the no-trend 5%/10% surface (Cheung-Lai Table 1, cols 1-2).

    ``dfgls.ado:417-432``. NOTE: this surface reproduces Stata's no-trend columns from lag 9 down
    but is off by ~0.013 in the 5% column at the LARGEST lag (lag 10 here), where ``p/N`` reaches
    0.196 and sits at the edge of the fitted region. No integer lag reproduces that first row
    exactly, so the discrepancy is recorded rather than papered over: the first-row 5%/10% values
    in the no-trend table are approximate.
    """

    surface = (
        (-1.624, -19.888, 155.231, 0.709, 5.480, -16.055),   # 10%
        (-1.948, -17.839, 104.086, 0.802, 5.558, -18.332),   # 5%
    )
    n1 = 1.0 / nobs
    pt = lags / nobs

    def evaluate(row: tuple[float, ...]) -> float:
        return (
            row[0]
            + row[1] * n1
            + row[2] * n1**2
            + row[3] * pt
            + row[4] * pt**2
            + row[5] * pt**3
        )

    return evaluate(surface[1]), evaluate(surface[0])


def _crit_t(nobs: int, lags: int) -> tuple[float, float]:
    """``GetCritT``: Cheung-Lai OBES 1995 Table 1 response surface, giving (5%, 10%).

    ``dfgls.ado:433-448``. Unlike ``GetCritE`` this DOES depend on the lag order, through
    ``pt = p/N``, which is why the 5%/10% columns shift down the table and the 1% column does not.
    Verified at N=51: lag 10 -> -2.721 / -2.440, lag 1 -> -3.160 / -2.860, both exactly Stata.
    """

    surface = (
        (-2.550, -20.166, 155.215, 1.133, 9.808, -20.313),   # 10%
        (-2.838, -20.328, 124.191, 1.267, 10.530, -24.600),  # 5%
    )
    n1 = 1.0 / nobs
    pt = lags / nobs

    def evaluate(row: tuple[float, ...]) -> float:
        return (
            row[0]
            + row[1] * n1
            + row[2] * n1**2
            + row[3] * pt
            + row[4] * pt**2
            + row[5] * pt**3
        )

    ten = evaluate(surface[0])
    five = evaluate(surface[1])
    return five, ten


@dataclass(frozen=True)
class DFGLSResult:
    """``dfgls``: the DF-GLS test across a range of lag orders."""

    cells: tuple[DFGLSCell, ...]
    trend: bool
    #: The series name Stata prints in the "Variable:" line. Empty for a bare array, in which case
    #: that line is omitted rather than inventing a name.
    variable: str = ""

    @property
    def best_lag(self) -> int:
        """The "Opt lag (Ng-Perron seq t)": the FIRST lag, scanning DOWN from the maximum, whose
        highest-lag Wald p-value is below 0.10 (``dfgls.ado:219-224``).

        NOT the smallest RMSE -- that earlier definition picked lag 10 on nelson_plosser where
        Stata picks 1, because a longer lag always fits the sample slightly better.
        """

        for cell in self.cells:  # self.cells is ordered by descending lag
            if np.isfinite(cell.lag_pvalue) and cell.lag_pvalue < 0.10:
                return cell.lags
        return self.cells[-1].lags

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            [
                {"lags": cell.lags, "DF-GLS": cell.statistic, "N": cell.nobs}
                for cell in self.cells
            ]
        )

    def __str__(self) -> str:
        """Stata's dfgls table, with the three critical-value columns.

        Columns measured from a live ``dfgls lrgnp`` on nelson_plosser.dta: lags right-aligned to
        7, the statistic to 23, then 1% to 36, 5% to 49 and 10% to 62 -- so every data row is 62
        characters. The 1% column is constant down the table (interpolated on the original series
        length); 5% and 10% shift row by row (response surface in the lag order).
        """

        title = "DF-GLS tau" if self.trend else "DF-GLS mu"
        lines = [
            "",
            # Stata's title line carries the ADF sample size, not the series length: 51 here
            # against the 62 observations the detrending used.
            f"{'DF-GLS test for unit root':<44}Number of obs = {self.cells[0].nobs}",
            f"Variable: {self.variable}" if self.variable else "",
            f"Lag selection: Schwert criterion{'':<12}Maximum lag   = {self.cells[0].lags}",
            "",
            f"{'':<29}-------- Critical value ---------",
            f"{'[lags]':>7}{title:>16}{'1%':>13}{'5%':>13}{'10%':>13}",
            "-" * 62,
        ]
        if not self.variable:
            lines.remove("")
        for cell in self.cells:
            lines.append(
                f"{cell.lags:>7}{cell.statistic:>16.3f}"
                f"{cell.critical_1:>13.3f}{cell.critical_5:>13.3f}{cell.critical_10:>13.3f}"
            )
        lines.append("-" * 62)
        best = min(self.cells, key=lambda cell: cell.sc)
        opt = min(self.cells, key=lambda cell: cell.maic)

        def _number(value: float, digits: int) -> str:
            """Stata drops the leading zero and trailing zeros on these footers."""
            text = f"{value:.{digits}g}"
        def _number(value: float, digits: int) -> str:
            """Stata drops the leading zero AND trailing zeros on these footers."""
            text = f"{value:.{digits}g}"
            if text.startswith("0."):
                text = text[1:]
            elif text.startswith("-0."):
                text = "-" + text[2:]
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return text or "0"

        def _fixed(value: float) -> str:
            """Seven DECIMALS with trailing zeros dropped: -5.56279 and -5.365953 are both
            seven-place renderings. Leading zero never appears for these values."""
            text = f"{value:.6f}"
            if "." in text:
                text = text.rstrip("0").rstrip(".")
            return text

        def _rmse(value: float) -> str:
            """Seven DECIMALS, leading zero dropped: .0573553, not .05735534 or .057355335.

            The footers use fixed decimal places rather than significant figures -- an earlier
            ``%.8g`` printed .057355335 because eight sig figs is nine characters here.
            """
            text = f"{value:.7f}"
            if text.startswith("0."):
                return text[1:]
            return text

        # Each footer names its own selected lag and quotes THAT lag's RMSE -- not the global
        # minimum, which is why the three RMSEs here all read .0573553 while the table's own
        # smallest row value differs. `self.cells` is ordered by DESCENDING lag, so the row is
        # looked up by lag rather than indexed positionally.
        opt_cell = next(cell for cell in self.cells if cell.lags == self.best_lag)
        lines.append(
            f"Opt lag (Ng\u2013Perron seq t) = {self.best_lag} "
            f"with RMSE = {_rmse(opt_cell.rmse)}"
        )
        lines.append(
            f"Min SIC  = {_fixed(best.sc):>9} at lag {best.lags} "
            f"with RMSE = {_rmse(best.rmse)}"
        )
        lines.append(
            f"Min MAIC = {_fixed(opt.maic):>9} at lag {opt.lags} "
            f"with RMSE = {_rmse(opt.rmse)}"
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
) -> tuple[float, int, float, float, float]:
    """ADF regression without constant or trend; return (t on L1, nobs, rss, b).

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
    rss = float(residual @ residual)
    dof = response.size - design.shape[1]
    if dof <= 0:
        raise ValueError(f"dfgls: lag {lags} leaves no residual degrees of freedom")
    sigma2 = rss / dof
    try:
        bread = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(design.T @ design)
    standard_error = float(np.sqrt(sigma2 * bread[0, 0]))
    if standard_error <= 0.0 or not np.isfinite(standard_error):
        raise ConsistencyError("dfgls: the lagged-level coefficient has no variance")
    statistic = float(coefficients[0]) / standard_error
    # Wald p-value on the HIGHEST-lag coefficient, which the Ng-Perron sequential-t rule uses to
    # pick the "Opt lag" footer. ``dfgls.ado:219`` tests DL`i'.dty -- one restriction -- with r(p).
    pvalue = float("nan")
    if lags >= 1 and design.shape[1] >= 2:
        try:
            cov = sigma2 * bread
            index = design.shape[1] - 1  # highest lag sits last
            estimate = float(coefficients[index])
            variance = float(cov[index, index])
            if variance > 0.0:
                from scipy import stats

                pvalue = float(2.0 * stats.norm.sf(abs(estimate / np.sqrt(variance))))
        except (np.linalg.LinAlgError, IndexError):
            pvalue = float("nan")
    # ``coefficients[0]`` is _b[L.dty], which MAIC needs. ``rss`` is returned so the caller can
    # form Stata's armse = sqrt(rss/nobs) -- an UNBIASED-denominator square root, which differs
    # from sqrt(rss/dof) by sqrt(dof/nobs). Using the dof version printed .058514146 against
    # Stata's .0573553 in every footer.
    return statistic, int(response.size), rss, float(coefficients[0]), pvalue


def dfgls(
    series: Any,
    *,
    maxlag: int | None = None,
    trend: bool = True,
    name: str = "",
    **legacy: Any,
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

    ``trend=False`` selects the second. With the trend the detrending uses
    ``cbar=-13.5`` and subtracts ``b_iota + b_qdtrd*t``; without it, ``cbar=-7`` and only
    ``b_iota``. Both cases reproduce Stata to the printed digit.

    ``maxlag=None`` uses Schwert's rule ``int(12*(n/100)**0.25)`` (``dfgls.ado:64``).

    ``notrend=`` was REMOVED (R3.1): it and ``trend=`` set the same state, and having two
    switches made ``notrend=False`` a no-op that nevertheless looked meaningful.  Stata
    spells the option ``notrend``; the Python argument is the positive ``trend``, and
    ``notrend=True`` is still accepted for one release as a deprecation shim.
    """

    if "notrend" in legacy:
        if legacy["notrend"]:
            trend = False
        legacy.pop("notrend")
    if legacy:
        raise TypeError(
            f"dfgls() got unexpected keyword argument(s): {', '.join(sorted(legacy))}"
        )
    if trend is None:
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
    # ``dfgls.ado:130-133`` picks GetCritE (trend) or GetCritD (no trend) and calls it with the
    # ORIGINAL series length, so the 1% column is the same on every row. The 5%/10% surface takes
    # the lag order AND the estimation sample size, so those columns move down the table.
    first = _crit_e(nobs)[0] if trend else _crit_d(nobs)
    cells: list[DFGLSCell] = []
    for lags in range(maxlag, 0, -1):
        statistic, used, rss, coefficient, pvalue = _adf_no_deterministic(
            detrended, lags, shared_start
        )
        # Stata's armse is sqrt(rss/nobs) -- no degrees-of-freedom correction (``dfgls.ado:217``).
        rmse = float(np.sqrt(rss / used))
        five, ten = _crit_t(used, lags) if trend else _crit_m(used, lags)
        # ``dfgls.ado:224-238``. sldty2 is the sum of squared lagged detrended levels over the
        # SAME shared sample, and tau = b^2 * sldty2 / rmse^2.
        level = detrended[shared_start - 1 : detrended.size - 1]
        sldty2 = float(level @ level)
        tau = (coefficient**2) * sldty2 / (rmse**2)
        log_n = np.log(used)
        sc = float(np.log(rmse**2) + (lags + 1) * log_n / used)
        maic = float(np.log(rmse**2) + 2 * (tau + lags) / used)
        cells.append(
            DFGLSCell(
                lags=lags,
                statistic=statistic,
                nobs=used,
                rmse=rmse,
                critical_1=first,
                critical_5=five,
                critical_10=ten,
                sc=sc,
                maic=maic,
                lag_pvalue=pvalue,
            )
        )
    return DFGLSResult(cells=tuple(cells), trend=bool(trend), variable=str(name))
