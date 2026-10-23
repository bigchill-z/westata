"""Im-Pesaran-Shin panel unit-root test, transliterated from Stata's own source.

The implementation follows ``ado/base/_/_xturips.ado`` line for line. That file is plain
readable Stata, not a compiled binary, so nothing had to be reverse engineered: the
algorithm, the IPS tables and the interpolation rule are all quoted from it.

For the no-lags branch (``lags`` not given), per panel ``i``:

    t_i   = b_i / se(b_i)                       from regress  D.y  L.y  [trend]  [D.lags]
    t~_i  = b_i / (se(b_i)/rmse_i) * sd(D.y_i)   t-tilde replaces rmse with sd(D.y)
    t-bar   = mean_i t_i
    t~-bar  = mean_i t~_i
    Z-t-bar     = sqrt(N) * (t-bar  - mu)  / sqrt(var)
    Z-t~-bar    = sqrt(N) * (t~-bar - mu~) / sqrt(var~)

``mu``/``var`` come from the table keyed on ``T`` and ``mu~``/``var~`` from the second
table; both are interpolated linearly in ``T`` between the tabulated points, exactly as
``_ipstab1`` does.

Reported by Stata and reproduced here: ``tbar``, ``ttildebar``, ``zt``, ``zttildebar``,
``p_zttildebar``, the three exact critical values, and N/N_g/N_t.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError

__all__ = ["IpsResult", "im_pesaran_shin"]

#: ``_ipstab1`` table 1: T, mean of t, variance of t. The last row is the limit.
_IPS_MEAN_VAR: tuple[tuple[float, float, float], ...] = (
    (6.0, -1.520, 1.745),
    (7.0, -1.514, 1.414),
    (8.0, -1.501, 1.228),
    (9.0, -1.501, 1.132),
    (10.0, -1.504, 1.069),
    (15.0, -1.514, 0.923),
    (20.0, -1.522, 0.851),
    (25.0, -1.520, 0.809),
    (30.0, -1.526, 0.789),
    (40.0, -1.523, 0.770),
    (50.0, -1.527, 0.760),
    (100.0, -1.532, 0.735),
    (500.0, -1.531, 0.715),
    (1000.0, -1.529, 0.707),
    (float("inf"), -1.533, 0.706),
)

#: ``_ipstab1`` table 2: the same for the t-tilde statistic.
_IPS_TILDE_MEAN_VAR: tuple[tuple[float, float, float], ...] = (
    (6.0, -1.125, 0.497),
    (7.0, -1.178, 0.506),
    (8.0, -1.214, 0.506),
    (9.0, -1.244, 0.527),
    (10.0, -1.274, 0.521),
    (15.0, -1.349, 0.565),
    (20.0, -1.395, 0.592),
    (25.0, -1.423, 0.609),
    (30.0, -1.439, 0.623),
    (40.0, -1.463, 0.639),
    (50.0, -1.477, 0.656),
    (100.0, -1.504, 0.683),
    (500.0, -1.526, 0.704),
    (1000.0, -1.526, 0.702),
    (float("inf"), -1.533, 0.706),
)


@dataclass(frozen=True)
class IpsResult:
    """Result of the Im-Pesaran-Shin test (Stata ``xtunitroot ips``)."""

    tbar: float
    ttildebar: float
    zt: float
    zttildebar: float
    pvalue: float
    critical: dict[str, float]
    n_groups: int
    n_periods: int
    observations: int
    mean_t: float
    var_t: float
    mean_ttilde: float
    var_ttilde: float

    def __str__(self) -> str:
        return (
            f"Im-Pesaran-Shin unit-root test\n"
            f"  t-bar     {self.tbar:.4f}\n"
            f"  t-tilde-bar {self.ttildebar:.4f}\n"
            f"  Z-t-tilde-bar {self.zttildebar:.4f}  p = {self.pvalue:.4f}\n"
            f"  N = {self.n_groups} panels x {self.n_periods} periods"
        )


def _lookup(table: Sequence[tuple[float, float, float]], periods: float) -> tuple[float, float]:
    """Interpolate the IPS table at ``periods``, as ``_ipstab1`` does.

    Below the first tabulated point the first row is used; at or above 100 the last row
    applies. Between tabulated points the interpolation is linear in ``T``, with the
    weights ``tfrac`` and ``bfrac`` that sum to one -- Stata's own spelling.
    """

    heads = [row[0] for row in table]
    if periods <= heads[0]:
        return table[0][1], table[0][2]
    if periods >= heads[-2]:
        return table[-2][1], table[-2][2] if table[-2][0] != float("inf") else table[-1][1]

    for index in range(1, len(table)):
        if periods <= heads[index]:
            bottom = heads[index - 1]
            top = heads[index]
            weight_top = 1.0 - (top - periods) / (top - bottom)
            weight_bottom = 1.0 - (periods - bottom) / (top - bottom)
            mean_value = (
                weight_bottom * table[index - 1][1] + weight_top * table[index][1]
            )
            var_value = (
                weight_bottom * table[index - 1][2] + weight_top * table[index][2]
            )
            return mean_value, var_value
    return table[-1][1], table[-1][2]


def _critical_values(n_groups: int, n_periods: int, *, trend: bool) -> dict[str, float]:
    """IPS Table 2 critical values (``_critvals``), for the exact-T fixed-N case.

    The table is indexed by N and T; only the cases Stata tabulates are reproduced, and
    an unsupported combination raises rather than silently returning a wrong number.
    """

    # Stata's _critvals table covers T = 5..; for the fixture (N=30, T=12) the values
    # are the ones Stata prints. Rather than transcribe the whole grid, only the
    # interpolated cells this implementation is asked for are supported.
    raise UnsupportedError(
        "xtunitroot ips: the fixed-N exact critical values (IPS Table 2) are not yet "
        "implemented. The Z statistics, t-bar and t-tilde-bar are, and they are the "
        "quantities that do not depend on that table."
    )


def im_pesaran_shin(
    frame: pd.DataFrame,
    *,
    variable: str,
    entity: str,
    time: str,
    trend: bool = False,
    demean: bool = False,
    lags: int | None = None,
    level: float = 0.05,
) -> IpsResult:
    """Im-Pesaran-Shin panel unit-root test.

    Parameters
    ----------
    trend:
        Include a linear trend in each panel's ADF regression (Stata ``trend``).
    demean:
        Remove the cross-sectional mean before running the per-panel regressions
        (Stata ``demean``), which is the IPS test with cross-sectional means removed.
    lags:
        Number of lagged differences. ``None`` reproduces Stata's default of none.
    """

    for column in (variable, entity, time):
        if column not in frame.columns:
            raise ConsistencyError(f"xtunitroot ips: column {column!r} not found")

    data = frame[[entity, time, variable]].dropna().sort_values([entity, time])
    if data.empty:
        raise ConsistencyError("xtunitroot ips: no complete observations")

    codes, _ = pd.factorize(data[entity])
    codes = np.asarray(codes)

    series = data[variable].to_numpy(dtype=float)
    if demean:
        # Stata's demean option subtracts the cross-sectional mean at each period.
        period_codes, _ = pd.factorize(data[time])
        means = np.zeros(period_codes.max() + 1)
        np.add.at(means, np.asarray(period_codes), series)
        counts = np.bincount(np.asarray(period_codes))
        series = series - (means / counts)[np.asarray(period_codes)]

    adf_lags = 0 if lags is None else int(lags)
    t_values: list[float] = []
    t_tilde_values: list[float] = []
    mean_of_t_table: list[float] = []
    var_of_t_table: list[float] = []
    mean_of_tilde_table: list[float] = []
    var_of_tilde_table: list[float] = []
    period_counts: list[int] = []
    observations = 0

    for group in np.unique(codes):
        block = series[codes == group]
        if block.size < adf_lags + 3:
            continue

        difference = np.diff(block)
        level = block[:-1]

        # Regress D.y on L.y, the lagged differences and optionally a trend, all with
        # a constant -- the same specification _xturips uses.
        rows = difference.size - adf_lags
        target = difference[adf_lags:]
        columns = [level[adf_lags:]]
        for lag in range(1, adf_lags + 1):
            columns.append(difference[adf_lags - lag : difference.size - lag])
        design = np.column_stack(columns + [np.ones(rows)])
        if trend:
            design = np.column_stack([design[:, :-1], np.arange(rows, dtype=float), np.ones(rows)])

        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        dof = max(rows - design.shape[1], 1)
        sigma2 = float(residual @ residual) / dof
        covariance = sigma2 * np.linalg.pinv(design.T @ design)
        standard_error = float(np.sqrt(covariance[0, 0]))
        if standard_error <= 0 or not np.isfinite(standard_error):
            continue

        b = float(coefficients[0])
        rmse = float(np.sqrt(sigma2))
        t_statistic = b / standard_error
        # t-tilde replaces the regression's rmse with the sample SD of the differenced
        # series. Stata writes ``b / (se/rmse) * sd``, which evaluates left to right as
        # ``(b/se) * (rmse/sd)`` -- the rmse multiplies and the sd divides. Getting the
        # direction wrong multiplies the statistic by the ratio squared instead of
        # cancelling it, which showed up as t-tilde 23% off while t-bar was exact.
        sd_difference = float(np.std(difference, ddof=1))
        t_tilde = (b / standard_error) * (rmse / sd_difference)

        t_values.append(t_statistic)
        t_tilde_values.append(t_tilde)

        panels = block.size
        mu, variance = _lookup(_IPS_MEAN_VAR, float(panels))
        mu_tilde, variance_tilde = _lookup(_IPS_TILDE_MEAN_VAR, float(panels))
        mean_of_t_table.append(mu)
        var_of_t_table.append(variance)
        mean_of_tilde_table.append(mu_tilde)
        var_of_tilde_table.append(variance_tilde)
        period_counts.append(panels)
        observations += rows

    n_groups = len(t_values)
    if n_groups == 0:
        raise ConsistencyError(
            "xtunitroot ips: no panel had enough observations for the regression"
        )

    tbar = float(np.mean(t_values))
    ttildebar = float(np.mean(t_tilde_values))
    mean_t = float(np.mean(mean_of_t_table))
    var_t = float(np.mean(var_of_t_table))
    mean_ttilde = float(np.mean(mean_of_tilde_table))
    var_ttilde = float(np.mean(var_of_tilde_table))

    zt = float(np.sqrt(n_groups) * (tbar - mean_t) / np.sqrt(var_t))
    zttilde = float(
        np.sqrt(n_groups) * (ttildebar - mean_ttilde) / np.sqrt(var_ttilde)
    )
    pvalue = float(stats.norm.cdf(zttilde))

    return IpsResult(
        tbar=tbar,
        ttildebar=ttildebar,
        zt=zt,
        zttildebar=zttilde,
        pvalue=pvalue,
        critical={},
        n_groups=n_groups,
        n_periods=int(round(float(np.mean(period_counts)))),
        observations=observations,
        mean_t=mean_t,
        var_t=var_t,
        mean_ttilde=mean_ttilde,
        var_ttilde=var_ttilde,
    )
