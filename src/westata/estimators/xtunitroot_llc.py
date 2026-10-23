"""Levin-Lin-Chu panel unit-root test, transliterated from Stata's own source.

Follows ``ado/base/_/_xturllc.ado``. As with IPS, the file is plain readable Stata, so
the three-step procedure, the finite-sample adjustment tables and the HAC bandwidth rule
are all quoted rather than inferred.

The procedure
-------------
Per panel ``i``, on ``T_i`` periods:

1. Auxiliary ADF regression of ``D.y`` on ``L.y`` (plus lagged differences, plus a trend),
   obtaining ``delta_i`` (the coefficient on ``L.y``) and ``p_i`` lags.

2. Two "orthogonalising" regressions that strip the deterministic part:
       ``D.y`` on [lagged ``D.y``, trend]  -> residual ``e_it``
       ``L.y`` on [lagged ``D.y``, trend]  -> residual ``nu_it``

3. Regress ``e`` on ``nu`` with no constant, take ``rss``, and set
       ``mrssdf = rss / (T_i - p_i - 1)``
   then standardise ``e`` and ``nu`` by ``sqrt(mrssdf)``.

The long-run variance of ``D.y`` comes from a Bartlett-kernel HAC estimator with
``int(3.21 * T^(1/3))`` lags by default, giving ``s_i = sqrt(sigma_i^2 / mrssdf)``.

Pooled statistics
-----------------
A single common AR parameter is then estimated from the stacked standardised series::

    nu_hat = regress e on nu, noconstant        (over all standardised observations)
    nu_se  = se(nu_hat) * sqrt((n-1)/n)
    td     = nu_hat / nu_se
    tds    = (td - sqrt(N)*sbar/Var_ep * nu_se * muadj) / sadj

where ``sbar`` is the mean of ``s_i``, ``Var_ep`` the residual variance of that last
regression, and ``muadj``/``sadj`` come from the finite-sample adjustment tables keyed on
the AVERAGE T and the deterministic specification.

``tds`` is the adjusted statistic; ``p_tds = Phi(tds)`` is the p-value Stata prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError

__all__ = ["LlcResult", "levin_lin_chu"]

#: ``_get_adj`` tables: model 1 = no constant, 2 = constant, 3 = constant + trend.
#: Each row is (T, mu adjustment, sigma adjustment); the last row is the limit.
_ADJUSTMENT: dict[int, tuple[tuple[float, float, float], ...]] = {
    1: (
        (25.0, 0.004, 1.049),
        (30.0, 0.003, 1.035),
        (35.0, 0.002, 1.027),
        (40.0, 0.002, 1.021),
        (45.0, 0.001, 1.017),
        (50.0, 0.001, 1.014),
        (60.0, 0.001, 1.011),
        (70.0, 0.000, 1.008),
        (80.0, 0.000, 1.007),
        (90.0, 0.000, 1.006),
        (100.0, 0.000, 1.005),
        (250.0, 0.000, 1.001),
        (float("inf"), 0.000, 1.000),
    ),
    2: (
        (25.0, -0.554, 0.919),
        (30.0, -0.546, 0.889),
        (35.0, -0.541, 0.867),
        (40.0, -0.537, 0.850),
        (45.0, -0.533, 0.837),
        (50.0, -0.531, 0.826),
        (60.0, -0.527, 0.810),
        (70.0, -0.524, 0.798),
        (80.0, -0.521, 0.789),
        (90.0, -0.520, 0.782),
        (100.0, -0.518, 0.776),
        (250.0, -0.509, 0.742),
        (float("inf"), -0.500, 0.707),
    ),
    3: (
        (25.0, -0.703, 1.003),
        (30.0, -0.674, 0.949),
        (35.0, -0.653, 0.906),
        (40.0, -0.637, 0.971),
        (45.0, -0.624, 0.842),
        (50.0, -0.614, 0.818),
        (60.0, -0.598, 0.780),
        (70.0, -0.587, 0.751),
        (80.0, -0.578, 0.728),
        (90.0, -0.571, 0.710),
        (100.0, -0.566, 0.695),
        (250.0, -0.533, 0.603),
        (float("inf"), -0.500, 0.500),
    ),
}


@dataclass(frozen=True)
class LlcResult:
    """Result of the Levin-Lin-Chu test (Stata ``xtunitroot llc``)."""

    tds: float
    td: float
    ttilde: float
    sbar: float
    nu_hat: float
    nu_se: float
    var_ep: float
    mu_adj: float
    sig_adj: float
    pvalue_tds: float
    pvalue_td: float | None
    n_groups: int
    n_periods: int
    observations: int
    adf_lags: int
    hac_lags: int

    def __str__(self) -> str:
        return (
            f"Levin-Lin-Chu unit-root test\n"
            f"  Unadjusted t   {self.td:.4f}\n"
            f"  Adjusted t*    {self.tds:.4f}   p = {self.pvalue_tds:.4f}\n"
            f"  {self.n_groups} panels x {self.n_periods} periods"
        )


def _adjustment(model: int, periods: float) -> tuple[float, float]:
    """Interpolate the finite-sample adjustment table in T."""

    table = _ADJUSTMENT[model]
    heads = [row[0] for row in table]
    if periods <= heads[0]:
        return table[0][1], table[0][2]
    if periods >= heads[-2]:
        return table[-2][1], table[-2][2]

    for index in range(1, len(table)):
        if periods <= heads[index]:
            bottom = heads[index - 1]
            top = heads[index]
            weight_top = 1.0 - (top - periods) / (top - bottom)
            weight_bottom = 1.0 - (periods - bottom) / (top - bottom)
            mu = weight_bottom * table[index - 1][1] + weight_top * table[index][1]
            sigma = weight_bottom * table[index - 1][2] + weight_top * table[index][2]
            return mu, sigma
    return table[-1][1], table[-1][2]


def _bartlett_variance(values: np.ndarray, lags: int) -> float:
    """Bartlett-kernel long-run variance, as ``_gethacvar`` computes it."""

    values = np.asarray(values, dtype=float).reshape(-1)
    n = values.size
    if n == 0:
        return float("nan")
    variance = float(values @ values) / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(values[lag:] @ values[:-lag]) / n
        variance += 2.0 * weight * covariance
    return variance


def _regress(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS returning coefficients and residuals."""

    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients, target - design @ coefficients


def levin_lin_chu(
    frame: pd.DataFrame,
    *,
    variable: str,
    entity: str,
    time: str,
    trend: bool = False,
    constant: bool = True,
    demean: bool = False,
    lags: int | None = None,
    hac_lags: int | None = None,
) -> LlcResult:
    """Levin-Lin-Chu panel unit-root test.

    ``constant`` and ``trend`` choose the deterministic specification, matching Stata's
    ``noconstant`` and ``trend`` options: neither gives model 1, constant only model 2 (the
    default), both model 3.
    """

    for column in (variable, entity, time):
        if column not in frame.columns:
            raise ConsistencyError(f"xtunitroot llc: column {column!r} not found")

    data = frame[[entity, time, variable]].dropna().sort_values([entity, time])
    if data.empty:
        raise ConsistencyError("xtunitroot llc: no complete observations")

    codes = np.asarray(pd.factorize(data[entity])[0])
    series = data[variable].to_numpy(dtype=float)

    if demean:
        period_codes = np.asarray(pd.factorize(data[time])[0])
        sums = np.zeros(period_codes.max() + 1)
        np.add.at(sums, period_codes, series)
        counts = np.bincount(period_codes)
        series = series - (sums / counts)[period_codes]

    model = 1 if not constant else (3 if trend else 2)
    # LLC's default is ONE lagged difference, unlike IPS which defaults to none.
    # [_xturllc.ado line 32: local adf_lags = 1]. Getting this wrong shifts the per-panel
    # sample by one observation, so N came out 330 against Stata's 300 while every other
    # shape number still looked plausible.
    adf_lags = 1 if lags is None else int(lags)

    pooled_e: list[np.ndarray] = []
    pooled_nu: list[np.ndarray] = []
    s_values: list[float] = []
    t_values: list[float] = []
    adf_used: list[int] = []
    hac_used: list[int] = []
    panel_sizes: list[int] = []
    raw_periods: list[int] = []
    total_observations = 0

    for group in np.unique(codes):
        block = series[codes == group]
        if block.size < adf_lags + 4:
            continue
        difference = np.diff(block)
        level = block[:-1]

        # Step 1: the auxiliary ADF regression, which supplies the lag count and the
        # per-panel t statistic Stata reports as ttilde.
        rows = difference.size - adf_lags
        target = difference[adf_lags:]
        columns = [level[adf_lags:]]
        for lag in range(1, adf_lags + 1):
            columns.append(difference[adf_lags - lag : difference.size - lag])
        extras = []
        if constant:
            extras.append(np.ones(rows))
        if trend:
            extras.append(np.arange(1, rows + 1, dtype=float))
        design = np.column_stack(columns + extras)
        coefficients, residual = _regress(design, target)
        dof = max(rows - design.shape[1], 1)
        sigma2 = float(residual @ residual) / dof
        covariance = sigma2 * np.linalg.pinv(design.T @ design)
        se = float(np.sqrt(covariance[0, 0]))
        if se > 0 and np.isfinite(se):
            t_values.append(float(coefficients[0]) / se)
        adf_used.append(adf_lags)
        # tivar in Stata's source: each panel's own regression sample size.
        panel_sizes.append(rows)
        raw_periods.append(block.size)
        total_observations += rows

        # Step 2: strip the deterministic part from D.y and from L.y separately.
        period_count = block.size

        def deterministic(length: int) -> list[np.ndarray]:
            pieces = []
            if constant:
                pieces.append(np.ones(length))
            if trend:
                pieces.append(np.arange(1, length + 1, dtype=float))
            return pieces

        # D.y on [lagged D.y, deterministic]
        index_start = adf_lags
        dy = difference[index_start:]
        length = dy.size
        lagged_terms = [
            difference[index_start - lag : difference.size - lag]
            for lag in range(1, adf_lags + 1)
        ]
        design_dy = np.column_stack(lagged_terms + deterministic(length))
        _, e_residual = _regress(design_dy, dy)

        # L.y on [lagged D.y, deterministic]
        lv = level[index_start:]
        design_lv = np.column_stack(
            [term[: length] for term in lagged_terms] + deterministic(length)
        )
        _, nu_residual = _regress(design_lv, lv)

        # Step 3: regress e on nu without a constant to set the scale.
        _, scale_residual = _regress(nu_residual.reshape(-1, 1), e_residual)
        mrssdf = float(scale_residual @ scale_residual) / max(
            period_count - adf_lags - 1, 1
        )
        if mrssdf <= 0:
            continue

        root = np.sqrt(mrssdf)
        pooled_e.append(e_residual / root)
        pooled_nu.append(nu_residual / root)

        # HAC long-run variance of the deterministically adjusted D.y.
        if model == 1:
            adjusted = dy
        elif model == 2:
            adjusted = dy - dy.mean()
        else:
            trend_design = np.column_stack(deterministic(length))
            _, adjusted = _regress(trend_design, dy)

        # Stata counts the panel's RAW periods here, not the post-differencing length:
        # int(3.21 * 12^(1/3)) = 7, which is the 'Bartlett kernel, 7.00 lags average'
        # printed in its output. Using the differenced length (11) gives 6 and moves
        # sbar by about 1%.
        bandwidth = (
            int(hac_lags)
            if hac_lags is not None
            else int(3.21 * block.size ** (1.0 / 3.0))
        )
        hac_used.append(bandwidth)
        sigma_i2 = _bartlett_variance(adjusted, bandwidth)
        s_values.append(float(np.sqrt(max(sigma_i2, 0.0) / mrssdf)))

    if not pooled_e:
        raise ConsistencyError(
            "xtunitroot llc: no panel had enough observations for the procedure"
        )

    e_all = np.concatenate(pooled_e)
    nu_all = np.concatenate(pooled_nu)
    n_groups = len(pooled_nu)
    n_observations = e_all.size

    # The pooled common-AR regression, no constant.
    coefficients, residual = _regress(nu_all.reshape(-1, 1), e_all)
    nu_hat = float(coefficients[0])
    rss = float(residual @ residual)
    var_ep = rss / n_observations
    nu_se = float(np.sqrt(var_ep / float(nu_all @ nu_all)))
    nu_se *= float(np.sqrt((n_observations - 1) / n_observations))
    td = nu_hat / nu_se

    sbar = float(np.mean(s_values))

    # Stata's r(ttilde) is the MEAN PER-PANEL REGRESSION SIZE, not a t statistic: it is
    # sum tivar / N_g where tivar holds each panel's e(N). For 12 raw periods that is 10,
    # because differencing costs one and the lag costs another.
    ttilde = float(np.mean(t_values)) if t_values else float("nan")
    mean_panel_n = float(np.mean(panel_sizes)) if panel_sizes else 0.0

    # _get_adj is keyed on that same mean, not on the raw period count.
    muadj, sadj = _adjustment(model, mean_panel_n)
    tds = (td - np.sqrt(n_groups) * sbar / var_ep * nu_se * muadj) / sadj
    pvalue_tds = float(stats.norm.cdf(tds))
    pvalue_td = float(stats.norm.cdf(td)) if model == 1 else None

    return LlcResult(
        tds=float(tds),
        td=float(td),
        ttilde=ttilde,
        sbar=sbar,
        nu_hat=nu_hat,
        nu_se=nu_se,
        var_ep=var_ep,
        mu_adj=muadj,
        sig_adj=sadj,
        pvalue_tds=pvalue_tds,
        pvalue_td=pvalue_td,
        n_groups=n_groups,
        # N_t is the RAW periods per panel (Stata: r(N)/N_g on the untransformed data).
        n_periods=int(round(float(np.mean(raw_periods)))) if raw_periods else 0,
        observations=total_observations,
        adf_lags=int(round(float(np.mean(adf_used)))) if adf_used else 0,
        hac_lags=int(round(float(np.mean(hac_used)))) if hac_used else 0,
    )
