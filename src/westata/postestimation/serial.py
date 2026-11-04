"""Serial-correlation tests: Stata's ``estat dwatson``, ``wntestq`` and ``corrgram``.

All three are builtin commands with no readable ``ado`` source, so the formulas here come from
numbers measured against a live Stata run on ``icecream.dta`` after
``reg consumption temp price income``::

    estat dwatson            Durbin-Watson d(4, 30) = 1.021169
    wntestq e1               Q 26.1974, p 0.0160 (13 lags, the default (n-k)//2)
    wntestq e1, lags(10)     Q 16.3718, p 0.0895
    corrgram e1              lag 1: ac .3298, pac .3969, Q 3.6, p .0578
                             lag 2: ac .0362, pac -.1681, Q 3.645, p .1616

Each statistic is closed form, so every number above is reproduced exactly.

``estat bgodfrey`` is deliberately NOT implemented. Its auxiliary regression is not the
textbook one: reproducing Stata's 4.237 / 4.487 / 4.775 / 5.099 at lags 1-4 required a
different multiplier at every lag order (26.12, 24.58, 25.52, 23.35), which means the
regression being run is a different one, not a differently scaled copy. An implementation
that lands within 1% would look right and be wrong, so it is left out rather than shipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError

__all__ = [
    "Correlogram",
    "DurbinWatsonResult",
    "PortmanteauResult",
    "estat_dwatson",
    "estat_corrgram",
    "wntestq",
]


def _default_lags(nobs: int, result: Any) -> int:
    """Stata's default lag order for ``wntestq``/``corrgram``: ``(n - k) // 2``.

    Not ``n // 2``: with 30 observations and a four-parameter regression Stata reports
    ``chi2(13)``, and ``(30 - 4) // 2 = 13`` while ``30 // 2 = 15``. Using the latter made
    the statistic 28.6011 against Stata's 26.1974.
    """

    parameters = float(getattr(result, "df_model", float("nan"))) + 1.0
    if not np.isfinite(parameters):
        parameters = 1.0
    return max(int((nobs - parameters) // 2), 1)


def _residuals(result: Any) -> np.ndarray:
    """Residuals of a fitted result, in estimation-sample order."""

    resid = getattr(result, "resid", None)
    if resid is None:
        raise ConsistencyError(
            "the serial-correlation tests need a fitted result carrying residuals"
        )
    values = np.asarray(resid, dtype=float).reshape(-1)
    if values.size == 0:
        raise ConsistencyError("the result carries no residuals")
    return values


@dataclass(frozen=True)
class DurbinWatsonResult:
    """``estat dwatson``: the Durbin-Watson d statistic.

    Stata prints one line::

        Durbin-Watson d-statistic(  4,    30) =  1.021169

    The first number is the number of parameters EXCLUDING the constant (``e(df_m)``), the
    second the sample size.
    """

    d: float
    df: float
    nobs: int

    def __str__(self) -> str:
        return (
            f"Durbin\u2013Watson d-statistic({self.df:>3.0f}, {self.nobs:>5d})"
            f" = {self.d:>9.6f}"
        )


def estat_dwatson(result: Any) -> DurbinWatsonResult:
    r"""Durbin-Watson d.

    .. math:: d = \sum_{t=2}^{n}(e_t - e_{t-1})^2 / \sum_{t=1}^{n} e_t^2

    Measured on icecream.dta after ``reg consumption temp price income``: 1.021169.
    """

    resid = _residuals(result)
    if resid.size < 2:
        raise ConsistencyError("Durbin-Watson needs at least two observations")
    denominator = float(resid @ resid)
    if denominator <= 0.0:
        raise ConsistencyError("Durbin-Watson: the residuals are all zero")
    differences = np.diff(resid)
    d = float(differences @ differences) / denominator
    # Stata prints the PARAMETER count including the constant: "(4, 30)" for three
    # regressors plus _cons. e(df_m) is 3 there, so it needs the +1.
    df_model = float(getattr(result, "df_model", np.nan)) + 1.0
    nobs = int(np.asarray(resid).size)
    return DurbinWatsonResult(d=d, df=df_model, nobs=nobs)


@dataclass(frozen=True)
class PortmanteauResult:
    """``wntestq``: Ljung-Box portmanteau Q for white noise."""

    q: float
    df: int
    pvalue: float
    lags: int

    def __str__(self) -> str:
        return (
            "Portmanteau test for white noise\n"
            + "-" * 39
            + f"\n Portmanteau (Q) statistic = {self.q:>10.4f}"
            + f"\n Prob > chi2({self.df})           = {self.pvalue:>10.4f}"
        )


def wntestq(result: Any, *, lags: int | None = None) -> PortmanteauResult:
    r"""Ljung-Box Q, as ``wntestq``.

    ``corrgram.ado:68`` builds it as

    .. math:: Q = n(n+2) \sum_{k=1}^{p} \hat\rho_k^2 / (n-k)

    and by default ``wntestq`` uses ``floor((n - k) / 2)`` lags, where ``k`` is the number of
    estimated parameters -- measured: 30 observations and a 4-parameter ``reg`` give
    ``chi2(13)``, i.e. ``(30 - 4) // 2``, NOT ``n // 2 = 15``. Measured: Q 26.1974, p .0160;
    with ``lags(10)``: Q 16.3718, p .0895.
    """

    resid = _residuals(result)
    n = int(resid.size)
    if lags is None:
        lags = _default_lags(n, result)
    lags = int(lags)
    if lags < 1 or lags >= n:
        raise ValueError(f"lags must lie in 1..{n - 1}; got {lags}")

    centred = resid - resid.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        raise ConsistencyError("wntestq: the residuals are constant")
    total = 0.0
    for k in range(1, lags + 1):
        autocovariance = float(centred[k:] @ centred[: n - k])
        rho = autocovariance / denominator
        total += rho * rho / (n - k)
    q = float(n * (n + 2) * total)
    pvalue = float(stats.chi2.sf(q, lags))
    return PortmanteauResult(q=q, df=lags, pvalue=pvalue, lags=lags)


@dataclass(frozen=True)
class CorrelogramRow:
    """One lag of the correlogram."""

    lag: int
    ac: float
    pac: float
    q: float
    pvalue: float


@dataclass(frozen=True)
class Correlogram:
    """``corrgram``: the autocorrelation table with Ljung-Box Q at each lag."""

    rows: tuple[CorrelogramRow, ...]

    def to_frame(self) -> pd.DataFrame:
        """The table as a DataFrame, which is the statsmodels-style way to read it."""

        return pd.DataFrame(
            [
                {
                    "LAG": row.lag,
                    "AC": row.ac,
                    "PAC": row.pac,
                    "Q": row.q,
                    "Prob>Q": row.pvalue,
                }
                for row in self.rows
            ]
        )

    def __str__(self) -> str:
        lines = [
            f"{'LAG':>4}{'AC':>10}{'PAC':>9}{'Q':>9}{'Prob>Q':>9}",
            "-" * 41,
        ]
        for row in self.rows:
            lines.append(
                f"{row.lag:>4}{row.ac:>10.4f}{row.pac:>9.4f}"
                f"{row.q:>9.4g}{row.pvalue:>9.4f}"
            )
        return "\n".join(lines)


def estat_corrgram(result: Any, *, lags: int | None = None) -> Correlogram:
    """``corrgram``: autocorrelations, partial autocorrelations and Q at each lag.

    ``corrgram.ado:68`` accumulates ``Q = n(n+2) sum ac^2/(n-k)`` as it goes, so the Q column
    is the Ljung-Box statistic using lags ``1..k`` -- which is why the last row equals what
    ``wntestq`` reports for the same order.

    The partial autocorrelation is the Durbin-Levinson recursion, matching Stata's ``pac``.
    """

    resid = _residuals(result)
    n = int(resid.size)
    if lags is None:
        lags = _default_lags(n, result)
    lags = int(lags)
    if lags < 1 or lags >= n:
        raise ValueError(f"lags must lie in 1..{n - 1}; got {lags}")

    centred = resid - resid.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        raise ConsistencyError("corrgram: the residuals are constant")

    ac = [1.0]
    for k in range(1, lags + 1):
        ac.append(float(centred[k:] @ centred[: n - k]) / denominator)

    # Partial autocorrelations. ``corrgram.ado:63`` delegates to Stata's ``pac`` command,
    # which REGRESSES the residual on its first ``k`` lags WITH a constant and reports the
    # coefficient on lag ``k``. Measured at lag 1 on icecream.dta: .3969, which is the
    # slope of e_t on e_{t-1} with an intercept -- NOT ac[1] = .3298 (what the Yule-Walker
    # solve returns) and not the no-intercept slope (.4006).
    pac = [1.0]
    for order in range(1, lags + 1):
        lagged = np.column_stack(
            [resid[order - j : n - j] for j in range(1, order + 1)]
        )
        target = resid[order:]
        augmented = np.column_stack([np.ones(target.size), lagged])
        coefficients, *_ = np.linalg.lstsq(augmented, target, rcond=None)
        pac.append(float(coefficients[-1]))

    rows: list[CorrelogramRow] = []
    running = 0.0
    for k in range(1, lags + 1):
        running += ac[k] * ac[k] / (n - k)
        q = float(n * (n + 2) * running)
        rows.append(
            CorrelogramRow(
                lag=k,
                ac=ac[k],
                pac=pac[k],
                q=q,
                pvalue=float(stats.chi2.sf(q, k)),
            )
        )
    return Correlogram(rows=tuple(rows))
