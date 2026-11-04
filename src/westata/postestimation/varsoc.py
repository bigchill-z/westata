"""Lag-order selection for a VAR: Stata's ``varsoc``.

Stata's reference values on ``macro_swatson.dta`` (``tsset``, ``varsoc dinf unem,
maxlag(4)``, N = 162) are::

     Lag |    LL      LR      df    p     FPE       AIC      HQIC      SBIC
    ----+-----------------------------------------------------------------
       0 |  881.581                     6.6e-08   -10.859  -10.8435  -10.8209
       1 |   1141.4  519.63    4  0.000 2.8e-09  -14.0172  -13.9708  -13.9029
       2 |  1213.86  144.93*   4  0.000 1.2e-09* -14.8625* -14.7851* -14.6719*
       3 |  1217.32  6.9224    4  0.140 1.2e-09  -14.8558  -14.7475   -14.589
       4 |  1219.45  4.2537    4  0.373 1.2e-09  -14.8327  -14.6934  -14.4896

The detail that decides every number: **all candidate lag orders are evaluated on the SAME
estimation sample**, namely the ``n - maxlag`` observations left after fitting the LONGEST
lag. Evaluating each order on ``n - p`` observations instead (the obvious reading) gives
-10.8815 / -14.8104 for p = 0 / 2 against Stata's -10.859 / -14.8625 -- right trend, wrong
values. With the shared sample all four criteria reproduce exactly.

The criteria are, with ``Sigma`` the ML residual covariance (divide by T, not T - k), ``k``
the number of regressors per equation and ``K`` the number of equations::

    AIC  = -2 lnL / T + 2 k K / T
    HQIC = -2 lnL / T + 2 ln(ln T) k K / T
    SBIC = -2 lnL / T + ln(T) k K / T
    FPE  = ((T + kK + 1) / T)^K * det(Sigma)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError

__all__ = ["VarSocResult", "varsoc"]


@dataclass(frozen=True)
class VarSocRow:
    """One lag order of the ``varsoc`` table."""

    lag: int
    ll: float
    lr: float
    lr_df: int
    lr_pvalue: float
    fpe: float
    aic: float
    hqic: float
    sbic: float


@dataclass(frozen=True)
class VarSocResult:
    """``varsoc``: lag-order selection criteria for a VAR."""

    rows: tuple[VarSocRow, ...]
    nobs: int
    equations: int
    sample_start: int
    sample_end: int

    def to_frame(self) -> pd.DataFrame:
        """The table as a DataFrame."""

        return pd.DataFrame(
            [
                {
                    "lag": row.lag,
                    "LL": row.ll,
                    "LR": row.lr,
                    "df": row.lr_df,
                    "p": row.lr_pvalue,
                    "FPE": row.fpe,
                    "AIC": row.aic,
                    "HQIC": row.hqic,
                    "SBIC": row.sbic,
                }
                for row in self.rows
            ]
        ).set_index("lag")

    @property
    def optimal(self) -> int:
        """The lag that minimises all four criteria, if they agree."""

        minima = {
            row.lag
            for row in self.rows
            if row.aic == min(r.aic for r in self.rows)
            and row.sbic == min(r.sbic for r in self.rows)
            and row.hqic == min(r.hqic for r in self.rows)
            and row.fpe == min(r.fpe for r in self.rows)
        }
        return min(minima) if minima else -1

    def __str__(self) -> str:
        lines = [
            "Lag-order selection criteria",
            "",
            f"{'Lag':>3} |{'LL':>10}{'LR':>9}{'df':>6}{'p':>8}"
            f"{'FPE':>10}{'AIC':>10}{'HQIC':>10}{'SBIC':>10}",
            "-" * 4 + "+" + "-" * 72,
        ]
        best_aic = min(row.aic for row in self.rows)
        best_hqic = min(row.hqic for row in self.rows)
        best_sbic = min(row.sbic for row in self.rows)
        best_fpe = min(row.fpe for row in self.rows)
        best_ll = max(row.ll for row in self.rows)
        best_lr = max(row.lr for row in self.rows if row.lr > 0)

        def mark(value: float, best: float, digits: int = 4) -> str:
            text = f"{value:.{digits}g}".rjust(10)
            return text + ("*" if value == best else " ")

        for row in self.rows:
            lines.append(
                f"{row.lag:>3} |"
                + (f"{row.ll:>10.6g}" + ("*" if row.ll == best_ll else " "))
                + (
                    f"{row.lr:>9.5g}" + ("*" if row.lr == best_lr else " ")
                    if row.lr > 0
                    else " " * 10
                )
                + (f"{row.lr_df:>6}" if row.lr > 0 else " " * 6)
                + (f"{row.lr_pvalue:>7.3f} " if row.lr > 0 else " " * 8)
                + mark(row.fpe, best_fpe)
                + mark(row.aic, best_aic)
                + mark(row.hqic, best_hqic)
                + mark(row.sbic, best_sbic)
            )
        lines.append("-" * 4 + "+" + "-" * 72)
        lines.append("  * optimal lag")
        return "\n".join(lines)


def varsoc(
    data: np.ndarray | pd.DataFrame,
    *,
    maxlag: int = 4,
    trend: str = "c",
    names: list[str] | None = None,
) -> VarSocResult:
    """Lag-order selection, as Stata's ``varsoc``.

    ``data`` holds the endogenous series in time order (rows) by equation (columns). Missing
    rows must already be dropped. ``trend="c"`` fits a constant, which is Stata's default.
    """

    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ConsistencyError(
            f"varsoc: the data must be a 2-D array (rows over time); got {values.ndim}-D"
        )
    if not np.all(np.isfinite(values)):
        raise ConsistencyError(
            "varsoc: the data contain non-finite values; drop the missing rows first"
        )
    nobs_total, equations = values.shape
    if maxlag < 1:
        raise ValueError(f"maxlag must be at least 1; got {maxlag}")
    if nobs_total <= maxlag + 1:
        raise ValueError(
            f"varsoc: {nobs_total} observations are too few for maxlag={maxlag}"
        )

    # Every candidate order shares the sample left by the LONGEST lag. This is the single
    # detail that makes the numbers match Stata; see the module docstring.
    sample_size = nobs_total - maxlag
    response = values[maxlag:]

    rows: list[VarSocRow] = []
    previous_ll: float | None = None
    previous_k: int | None = None
    for lag in range(maxlag + 1):
        columns = [np.ones((sample_size, 1))]
        for depth in range(1, lag + 1):
            columns.append(values[maxlag - depth : maxlag - depth + sample_size])
        design = np.column_stack(columns)
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        residual = response - design @ coefficients
        # ML residual covariance: divide by T, which is what Stata's var stores.
        sigma = (residual.T @ residual) / sample_size
        determinant = float(np.linalg.det(sigma))
        if determinant <= 0.0:
            raise ConsistencyError(
                f"varsoc: the residual covariance at lag {lag} is singular"
            )
        loglikelihood = -0.5 * sample_size * (
            equations * np.log(2.0 * np.pi) + np.log(determinant) + equations
        )
        k = design.shape[1]
        aic = -2.0 * loglikelihood / sample_size + 2.0 * k * equations / sample_size
        hqic = (
            -2.0 * loglikelihood / sample_size
            + 2.0 * np.log(np.log(sample_size)) * k * equations / sample_size
        )
        sbic = (
            -2.0 * loglikelihood / sample_size
            + np.log(sample_size) * k * equations / sample_size
        )
        fpe = ((sample_size + k * equations + 1) / sample_size) ** equations * determinant

        lr = 0.0
        lr_df = 0
        lr_p = float("nan")
        if previous_ll is not None and previous_k is not None:
            lr = 2.0 * (loglikelihood - previous_ll)
            lr_df = (k - previous_k) * equations
            lr_p = float(stats.chi2.sf(lr, lr_df)) if lr_df > 0 else float("nan")

        rows.append(
            VarSocRow(
                lag=lag,
                ll=float(loglikelihood),
                lr=float(lr),
                lr_df=int(lr_df),
                lr_pvalue=float(lr_p),
                fpe=float(fpe),
                aic=float(aic),
                hqic=float(hqic),
                sbic=float(sbic),
            )
        )
        previous_ll = loglikelihood
        previous_k = k

    return VarSocResult(
        rows=tuple(rows),
        nobs=int(sample_size),
        equations=int(equations),
        sample_start=int(maxlag + 1),
        sample_end=int(nobs_total),
    )
