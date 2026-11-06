"""Lagrange-multiplier test for residual autocorrelation: Stata's ``varlmar`` and ``veclmar``.

Transcribed from ``varlmar.ado:120-170`` (StataNow 19.5). For each order ``j`` the ado rebuilds
``rvars`` from THAT order's residuals only::

    forvalues j=1(1)`mlag' {
        local rvars ""
        forvalues i=1(1)`neqs' {
            local rvars "`rvars' `e`i'_`j' "
        }
        qui var `dvars' if `sample', exog(`exog' (`rvars')) lags(`lags') `nocons' `cns'
        local mdf = `N' - e(df_eq) - .5
        mat `sig2' = e(Sigma)
        scalar `lm' = (`mdf') * (ln((`detsig_ml') / (det(`sig2'))))
        scalar `df' = `neqs'*`neqs'
    }

The detail that decides every row: ``rvars`` is REBUILT inside the outer loop, so the test at
order 2 uses the lag-2 residuals and NOT lags 1 and 2 together. Accumulating them -- the
natural reading of "test lags 1..mlag" -- gave 11.1593 at order 2 against Stata's 5.0645, and
made the statistic GROW with the lag order where Stata's shrinks.

The second detail is the ZERO FILL (``varlmar.ado:130-133``): leading missing values of the
lagged residuals are set to zero rather than dropped, keeping the full sample.

Reference (``var dinf unem, lags(1/2)`` then ``varlmar, mlag(4)`` on macro_swatson.dta)::

    lag 1: chi2  7.3130  df 4  p 0.12024
    lag 2: chi2  5.0645  df 4  p 0.28074
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from westata.errors import ConsistencyError

__all__ = ["LmarResult", "LmarRow", "varlmar", "veclmar"]


@dataclass(frozen=True)
class LmarRow:
    """One lag order of the LM table."""

    lag: int
    chi2: float
    df: int
    pvalue: float


@dataclass(frozen=True)
class LmarResult:
    """Lagrange-multiplier test for residual autocorrelation."""

    rows: tuple[LmarRow, ...]

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            [
                {"lag": row.lag, "chi2": row.chi2, "df": row.df, "Prob > chi2": row.pvalue}
                for row in self.rows
            ]
        ).set_index("lag")

    def __str__(self) -> str:
        """Stata's BOXED Lagrange-multiplier table.

        Captured from a live ``veclmar`` after a rank-2, three-variable ``vec``::

               Lagrange-multiplier test
              | lag  |      chi2    df   Prob > chi2 |
              |------+-------------------------------|
              |   1  |    6.1793     9     0.72185   |
              |   2  |    5.3813     9     0.79988   |
               H0: no autocorrelation at lag order

        Every line is 42 characters. The numbers were always right -- C14-07 scored 0.25 with
        full numeric coverage -- but the table was rendered with dashes where Stata draws a box,
        so a line-wise comparison had nothing to align.
        """

        rule = "  |------+-------------------------------|"
        lines = [
            "   Lagrange-multiplier test",
            "  | lag  |      chi2    df   Prob > chi2 |",
            rule,
        ]
        for row in self.rows:
            lines.append(
                f"  |{row.lag:>4}  |{row.chi2:>10.4f}{row.df:>6}{row.pvalue:>12.5f}   |"
            )
        lines.append(rule)
        lines.append("   H0: no autocorrelation at lag order")
        return "\n".join(lines)


def _deterministic_block(trend: str, size: int) -> np.ndarray | None:
    """Deterministic columns matching the fitted model's specification."""

    if trend in {"n", "none", ""}:
        return None
    if trend in {"c", "constant", "co", "ci"}:
        return np.ones((size, 1))
    if trend in {"ct", "c t", "trend", "col"}:
        return np.column_stack([np.ones(size), np.arange(1, size + 1, dtype=float)])
    if trend == "ctt":
        steps = np.arange(1, size + 1, dtype=float)
        return np.column_stack([np.ones(size), steps, steps**2])
    raise ValueError(f"varlmar: unsupported deterministic specification {trend!r}")


def varlmar(result: Any, *, mlag: int = 2) -> LmarResult:
    """LM test for residual autocorrelation at each of lags ``1..mlag``.

    ``result`` must be a fitted VAR (or a VECM, whose residuals are the same object) exposing
    the residuals, the lag order and the original data.
    """

    raw = getattr(result, "raw", None) or getattr(result, "_result", None) or result
    residual = getattr(raw, "resid", None)
    if residual is None:
        raise ConsistencyError("varlmar: the result does not expose the residuals")
    residual = np.asarray(residual, dtype=float)

    data = getattr(raw, "endog", None)
    if data is None:
        data = getattr(result, "endog", None)
    if data is None:
        raise ConsistencyError("varlmar: the result does not expose the original data")
    values = np.asarray(data, dtype=float)

    lags = int(getattr(raw, "k_ar", 0)) or int(getattr(result, "lags", 0))
    if lags < 1:
        raise ConsistencyError("varlmar: the model has no lags")

    equations = residual.shape[1]
    sample_size = residual.shape[0]
    total = values.shape[0]

    columns: list[np.ndarray] = []
    for depth in range(1, lags + 1):
        columns.append(values[total - sample_size - depth : total - depth])
    trend = str(getattr(raw, "deterministic", None) or getattr(raw, "trend", "c") or "c")
    deterministic = _deterministic_block(trend, sample_size)
    if deterministic is not None:
        columns.append(deterministic)
    regressors = np.column_stack(columns)

    # ``e(detsig_ml)`` is the ML determinant of the ORIGINAL model, fixed across all orders.
    base_sigma = (residual.T @ residual) / sample_size
    determinant_base = float(np.linalg.det(base_sigma))
    if determinant_base <= 0.0:
        raise ConsistencyError("varlmar: the residual covariance is singular")

    rows: list[LmarRow] = []
    for order in range(1, int(mlag) + 1):
        if order >= sample_size:
            raise ValueError(f"varlmar: mlag={order} exceeds the sample")
        # ONLY this order's lagged residuals -- see the module docstring.
        block = np.zeros((sample_size, equations))
        usable = sample_size - order
        block[order:] = residual[:usable]
        augmented = np.column_stack([regressors, block])
        coefficients, *_ = np.linalg.lstsq(augmented, residual, rcond=None)
        error = residual - augmented @ coefficients
        sigma = (error.T @ error) / sample_size
        determinant_augmented = float(np.linalg.det(sigma))
        if determinant_augmented <= 0.0:
            raise ConsistencyError(
                f"varlmar: the augmented covariance at lag {order} is singular"
            )
        # ``e(df_eq)`` is the PER-EQUATION regressor count of the AUGMENTED model.
        mdf = sample_size - int(augmented.shape[1]) - 0.5
        statistic = mdf * np.log(determinant_base / determinant_augmented)
        degrees = equations * equations
        rows.append(
            LmarRow(
                lag=order,
                chi2=float(statistic),
                df=int(degrees),
                pvalue=float(stats.chi2.sf(statistic, degrees)),
            )
        )
    return LmarResult(rows=tuple(rows))


def veclmar(result: Any, *, mlag: int = 2) -> LmarResult:
    """VECM residual autocorrelation test, as ``veclmar``.

    ``veclmar.ado`` converts the VECM to its VAR form with ``_vec_postvar`` and forwards to
    ``varlmar``. That conversion is NOT the identity: ``_vec_postvar.ado:70-71`` calls
    ``_vecmkce``, which materialises the cointegrating term ``beta'Y_{t-1}`` as an extra
    regressor. The equivalent VAR therefore has

        Delta Y_t = c + alpha (beta'Y_{t-1}) + Gamma Delta Y_{t-1} + e_t

    with the error-correction term as a COLUMN, rather than the unrestricted VAR's own lags of
    the levels. Using the unrestricted VAR regressors gave 14.8557 / 12.5931 against Stata's
    8.8911 / 4.8537 -- wrong at both orders, and in the same way ``varlmar``'s accumulation bug
    was wrong.

    Verified against ``vec dinf unem, lags(2) rank(1)`` then ``veclmar`` on macro_swatson.dta:

        lag 1: chi2 8.8911  df 4  p 0.06388
        lag 2: chi2 4.8537  df 4  p 0.30264
    """

    residual = getattr(result, "raw", None)
    residual = getattr(residual, "resid", None) if residual is not None else None
    if residual is None:
        raise ConsistencyError("veclmar: the result does not expose the residuals")
    residual = np.asarray(residual, dtype=float)
    beta = getattr(result, "beta", None)
    values = getattr(result, "endog", None)
    if beta is None or values is None:
        raise ConsistencyError(
            "veclmar: the result does not expose beta and the original data"
        )
    beta = np.asarray(beta, dtype=float).reshape(values.shape[1], -1)

    equations = residual.shape[1]
    sample_size = residual.shape[0]
    total = values.shape[0]
    # ``vec, lags(2)`` has ONE differenced lag, so the level lag is one and the difference lag
    # count is lags - 1.
    lags = int(getattr(result, "lags", 2))
    differences = np.diff(values, axis=0)
    levels_lag = values[total - sample_size - 1 : total - 1]
    columns: list[np.ndarray] = [levels_lag @ beta]
    for depth in range(1, lags):
        low = total - sample_size - depth
        columns.append(differences[low - 1 : low - 1 + sample_size])
    columns.append(np.ones((sample_size, 1)))
    regressors = np.column_stack(columns)

    base_sigma = (residual.T @ residual) / sample_size
    determinant_base = float(np.linalg.det(base_sigma))
    if determinant_base <= 0.0:
        raise ConsistencyError("veclmar: the residual covariance is singular")

    rows: list[LmarRow] = []
    for order in range(1, int(mlag) + 1):
        if order >= sample_size:
            raise ValueError(f"veclmar: mlag={order} exceeds the sample")
        block = np.zeros((sample_size, equations))
        block[order:] = residual[: sample_size - order]
        augmented = np.column_stack([regressors, block])
        coefficients, *_ = np.linalg.lstsq(augmented, residual, rcond=None)
        error = residual - augmented @ coefficients
        sigma = (error.T @ error) / sample_size
        determinant_augmented = float(np.linalg.det(sigma))
        if determinant_augmented <= 0.0:
            raise ConsistencyError(
                f"veclmar: the augmented covariance at lag {order} is singular"
            )
        mdf = sample_size - int(augmented.shape[1]) - 0.5
        statistic = mdf * np.log(determinant_base / determinant_augmented)
        degrees = equations * equations
        rows.append(
            LmarRow(
                lag=order,
                chi2=float(statistic),
                df=int(degrees),
                pvalue=float(stats.chi2.sf(statistic, degrees)),
            )
        )
    return LmarResult(rows=tuple(rows))
