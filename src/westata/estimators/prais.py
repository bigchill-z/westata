"""Implementation of Stata's ``prais`` (Prais-Winsten AR(1) regression).

Stata's reference values on icecream.dta after ``prais consumption temp price income``:

    Iteration 40: rho = 0.8002          (iterated until |change| < 1e-6)
    temp    .0029541
    price  -1.048854
    income -.0008022
    _cons   .5870049
    rho     .8002264
    DW original    1.021169
    DW transformed 1.846795
    R-squared      0.6234
    Root MSE       .03232

The algorithm Stata documents: estimate by OLS, take rho from the residuals, transform the
data by the Prais-Winsten rule (the first observation is scaled by sqrt(1 - rho^2) rather than
dropped), re-estimate, and repeat until rho stops moving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, ConvergenceError

__all__ = ["PraisResult", "prais", "prais_winsten"]


def _rho_from_residuals(resid: np.ndarray) -> float:
    """AR(1) coefficient from a residual series: ``sum e_t e_{t-1} / sum e_{t-1}^2``."""

    lagged = resid[:-1]
    current = resid[1:]
    denominator = float(lagged @ lagged)
    if denominator <= 0.0:
        return 0.0
    return float(lagged @ current) / denominator


def _transform(values: np.ndarray, rho: float) -> np.ndarray:
    """Prais-Winsten transform of a 1-D series."""

    out = np.empty_like(values)
    out[0] = np.sqrt(max(1.0 - rho * rho, 0.0)) * values[0]
    out[1:] = values[1:] - rho * values[:-1]
    return out


def prais_winsten(
    exog: np.ndarray,
    endog: np.ndarray,
    *,
    maxiter: int = 100,
    tolerance: float = 1e-8,
    rho: float | None = None,
) -> tuple[np.ndarray, float, int]:
    """Iterated Prais-Winsten estimate.

    Returns ``(coefficients, rho, iterations)``. ``rho`` may be given to run a single
    Cochrane-Orcutt style pass instead of iterating.
    """

    x = np.asarray(exog, dtype=float)
    y = np.asarray(endog, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ConsistencyError(
            f"prais: design is {x.shape} but the response has {y.size} values"
        )

    def ols(design: np.ndarray, response: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(design, response, rcond=None)[0]

    coefficients = ols(x, y)
    current_rho = _rho_from_residuals(y - x @ coefficients) if rho is None else float(rho)

    iterations = 0
    for iterations in range(1, maxiter + 1):
        if abs(current_rho) >= 1.0:
            raise ConvergenceError(
                f"prais: the estimated AR(1) coefficient {current_rho:.6f} is not inside "
                "(-1, 1); the series is not stationary"
            )
        transformed_x = np.column_stack(
            [_transform(x[:, j], current_rho) for j in range(x.shape[1])]
        )
        transformed_y = _transform(y, current_rho)
        updated = ols(transformed_x, transformed_y)
        residual = y - x @ updated
        next_rho = _rho_from_residuals(residual)
        if rho is not None or abs(next_rho - current_rho) < tolerance:
            coefficients = updated
            current_rho = next_rho if rho is None else current_rho
            break
        coefficients = updated
        current_rho = next_rho
    else:
        raise ConvergenceError(
            f"prais: rho did not converge in {maxiter} iterations"
        )

    return coefficients, current_rho, iterations


class PraisResult:
    """Results of a Prais-Winsten AR(1) regression."""

    __slots__ = (
        "_params",
        "_names",
        "_bse",
        "_tvalues",
        "_pvalues",
        "_conf_int",
        "_rho",
        "_nobs",
        "_df_resid",
        "_df_model",
        "_mss",
        "_rss",
        "_tss",
        "_r2",
        "_r2_adj",
        "_rmse",
        "_fvalue",
        "_p_f",
        "_dw_original",
        "_dw_transformed",
        "_iterations",
    )

    def __init__(self, **values: object) -> None:
        for key, value in values.items():
            setattr(self, f"_{key}", value)

    @property
    def params(self) -> pd.Series:
        return pd.Series(self._params, index=self._names, name="params")

    @property
    def bse(self) -> pd.Series:
        return pd.Series(self._bse, index=self._names, name="bse")

    @property
    def tvalues(self) -> pd.Series:
        return pd.Series(self._tvalues, index=self._names, name="tvalues")

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(self._pvalues, index=self._names, name="pvalues")

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def nobs(self) -> int:
        return self._nobs

    @property
    def df_resid(self) -> int:
        return self._df_resid

    @property
    def rsquared(self) -> float:
        return self._r2

    @property
    def rsquared_adj(self) -> float:
        return self._r2_adj

    @property
    def rmse(self) -> float:
        return self._rmse

    @property
    def fvalue(self) -> float:
        return self._fvalue

    @property
    def dw_original(self) -> float:
        return self._dw_original

    @property
    def dw_transformed(self) -> float:
        return self._dw_transformed

    @property
    def iterations(self) -> int:
        return self._iterations

    def __str__(self) -> str:
        width = 78
        lines = ["Prais\u2013Winsten AR(1) regression with iterated estimates", ""]
        lines.append(
            f"{'':>13}{'SS':>13}{'df':>9}{'MS':>13}"
            f"{'Number of obs':>19}   ={self._nobs:>9}"
        )
        lines.append("-" * 14 + "+" + "-" * 35 + "   " + "-" * 15 + "-" * 9)
        f_stat = f"F({self._df_model:>2}, {self._df_resid:>3})"
        lines.append(
            f"{'Model':>12} |{self._mss:>13.7g}{self._df_model:>9}"
            f"{self._mss / max(self._df_model, 1):>13.7g}"
            f"{f_stat:>19}   ={self._fvalue:>9.2f}"
        )
        lines.append(
            f"{'Residual':>12} |{self._rss:>13.7g}{self._df_resid:>9}"
            f"{self._rss / max(self._df_resid, 1):>13.7g}"
            f"{'Prob > F':>19}   ={self._p_f:>9.4f}"
        )
        lines.append("-" * 14 + "+" + "-" * 35 + "   " + "-" * 23)
        lines.append(
            f"{'Total':>12} |{self._tss:>13.7g}{self._nobs - 1:>9}"
            f"{self._tss / max(self._nobs - 1, 1):>13.7g}"
            f"{'R-squared':>19}   ={self._r2:>9.4f}"
        )
        lines.append(
            f"{'':>13}{'':>13}{'':>9}{'':>13}"
            f"{'Adj R-squared':>19}   ={self._r2_adj:>9.4f}"
        )
        lines.append(
            f"{'':>13}{'':>13}{'':>9}{'':>13}"
            f"{'Root MSE':>19}   ={self._rmse:>9.5f}"
        )
        lines.append("")
        lines.append("-" * width)
        lines.append(
            f"{'consumption':>13} |{'Coefficient':>11}{'Std. err.':>11}"
            f"{'t':>7}{'P>|t|':>8}{'[95% conf. interval]':>22}"
        )
        lines.append("-" * 14 + "+" + "-" * 62)
        for index, name in enumerate(self._names):
            low, high = self._conf_int[index]
            lines.append(
                f"{name:>13} |{self._params[index]:>11.7g}{self._bse[index]:>11.7g}"
                f"{self._tvalues[index]:>7.2f}{self._pvalues[index]:>8.3f}"
                f"{low:>11.7g}{high:>11.7g}"
            )
        lines.append("-" * 14 + "+" + "-" * 62)
        lines.append(f"{'rho':>13} |{self._rho:>11.7g}")
        lines.append("-" * width)
        lines.append(
            f"Durbin\u2013Watson statistic (original)    = {self._dw_original:.6f}"
        )
        lines.append(
            f"Durbin\u2013Watson statistic (transformed) = {self._dw_transformed:.6f}"
        )
        return "\n".join(lines)


def prais(
    frame: pd.DataFrame,
    formula: str,
    *,
    maxiter: int = 100,
    tolerance: float = 1e-8,
) -> PraisResult:
    """Fit a Prais-Winsten AR(1) regression, as Stata's ``prais``.

    ``formula`` is a statsmodels/patsy style spec such as ``"y ~ x1 + x2"``. The data must
    already be in time order through ``frame``'s row order.
    """

    from westata.formula.factor import build_design

    design, names, kept_index = build_design(formula, frame)
    lhs = formula.split("~", 1)[0].strip()
    if lhs not in frame.columns:
        raise ConsistencyError(
            f"prais: the response {lhs!r} is not a column of the data"
        )
    y = frame.loc[kept_index, lhs].to_numpy(dtype=float)
    x = np.asarray(design, dtype=float)

    coefficients, rho, iterations = prais_winsten(
        x, y, maxiter=maxiter, tolerance=tolerance
    )

    resid = y - x @ coefficients
    df_resid = y.size - x.shape[1]
    # Stata reports R-squared, Adjusted R-squared and Root MSE on the TRANSFORMED scale,
    # because the estimated model is the one fitted to the rho-transformed data. Reporting
    # the original-scale sums of squares gave R2 0.521716 and RMSE 0.048053 against Stata's
    # 0.6234 and .03232.
    transformed_x = np.column_stack([_transform(x[:, j], rho) for j in range(x.shape[1])])
    transformed_y = _transform(y, rho)
    trans_resid = transformed_y - transformed_x @ coefficients
    rss = float(trans_resid @ trans_resid)
    tss = float(np.sum((transformed_y - transformed_y.mean()) ** 2))
    mss = tss - rss
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    r2_adj = 1.0 - (1.0 - r2) * (y.size - 1) / df_resid if df_resid > 0 else 0.0
    rmse = float(np.sqrt(rss / df_resid)) if df_resid > 0 else float("nan")

    sigma2 = rss / df_resid if df_resid > 0 else float("nan")
    try:
        bread = np.linalg.inv(transformed_x.T @ transformed_x)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(transformed_x.T @ transformed_x)
    covariance = sigma2 * bread
    bse = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = coefficients / bse
    pvalues = 2.0 * stats.t.sf(np.abs(tvalues), df_resid)
    critical = stats.t.ppf(0.975, df_resid)
    conf = np.column_stack(
        [coefficients - critical * bse, coefficients + critical * bse]
    )

    fvalue = (mss / (x.shape[1] - 1)) / (rss / df_resid) if rss > 0 and df_resid > 0 and x.shape[1] > 1 else float("nan")
    p_f = float(stats.f.sf(fvalue, x.shape[1] - 1, df_resid)) if np.isfinite(fvalue) else float("nan")

    # Stata's "original" Durbin-Watson is computed from the residuals of the ORDINARY
    # (untransformed) regression of y on X, not from the Prais-Winsten residual of the
    # transformed fit. Using the latter gave 0.482716 against Stata's 1.021169.
    ols_resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    ols_rss = float(ols_resid @ ols_resid)
    dw_original = (
        float(np.sum(np.diff(ols_resid) ** 2) / ols_rss) if ols_rss > 0 else float("nan")
    )
    dw_transformed = (
        float(np.sum(np.diff(trans_resid) ** 2) / float(trans_resid @ trans_resid))
        if float(trans_resid @ trans_resid) > 0
        else float("nan")
    )

    return PraisResult(
        params=coefficients,
        names=list(names),
        bse=bse,
        tvalues=tvalues,
        pvalues=pvalues,
        conf_int=conf,
        rho=rho,
        nobs=int(y.size),
        df_resid=int(df_resid),
        df_model=int(x.shape[1] - 1),
        r2=r2,
        r2_adj=r2_adj,
        rmse=rmse,
        fvalue=fvalue,
        p_f=p_f,
        mss=mss,
        rss=rss,
        tss=tss,
        dw_original=dw_original,
        dw_transformed=dw_transformed,
        iterations=iterations,
    )
