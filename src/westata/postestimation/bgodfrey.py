"""Breusch-Godfrey LM test for autocorrelation: Stata's ``estat bgodfrey``.

Transcribed from ``bgodfrey.ado:163-196`` (StataNow 19.5, v1.3.1). The ``Calc`` subroutine is
the whole definition::

    tsrevar l(1/`lags').`res'
    foreach i of local reslist {
        qui replace `i' = 0 if `i' >= . & `touse'
    }
    qui regress `res' `reslist' `varlist' if `touse', `cons'
    return scalar chi2 = e(N)*e(r2)
    return scalar df   = `lags'

So the auxiliary regression puts the residual on its own ``p`` lags AND the original
regressors, and the statistic is ``e(N) * e(r2)`` with ``e(N)`` the FULL estimation sample.

The detail that decides every number: the missing leading values of the lagged residuals are
replaced by **ZERO**, not dropped (``bgodfrey.ado:178-180``). Truncating the sample by ``p``
rows instead gives 4.704 / 5.111 / 5.053 / 5.679 on icecream.dta where Stata reports
4.237 / 4.487 / 4.775 / 5.099 -- close enough to look plausible at lag 1 and wrong at every
order.

Reference values (``reg consumption temp price income`` then ``estat bgodfrey, lags(1/4)``)::

    lag 1: chi2 4.237  df 1  p 0.0396
    lag 2: chi2 4.487  df 2  p 0.1061
    lag 3: chi2 4.775  df 3  p 0.1891
    lag 4: chi2 5.099  df 4  p 0.2773
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import stats

from westata.errors import ConsistencyError

__all__ = ["BgTestResult", "BgTestRow", "estat_bgodfrey"]


@dataclass(frozen=True)
class BgTestRow:
    """One lag order of the Breusch-Godfrey table."""

    lags: int
    chi2: float
    df: int
    pvalue: float


@dataclass(frozen=True)
class BgTestResult:
    """``estat bgodfrey``: Breusch-Godfrey LM test for autocorrelation."""

    rows: tuple[BgTestRow, ...]
    nobs: int

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            [
                {"lags(p)": row.lags, "chi2": row.chi2, "df": row.df, "Prob > chi2": row.pvalue}
                for row in self.rows
            ]
        ).set_index("lags(p)")

    def __str__(self) -> str:
        lines = [
            "Breusch\u2013Godfrey LM test for autocorrelation",
            "-" * 74,
            f"{'lags(p)':>12}  |{'chi2':>13}{'df':>17}{'Prob > chi2':>18}",
            "-" * 13 + "+" + "-" * 60,
        ]
        for row in self.rows:
            lines.append(
                f"{row.lags:>12}  |{row.chi2:>13.3f}{row.df:>17}{row.pvalue:>18.4f}"
            )
        lines.append("-" * 74)
        lines.append(f"{'H0: no serial correlation':>50}")
        return "\n".join(lines)


def estat_bgodfrey(result: Any, *, lags: int | Sequence[int] = 1) -> BgTestResult:
    """Breusch-Godfrey LM test, as ``estat bgodfrey, lags(...)``.

    ``result`` must be an OLS-style fit carrying residuals and the estimation design; Stata
    refuses the test after ``regress, robust`` and so does this.
    """

    if isinstance(lags, int):
        if lags < 1:
            raise ValueError(f"lags must be at least 1; got {lags}")
        orders = [lags]
    else:
        orders = sorted({int(value) for value in lags})
        if not orders or orders[0] < 1:
            raise ValueError("lags must contain at least one positive order")

    resid = getattr(result, "resid", None)
    design = getattr(result, "design_matrix", None)
    if resid is None or design is None:
        raise ConsistencyError(
            "estat bgodfrey needs a fitted result carrying residuals and the design matrix"
        )
    # ``bgodfrey.ado:5-10`` refuses to run after a robust regression. Match the name exactly:
    # the non-robust type is spelled "nonrobust" here, so a substring test on "robust" would
    # reject every ordinary fit.
    cov_type = str(getattr(result, "cov_type", "") or "").strip().lower()
    robust_names = {"robust", "hc0", "hc1", "hc2", "hc3", "hc4", "hc5"}
    if cov_type in robust_names or "robust" in cov_type and not cov_type.startswith("non"):
        raise ConsistencyError(
            "estat bgodfrey: this command does not work after regress, robust"
        )

    values = np.asarray(resid, dtype=float).reshape(-1)
    exog = np.asarray(design[0], dtype=float)
    if exog.ndim != 2 or exog.shape[0] != values.size:
        raise ConsistencyError(
            f"estat bgodfrey: design is {exog.shape} against {values.size} residuals"
        )
    nobs = int(values.size)

    rows: list[BgTestRow] = []
    for order in orders:
        if order >= nobs - 1:
            raise ValueError(
                f"estat bgodfrey: lags({order}) is too large for {nobs} observations"
            )
        lagged: list[np.ndarray] = []
        for depth in range(1, order + 1):
            block = np.zeros(nobs)
            # ``bgodfrey.ado:178-180``: leading missing values become ZERO, keeping the full
            # sample. Dropping them instead shifts every statistic.
            block[depth:] = values[: nobs - depth]
            lagged.append(block)
        augmented = np.column_stack([exog, *lagged])
        coefficients, *_ = np.linalg.lstsq(augmented, values, rcond=None)
        fitted = augmented @ coefficients
        residual = values - fitted
        total = float(np.sum((values - values.mean()) ** 2))
        if total <= 0.0:
            raise ConsistencyError("estat bgodfrey: the residuals are constant")
        r2 = 1.0 - float(residual @ residual) / total
        # ``return scalar chi2 = e(N)*e(r2)`` -- e(N), the full sample.
        statistic = float(nobs * r2)
        rows.append(
            BgTestRow(
                lags=order,
                chi2=statistic,
                df=order,
                pvalue=float(stats.chi2.sf(statistic, order)),
            )
        )
    return BgTestResult(rows=tuple(rows), nobs=nobs)
