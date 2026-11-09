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
        """Reproduce ``estat bgodfrey`` exactly, INCLUDING the leading blank line::

            (blank)
            Breusch–Godfrey LM test for autocorrelation
            ---------------------------------------------------------------------------
                lags(p)  |          chi2               df                 Prob > chi2
            -------------+-------------------------------------------------------------
                   1     |          4.237               1                   0.0396
            ---------------------------------------------------------------------------
                                    H0: no serial correlation

        The raw capture begins with a newline, so the block opens with an empty line -- measured in
        round 192 while walking every renderer's prefix. Omitting it shifted the whole block one
        line up against Stata's, which is why a line-by-line comparison failed on EVERY row even
        though all the geometry below was already right.

        Geometry measured from that output: the rules are 75 characters, the pipe sits at
        column 13, the chi2 value ends at 29, the df at 45 and the probability at 70. The
        lag label is left-aligned within its field with trailing spaces before the pipe
        (``       1     |``), which is why it is not simply right-aligned to 13.

        Before this the rules were 74 characters and every column was a different width,
        so the numbers were right and the table lines did not line up with Stata's.
        """

        lines = [
            "",
            "Breusch\u2013Godfrey LM test for autocorrelation",
            "-" * 75,
            f"{'lags(p)':>11}  |{'chi2':>14}{'df':>17}{'Prob > chi2':>28}",
            "-" * 13 + "+" + "-" * 61,
        ]
        for row in self.rows:
            lines.append(
                f"{row.lags:>8}     |{row.chi2:>15.3f}{row.df:>16.0f}"
                f"{row.pvalue:>25.4f}"
            )
        lines.append("-" * 75)
        lines.append(f"{'H0: no serial correlation':>49}")
        return chr(10).join(lines)


def estat_bgodfrey(
    result: Any, *, lags: int | Sequence[int] = 1, nomiss0: bool = False
) -> BgTestResult:
    """Breusch-Godfrey LM test, as ``estat bgodfrey, lags(...)``.

    ``result`` must be an OLS-style fit carrying residuals and the estimation design; Stata
    refuses the test after ``regress, robust`` and so does this.

    ``nomiss0`` keeps the leading missing values of the lagged residuals MISSING instead of
    replacing them with zero, which shortens the auxiliary regression by ``p`` rows.
    ``bgodfrey.ado:180-184`` is the whole difference::

        if "`dm'"=="" {
            foreach i of local reslist {
                qui replace `i' = 0 if `i' >= . & `touse'
            }
        }

    The option is called ``dm`` inside the ado -- "Davidson-MacKinnon method is now
    miss0" -- so an empty macro means the default (replace with zero) and a set one means
    keep them missing. Measured on a 200-row series, default against ``nomiss0``:

        lag 1: chi2 187.976  ->  191.674
        lag 2: chi2 188.371  ->  191.009
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
            # ``bgodfrey.ado:180-184``: by DEFAULT the leading missing values of the lagged
            # residual are replaced by ZERO, keeping the full sample; ``nomiss0`` leaves them
            # missing so the auxiliary regression loses those rows. Dropping them in the
            # default case shifts every statistic -- it gives 4.704 where Stata reports 4.237
            # on icecream.dta -- so the two paths are not interchangeable.
            if nomiss0:
                block = np.full(nobs, np.nan)
            else:
                block = np.zeros(nobs)
            block[depth:] = values[: nobs - depth]
            lagged.append(block)
        augmented = np.column_stack([exog, *lagged])
        if nomiss0:
            keep = np.isfinite(augmented).all(axis=1) & np.isfinite(values)
            augmented = augmented[keep]
            target = values[keep]
        else:
            target = values
        used = int(target.size)
        coefficients, *_ = np.linalg.lstsq(augmented, target, rcond=None)
        fitted = augmented @ coefficients
        residual = target - fitted
        total = float(np.sum((target - target.mean()) ** 2))
        if total <= 0.0:
            raise ConsistencyError("estat bgodfrey: the residuals are constant")
        r2 = 1.0 - float(residual @ residual) / total
        # ``return scalar chi2 = e(N)*e(r2)`` -- e(N) of the AUXILIARY regression, which is
        # the full sample by default and ``nobs - order`` under ``nomiss0``.
        statistic = float(used * r2)
        rows.append(
            BgTestRow(
                lags=order,
                chi2=statistic,
                df=order,
                pvalue=float(stats.chi2.sf(statistic, order)),
            )
        )
    return BgTestResult(rows=tuple(rows), nobs=nobs)
