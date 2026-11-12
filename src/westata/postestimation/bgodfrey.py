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
regressors, and the statistic is ``e(N) * e(r2)`` with ``e(N)`` the full estimation sample.

The detail that decides every number: the missing leading values of the lagged residuals are
replaced by **ZERO**, not dropped (``bgodfrey.ado:178-180``). Truncating the sample by ``p``
rows instead gives 4.704 / 5.111 / 5.053 / 5.679 on icecream.dta where Stata reports
4.237 / 4.487 / 4.775 / 5.099 -- close enough to look plausible at lag 1 and wrong at every
order.

**The lags are on the TIME axis, not the row axis.** ``bgodfrey.ado:59-60`` calls
``_ts timevar panelvar if `touse', sort onepanel`` and the lags come from ``tsrevar``, so a
row whose predecessor period is missing from the sample gets a MISSING lag. Dropping
``time == 5`` from icecream.dta therefore changes the answer to 4.220 / 4.476 / 4.787 /
5.048, and lagging by row position gives 3.286 instead. Pass ``time=`` to reproduce it.

Reference values (``reg consumption temp price income`` then ``estat bgodfrey, lags(1/4)``)::

    lag 1: chi2 4.237  df 1  p 0.0396
    lag 2: chi2 4.487  df 2  p 0.1061
    lag 3: chi2 4.775  df 3  p 0.1891
    lag 4: chi2 5.099  df 4  p 0.2773

and with ``nomiss0``::

    lag 1: chi2 4.704  p 0.0301      lag 3: chi2 5.053  p 0.1680
    lag 2: chi2 5.111  p 0.0777      lag 4: chi2 5.679  p 0.2244
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import stats

from westata.errors import ConsistencyError
from westata.postestimation.ts_lags import (
    TimeAxis,
    refuse_clustered_vce,
    refuse_panel_fit,
    resolve_time_axis,
)

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
    #: ``r(N_gaps)``: periods inside the sample's span with no observation. Zero means the
    #: table is what a contiguous series would give, which is the only case that was
    #: reachable before ``time=`` existed.
    n_gaps: int = 0

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

        With gaps, ``bgodfrey.ado:44`` prints ``Number of gaps in sample = N`` BEFORE the
        blank line that opens the table -- captured byte for byte::

            Number of gaps in sample = 1

            Breusch–Godfrey LM test for autocorrelation
        """

        lines: list[str] = []
        if self.n_gaps:
            lines.append(f"Number of gaps in sample = {self.n_gaps}")
        lines.extend(
            [
                "",
                "Breusch\u2013Godfrey LM test for autocorrelation",
                "-" * 75,
                f"{'lags(p)':>11}  |{'chi2':>14}{'df':>17}{'Prob > chi2':>28}",
                "-" * 13 + "+" + "-" * 61,
            ]
        )
        for row in self.rows:
            lines.append(
                f"{row.lags:>8}     |{row.chi2:>15.3f}{row.df:>16.0f}"
                f"{row.pvalue:>25.4f}"
            )
        lines.append("-" * 75)
        lines.append(f"{'H0: no serial correlation':>49}")
        return chr(10).join(lines)


def _lagged_residuals(
    values: np.ndarray, order: int, axis: TimeAxis | None, nomiss0: bool
) -> list[np.ndarray]:
    """``tsrevar l(1/`order').`res'`` plus the zero-fill decision.

    Without ``axis`` the lags are taken by row position, which is this package's historical
    behaviour and is what a caller with no ``tsset`` equivalent gets. With one, a period that
    is absent yields a MISSING lag -- exactly what ``L.`` does -- and the two branches below
    decide what happens to those missing values.
    """

    nobs = values.size
    blocks: list[np.ndarray] = []
    for depth in range(1, order + 1):
        # ``bgodfrey.ado:180-184``: by DEFAULT the leading missing values of the lagged
        # residual are replaced by ZERO, keeping the full sample; ``nomiss0`` leaves them
        # missing so the auxiliary regression loses those rows. Dropping them in the
        # default case shifts every statistic -- it gives 4.704 where Stata reports 4.237
        # on icecream.dta -- so the two paths are not interchangeable.
        block = np.full(nobs, np.nan) if nomiss0 else np.zeros(nobs)
        if axis is None:
            block[depth:] = values[: nobs - depth]
        else:
            source = axis.lag_indices(depth)
            present = source >= 0
            block[present] = values[source[present]]
            if not nomiss0:
                # The zero fill covers GAPS as well as the leading edge: Stata replaces
                # every missing lag inside e(sample), and a gap is one of those.
                block[~present] = 0.0
        blocks.append(block)
    return blocks


def estat_bgodfrey(
    result: Any,
    *,
    lags: int | Sequence[int] = 1,
    nomiss0: bool = False,
    time: str | Sequence[Any] | None = None,
    delta: float = 1.0,
) -> BgTestResult:
    """Breusch-Godfrey LM test, as ``estat bgodfrey, lags(...)``.

    ``result`` must be an OLS-style fit carrying residuals and the estimation design; Stata
    refuses the test after ``regress, robust`` and so does this -- including
    ``vce(cluster)``, which Stata treats as robust.

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

    Parameters
    ----------
    time:
        Stands in for ``tsset``. Either the NAME of a column in the fit's estimation frame
        or the period values themselves, one per estimation row. When given, the lags are
        taken on the time axis: a period that is absent from the sample produces a MISSING
        lag instead of silently pairing the rows on either side. Omitting it lags by row
        order, which is only correct when the rows are already a contiguous period
        sequence. Measured on icecream.dta with ``time == 5`` dropped, Stata's
        4.220 / 4.476 / 4.787 / 5.048 against 3.286 / 3.363 / 3.403 / 5.084 by row order.
    delta:
        The lag step, ``tsset``'s ``Delta:``. One unless the data declare otherwise; an
        explicit ``tsset t, delta(2)`` is the only case that differs. Must be a positive
        integer, which is ``tsset``'s own rule (r(451) for fractional periods).

    Raises
    ------
    ConsistencyError
        After a robust or clustered VCE, on a panel spanning several units, or when
        ``time`` cannot be resolved against the estimation sample.
    ValueError
        For a bad lag order or a non-integer ``delta``.
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
    refuse_clustered_vce(result, "estat bgodfrey")
    refuse_panel_fit(result, "estat bgodfrey")
    # ``bgodfrey.ado:5-10`` restricts the test to ``regress``; ``areg``, ``newey`` and
    # ``xtreg`` answer "estat bgodfrey not valid" (r(321)). This package's linear results
    # carry the model name, so the same gate can be applied rather than assumed away.
    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name and not model_name.startswith("regress"):
        raise ConsistencyError(
            f"estat bgodfrey: not valid after {model_name!r}. bgodfrey.ado:11-15 accepts "
            "only e(cmd)=='regress'; Stata answers r(321) for areg, newey and xtreg."
        )

    axis = resolve_time_axis(result, time, delta=delta, command="estat bgodfrey")

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
        lagged = _lagged_residuals(values, order, axis, nomiss0)
        augmented = np.column_stack([exog, *lagged])
        if nomiss0:
            # Under nomiss0 the missing lags are DROPPED (the auxiliary regression runs on
            # e(sample) minus the rows whose lags are missing); under the default they were
            # already zero-filled above, which is the whole point of the option.
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
    return BgTestResult(
        rows=tuple(rows), nobs=nobs, n_gaps=0 if axis is None else axis.n_gaps
    )
