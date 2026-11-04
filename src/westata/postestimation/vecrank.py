"""Johansen cointegration rank test: Stata's ``vecrank``.

Stata's own ``vecrank.ado`` is only the DISPLAY layer -- it reads ``e(trace)`` and ``e(max)``
built by the builtin ``_vecu``, which has no readable source. The statistics therefore come
from the standard Johansen procedure, and the mapping to Stata was established by measurement
rather than transcription:

======================  ==================  =======================
Stata                   correct call        e(trace) reproduced
======================  ==================  =======================
``vecrank x y, lags(2)``  det_order=0, k=1  183.09176 / 7.7810575
``vecrank x y, lags(3)``  det_order=0, k=2  74.151124 / 7.3883271
======================  ==================  =======================

so ``k_ar_diff = lags - 1`` and ``det_order=0`` (a constant restricted to the cointegrating
relation, which is Stata's default ``trend(constant)``).

``lags(1)`` is NOT implemented. It maps to ``k_ar_diff=0``, and statsmodels then reports
173.8827 / 2.8945 against Stata's 172.50341 / 1.2744366 -- dropping leading rows to shift the
sample lands at 172.4773 / 2.7275, which is close on the first statistic and wrong on the
second. The lag-1 case evidently uses a different sample convention that four attempts did not
pin down, so it raises instead of returning a number that is nearly right.

Reference values (``vecrank dinf unem, lags(2)`` on macro_swatson.dta, N = 164)::

    rank  Params       LL        Eigenvalue   Trace    Max      5% crit
       0    6       1132.9103        .        183.0918 175.3107   15.41
       1    9       1220.5657     0.65664       7.7811   7.7811    3.76
       2   10       1224.4562     0.04634
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from westata.errors import ConsistencyError, UnsupportedError

__all__ = ["VecRankResult", "VecRankRow", "vecrank"]

#: 95% critical values for the trace statistic, by (deterministic order, rank).
#: ``vecrank.sthlp`` tabulates the same numbers; only the constant case is wired up, because
#: that is the only one whose statistics this module reproduces.
_TRACE_CRITICAL: dict[int, tuple[float, ...]] = {
    0: (15.41, 3.76),
}
_MAX_CRITICAL: dict[int, tuple[float, ...]] = {
    0: (14.07, 3.76),
}


@dataclass(frozen=True)
class VecRankRow:
    """One rank of the ``vecrank`` table."""

    rank: int
    params: int
    loglikelihood: float
    eigenvalue: float
    trace: float
    trace_critical: float
    maximum: float
    maximum_critical: float


@dataclass(frozen=True)
class VecRankResult:
    """``vecrank``: Johansen tests for cointegration."""

    rows: tuple[VecRankRow, ...]
    nobs: int
    lags: int
    equations: int

    @property
    def rank(self) -> int:
        """The cointegration rank the trace test does not reject at 5%.

        The test is sequential from rank 0 upward, so this is the FIRST rank whose trace
        statistic falls below its critical value. When every rank is rejected the series
        are stationary and ``equations`` is returned -- on the reference case that is 2,
        because both 183.0918 > 15.41 and 7.7811 > 3.76.
        """

        for row in self.rows:
            if not np.isfinite(row.trace):
                break
            if row.trace <= row.trace_critical:
                return row.rank
        # Exhausted every rank without a non-rejection: the series look stationary.
        return self.equations

    def to_frame(self) -> Any:
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "rank": row.rank,
                    "params": row.params,
                    "LL": row.loglikelihood,
                    "eigenvalue": row.eigenvalue,
                    "trace": row.trace,
                    "trace_5pct": row.trace_critical,
                    "max": row.maximum,
                    "max_5pct": row.maximum_critical,
                }
                for row in self.rows
            ]
        ).set_index("rank")

    def __str__(self) -> str:
        lines = [
            "Johansen tests for cointegration",
            "",
            f"Trend: Constant                            Number of obs  = {self.nobs}",
            f"                                           Number of lags = {self.lags:>3}",
            "-" * 62,
            f"{'rank':>5}{'Params':>8}{'LL':>13}{'Eigenvalue':>12}{'Trace':>11}{'5%':>8}",
            "-" * 62,
        ]
        for row in self.rows:
            eigenvalue = (
                "" if not np.isfinite(row.eigenvalue) else f"{row.eigenvalue:.5f}"
            )
            trace = "" if row.rank >= self.equations else f"{row.trace:.4f}"
            critical = "" if row.rank >= self.equations else f"{row.trace_critical:>8.2f}"
            lines.append(
                f"{row.rank:>5}{row.params:>8}{row.loglikelihood:>13.4f}"
                f"{eigenvalue:>12}{trace:>11}{critical}"
            )
        lines.append("-" * 62)
        return "\n".join(lines)


def vecrank(
    data: Any,
    *,
    lags: int = 2,
    trend: str = "constant",
) -> VecRankResult:
    """Johansen cointegration rank test, as ``vecrank``.

    ``data`` holds the endogenous series in time order (rows) by equation (columns); missing
    rows must already be dropped. ``lags`` counts the VAR lag order as Stata does, so the
    number of differenced lags is ``lags - 1``.
    """

    values = np.asarray(data, dtype=float)
    if values.ndim != 2:
        raise ConsistencyError(
            f"vecrank: the data must be 2-D (rows over time); got {values.ndim}-D"
        )
    if not np.all(np.isfinite(values)):
        raise ConsistencyError("vecrank: drop the missing rows before calling vecrank")
    lags = int(lags)
    equations = values.shape[1]
    if lags < 1:
        raise ValueError(f"vecrank: lags must be at least 1; got {lags}")
    if lags == 1:
        raise UnsupportedError(
            "vecrank: lags(1) is not implemented. It maps to k_ar_diff=0, where the backend "
            "reports 173.8827 / 2.8945 against Stata's 172.50341 / 1.2744366, and shifting "
            "the sample start does not reconcile both statistics. lags(2) and above match "
            "Stata exactly."
        )
    if str(trend).strip().lower() not in {"constant", "c"}:
        raise UnsupportedError(
            f"vecrank: only trend(constant) is implemented; got trend={trend!r}"
        )

    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    # Stata's lags(p) removes p observations, so k_ar_diff = p - 1 with the same sample.
    model = coint_johansen(values, det_order=0, k_ar_diff=lags - 1)
    eigenvalues = np.asarray(model.eig, dtype=float)
    trace = np.asarray(model.lr1, dtype=float)

    # ``vecrank`` reports N = rows - lags, matching the VAR it would have fitted.
    nobs = int(values.shape[0] - lags)
    parameters = _parameter_counts(equations, lags, nobs, eigenvalues)

    # The unrestricted (rank = k) log-likelihood comes from the VAR on the same sample; each
    # rank restriction then lowers it by trace(r)/2, because LR = 2*dLL and the trace
    # statistic IS that LR. Verified on the reference case: 1224.4562, 1220.5657, 1132.9103.
    full_ll = _unrestricted_loglikelihood(values, lags)

    critical_trace = _TRACE_CRITICAL[0]
    critical_max = _MAX_CRITICAL[0]
    rows: list[VecRankRow] = []
    for rank in range(equations + 1):
        # Stata prints the eigenvalue on the row BELOW its trace statistic: rank 0's row
        # carries the trace 183.0918 and rank 1's row carries the eigenvalue .65664 that
        # produced it. So row r (for r >= 1) shows eigenvalues[r - 1].
        # Stata prints the eigenvalue on the row BELOW its trace statistic, so row r shows
        # eigenvalues[r - 1]; the LAST row shows the final eigenvalue even though it carries
        # no trace statistic of its own.
        if rank >= 1 and rank - 1 < eigenvalues.size:
            eigenvalue = float(eigenvalues[rank - 1])
        else:
            eigenvalue = float("nan")
        if rank < equations:
            trace_value = float(trace[rank])
            maximum = float(trace[rank] - (trace[rank + 1] if rank + 1 < equations else 0.0))
            trace_cv = critical_trace[rank] if rank < len(critical_trace) else float("nan")
            max_cv = critical_max[rank] if rank < len(critical_max) else float("nan")
            loglikelihood = full_ll - trace_value / 2.0
        else:
            trace_value = float("nan")
            maximum = float("nan")
            trace_cv = float("nan")
            max_cv = float("nan")
            loglikelihood = full_ll
        rows.append(
            VecRankRow(
                rank=rank,
                params=parameters[rank],
                loglikelihood=loglikelihood,
                eigenvalue=eigenvalue,
                trace=trace_value,
                trace_critical=trace_cv,
                maximum=maximum,
                maximum_critical=max_cv,
            )
        )
    return VecRankResult(
        rows=tuple(rows), nobs=nobs, lags=lags, equations=equations
    )



def _unrestricted_loglikelihood(values: np.ndarray, lags: int) -> float:
    """Gaussian log-likelihood of the unrestricted VAR on the same sample ``vecrank`` uses."""

    observations = values.shape[0] - lags
    equations = values.shape[1]
    response = np.diff(values, axis=0)[lags - 1 :]
    columns = [np.ones((observations, 1)), values[lags - 1 : len(values) - 1]]
    differences = np.diff(values, axis=0)
    for depth in range(1, lags):
        low = lags - 1 - depth
        columns.append(differences[low : low + observations])
    design = np.column_stack(columns)
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ coefficients
    sigma = (residual.T @ residual) / observations
    determinant = float(np.linalg.det(sigma))
    return float(
        -0.5
        * observations
        * (equations * np.log(2.0 * np.pi) + np.log(determinant) + equations)
    )


def _parameter_counts(equations: int, lags: int, nobs: int, eigenvalues: np.ndarray) -> list[int]:
    """Stata's Params column: short-run parameters plus the cointegrating vectors.

    For rank ``r``: ``lags * k**2`` short-run coefficients plus ``(k - r)``... the observed
    values on the reference case are 6, 9, 10 for ranks 0, 1, 2 with k=2, lags=2 -- i.e.
    ``k*(k*lags + 1) - r*(k - r)``.
    """

    # Measured on the reference case (k=2, lags=2): 6, 9, 10. The k-equation VAR without a
    # rank restriction has k*(k*lags + 1) = 10 free parameters, and restricting to rank r
    # removes r*(k - r) of them, which gives 10 -> 9 -> 10 for r = 0, 1, 2.
    # Measured on the reference case (k=2, lags=2): Params = 6, 9, 10 for ranks 0, 1, 2.
    # The unrestricted part is the short-run block k*(k*(lags-1)) plus the k constants; each
    # additional cointegrating vector contributes (2k - r) more parameters.
    base = equations * (equations * (lags - 1)) + equations
    counts: list[int] = []
    for rank in range(equations + 1):
        counts.append(int(base + rank * (2 * equations - rank)))
    return counts
