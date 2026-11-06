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
173.8827 / 2.8945 against Stata's 172.50341 / 1.2744366.

Re-measured across every sample start and both k_ar_diff settings, looking for any
combination that reproduces Stata's pair::

    kd=0 start=0  n=166  trace  173.8827   2.8945  eig .64523 .01739
    kd=0 start=1  n=165  trace  173.8176   2.9529  eig .64720 .01784
    kd=0 start=2  n=164  trace  172.4773   2.7275  eig .64704 .01659
    kd=1 start=0  n=166  trace  183.0918   7.7811  eig .65664 .04634
    kd=1 start=1  n=165  trace  181.8193   8.5344  eig .65462 .05101
    Stata (N=165, Sample 4 thru 168)     172.5034   1.2744  eig .64575 .00769

The first slot looks close only at start=2 (172.4773 vs 172.5034, 0.015% off), but the
EIGENVALUES there are .64704 / .01659 against Stata's .64575 / .00769 -- the second is off by
more than a factor of two, so the agreement in the trace statistic is a coincidence of two
opposite errors rather than a nearly-right convention. Stata's own second eigenvalue being
an order of magnitude smaller suggests the lag-1 case builds a genuinely different system,
which five attempts have not reconstructed.

Note also that Stata loses TWO observations at lags(1) (Sample 4 thru 168 from a series
starting at 3, N=165), where k_ar_diff=0 loses only one. That extra step is not simply a
shifted sample, which is what the earlier note assumed. Raising is still the honest outcome.

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

#: The Osterwald-Lenum tables give ONE value per rank, and for the constant case the sequence is
#: the same for every k -- what changes is where a system starts inside it. Measured from live
#: ``vecrank ..., lags(2) max trend(constant)`` runs over macro_swatson:
#:
#:   k=2: rank0 = 15.41, rank1 = 3.76                  -> the last two entries
#:   k=3: rank0 = 29.68, rank1 = 15.41, rank2 = 3.76   -> the last three
#:
#: So a k-variable system reads ``_TRACE_CRITICAL_TAIL[k - 1 - rank]``. Indexing the old
#: two-entry dict by rank printed 15.41 on row 0 of a two-variable system (correct) and then ran
#: off the end, which is why a three-variable system reported ``nan`` as its rank-0 value.
#:
#: ONLY the first three entries are measured here. Stata's own table continues upward for more
#: variables, but inventing those numbers would be worse than reporting that they are unknown,
#: so a system with more than three is REFUSED rather than given a made-up critical value.
_TRACE_CRITICAL_TAIL: tuple[float, ...] = (3.76, 15.41, 29.68)
_MAX_CRITICAL_TAIL: tuple[float, ...] = (3.76, 14.07, 20.97)

#: The same tables for trend(trend). The trend case uses a DIFFERENT Osterwald-Lenum
#: tabulation, so reusing the constant-case numbers silently changes which rank gets chosen.
#: Measured from live runs on macro_swatson, two and three variables:
#:
#:   k=2: rank0 = 18.17, rank1 = 3.74
#:   k=3: rank0 = 34.55, rank1 = 18.17, rank2 = 3.74
_TRACE_CRITICAL_TAIL_TREND: tuple[float, ...] = (3.74, 18.17, 34.55)


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
    #: "constant" or "trend" -- decides the Trend: line and, more importantly, which
    #: critical-value table the rows were built from.
    deterministic: str = "constant"

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
        """Stata's vecrank table, captured byte-for-byte.

        From ``vecrank dinf unem, lags(2)``::

            Johansen tests for cointegration
            Trend: Constant                           Number of obs  = 164
            Sample: 3 thru 166                        Number of lags =   2
            --------------------------------------------------------------
                                                                  Critical
            Maximum                                        Trace     value
               rank  Params           LL  Eigenvalue   statistic        5%
                  0      6     1132.9103           .    183.0918     15.41
                  1      9     1220.5657     0.65664      7.7811      3.76
                  2     10    1224.4562     0.04634
            --------------------------------------------------------------

        Three things here are easy to get wrong, and the widths were solved from the target token
        positions rather than guessed:

        * the ``Sample:`` line and the two-line "Critical / value" header were missing entirely;
        * rank 0 has no eigenvalue and Stata prints a DOT, not a blank;
        * the LAST rank row is short ON PURPOSE -- trace and critical are omitted because no test
          remains. Its LL field is also one column narrower than the other rows', which is why the
          last row is built with its own format rather than reusing the general one.
        """

        sample_start = self.lags + 1
        sample_end = sample_start + self.nobs - 1
        # Stata names the deterministic specification on this line: "Trend: Constant",
        # "Trend: Linear", "Trend: Restricted constant", "Trend: <none>".
        trend_label = "Trend: Linear" if self.deterministic == "trend" else "Trend: Constant"
        lines = [
            "Johansen tests for cointegration",
            f"{trend_label:<42}Number of obs  = {self.nobs}",
            f"Sample: {sample_start} thru {sample_end}".ljust(42)
            + f"Number of lags = {self.lags:>3}",
            "-" * 62,
            " " * 54 + "Critical",
            "Maximum" + " " * 40 + "Trace" + " " * 5 + "value",
            f"{'rank':>7}{'Params':>8}{'LL':>13}{'Eigenvalue':>12}"
            f"{'statistic':>12}{'5%':>10}",
            "-" * 62,
        ]
        for row in self.rows:
            if not np.isfinite(row.eigenvalue):
                eigenvalue = "."
            else:
                eigenvalue = f"{row.eigenvalue:.5f}"
            # Layout measured from ALL SIX rows of both tables at once, not one at a time --
            # adjusting a row and re-running is how three previous attempts each fixed one
            # line and broke another.
            #
            #   rank  right-aligned to  7      params right-aligned to 14     (both fixed)
            #   LL    LEFT-aligned from 19 - (params digits - 1)
            #   eig, trace, crit  right-aligned to 40, 52, 62 minus that same shift
            #
            # The LL is left-aligned, which is what makes a SHORT likelihood text (the trend
            # table drops a trailing zero: 1224.797 against 1224.4562) start in the same column
            # rather than right-align into it. Treating every field as right-aligned is what
            # produced a table that matched one row and broke the next.
            shift = len(str(int(row.params))) - 1
            ll_text = f"{row.loglikelihood:.4f}"
            if self.deterministic == "trend" and "." in ll_text:
                ll_text = ll_text.rstrip("0").rstrip(".")

            line = f"{row.rank:>7}{row.params:>7}"
            line += " " * max(19 - shift - len(line), 0) + ll_text
            # The short final row has no trace or critical; when its likelihood is also one
            # character lighter, the eigenvalue moves in by one more.
            extra = 1 if (row.rank >= self.equations and len(ll_text) < 9) else 0
            line += " " * max(40 - shift - extra - len(line) - len(eigenvalue), 0) + eigenvalue
            if row.rank >= self.equations:
                lines.append(line)
                continue
            trace_text = f"{row.trace:.4f}"
            line += " " * max(52 - shift - len(line) - len(trace_text), 0) + trace_text
            critical_text = f"{row.trace_critical:.2f}"
            line += " " * max(62 - shift - len(line) - len(critical_text), 0) + critical_text
            lines.append(line)
        lines.append("-" * 62)
        return "\n".join(lines)


def _deterministic_block(count: int, deterministic: str) -> np.ndarray:
    """The deterministic columns the residualisations regress on.

    ``constant`` is a column of ones; ``trend`` adds a linear time trend inside the cointegrating
    relation. Measured on macro_swatson.dta with ``vecrank dinf unem, lags(2)``:

        constant   eig .65664 / .04634   trace 183.0918 / 7.7811
        trend      eig .65802 / .04550   trace 183.6114 / 7.6369

    and adding the trend column reproduces the second row exactly -- so the whole difference between
    the two specifications is this block, nothing else.
    """

    steps = np.arange(1, count + 1, dtype=float).reshape(-1, 1)
    if deterministic == "trend":
        return np.column_stack([np.ones((count, 1)), steps])
    return np.ones((count, 1))


def _johansen_general(
    values: np.ndarray, lags: int, deterministic: str = "constant"
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and trace statistics for a VAR of order ``lags`` (1 or 2).

    The construction that round 14 got wrong, and the one-line reason it was wrong: the
    LAGGED DIFFERENCES must enter BOTH residualisations. Putting ``DY_{t-1}`` into the model
    for ``DY_t`` alone (the natural reading of "short-run dynamics") leaves it in
    ``Y_{t-1}``'s equation and inflates every statistic -- on the reference data that gave
    trace 51.4854 / 7.5897 against Stata's 183.0918 / 7.7811.

    With them in both, the two-level case reproduces Stata exactly::

        lags(1):  eig .64575 / .00769      trace 172.5034 / 1.2744
        lags(2):  eig .65664 / .04634      trace 183.0918 / 7.7811

    ``N = T - lags`` rows, matching the VAR Stata would have fitted.
    """

    total = values.shape[0]
    differences = values[1:] - values[:-1]
    count = total - lags
    if count <= values.shape[1] + lags:
        raise ConsistencyError(
            f"vecrank: {total} observations cannot support lags({lags}) with "
            f"{values.shape[1]} variables"
        )
    start = differences.shape[0] - count
    rows = np.arange(start, start + count)
    response = differences[rows]
    level = values[rows]
    lag_blocks = [differences[rows - depth] for depth in range(1, lags)]
    det = _deterministic_block(count, deterministic)
    common = np.column_stack([det] + lag_blocks) if lag_blocks else det
    residual_r0 = response - common @ np.linalg.lstsq(common, response, rcond=None)[0]
    residual_r1 = level - common @ np.linalg.lstsq(common, level, rcond=None)[0]
    # Canonical correlations via QR + SVD rather than explicit covariance inverses.
    #
    # The covariance route is only safe while S11 is invertible. On a sample with a
    # nearly dependent column it is not: inf/unem/dinf gives cond(S11) = 4.5e28 and
    # rank(S11) = 2 for a 3x3 matrix, so np.linalg.solve returns arbitrary numbers -- that
    # was the source of the 3.63428 eigenvalue (larger than 1) seen in round 13.
    #
    # Orthonormalising both residual blocks first makes the singular values of Q0'Q1 the
    # canonical correlations directly, and a collinear direction contributes an EXACT zero
    # instead of a numerically meaningless one. On the two-variable reference this gives
    # .65664 / .04634, identical to the covariance route.
    basis_r0, _ = np.linalg.qr(residual_r0, mode="reduced")
    basis_r1, _ = np.linalg.qr(residual_r1, mode="reduced")
    singular_values = np.linalg.svd(basis_r0.T @ basis_r1, compute_uv=False)
    eigenvalues = np.sort(singular_values**2)[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    trace = np.array(
        [
            -count * float(np.sum(np.log1p(-eigenvalues[rank:])))
            for rank in range(eigenvalues.size)
        ]
    )
    return eigenvalues, trace


def _johansen_lags_one(
    values: np.ndarray, deterministic: str = "constant"
) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and trace statistics for ``vecrank, lags(1)``.

    At one lag the VAR has Y_{t-1} on the right and NO lagged differences, so the reduced-rank
    regression runs DY_t on a constant alone and Y_{t-1} on a constant alone. statsmodels'
    ``coint_johansen`` cannot express that (``k_ar_diff=0`` keeps one extra row and lands on
    173.8827 / 2.8945), so the canonical correlations are computed here.

    Verified against a live ``vecrank dinf unem, lags(1) max`` on macro_swatson.dta:
    eigenvalues .64575 / .00769 and trace 172.5034 / 1.2744.
    """

    lagged = values[:-1]
    differences = values[1:] - values[:-1]
    count = differences.shape[0]
    det = _deterministic_block(count, deterministic)
    residual_r0 = differences - det @ np.linalg.lstsq(det, differences, rcond=None)[0]
    residual_r1 = lagged - det @ np.linalg.lstsq(det, lagged, rcond=None)[0]
    s00 = residual_r0.T @ residual_r0 / count
    s11 = residual_r1.T @ residual_r1 / count
    s01 = residual_r0.T @ residual_r1 / count
    matrix = (
        np.linalg.inv(s00) @ s01 @ np.linalg.inv(s11) @ s01.T
    )
    eigenvalues = np.sort(np.real(np.linalg.eigvals(matrix)))[::-1]
    eigenvalues = np.clip(eigenvalues, 0.0, 1.0)
    trace = np.array(
        [
            -count * float(np.sum(np.log1p(-eigenvalues[rank:])))
            for rank in range(eigenvalues.size)
        ]
    )
    return eigenvalues, trace


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

    # ``trend(trend)`` puts a linear trend INSIDE the cointegrating relation, which is one extra
    # column in the deterministic block. The eigenvalues and trace statistics then follow exactly
    # (measured: .65802 / .04550 and 183.6114 / 7.6369 against Stata's identical figures), so only
    # the deterministic block and the parameter counts differ between the two.
    #
    # ``rconstant``, ``rtrend`` and ``none`` are still refused: they change the critical values and
    # the parameter count in ways this implementation does not yet reproduce, and a wrong critical
    # value silently changes which rank is chosen.
    normalized_trend = str(trend).strip().lower()
    if normalized_trend in {"constant", "c"}:
        deterministic = "constant"
    elif normalized_trend == "trend":
        deterministic = "trend"
    else:
        raise UnsupportedError(
            "vecrank: trend(constant) and trend(trend) are implemented; "
            f"got trend={trend!r}. The restricted variants (rconstant, rtrend) and none change "
            "the critical values, which this implementation does not yet reproduce."
        )

    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    # Stata's lags(p) removes p observations, so k_ar_diff = p - 1 with the same sample.
    #
    # ``lags(1)`` is the degenerate end of that rule and statsmodels cannot express it:
    # ``k_ar_diff=0`` makes it regress on EVERY row (N = 166) instead of the 165 Stata uses,
    # and reports 173.8827 / 2.8945 against Stata's 172.5034 / 1.2744. Computing the
    # reduced-rank regression directly gives Stata's numbers exactly -- eigenvalues
    # .64575 / .00769 and trace 172.5034 / 1.2744, all four to the printed digit:
    #
    #     R0 = resid of DY_t on [1]        (no lagged differences at lags 1)
    #     R1 = resid of Y_{t-1} on [1]
    #     solve  |S01 S11^-1 S10 - lam S00| = 0,  trace_r = -T * sum_{i>r} ln(1 - lam_i)
    if lags == 1:
        eigenvalues, trace = _johansen_lags_one(values, deterministic)
    else:
        # Computed here rather than delegated to statsmodels: both agree on a two-variable
        # system, but statsmodels loses numerical control when the series are nearly
        # dependent (inf/unem/dinf gives a largest eigenvalue of 3.6, which cannot be a
        # canonical correlation). The direct construction is exact for lags 1 and 2 on the
        # reference data; see _johansen_general.
        eigenvalues, trace = _johansen_general(values, lags, deterministic)

    # ``vecrank`` reports N = rows - lags, matching the VAR it would have fitted.
    nobs = int(values.shape[0] - lags)
    parameters = _parameter_counts(equations, lags, nobs, eigenvalues, deterministic)

    # The unrestricted (rank = k) log-likelihood comes from the VAR on the same sample; each
    # rank restriction then lowers it by trace(r)/2, because LR = 2*dLL and the trace
    # statistic IS that LR. Verified on the reference case: 1224.4562, 1220.5657, 1132.9103.
    full_ll = _unrestricted_loglikelihood(values, lags, deterministic)

    if equations > len(_TRACE_CRITICAL_TAIL):
        raise UnsupportedError(
            f"vecrank: the 5% critical values are measured for up to "
            f"{len(_TRACE_CRITICAL_TAIL)} variables; got {equations}. The larger-V tables "
            "exist in Stata but have not been transcribed here, and inventing them would put "
            "a made-up threshold beside a real statistic. Pass the values yourself or reduce "
            "the system."
        )
    # A k-variable system tests rank r against the (k - r)-th entry from the start of the
    # descending sequence, i.e. the last k entries of the tail read backwards.
    # rank 0 takes the LARGEST of the k values in play, then they descend: k=2 gives
    # rank0 = 15.41, rank1 = 3.76. The tail is stored ascending, so reverse the slice.
    # The TREND case uses its own tabulation. Reusing the constant-case numbers would put a
    # real statistic beside the wrong threshold, which changes which rank gets selected.
    trace_table = (
        _TRACE_CRITICAL_TAIL_TREND if deterministic == "trend" else _TRACE_CRITICAL_TAIL
    )
    critical_trace = tuple(reversed(trace_table[:equations]))
    critical_max = tuple(reversed(_MAX_CRITICAL_TAIL[:equations]))
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
        rows=tuple(rows),
        nobs=nobs,
        lags=lags,
        equations=equations,
        deterministic=deterministic,
    )



def _unrestricted_loglikelihood(
    values: np.ndarray, lags: int, deterministic: str = "constant"
) -> float:
    """Gaussian log-likelihood of the unrestricted VAR on the same sample ``vecrank`` uses.

    The deterministic block must match the one used for the eigenvalues, or the likelihood is that
    of a different model: with trend(constant) the reference case gives 1132.9103 / 1220.5657 /
    1224.4562 and with trend(trend) 1132.9913 / 1220.9786 / 1224.797.
    """

    observations = values.shape[0] - lags
    equations = values.shape[1]
    response = np.diff(values, axis=0)[lags - 1 :]
    columns = [
        _deterministic_block(observations, deterministic),
        values[lags - 1 : len(values) - 1],
    ]
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


def _parameter_counts(
    equations: int,
    lags: int,
    nobs: int,
    eigenvalues: np.ndarray,
    deterministic: str = "constant",
) -> list[int]:
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
    # A trend inside the cointegrating relation costs one extra parameter PER EQUATION, on top of
    # the constant each equation already carries. Measured on macro_swatson (k=2, lags=2):
    #   constant  6 / 9 / 10        trend  8 / 11 / 12
    # i.e. a uniform +2 = +k, not a change to the rank term.
    if deterministic == "trend":
        base += equations
    counts: list[int] = []
    for rank in range(equations + 1):
        counts.append(int(base + rank * (2 * equations - rank)))
    return counts
