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
    #: Whether ``max`` was requested. Stata then prints the trace table FOLLOWED BY a second table
    #: of maximum-eigenvalue statistics, which is what ``vecrank ..., max`` always does.
    show_max: bool = False

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
        # Stata names the deterministic specification on this line. With all five labels measured,
        # "Number" starts at column 42 for EVERY case -- a plain ljust(42). Round 137 inferred a
        # centring formula and a special 41-column Linear case from two labels and one
        # mistyped reference line; five labels settle it.
        trend_label = _TREND_LABELS.get(self.deterministic, "Trend: Constant")
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
            #   rank      right-aligned to 7
            #   params    right-aligned to 7 + (k + 5)   -- the field widens with the system
            #   LL        LEFT-aligned from 19 - (params digits - 1)
            #   eig/trace/crit  right-aligned to 40 / 52 / 62 minus that same shift
            #
            # The Params width is k + 5, not a constant: k=2 ends the field at 14 and k=3 at 15.
            # Hard-coding 7 put every k=3 table one column out. The LL is RIGHT-ALIGNED to
            # column 28 - shift; round 138 inferred "LEFT-aligned" from six rows whose likelihoods
            # all happened to be nine characters wide, which hid the rule. With the five trend
            # specifications the likelihoods are 8 or 9 characters (1217.468 vs 1224.4562) and
            # both still END at 28, proving right alignment.
            #
            # Stata also STARTS the selected rank with "*" and prints "* selected rank" below the
            # rule. On a three-variable system: "     17.7528*    29.68".
            # The Params field is 7 columns plus one for each extra digit of the value. Stata
            # right-aligns the value to column 14 for one digit and 15 for two; the following
            # columns stay at their ABSOLUTE positions (28 / 40 / 52 / 62), so a wider Params
            # value simply leaves one fewer space before the likelihood.
            params_width = 7 + len(str(int(row.params))) - 1
            # EIGHT SIGNIFICANT FIGURES, not four decimals. "%.4f" happened to be right while
            # every likelihood was positive and four-digit (1132.9103) and wrong the moment a
            # three-variable system produced a negative one: Stata prints -628.85566, not
            # -628.8557. Trailing zeros are then dropped, which is why 4500.724 shows seven.
            ll_text = _vecrank_significant(row.loglikelihood, 8)

            line = f"{row.rank:>7}{row.params:>{params_width}}"
            # EVERY column has a FIXED end position: LL 28, eigenvalue 40, trace 52, critical 62.
            # This is the rule that all five trend specifications and both k=2 and k=3 agree on.
            # Round 138 inferred that the later columns moved with the Params width; that was an
            # artefact of a left-aligned LL. With right alignment the Params field may widen by
            # one column but the fields after it do not move at all.
            line += " " * max(28 - len(line) - len(ll_text), 0) + ll_text
            # Every later field is at a FIXED end column, minus the same shift: 40 (eigenvalue),
            # 52 (trace), 62 (critical). Round 138's "extra" rule for short rows was an artefact of
            # a left-aligned LL; with right alignment all five specifications put the short row's
            # eigenvalue at 40 - shift.
            line += " " * max(40 - len(line) - len(eigenvalue), 0) + eigenvalue
            if row.rank >= self.equations:
                lines.append(line)
                continue
            trace_text = f"{row.trace:.4f}"
            # The star rides ON the trace field: it does not consume a column, so the field is
            # padded using the WIDTH WITHOUT the star and the star is appended afterwards.
            # Padding with it included pushed the critical value one column right (63 instead of
            # Stata's 62) on exactly the row that carried the marker.
            selected = row.rank == self.rank
            line += " " * max(52 - len(line) - len(trace_text), 0) + trace_text
            if selected:
                line += "*"
            critical_text = f"{row.trace_critical:.2f}"
            line += " " * max(62 - len(line) - len(critical_text), 0) + critical_text
            lines.append(line)
        lines.append("-" * 62)
        if self.rank < self.equations:
            lines.append("* selected rank")
        if self.show_max:
            # ``vecrank ..., max`` prints a SECOND table: the maximum-eigenvalue statistic, which
            # is -N*ln(1 - lambda_i) rather than the trace's sum over the remaining eigenvalues.
            # Same columns and row layout, with the trace swapped for the maximum and its own
            # critical value. Values verified against Stata: 175.3107 / 14.07 at k=2.
            lines.extend(
                [
                    "-" * 62,
                    " " * 54 + "Critical",
                    "Maximum" + " " * 24 + "------Eigenvalue-----" + " " * 5 + "value",
                    # Columns measured against Stata: "Maximum" ends at 52 and "5%" at 62,
                    # which a plain :>8 / :>7 pair does not produce.
                    f"{'rank':>7}{'Params':>8}{'LL':>13}"
                    + " " * 17
                    + "Maximum"
                    + " " * 8
                    + "5%",
                    "-" * 62,
                ]
            )
            for row in self.rows:
                eigenvalue = (
                    "." if not np.isfinite(row.eigenvalue) else f"{row.eigenvalue:.5f}"
                )
                params_width = 7 + len(str(int(row.params))) - 1
                ll_text = _vecrank_significant(row.loglikelihood, 8)
                line = f"{row.rank:>7}{row.params:>{params_width}}"
                line += " " * max(28 - len(line) - len(ll_text), 0) + ll_text
                line += " " * max(40 - len(line) - len(eigenvalue), 0) + eigenvalue
                if row.rank < self.equations:
                    statistic_text = f"{row.maximum:.4f}"
                    line += " " * max(52 - len(line) - len(statistic_text), 0) + statistic_text
                    critical_text = f"{row.maximum_critical:.2f}"
                    line += " " * max(62 - len(line) - len(critical_text), 0) + critical_text
                lines.append(line)
            lines.append("-" * 62)
        return "\n".join(lines)


def _vecrank_significant(value: float, digits: int) -> str:
    """Format ``value`` to ``digits`` significant figures, dropping trailing zeros.

    Python's ``g`` presentation does NOT do this: it prints as many digits as the float carries
    once they fit inside the requested precision, so ``f"{v:.9g}"`` on -628.855659 returns
    "-628.855659" rather than a nine- or eight-digit form. Stata prints a fixed number of
    significant figures, so the rounding has to be explicit.

    Verified against live ``vecrank`` output, which uses EIGHT:

        1132.9103   (8 sig figs, 4 decimals)
        -628.85566  (8 sig figs, 5 decimals -- the failure "%.4f" produced as -628.8557)
        4500.724    (8 sig figs, trailing zero dropped, so seven digits show)
    """

    from decimal import Decimal, ROUND_HALF_UP

    number = Decimal(repr(float(value)))
    if number == 0:
        return "0"
    quantum = Decimal(1).scaleb(number.adjusted() - (digits - 1))
    text = format(number.quantize(quantum, rounding=ROUND_HALF_UP), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


#: Johansen's five deterministic specifications. ``unrestricted`` is the number of deterministic
#: columns regressed out of BOTH dY_t and Y_{t-1} (0, 1 for a constant, 2 for constant + trend).
#: ``restricted`` is the term that belongs INSIDE the cointegrating relation and is appended to R1
#: as an extra column before the canonical correlations -- NOT a different residualising block.
#:
#: Round 142 proved this by construction: two probes over all five cases, one with asymmetric
#: residualising blocks (3 of 5 matched) and one with the augmented R1 (5 of 5 matched), against
#: live `vecrank dinf unem, lags(2) trend(<case>)` on macro_swatson.dta.
_DETERMINISTIC_CASES: dict[str, tuple[int, str | None]] = {
    "none": (0, None),
    "rconstant": (0, "constant"),
    "constant": (1, None),
    "rtrend": (1, "trend"),
    "trend": (2, None),
}

#: 5% trace critical values, per specification, in the same ascending "tail" order as the
#: constant table. Measured from live runs at k=2 and k=3, which is enough for the three-variable
#: guard; a larger system is refused rather than given invented thresholds.
_TRACE_CRITICAL_TAILS: dict[str, tuple[float, ...]] = {
    "none": (3.84, 12.53, 24.31),
    "rconstant": (9.42, 19.96, 34.91),
    "constant": (3.76, 15.41, 29.68),
    "rtrend": (12.25, 25.32, 42.44),
    "trend": (3.74, 18.17, 34.55),
}

#: The maximum-eigenvalue 5% critical tables use the SAME tail values but a different second
#: entry for the restricted cases (9.24 vs 9.42, 12.52 vs 12.25). Measured from live
#: `vecrank ..., max trend(<case>)` runs at k=2 and k=3. The k=3 entries here were measured,
#: NOT extrapolated: an earlier draft of this line guessed 21.12/27.42/31.79 and was wrong on
#: all three.
_MAX_CRITICAL_TAILS: dict[str, tuple[float, ...]] = {
    "none": (3.84, 11.44, 17.89),
    "rconstant": (9.24, 15.67, 22.00),
    "constant": (3.76, 14.07, 20.97),
    "rtrend": (12.52, 18.96, 25.54),
    "trend": (3.74, 16.87, 23.78),
}

#: Stata's "Trend:" label for each specification.
_TREND_LABELS: dict[str, str] = {
    "none": "Trend: <none>",
    "rconstant": "Trend: Restricted constant",
    "constant": "Trend: Constant",
    "rtrend": "Trend: Restricted",
    "trend": "Trend: Linear",
}


def _deterministic_block(count: int, deterministic: str) -> np.ndarray | None:
    """The UNRESTRICTED deterministic columns regressed out of both residualisations.

    ``None`` means no deterministic term at all (``trend(none)`` and ``trend(rconstant)``). For
    ``rconstant``/``rtrend`` the restricted term is handled separately by ``_johansen_general``,
    which appends it to R1 -- that asymmetry is what makes those two cases different, and why an
    earlier attempt that only swapped the residualising block reproduced 3 of 5 cases, not 5.
    """

    steps = np.arange(1, count + 1, dtype=float).reshape(-1, 1)
    unrestricted, _ = _DETERMINISTIC_CASES[deterministic]
    if unrestricted == 0:
        return None
    if unrestricted == 1:
        return np.ones((count, 1))
    return np.column_stack([np.ones((count, 1)), steps])


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
    base_columns = ([det] if det is not None else []) + lag_blocks
    common = np.column_stack(base_columns) if base_columns else None

    def residualise(matrix: np.ndarray) -> np.ndarray:
        if common is None:
            return matrix
        return matrix - common @ np.linalg.lstsq(common, matrix, rcond=None)[0]

    residual_r0 = residualise(response)
    residual_r1 = residualise(level)

    # The RESTRICTED deterministic term belongs inside the cointegrating relation, so it is
    # APPENDED to R1 as an extra pseudo-variable. This is the step that separates rconstant and
    # rtrend from the symmetric cases. Verified on all five specifications.
    _, restricted = _DETERMINISTIC_CASES[deterministic]
    if restricted is not None:
        steps = np.arange(1, count + 1, dtype=float).reshape(-1, 1)
        term = np.ones((count, 1)) if restricted == "constant" else steps
        residual_r1 = np.column_stack([residual_r1, residualise(term)])
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
    # An augmented R1 (restricted cases) has more columns than equations; only the first k
    # canonical correlations are eigenvalues of the cointegrating system.
    eigenvalues = eigenvalues[: values.shape[1]]
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

    def residualise(matrix: np.ndarray) -> np.ndarray:
        if det is None:
            return matrix
        return matrix - det @ np.linalg.lstsq(det, matrix, rcond=None)[0]

    residual_r0 = residualise(differences)
    residual_r1 = residualise(lagged)
    _, restricted = _DETERMINISTIC_CASES[deterministic]
    if restricted is not None:
        steps = np.arange(1, count + 1, dtype=float).reshape(-1, 1)
        term = np.ones((count, 1)) if restricted == "constant" else steps
        residual_r1 = np.column_stack([residual_r1, residualise(term)])
    # QR + SVD, as in _johansen_general: an augmented R1 may have more columns than equations,
    # and the covariance route needs an invertible S11 that it no longer has.
    basis_r0, _ = np.linalg.qr(residual_r0, mode="reduced")
    basis_r1, _ = np.linalg.qr(residual_r1, mode="reduced")
    singular = np.linalg.svd(basis_r0.T @ basis_r1, compute_uv=False)
    eigenvalues = np.clip(np.sort(singular**2)[::-1], 0.0, 1.0)[: values.shape[1]]
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
    show_max: bool = False,
) -> VecRankResult:
    """Johansen cointegration rank test, as ``vecrank``.

    ``data`` holds the endogenous series in time order (rows) by equation (columns); missing
    rows must already be dropped. ``lags`` counts the VAR lag order as Stata does, so the
    number of differenced lags is ``lags - 1``.

    ``show_max=True`` is Stata's ``max`` option: the trace table is printed with a SECOND table of
    maximum-eigenvalue statistics below it. Stata always shows both when ``max`` is given, so this
    is a display flag -- the ``maximum`` and ``maximum_critical`` fields are populated either way.
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

    # All five of Stata's trend() specifications are Johansen's five deterministic cases. The
    # symmetric ones (none, constant, trend) only differ in how many columns are regressed out of
    # both residualisations; the RESTRICTED ones (rconstant, rtrend) additionally put a term
    # INSIDE the cointegrating relation by appending it to R1. Round 142 proved that construction
    # against live Stata on all five cases, so the earlier refusal is lifted.
    normalized_trend = str(trend).strip().lower()
    aliases = {"c": "constant", "rc": "rconstant", "rt": "rtrend", "n": "none"}
    deterministic = aliases.get(normalized_trend, normalized_trend)
    if deterministic not in _DETERMINISTIC_CASES:
        raise UnsupportedError(
            f"vecrank: unknown trend={trend!r}; use none, rconstant, constant, rtrend or trend"
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

    trace_table = _TRACE_CRITICAL_TAILS[deterministic]
    max_table = _MAX_CRITICAL_TAILS[deterministic]
    if equations > len(trace_table) or equations > len(max_table):
        raise UnsupportedError(
            f"vecrank: the 5% critical values are measured for up to "
            f"{min(len(trace_table), len(max_table))} variables; got {equations}. The larger-V "
            "tables exist in Stata but have not been transcribed here, and inventing them would "
            "put a made-up threshold beside a real statistic. Reduce the system."
        )
    # A k-variable system tests rank r against the (k - r)-th entry from the start of the
    # descending sequence, i.e. the last k entries of the tail read backwards.
    # rank 0 takes the LARGEST of the k values in play, then they descend: k=2 gives
    # rank0 = 15.41, rank1 = 3.76. The tail is stored ascending, so reverse the slice.
    # Each specification has its OWN tabulation; reusing the constant numbers would put a real
    # statistic beside the wrong threshold and silently change which rank gets selected.
    critical_trace = tuple(reversed(trace_table[:equations]))
    critical_max = tuple(reversed(max_table[:equations]))
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
        show_max=bool(show_max),
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
    # In the UNRESTRICTED (rank = k) VAR the restricted deterministic term is just another
    # unrestricted column: it is a constant for rconstant and a trend for rtrend. So the number
    # of deterministic columns here is unrestricted + n_restricted, NOT just unrestricted.
    # Using only the unrestricted block gave full_ll 1217.468 for rconstant, which made rank 0's
    # LL 1125.8841 against Stata's 1132.8723 -- a 7-unit error from a mis-specified full model.
    unrestricted, restricted = _DETERMINISTIC_CASES[deterministic]
    n_columns = unrestricted + (0 if restricted is None else 1)
    steps = np.arange(1, observations + 1, dtype=float).reshape(-1, 1)
    det_columns: list[np.ndarray] = []
    if n_columns >= 1:
        det_columns.append(np.ones((observations, 1)))
    if n_columns >= 2:
        det_columns.append(steps)
    columns = det_columns + [values[lags - 1 : len(values) - 1]]
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
    """Stata's Params column for all five deterministic specifications.

    Round 142 measured all five on macro_swatson (k=2, lags=2) and fitted ONE formula:

        base    = k*(k*(lags-1)) + n_unrestricted*k
        rank r  = base + r*(2k - r + n_restricted)

    which gives 4/7/8 (none), 4/8/10 (rconstant), 6/9/10 (constant), 6/10/12 (rtrend) and
    8/11/12 (trend). The restricted term adds ONE to the rank coefficient, not to the base --
    that is the whole reason rconstant's second row is 8 where none's is 7.
    """

    unrestricted, restricted = _DETERMINISTIC_CASES[deterministic]
    base = equations * (equations * (lags - 1)) + unrestricted * equations
    n_restricted = 0 if restricted is None else 1
    counts: list[int] = []
    for rank in range(equations + 1):
        counts.append(int(base + rank * (2 * equations - rank + n_restricted)))
    return counts
