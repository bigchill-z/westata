"""Serial-correlation tests: Stata's ``estat dwatson``, ``wntestq`` and ``corrgram``.

All three are builtin commands, but their arithmetic is readable in ``ac.ado`` /
``pac.ado`` / ``corrgram.ado``, which is where the rules below come from -- and numbers
measured against a live Stata run on ``icecream.dta`` after
``reg consumption temp price income``::

    estat dwatson            Durbin-Watson d(4, 30) = 1.021169
    wntestq e1               Q 26.1974, chi2(13), p .0160
    wntestq e1, lags(10)     Q 16.3718, p .0895
    corrgram e1              lag 1: ac .3298, pac .3969, Q 3.6, p .0578
                             lag 2: ac .0362, pac -.1681, Q 3.645, p .1616

Three rules decide whether a reimplementation agrees with Stata, and all three were wrong
here at some point:

* **The default lag count is ``max(1, min(int(n/2) - 2, 40))``** -- ``ac.ado:71-73`` and
  ``corrgram.ado:18`` spell it identically. It does not depend on how many parameters the
  previous regression estimated. The earlier ``(n - k)//2`` happened to equal it whenever
  ``k == 4``, and every fixture here had four parameters, so the bug was invisible on the
  documented examples and 1 lag off on a one-regressor fit -- then far off once ``n``
  reached 84, where Stata's 40 cap bites.
* **``L.`` is on the TIME axis**, so a period missing from the sample makes that lag
  missing instead of pairing the neighbouring rows; see :mod:`westata.postestimation.ts_lags`.
* **``corrgram``'s PAC is the coefficient on lag ``k``** in a regression of the residual on
  its first ``k`` lags WITH a constant (``pac.ado:128-156``, the non-``yw`` branch), not the
  Yule-Walker recursion. On icecream.dta at lag 1 that is .3969, against .3298 for the
  autocorrelation and .4006 for the no-intercept slope.

``estat bgodfrey`` lives in :mod:`westata.postestimation.bgodfrey`; this module's docstring
used to declare it unimplementable, which was wrong -- the formula was in
``ado/base/b/bgodfrey.ado`` the whole time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError
from westata.postestimation.ts_lags import (
    TimeAxis,
    refuse_panel_fit,
    resolve_time_axis,
)

__all__ = [
    "Correlogram",
    "DurbinWatsonResult",
    "PortmanteauResult",
    "estat_dwatson",
    "estat_corrgram",
    "wntestq",
]

#: ``ac.ado:71-73`` and ``corrgram.ado:18``: ``max(1, min(int(n/2) - 2, 40))``.
#:
#: The 40 is a hard ceiling in both, and it is what the earlier ``(n - k)//2`` missed
#: entirely: at ``n = 758`` Stata reports ``chi2(40)`` while that formula reported 377.
MAX_DEFAULT_LAGS = 40


def _default_lags(nobs: int) -> int:
    """Stata's default lag order: ``max(1, min(int(n/2) - 2, 40))``.

    Measured at ``n = 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 101, 120, 200, 500``:
    the last lag is 3, 8, 13, 18, 23, 28, 33, 38, 40, 40, 40, 40, 40, 40, 40. The count is
    independent of the previous regression's parameter count -- ``reg u x1`` (2 parameters)
    and ``reg u x1 ... x6`` (7) both give 13 at ``n = 30``.
    """

    return max(1, min(int(nobs // 2) - 2, MAX_DEFAULT_LAGS))


def _residuals(result: Any, command: str) -> np.ndarray:
    """Residuals of a fitted result, or a bare series passed through unchanged.

    ``wntestq e1`` / ``corrgram e1`` take a VARIABLE in Stata, so the series form is the
    primary one; accepting a result object is this package's convenience. A result carrying
    residuals is used for those, which is what every existing caller passes.
    """

    resid = getattr(result, "resid", None)
    if resid is not None:
        values = np.asarray(resid, dtype=float).reshape(-1)
    else:
        try:
            values = np.asarray(result, dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ConsistencyError(
                f"{command}: needs a fitted result carrying residuals or a numeric series"
            ) from exc
    if values.size == 0:
        raise ConsistencyError(f"{command}: the series is empty")
    if not np.isfinite(values).all():
        raise ConsistencyError(
            f"{command}: the series contains missing values; Stata drops them with "
            "`if`, which is not assumed here"
        )
    return values


@dataclass(frozen=True)
class DurbinWatsonResult:
    """``estat dwatson``: the Durbin-Watson d statistic.

    Stata prints one line::

        Durbin-Watson d-statistic(  4,    30) =  1.021169

    The first number is the number of parameters EXCLUDING the constant (``e(df_m)``), the
    second the sample size.

    With gaps in the sample, ``estat dwatson`` prints ``Number of gaps in sample = N`` on
    its own line first (measured: ``drop if time == 5`` gives ``= 1``, and removing two
    periods gives ``= 2``), then a blank line, then the statistic.
    """

    d: float
    df: float
    nobs: int
    #: ``r(N_gaps)``; zero for a contiguous sample, which prints no note.
    n_gaps: int = 0

    def __str__(self) -> str:
        """Stata's single-line form, with a LEADING blank line.

        Two details measured in round 192:

        * the raw capture begins with a newline, so the block is two lines -- blank, then the
          statistic. Emitting one line put it where Stata's blank line should be;
        * the value is SIX DECIMALS with a trailing zero dropped, not six significant figures.
          Three fits settle it: 1.021169 (icecream), 1.704354 (grilic/3 regressors) and 1.75295
          (grilic/5 regressors) -- the last lost its trailing zero from 1.752950. '%.6g' gave
          1.70435 for the middle one, which is six FIGURES rather than six decimals.
        """

        # TWO RULES, SPLIT AT 1 -- the same shape this package has had to apply to several other tables.
        # Measured at two sample sizes, where the statistic is small:
        #
        #     0.0127663  -> .0127663     seven SIGNIFICANT figures, leading zero dropped
        #     0.0070096  -> .0070096
        #
        # while the earlier captures all above 1 were six DECIMALS with the trailing zero trimmed:
        #
        #     1.021169   1.704354   1.75295
        #
        # Six decimals below 1 gave '.012766' where Stata prints '.0127663'.
        if abs(float(self.d)) < 1.0:
            # SEVEN DECIMALS WITH THE TRAILING ZEROS TRIMMED, then the leading zero dropped. Not seven
            # significant figures: 0.007009565 renders as .0070096 (five significant figures), which
            # ".6g" shortens to .00700956 and ".7g" lengthens to .007009565. Three values settle it:
            #
            #     0.007009565 -> .0070096
            #     0.01276632  -> .0127663
            #     0.012766    -> .012766
            value = f"{abs(float(self.d)):.7f}".rstrip("0")
            if value.startswith("0."):
                value = value[1:]
            value = ("-" if float(self.d) < 0 else "") + value
        else:
            value = f"{self.d:.6f}".rstrip("0")
            if value.endswith("."):
                value += "0"
        prefix = f"Number of gaps in sample = {self.n_gaps}\n" if self.n_gaps else ""
        return (
            prefix
            + "\n"
            f"Durbin\u2013Watson d-statistic({self.df:>3.0f}, {self.nobs:>5d})"
            f" = {value:>9}"
        )


def estat_dwatson(
    result: Any,
    *,
    time: str | Any | None = None,
    delta: float = 1.0,
) -> DurbinWatsonResult:
    r"""Durbin-Watson d.

    .. math:: d = \sum_{t=2}^{n}(e_t - e_{t-1})^2 / \sum_{t=1}^{n} e_t^2

    Measured on icecream.dta after ``reg consumption temp price income``: 1.021169.

    ``time`` stands in for ``tsset`` and is what decides which differences exist: with a
    period missing from the sample, the pair straddling the gap is dropped rather than
    differenced, because ``L.`` is undefined across it. Measured on icecream.dta with
    ``time == 5`` dropped: Stata reports ``.8930395``, and differencing adjacent ROWS gives
    ``1.0868969``.
    """

    resid = _residuals(result, "estat dwatson")
    if resid.size < 2:
        raise ConsistencyError("Durbin-Watson needs at least two observations")
    refuse_panel_fit(result, "estat dwatson")
    axis = resolve_time_axis(result, time, delta=delta, command="estat dwatson")

    denominator = float(resid @ resid)
    if denominator <= 0.0:
        raise ConsistencyError("Durbin-Watson: the residuals are all zero")
    if axis is None:
        differences = np.diff(resid)
        numerator = float(differences @ differences)
        n_gaps = 0
    else:
        # Only consecutive PERIODS contribute; a row whose predecessor is absent is skipped.
        source = axis.lag_indices(1)
        present = source >= 0
        numerator = float(np.sum((resid[present] - resid[source[present]]) ** 2))
        n_gaps = axis.n_gaps
    d = numerator / denominator
    # Stata prints the PARAMETER count including the constant: "(4, 30)" for three
    # regressors plus _cons. e(df_m) is 3 there, so it needs the +1.
    df_model = float(getattr(result, "df_model", np.nan)) + 1.0
    nobs = int(np.asarray(resid).size)
    return DurbinWatsonResult(d=d, df=df_model, nobs=nobs, n_gaps=n_gaps)


def _autocorrelations(resid: np.ndarray, lags: int, axis: TimeAxis | None) -> list[float]:
    """``ac``'s ``ac_1..ac_lags``: ``sum(xm * L^k.xm) / sum(xm^2)``.

    ``ac.ado:133-136`` puts the product ``xm*L`i'.xm`` through ``summarize`` -- which skips
    missing -- and divides by ``R0 = r(Var)*(n-1)``, i.e. the sum of squared deviations. So
    a gap removes those pairs from the NUMERATOR and leaves the denominator alone.
    """

    nobs = resid.size
    centred = resid - resid.mean()
    denominator = float(centred @ centred)
    if denominator <= 0.0:
        raise ConsistencyError("the residuals are constant")
    values: list[float] = []
    for k in range(1, lags + 1):
        if axis is None:
            numerator = float(centred[k:] @ centred[: nobs - k])
        else:
            source = axis.lag_indices(k)
            present = source >= 0
            numerator = float(centred[present] @ centred[source[present]])
        values.append(numerator / denominator)
    return values


def _partial_autocorrelations(
    resid: np.ndarray, lags: int, axis: TimeAxis | None
) -> list[float]:
    """``pac``'s ``pac_k``: the coefficient on lag ``k``, with a constant.

    ``pac.ado:128-156`` runs ``reg xm L(1/i).xm`` for each ``i`` and keeps
    ``_b[L`i'.xm]``. On icecream.dta lag 1 gives .3969 -- the intercept-included slope, NOT
    the Yule-Walker value ``ac_1`` = .3298 (``yw`` is a separate option) and not the
    no-intercept slope .4006.

    A lag whose regression cannot be estimated is reported as ``NaN``, which is what Stata
    prints as ``.``: measured at lag 12 on the gapped icecream sample (28 usable rows,
    13 parameters), where Stata's ``corrgram`` shows a blank PAC and no bar.
    """

    nobs = resid.size
    values: list[float] = []
    for order in range(1, lags + 1):
        columns: list[np.ndarray] = []
        for depth in range(1, order + 1):
            column = np.full(nobs, np.nan)
            if axis is None:
                column[depth:] = resid[: nobs - depth]
            else:
                source = axis.lag_indices(depth)
                present = source >= 0
                column[present] = resid[source[present]]
            columns.append(column)
        augmented = np.column_stack([np.ones(nobs), *columns])
        keep = np.isfinite(augmented).all(axis=1)
        used = int(keep.sum())
        if used <= order + 1:
            values.append(float("nan"))
            continue
        block = augmented[keep]
        if np.linalg.matrix_rank(block) < block.shape[1]:
            values.append(float("nan"))
            continue
        coefficients, *_ = np.linalg.lstsq(block, resid[keep], rcond=None)
        values.append(float(coefficients[-1]))
    return values


@dataclass(frozen=True)
class PortmanteauResult:
    """``wntestq``: Ljung-Box portmanteau Q for white noise."""

    q: float
    df: int
    pvalue: float
    lags: int
    #: ``r(N_gaps)``; ``wntestq`` announces a gap on its own line first.
    n_gaps: int = 0

    def __str__(self) -> str:
        note = ""
        if self.n_gaps:
            note = (
                "(note: time series has 1 gap)"
                if self.n_gaps == 1
                else f"(note: time series has {self.n_gaps} gaps)"
            ) + "\n\n"
        return (
            note
            + "Portmanteau test for white noise\n"
            + "-" * 39
            + f"\n Portmanteau (Q) statistic = {self.q:>10.4f}"
            + f"\n Prob > chi2({self.df})           = {self.pvalue:>10.4f}"
        )


def wntestq(
    result: Any,
    *,
    lags: int | None = None,
    time: str | Any | None = None,
    delta: float = 1.0,
) -> PortmanteauResult:
    r"""Ljung-Box Q, as ``wntestq``.

    ``corrgram.ado:68`` builds it as

    .. math:: Q = n(n+2) \sum_{k=1}^{p} \hat\rho_k^2 / (n-k)

    ``wntestq.ado:16-24`` takes the default lag count from ``ac``, so it is
    ``max(1, min(int(n/2) - 2, 40))`` -- measured: 30 observations give ``chi2(13)``, and
    758 give ``chi2(40)`` rather than the 377 an ``(n-k)//2`` rule would claim. Measured
    numbers: Q 26.1974, p .0160; with ``lags(10)``: Q 16.3718, p .0895.

    ``time`` stands in for ``tsset``; see :func:`estat_dwatson`.
    """

    resid = _residuals(result, "wntestq")
    refuse_panel_fit(result, "wntestq")
    axis = resolve_time_axis(result, time, delta=delta, command="wntestq")
    n = int(resid.size)
    if lags is None:
        lags = _default_lags(n)
    lags = int(lags)
    if lags < 1 or lags >= n:
        raise ValueError(f"lags must lie in 1..{n - 1}; got {lags}")

    ac = _autocorrelations(resid, lags, axis)
    total = 0.0
    for k, rho in enumerate(ac, start=1):
        total += rho * rho / (n - k)
    q = float(n * (n + 2) * total)
    pvalue = float(stats.chi2.sf(q, lags))
    return PortmanteauResult(
        q=q, df=lags, pvalue=pvalue, lags=lags, n_gaps=0 if axis is None else axis.n_gaps
    )


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
    #: ``r(N_gaps)``; ``corrgram`` prints ``(note: time series has N gap(s))`` before the
    #: table when it is non-zero.
    n_gaps: int = 0

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
        """Stata's ``corrgram`` block, captured byte for byte in round 185.

        Every line is 79 characters. Measured column ends: AC at 14, PAC at 23, Q at 33,
        Prob>Q at 41, and the two bar charts are centred on columns 51 (autocorrelation) and
        70 (partial autocorrelation), with the '-' strokes filling leftward or rightward.

        THE BAR SCALE IS ONE CHARACTER PER 0.1 of the statistic, capped at five, which is why
        ac .6627 prints '|-----' while ac .0362 prints a bare '|'.

        A missing PAC (too few usable rows, which a gap can cause) prints '.' in its column
        and no bar. Stata's line there stops after the AC bar -- 62 characters on the gapped
        icecream capture -- while this renderer keeps the 79-column frame; the numbers agree
        either way, and the trailing spaces are the only difference.
        """

        centre_ac, centre_pac = 51, 70

        def paint(canvas: list[str], value: float, centre: int) -> None:
            """Draw one bar chart into ``canvas`` at the absolute column ``centre``.

            Transcribed from ``corrgram.ado``'s ``MkString`` (round 186) rather than fitted:

                if corr >= 0:  vb = 9;                  ve = int(8*corr) + 9
                else:          ve = 9;                  vb = int(8*corr) + 9

            and then every position from ``vb`` to ``ve`` is stroked, with the pipe at position 9.
            So the scale is EIGHT characters per unit of correlation, and the boundary is a
            TRUNCATION of ``8*corr``, not a rounding of ``10*corr``.

            Round 185 spent a whole round on this and concluded "neither pure rounding nor pure
            truncation reproduces the column" -- both guesses were of the wrong FORM (10 per
            unit, rounded) and the reference value I was checking against (-.0934 -> 1 stroke) was
            mis-transcribed; Stata draws zero strokes there. Reading the ado answered it in one
            step, which is the third time that has been the difference (varsoc's FPE, round 180;
            this, round 186).
            """
            if not np.isfinite(value):
                return
            scaled = int(8.0 * float(value))
            start = 9 if value >= 0 else 9 + scaled
            finish = 9 + scaled if value >= 0 else 9
            canvas[centre] = "|"
            for position in range(min(start, finish), max(start, finish) + 1):
                if position != 9:
                    canvas[centre + position - 9] = "-"

        def cell(text: str, end: int, current: int) -> str:
            return " " * max(end - current - len(text), 0) + text

        width = 79
        lines: list[str] = []
        if self.n_gaps:
            lines.append(
                "(note: time series has 1 gap)"
                if self.n_gaps == 1
                else f"(note: time series has {self.n_gaps} gaps)"
            )
        lines.extend(
            [
                " " * 42 + "-1" + " " * 7 + "0" + " " * 7 + "1 -1" + " " * 7 + "0" + " " * 7 + "1",
                # Header end columns measured from Stata: LAG starts at 1, AC ends at 13, PAC at 23,
                # Q at 30, Prob>Q at 41; the two chart labels start at 43 and 62.
                " "
                + "LAG".ljust(7)
                + "AC".rjust(5)
                + " " * 7
                + "PAC".rjust(3)
                + " " * 6
                + "Q"
                + " " * 5
                + "Prob>Q".rjust(6)
                + "  [Autocorrelation]  [Partial autocor]",
                "-" * width,
            ]
        )
        for row in self.rows:
            head = f"{row.lag:<3}"
            head += cell(f"{row.ac:.4f}", 15, len(head))
            # A missing PAC prints '.' at the same column, as Stata writes it.
            head += cell("." if not np.isfinite(row.pac) else f"{row.pac:.4f}", 24, len(head))
            head += cell(f"{row.q:.5g}", 33, len(head))
            head += cell(f"{row.pvalue:.4f}", 41, len(head))
            canvas = [" "] * width
            paint(canvas, row.ac, centre_ac)
            paint(canvas, row.pac, centre_pac)
            line = (head.ljust(42) + "".join(canvas)[42:])[:width]
            lines.append(line.ljust(width))
        return "\n".join(lines)


def estat_corrgram(
    result: Any,
    *,
    lags: int | None = None,
    time: str | Any | None = None,
    delta: float = 1.0,
) -> Correlogram:
    """``corrgram``: autocorrelations, partial autocorrelations and Q at each lag.

    ``corrgram.ado:68`` accumulates ``Q = n(n+2) sum ac^2/(n-k)`` as it goes, so the Q column
    is the Ljung-Box statistic using lags ``1..k`` -- which is why the last row equals what
    ``wntestq`` reports for the same order.

    The default order is ``max(1, min(int(n/2) - 2, 40))`` (``corrgram.ado:18``), not a
    function of the previous regression's parameter count.

    ``time`` stands in for ``tsset``; see :func:`estat_dwatson`.
    """

    resid = _residuals(result, "estat corrgram")
    refuse_panel_fit(result, "estat corrgram")
    axis = resolve_time_axis(result, time, delta=delta, command="estat corrgram")
    n = int(resid.size)
    if lags is None:
        lags = _default_lags(n)
    lags = int(lags)
    if lags < 1 or lags >= n:
        raise ValueError(f"lags must lie in 1..{n - 1}; got {lags}")

    ac = _autocorrelations(resid, lags, axis)
    pac = _partial_autocorrelations(resid, lags, axis)

    rows: list[CorrelogramRow] = []
    running = 0.0
    for k in range(1, lags + 1):
        running += ac[k - 1] * ac[k - 1] / (n - k)
        q = float(n * (n + 2) * running)
        rows.append(
            CorrelogramRow(
                lag=k,
                ac=ac[k - 1],
                pac=pac[k - 1],
                q=q,
                pvalue=float(stats.chi2.sf(q, k)),
            )
        )
    return Correlogram(
        rows=tuple(rows), n_gaps=0 if axis is None else axis.n_gaps
    )
