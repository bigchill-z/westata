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

import math

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
    #: The endogenous series names, for the Endogenous: footnote Stata prints.
    names: tuple[str, ...] = ()

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
        """Stata's bordered lag-order table.

        BUILT FROM ABSOLUTE COLUMN ENDS rather than a chain of field widths. The old version
        concatenated eight segments and every attempt to box it produced a different total width
        (77, 78, 81, 82, 84) because each segment's change shifted all the rest. With the ends fixed
        a row is a list of (end_column, text) pairs and cannot drift.

        Measured from `varsoc y1 y2, maxlag(2)`:

            '   Sample: 3 thru 200                                      Number of obs = 198'
            '  +---------------------------------------------------------------------------+'
            '  | Lag |    LL      LR      df    p     FPE       AIC      HQIC      SBIC    |'
            '  |-----+---------------------------------------------------------------------|'
            '  |   1 | -543.814  1.3069    4  0.860 .885083   5.55368   5.59401   5.65332  |'
            '  +---------------------------------------------------------------------------+'
            '   * optimal lag'
            '   Endogenous: y1 y2'
            '    Exogenous: _cons'

        Data-row fields end at 7, 18, 26, 31, 38, 46, 56, 66, 76 with the closing pipe at 78; the
        star replaces the last character of its field, so a starred row is the same width.
        """

        best_aic = min(row.aic for row in self.rows)
        best_hqic = min(row.hqic for row in self.rows)
        best_sbic = min(row.sbic for row in self.rows)
        best_fpe = min(row.fpe for row in self.rows)
        # The LR star marks the ORDER SELECTED BY THE SEQUENCE TESTS, not the largest LR: Stata
        # stars the last significant one, which can be a smaller statistic than an earlier lag's.
        best_lr = 0.0
        for row in self.rows:
            if row.lr > 0 and row.lr_pvalue < 0.05:
                best_lr = row.lr

        def place(cells: list[tuple[int, str]]) -> str:
            """Lay out (start_column, text) pairs so each text BEGINS at its column.

            STATA ANCHORS THESE FIELDS AT THEIR START, not their end -- measured on two rows of the
            same table:

                lag 0 (all starred)  '.855661*' spans columns 39..47
                lag 1 (unstarred)    '.885083'  spans columns 39..46

            Same start, one column shorter, and the unstarred row does NOT pad to match: the next
            field begins at 49 either way. A right-anchored layout cannot express that, which is why
            four earlier attempts each fixed one row and broke the other.
            """

            out = ""
            for start, text in cells:
                out += " " * max(0, start - len(out)) + text
            return out

        def number(value: float, digits: int = 6) -> str:
            """Significant digits with the leading zero dropped, as Stata prints them."""

            text = f"{value:.{digits}g}"
            if text.startswith("0."):
                return text[1:]
            if text.startswith("-0."):
                return "-" + text[2:]
            return text

        def starred(value: float, best: float) -> str:
            return number(value) + ("*" if value == best else "")

        def fpe_text(value: float, best: float) -> str:
            """FPE keeps ONE mantissa digit and TRUNCATES when the value is tiny.

            Stata prints 6.67195e-08 as 6.6e-08, so ``%.1e`` (which rounds to 6.7e-08) is wrong.
            Above 1e-3 it switches to six decimals with the leading zero dropped.
            """

            import math

            if not math.isfinite(value):
                body = "."
            elif value == 0.0:
                body = ".000000"
            elif abs(value) >= 1e-3:
                body = number(abs(value), 6) if abs(value) < 1 else f"{abs(value):.6g}"
                if value < 0:
                    body = "-" + body
            else:
                exponent = int(math.floor(math.log10(abs(value))))
                mantissa = math.trunc(abs(value) / 10.0**exponent * 10.0) / 10.0
                body = f"{'-' if value < 0 else ''}{mantissa:.1f}e{exponent:02d}"
            return body + ("*" if value == best else "")

        inner = "-" * 75
        rule = "  +" + inner + "+"
        lines = [
            "",
            "Lag-order selection criteria",
            "",
            # THE BANNER IS RIGHT-ALIGNED TO A 78-COLUMN ROW, not padded to 59. Measured on two tables:
            # 'Number of obs' begins at 59 when the count is 198 and at 57 when it is 1,496 -- the
            # separator makes the tail longer and the label moves left.
            ("   Sample: " + str(self.sample_start) + " thru " + str(self.sample_end)).ljust(
                78 - len(f"Number of obs = {self.nobs:,}")
            ) + f"Number of obs = {self.nobs:,}",
            rule,
            "  | Lag |    LL      LR      df    p     FPE       AIC      HQIC      SBIC    |",
            "  |-----+" + "-" * 69 + "|",
        ]

        for row in self.rows:
            if row.lr > 0:
                lr_cell = number(row.lr, 5) + ("*" if row.lr == best_lr else "")
                df_cell = str(int(row.lr_df))
                p_cell = f"{row.lr_pvalue:.3f}"
            else:
                lr_cell = df_cell = p_cell = ""
            # `place` counts from the first character AFTER "  |", so every end is three lower
            # than its absolute column. The lag field is followed by a pipe before the numbers.
            # ABSOLUTE COLUMNS, like round 96's border work. A dict of column -> text is placed
            # last, so an error in one column cannot shift any other. Every cursor-based attempt
            # drifted: 125 wide once, fields fused another time, unbalanced parens a third.
            #
            # MEASURED (79-wide rows):
            #   start-anchored : LL 10, LR 20, df 30, p 33, FPE 39
            #   end-anchored   : AIC 56, HQIC 66, SBIC 76  (lag 2's 5.574 starts at 51, lag 1's
            #                    5.55368 at 49, and BOTH end at 56)
            # THE CRITERIA START AT THE SAME COLUMN ON EVERY ROW -- 49, 59, 69 -- and only their
            # ENDS differ (56/57, 66/67, 76/77). Measured on both row kinds:
            #
            #     unstarred  FPE ends 46   AIC 49..56   HQIC 59..66   SBIC 69..76
            #     starred    FPE ends 47   AIC 49..57   HQIC 59..67   SBIC 69..77
            #
            # So placing them by START is correct for both, which the end-anchored arithmetic never
            # achieved: it put AIC on 48 and left HQIC a column behind.
            # THE LR CELL IS RIGHT-ALIGNED TO COLUMN 26, unlike the LL cell which is start-anchored.
            # Measured on two rows of the same table:
            #
            #     '-558.096    1412'     1412  ends at 26
            #     '-789.832   2.223'     2.223 ends at 26
            #
            # Placing it by its start (20) put a wide value one column right of a narrow one and pushed
            # the df column with it -- visible only at n=1500, where the LR statistic reaches five
            # digits.
            # THE STAR DOES NOT COUNT TOWARD THE WIDTH -- it spills past the field, exactly as it does
            # in the criteria columns. Measured: '14.108*' starts one column left of '1412' because only
            # the digits are aligned to column 26.
            if lr_cell:
                _lr_width = len(lr_cell.rstrip("*"))
                _lr_start = max(19, 26 - _lr_width)
            else:
                _lr_start = 20
            _cells: dict[int, str] = {
                # THE LL CELL IS RIGHT-ALIGNED TO COLUMN 17. Measured on two rows:
                #
                #     '-4206.62'  spans 10..17
                #     '   -4203'  spans 13..17
                #
                # A start-anchored 10 left a short value ragged on the left.
                max(9, 18 - len(number(row.ll))): number(row.ll),
                _lr_start: lr_cell,
                30: df_cell,
                33: p_cell,
                max(39, 46 - len(fpe_text(float(row.fpe), best_fpe).rstrip("*"))): fpe_text(float(row.fpe), best_fpe),
            }
            # The measured START column for each criterion's END column.
            _criteria_starts = {56: 49, 66: 59, 76: 69}
            _criteria = (
                (56, starred(float(row.aic), best_aic)),
                (66, starred(float(row.hqic), best_hqic)),
                (76, starred(float(row.sbic), best_sbic)),
            )

            _out = "  |" + str(int(row.lag)).rjust(4) + " | "
            for _column in sorted(_cells):
                _out += " " * max(0, _column - len(_out)) + _cells[_column]
            for _column, _text in _criteria:
                # Right-align to the end column, but never start before the measured start column.
                _by_end = _column - len(_out) - len(_text)
                _by_start = _criteria_starts[_column] - len(_out)
                _gap_width = max(1, _by_start, _by_end)
                _out += " " * _gap_width + _text
            # A starred SBIC ends one column further right, so it takes one less closing space.
            _close = 1 if float(row.sbic) == best_sbic else 2
            lines.append(_out + " " * _close + "|")

        lines.append(rule)
        lines.append("   * optimal lag")
        _names_for_footer = getattr(self, "names", None)
        if isinstance(_names_for_footer, (list, tuple)) and _names_for_footer:
            lines.append("   Endogenous: " + " ".join(str(n) for n in _names_for_footer))
            lines.append("    Exogenous: _cons")
        return chr(10).join(lines)


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

    # The series names come from a DataFrame, so the Endogenous footnote can list them.
    if names is None and hasattr(data, "columns"):
        names = [str(column) for column in data.columns]
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
        # STATA'S FPE FORMULA, transcribed from var.ado and already implemented correctly in
        # `westata.estimators.tseries._fpe` for the `var` command:
        #
        #     fpe = ((T + eqparm) / (T - eqparm))**K * detsig_ml
        #
        # with `eqparm` the per-equation parameter count INCLUDING the constant -- lag*K + 1, not
        # this table's `k*K + 1`. The previous line used the wrong numerator, the wrong denominator
        # and the wrong parameter count, which is why the column sat about 1% high while every
        # other criterion agreed.
        #
        # THE SAMPLE ALSO DIFFERS BY ROW: lag 0 is evaluated on the common window, lag >= 1 on its
        # OWN sample. That is the second half of the fix and the reason the gap was not constant
        # across lags.
        _eqparm = lag * equations + 1
        # ALL ROWS USE THE COMMON WINDOW. Stata's varsoc evaluates every lag on the same rows --
        # the ones left after the largest lag -- which is not the same as ar, whose _fpe uses the
        # per-order sample. Measured: with drop=maxlag all three lags match (.855661/.885083/.903268),
        # while drop=lag matches only lag 0 and 2.
        _drop = maxlag
        _usable = nobs_total - _drop
        if _usable > _eqparm > 0:
            _target = values[_drop:]
            _regressors = [np.ones((_usable, 1))]
            for _step in range(1, lag + 1):
                _regressors.append(values[_drop - _step : nobs_total - _step])
            _design = np.column_stack(_regressors)
            _coefficients, *_ = np.linalg.lstsq(_design, _target, rcond=None)
            _residuals = _target - _design @ _coefficients
            _sigma = _residuals.T @ _residuals / _usable
            fpe = float(
                ((_usable + _eqparm) / (_usable - _eqparm)) ** equations
                * np.linalg.det(_sigma)
            )
        else:  # pragma: no cover - guarded by the length check above
            fpe = float("nan")

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
        names=tuple(names) if names else (),
    )
