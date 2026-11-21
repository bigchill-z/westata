"""Stata's ``varbasic`` and ``irf table``: VAR estimation plus impulse-response tables.

``varbasic varlist, lags(p) irf`` fits a VAR(p) and REGISTERS an IRF; it prints only the estimation
block, never the responses. The table comes from ``irf table irf``:

    . varbasic dlny if year<2013, lags(1) irf

    Vector autoregression

    Sample: 1980 thru 2012                          Number of obs     =         33
    ...

    . irf table irf

    Results from varbasic

    ---------------------------------------------
             |      (1)         (1)         (1)
        Step |      irf       Lower       Upper
    ---------+-----------------------------------
           0 |        1           1           1
           1 |  .536273     .248498     .824048
    ...

EVERYTHING HERE WAS MEASURED, and two details differ from the obvious first guess:

* ``varbasic`` accepts a SINGLE variable. ``statsmodels`` refuses one, so the standard errors and the
  IRF are computed here directly rather than delegated.
* Stata normalises the residual covariance by **T**, not ``T - k``. Using ``T - k`` gives
  ``se = 0.1514890`` against Stata's ``0.1468267`` -- a 3.2% gap that shifts every band.

The confidence bands are the delta-method bands on ``a^h``: ``d(a^h)/da = h * a^(h-1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from westata.errors import ConsistencyError
from westata.plotting import make_axes, make_figure
from westata.results.serialisation import ResultSerialisation

__all__ = ["ImpulseResponse", "IrfTable", "IrfResults", "VarBasic", "varbasic", "irf_table"]


# ---------------------------------------------------------------- helpers
def _prepared(data: pd.DataFrame, variables: Sequence[str], lags: int) -> tuple[np.ndarray, int]:
    """The estimation sample after lags, as an (n, k) float array."""

    for name in variables:
        if name not in data.columns:
            raise ConsistencyError(
                f"varbasic: variable {name!r} not found; available columns: {list(data.columns)}"
            )
    frame = data[list(variables)].dropna()
    if len(frame) <= lags:
        raise ConsistencyError(
            f"varbasic: only {len(frame)} complete observations for lags({lags})"
        )
    return frame.to_numpy(dtype=float), len(frame)


def _period_ordinals(values: Any) -> np.ndarray:
    """Map timestamps to consecutive period ordinals, so a calendar axis has a uniform step.

    Quarters are 90 to 92 days apart and months 28 to 31, so elapsed days make an evenly spaced
    axis look gappy. The ordinal removes that: 1961q1 -> 1961q2 is 1, and a genuinely missing
    quarter is 2.
    """

    stamps = pd.to_datetime(pd.Series(values))
    ordinals = np.empty(len(stamps), dtype=np.int64)
    for index, stamp in enumerate(stamps):
        if pd.isna(stamp):
            ordinals[index] = -1
        elif stamp.day == 1 and stamp.month in (1, 4, 7, 10):
            ordinals[index] = stamp.year * 4 + (stamp.month - 1) // 3
        elif stamp.day == 1:
            ordinals[index] = stamp.year * 12 + stamp.month - 1
        else:
            ordinals[index] = int(stamp.toordinal())
    return ordinals


def _stata_period_label(value: Any) -> str:
    """Format a timestamp the way a Stata ``tsset`` axis labels it.

    Quarterly prints ``1961q2``, monthly ``1961m4``, and anything finer keeps its ISO date, which
    is what ``tsset`` shows for ``%td``. Stata never shows the underlying datetime in a ``Sample:``
    line.
    """

    stamp = pd.Timestamp(value)
    if stamp.day == 1 and stamp.month in (1, 4, 7, 10):
        return f"{stamp.year}q{(stamp.month - 1) // 3 + 1}"
    if stamp.day == 1:
        return f"{stamp.year}m{stamp.month}"
    return stamp.strftime("%Y-%m-%d")


def _lag_order(lags: "int | Sequence[int]") -> list[int]:
    """Normalise the ``lags`` argument to the ascending list of retained lags.

    ``varbasic`` has NO default of its own: ``varbasic.ado:19`` sets ``local lags "1 2"`` and passes
    it straight to ``var``, so a bare ``varbasic dlny, irf`` fits a VAR with L1 and L2. Passing an
    ``int`` p keeps only lag p (which is what ``lags(2)`` means on its own), matching the measured
    behaviour recorded below.
    """

    if isinstance(lags, int):
        return [lags]
    order = [int(value) for value in lags]
    if not order:
        raise ConsistencyError("varbasic: `lags` must name at least one lag")
    if min(order) < 1:
        raise ConsistencyError("varbasic: `lags` must be positive")
    return sorted(set(order))


def _var_design(
    values: np.ndarray, lags: "int | Sequence[int]", trend: str = "c"
) -> tuple[np.ndarray, np.ndarray]:
    """Stack the retained lag blocks; ``trend='c'`` prepends a constant."""

    n, k = values.shape
    order = _lag_order(lags)
    blocks = [np.ones((n, 1))] if trend == "c" else []

    # MEASURED: each requested lag contributes its own block, and the table labels them `L1.`,
    # `L2.` ... A bare `varbasic` keeps 1 AND 2; `lags(2)` alone keeps only 2.
    #
    # THE LAG IS BUILT ON THE FULL SERIES, THEN THE SAMPLE IS TRIMMED -- not the other way round.
    # `_qsur.ado:54` marks out ``l(`lags').(varlist)``, so `l(0 2).dlny` only drops rows whose
    # two-period lag is missing; building the lag after slicing to `year>=1985` throws away the two
    # extra rows and shifts L2 by -.05149 instead of Stata's -.05540.
    usable = np.ones(n, dtype=bool)
    for lag in order:
        lagged = np.full_like(values, np.nan)
        if n > lag:
            lagged[lag:, :] = values[: n - lag, :]
        blocks.append(lagged)
        usable &= ~np.isnan(lagged).any(axis=1)

    design = np.column_stack(blocks)
    return design[usable], values[usable]


def _ols(design: np.ndarray, response: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-equation OLS. Returns (coefficients, residuals, sigma) with Stata's /T normalisation."""

    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    residuals = response - design @ coefficients
    n = design.shape[0]
    # MEASURED: /T, not /(T - k). See the module docstring.
    sigma = residuals.T @ residuals / n
    return coefficients, residuals, sigma


def _cholesky(sigma: np.ndarray) -> np.ndarray:
    """Lower-triangular factor; ``numpy`` raises on a non-positive-definite matrix."""

    try:
        return np.linalg.cholesky(sigma)
    except np.linalg.LinAlgError as exc:
        raise ConsistencyError(
            "varbasic: the residual covariance matrix is not positive definite; "
            "the IRF is undefined for this ordering"
        ) from exc


@dataclass(frozen=True)
class ImpulseResponse(ResultSerialisation):
    """One response path: the point estimates and their delta-method bands."""

    impulse: str
    response: str
    steps: tuple[int, ...]
    values: tuple[float, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def at(self, step: int) -> tuple[float, float, float]:
        index = self.steps.index(step)
        return self.values[index], self.lower[index], self.upper[index]


@dataclass(frozen=True)
class IrfResults(ResultSerialisation):
    """The fitted VAR plus everything ``irf table`` needs."""

    names: list[str]
    lags: int
    nobs: int
    sample_first: Any = None
    sample_last: Any = None
    #: True when the ``time`` axis has an interior gap, so the ``Sample:`` line carries Stata's
    #: ``, but with a gap`` suffix. A property of the axis, not of the retained rows.
    sample_has_gap: bool = False
    coefficients: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    sigma: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    sigma_se: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    ll: float = float("nan")
    aic: float = float("nan")
    hqic: float = float("nan")
    sbic: float = float("nan")
    fpe: float = float("nan")
    det_sigma: float = float("nan")
    rmse: tuple[float, ...] = ()
    r2: tuple[float, ...] = ()
    chi2: tuple[float, ...] = ()
    chi2_p: tuple[float, ...] = ()
    parms: tuple[int, ...] = ()
    #: Filled in by :func:`irf_table`; ``varbasic`` itself leaves it empty, as Stata prints no table.
    responses: tuple[ImpulseResponse, ...] = ()
    #: The estimated regressor matrix, needed to rebuild each equation's coefficient covariance.
    design: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    #: Regressor names in design order -- ``["L1.dlny", "_cons"]`` for a one-lag VAR(1).
    regressor_names: tuple[str, ...] = ()
    #: The lag argument VERBATIM: an ``int`` p for a single block at distance p, or the tuple of
    #: retained lags. The IRF needs the distinction -- Stata's ``lags(2)`` alone is a 2-period
    #: recurrence (zero on odd steps) while ``lags(1 2)`` is an ordinary VAR(2).
    lag_orders: "int | tuple[int, ...]" = 1

    # ---------------------------------------------------------------- R4 result surface
    @property
    def params(self) -> "np.ndarray":
        """The VAR coefficients, flattened in Stata's (regressor, equation) order."""

        return np.asarray(self.coefficients, dtype=float).ravel()

    @property
    def bse(self) -> "np.ndarray":
        """Standard errors of those coefficients, from sigma/T and (X'X)^-1.

        Stata normalises the residual covariance by T, not T-k -- see the module docstring.
        """

        sigma = np.asarray(self.sigma, dtype=float)
        design = np.asarray(self.design, dtype=float)
        xtx_inv = np.linalg.pinv(design.T @ design)
        # Each column of `coefficients` is one equation; its own variance drives its own errors.
        diagonal = np.diag(sigma)
        per_equation = [
            np.sqrt(max(diagonal[column], 0.0) * np.diag(xtx_inv))
            for column in range(np.asarray(self.coefficients).shape[1])
        ]
        return np.concatenate(per_equation)

    @property
    def e(self) -> "dict[str, float]":
        """The scalars Stata stores in ``e()``, under their own names."""

        return {
            "N": float(self.nobs),
            "k": float(self.parms[0]) if self.parms else 0.0,
            "k_eq": float(len(self.names)),
            "ll": self.ll,
            "aic": self.aic,
            "hqic": self.hqic,
            "sbic": self.sbic,
            "fpe": self.fpe,
            "det_sigma": self.det_sigma,
            "lag": float(self.lags),
        }

    def __str__(self) -> str:
        """Stata's ``varbasic`` block, measured on gdp_china.dta."""

        lines = ["", "Vector autoregression", ""]
        sample = (
            # MEASURED, gdp_china.dta with `drop if year == 1989`:
            #     Sample: 1980 thru 2013, but with a gap          Number of obs     =         32
            # The suffix follows the LAST period, not the first, and the column width is measured
            # from the 48-character field Stata pads to before 'Number of obs'.
            f"Sample: {self.sample_first} thru {self.sample_last}"
            + (", but with a gap" if self.sample_has_gap else "")
            if self.sample_first is not None
            else "Sample: (not recorded)"
        )
        lines.append(f"{sample:<48}Number of obs     ={self.nobs:>11}")
        lines.append(
            f"{'Log likelihood = ' + f'{self.ll:>10.5f}':<48}AIC               ={_trim(self.aic):>11}"
        )
        # MEASURED: 'FPE            =   .0004889' and 'Det(Sigma_ml)  =   .0004331' -- the leading
        # zero is dropped, and the value sits in a fixed field ending at column 27.
        lines.append(
            f"{'FPE':<15}={'':>2}{_coef(self.fpe):>9}"
            f"{'':<21}HQIC              ={_trim(self.hqic):>11}"
        )
        lines.append(
            f"{'Det(Sigma_ml)':<15}={'':>2}{_coef(self.det_sigma):>9}"
            f"{'':<21}SBIC              ={_trim(self.sbic):>11}"
        )
        lines.append("")
        lines.append("Equation           Parms      RMSE     R-sq      chi2     P>chi2")
        lines.append("-" * 64)
        for index, name in enumerate(self.names):
            # MEASURED ends: name 0.., Parms 23, RMSE 35, R-sq 44, chi2 55, P>chi2 64.
            lines.append(
                f"{name:<16}{self.parms[index]:>7}"
                f"{_drop_leading_zero(('%.6f' % self.rmse[index])):>12}"
                f"{'%.4f' % self.r2[index]:>9}"
                f"{_sig(self.chi2[index], 7):>11}"
                f"{'%.4f' % self.chi2_p[index]:>9}"
            )
        lines.append("-" * 64)

        # The coefficient table below the summary, which `varbasic` prints for each equation.
        # MEASURED (78 wide, pipe at 13):
        #     '        dlny | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]'
        #     '         L1. |   .5362727   .1468267     3.65   0.000     .2484976    .8240478'
        #     '       _cons |   .0437698   .0143197     3.06   0.002     .0157037     .071836'
        lines.append("")
        lines.append("-" * 78)
        # THE HEADER NAMES THE RESPONSE ONLY IN A SINGLE-EQUATION VAR. With several equations
        # Stata leaves the name field blank, because no one variable owns the table. MEASURED:
        # one equation -> '        dlny | Coefficient ...'; two equations -> the same line with
        # twelve spaces before the pipe.
        header_name = self.names[0] if len(self.names) == 1 else ""
        lines.append(
            f"{header_name:>12} | Coefficient  Std. err.      z    P>|z|"
            "     [95% conf. interval]"
        )
        lines.append("-" * 13 + "+" + "-" * 64)
        # MEASURED column ends (78-wide rows, pipe at 13):
        #     estimate 25 | std.err 36 | z 45 | P>|z| 53 | lower 66 | upper 78
        # The design stores the constant first; the table prints the lags first, so the rows are
        # reordered here rather than in the estimator.
        coefficients = np.asarray(self.coefficients, dtype=float)
        errors = self.bse.reshape(len(self.names), -1) if self.bse.size else self.bse
        # The design puts the constant FIRST and the lags after it; the table prints the lags first.
        # Detect the constant by its NAME rather than by a hard-coded lag label, which is what broke
        # the lags(2) case (the label there is `L2.`, not `L1.`).
        has_constant = bool(self.regressor_names) and self.regressor_names[-1] == "_cons"
        if has_constant:
            # table order [L1., L2., ..., _cons] -> design rows [1, 2, ..., 0]
            n_lags = len(self.regressor_names) - 1
            design_order = list(range(1, n_lags + 1)) + [0]
        else:
            design_order = list(range(len(self.regressor_names)))

        for equation_index, equation in enumerate(self.names):
            lines.append(f"{equation:<13}|")
            # STATS GROUPS BY LAGGED VARIABLE, NOT BY LAG. For `dinf unem` with lags(1/2) the
            # design rows run lag-major (L1.dinf, L1.unem, L2.dinf, L2.unem), but the table walks
            # every lag of one variable, then every lag of the next, with a variable sub-header and
            # a blank spacer between groups. MEASURED (StataNow 19.5, macro_swatson.dta):
            #
            #     dinf         |
            #             dinf |
            #              L1. |  -.4709436 ...
            #              L2. |   -.401031 ...
            #                  |
            #             unem |
            #              L1. |   -2.24205 ...
            #              L2. |    2.03417 ...
            #                  |
            #            _cons |   .0123764 ...
            #
            # With ONE endogenous variable the grouping is a no-op -- the single group is the
            # equation itself -- so the single-variable block stays byte-identical.
            # `lag_orders` is the RETENTION LIST, not a count: `lags=(1, 2)` gives (1, 2) and
            # `lags(2)` gives (2,) -- a two-period recurrence with no L1 block at all. Deriving a
            # count from len(regressor_names) instead mixes the two up and overruns design_order.
            retained = (
                tuple(self.lag_orders)
                if isinstance(self.lag_orders, tuple)
                else (int(self.lag_orders),)
            )
            groups: list[tuple[str, list[tuple[str, int]]]] = []
            if len(self.names) > 1:
                names_list = list(self.names)
                for variable in names_list:
                    members: list[tuple[str, int]] = []
                    for lag in retained:
                        position = retained.index(lag) * len(names_list) + names_list.index(variable)
                        members.append((f"L{lag}.", design_order[position]))
                    groups.append((variable, members))
            else:
                members = []
                for lag in retained:
                    position = retained.index(lag)
                    members.append((f"L{lag}.", design_order[position]))
                groups.append((self.names[equation_index], members))

            for group_index, (group_name, members) in enumerate(groups):
                # The variable sub-header is always printed. With a single equation it repeats the
                # equation's own name -- MEASURED: 'dlny         |' then '        dlny |' then
                # 'L1.'; with several, it names which regressor's lags follow.
                lines.append(f"{group_name:>12} |")
                for label, row in members:
                    estimate = float(coefficients[row, equation_index])
                    error = float(errors[equation_index][row])
                    z_value = estimate / error if error else float("nan")
                    p_value = (
                        float(2.0 * (1.0 - _norm_cdf(abs(z_value))))
                        if np.isfinite(z_value)
                        else float("nan")
                    )
                    low = estimate - 1.959963985 * error
                    high = estimate + 1.959963985 * error
                    lines.append(
                        f"{label:>12} |"
                        f"{_coef(estimate):>11}"
                        f"{_coef(error):>11}"
                        f"{'%.2f' % z_value:>9}"
                        f"{'%.3f' % p_value:>8}"
                        f"{_bound(low):>13}"
                        f"{_bound(high):>12}"
                    )
                if group_index + 1 < len(groups):
                    lines.append(f"{'':>13}|")
            if has_constant:
                lines.append(f"{'':>13}|")
                row = design_order[-1]
                estimate = float(coefficients[row, equation_index])
                error = float(errors[equation_index][row])
                z_value = estimate / error if error else float("nan")
                p_value = (
                    float(2.0 * (1.0 - _norm_cdf(abs(z_value))))
                    if np.isfinite(z_value)
                    else float("nan")
                )
                low = estimate - 1.959963985 * error
                high = estimate + 1.959963985 * error
                lines.append(
                    f"{'_cons':>12} |"
                    f"{_coef(estimate):>11}"
                    f"{_coef(error):>11}"
                    f"{'%.2f' % z_value:>9}"
                    f"{'%.3f' % p_value:>8}"
                    f"{_bound(low):>13}"
                    f"{_bound(high):>12}"
                )
        lines.append("-" * 78)
        return "\n".join(lines)


class VarBasic:
    """Class form of :func:`varbasic`, following the package's estimator convention.

    ``VarBasic(frame, "dlny", lags=1, time="year").fit()`` is the same computation as
    ``varbasic(frame, "dlny", lags=1, time="year")``. The function stays as a thin wrapper because
    Stata's ``varbasic`` is itself a one-line command, and callers coming from ``ws.panel_test0(...)``
    style expect the bare call to work.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variables: Sequence[str] | str,
        *,
        lags: "int | Sequence[int]" = (1, 2),
        trend: str = "c",
        time: str | None = None,
        subset: "Any | None" = None,
    ) -> None:
        self._data = data
        self._variables = variables
        self._lags = lags
        self._trend = trend
        self._time = time
        self._subset = subset

    def fit(self) -> IrfResults:
        """Estimate the VAR and register the IRF (the responses need :func:`irf_table`)."""

        return varbasic(
            self._data,
            self._variables,
            lags=self._lags,
            trend=self._trend,
            time=self._time,
            subset=self._subset,
        )


def _norm_cdf(value: float) -> float:
    """Standard normal CDF, so the P>|z| column does not need a scipy import here."""

    import math as _math

    return 0.5 * (1.0 + _math.erf(value / _math.sqrt(2.0)))


def _bound(value: float) -> str:
    """A confidence-interval bound, in Stata's table format.

    MEASURED by asking Stata to print its own bounds with several formats:

        %9.0g  -> [ .0157037] [  .071836]     <- what the table shows
        %10.0g -> [ .01570365] [ .07183599]   <- too wide

    The rule that reproduces all thirteen captured bounds: SEVEN significant figures, but a result
    wider than EIGHT characters (nine when negative, since the sign occupies a column) drops to six.
    That is why 1.0835279 prints as 1.083528 while -.7577275 keeps seven digits.
    """

    if value is None or not np.isfinite(value):
        return "."
    # SEVEN SIGNIFICANT DIGITS, CAPPED AT SEVEN DECIMALS -- the same rule as `_coef`, and it
    # reproduces thirteen of the fourteen measured bounds. The one that differs, `.0035047`
    # against this function's `.0035046`, is not a formatting question: recomputing the bound
    # from Stata's own printout gives 0.00350462302, whose seventh decimal is 6, while the full
    # precision coefficient gives 0.00350464867, whose seventh decimal rounds to 6 as well. Stata
    # prints 7, so its internal value sits on the other side of the rounding boundary -- a
    # last-digit float difference, not a rule this function can apply.
    text = f"{float(value):.7g}"
    if "e" not in text and "E" not in text:
        if "." in text and len(text.split(".")[1]) > 7:
            text = f"{float(value):.7f}"
        text = text.rstrip("0").rstrip(".")
        if text in ("", "-"):
            text = "0"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _trim(value: float) -> str:
    """Six decimals with trailing zeros removed, as Stata's information-criteria column shows.

    MEASURED: ``-4.41417`` and ``-4.35399`` on the ``lags(2)`` run, where a fixed six-decimal format
    would print ``-4.414170`` and ``-4.353990``.
    """

    if value is None or not np.isfinite(value):
        return "."
    text = f"{float(value):.6f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def _sig(value: float, digits: int) -> str:
    """Stata's variable-precision column: ``digits`` significant figures, leading zero dropped.

    Used for the equation table's chi2 column, which prints ``13.34017`` (7 significant figures) and
    ``.0364487`` (also 7, once the leading zero is gone) -- a fixed decimal count reproduces one but
    not the other.
    """

    if value is None or not np.isfinite(value):
        return "."
    if value == 0:
        return "0"
    text = f"%.{digits}g" % float(value)
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _drop_leading_zero(text: str) -> str:
    """`0.0364487` -> `.0364487`; also trims trailing zeros so `.025460` prints as `.02546`."""

    trimmed = text.rstrip("0")
    if trimmed.endswith("."):
        trimmed += "0"
    if trimmed.startswith("0."):
        trimmed = trimmed[1:]
    elif trimmed.startswith("-0."):
        trimmed = "-" + trimmed[2:]
    return trimmed


def _coef(value: float) -> str:
    """Stata's coefficient-column format: seven decimals, leading zero dropped.

    MEASURED: ``.5362727``, ``.0437698``, ``.071836`` -- the last one shows only six decimals because
    Stata trims trailing zeros on this column.
    """

    if value is None or not np.isfinite(value):
        return "."
    # A VERY SMALL VALUE IS PRINTED IN SCIENTIFIC NOTATION, as Stata does. Fixed point at seven
    # decimals turns 1.44e-09 into "0.0000000", whose trimmed form is ".0" -- a number that reads
    # as zero and is off by nine orders of magnitude. Measured on macro_swatson.dta
    # (`varbasic dinf unem if quarter<tq(1999q1), lags(1/2)`): Stata prints 'FPE = 1.44e-09' and
    # 'Det(Sigma_ml) = 1.26e-09' where this returned '.0' for both.
    magnitude = abs(float(value))
    if magnitude != 0.0 and magnitude < 1e-6:
        mantissa, exponent = f"{float(value):.2e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        if mantissa in ("", "-"):
            return "0"
        return f"{mantissa}e{int(exponent):+03d}"
    text = f"{float(value):.7g}"
    if "e" not in text and "E" not in text:
        # SEVEN SIGNIFICANT DIGITS, BUT NEVER MORE THAN SEVEN DECIMALS. A fixed decimal count
        # prints -2.2420501 where Stata shows -2.24205; a fixed significant-digit count prints
        # .01237643 where Stata shows .0123764 (the decimals cap). Both rules are needed, and
        # min(sig, decimals) reproduces all ten measured values in one pass:
        #
        #     -2.24205008936655 -> -2.24205      (6 sig, 5 decimals)
        #      1.638030690829964 ->  1.638031    (7 sig, 6 decimals)
        #      0.01237642564918 ->  .0123764     (6 sig, capped at 7 decimals)
        #     -0.4709436484841  -> -.4709436     (7 sig)
        if "." in text and len(text.split(".")[1]) > 7:
            text = f"{float(value):.7f}"
        text = text.rstrip("0").rstrip(".")
        if text in ("", "-"):
            text = "0"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def varbasic(
    data: pd.DataFrame,
    variables: Sequence[str] | str,
    *,
    lags: "int | Sequence[int]" = (1, 2),
    trend: str = "c",
    time: str | None = None,
    subset: "Any | None" = None,
) -> IrfResults:
    """Stata's ``varbasic``: fit a VAR(p) and register an IRF.

    ``time`` names the time variable. It is OPTIONAL -- omitting it changes no number, only the
    ``Sample:`` line -- but when given it is load-bearing, because the lag is built on the TIME
    axis: a row enters only if the earlier period it needs actually exists. That is what Stata's
    ``tsset`` does, and skipping it silently pairs a row with a *different* year whenever the frame
    has an interior gap (measured: 33 observations against Stata's 32, with 1990's lag taken from
    1988). Passing ``time`` turns that into a refusal rather than a wrong number.

    ``varbasic`` itself prints only the estimation block -- the responses need :func:`irf_table`,
    which is also what Stata requires.

    Filtering is pandas' job: slice the frame and pass it in. ``subset`` follows R3.4 -- it is applied
    as ``data.loc[subset]`` BEFORE anything is estimated, so it agrees with pre-filtering the frame.

    That means a head-cutting condition reproduces Stata only up to the rows pandas keeps: Stata
    builds the lag over the complete time axis (`_qsur.ado:54`), so ``if year>=1985`` keeps 29
    observations there, while a frame sliced at 1985 has no value for 1985's lag and yields 28. If a
    series is available before the slice, slice only the TAIL (or pass the full frame and accept the
    smaller sample) -- matching Stata's head-cut sample is the caller's decision, not a hidden one.
    """

    names = [variables] if isinstance(variables, str) else list(variables)
    if not names:
        raise ConsistencyError("varbasic: at least one variable is required")

    # A `time` that does not name a column cannot be honoured, and silently falling back to
    # `Sample: (not recorded)` is indistinguishable from the caller omitting the argument on
    # purpose. It is also dangerous for a second reason: the output would claim a sample it never
    # verified. Refuse instead, and name what IS available so the typo is obvious.
    if time is not None and time not in data.columns:
        raise ConsistencyError(
            f"varbasic: `time` names {time!r}, which is not a column; "
            f"available columns: {list(data.columns)}"
        )

    # R3.4: `subset` is applied to the FRAME, before the lag is built, so `varbasic(data, subset=m)`
    # and `varbasic(data.loc[m], ...)` cannot disagree.
    if subset is not None:
        try:
            data = data.loc[subset]
        except (KeyError, IndexError) as exc:
            raise ConsistencyError(
                f"varbasic: `subset` does not align with the frame's index ({exc})"
            ) from None
        if data.empty:
            raise ConsistencyError("varbasic: `subset` selects no rows")

    for name in names:
        if name not in data.columns:
            raise ConsistencyError(
                f"varbasic: variable {name!r} not found; available columns: {list(data.columns)}"
            )

    # Build the engine's view of the data over EVERY row, so a retained row's lag can reach back
    # before the `if` condition. `_qsur.ado:54` marks out l(lags).(varlist) on the full sample.
    raw = data[list(names)].to_numpy(dtype=float)
    n_rows = raw.shape[0]
    order = _lag_order(lags)
    if n_rows <= max(order):
        raise ConsistencyError(
            f"varbasic: only {n_rows} rows for lags({', '.join(str(v) for v in order)})"
        )

    # THE LAG IS TAKEN ON THE TIME AXIS, NOT ON ROW POSITION -- and the difference is a wrong
    # number, not a cosmetic one. Stata's `tsset` makes `L1.y` mean "the previous PERIOD", so with
    # an interior gap a row whose predecessor period is missing cannot enter the sample.
    #
    # Measured on gdp_china.dta with `drop if year == 1989` (StataNow 19.5):
    #
    #     Sample: 1980 thru 2013, but with a gap   Number of obs = 32
    #     .5016988   L1      .049534   _cons
    #
    # and `levelsof year if e(sample)` is 1980..1988 then 1991..2013: 1990 is ABSENT because its L1
    # is 1989, which no longer exists. Shifting by row position instead keeps 33 rows and pairs 1990
    # with 1988 -- a different observation entirely -- producing .5017/.0495 shifted and nobs 33.
    #
    # Without `time` there is no axis to align to, so the historical position-based shift stays and
    # the caller accepts that a gapped frame is its own responsibility (documented on the function).
    axis = None
    if time is not None:
        axis = data[time].to_numpy()
        if pd.isna(axis).any():
            raise ConsistencyError(
                f"varbasic: `time` column {time!r} has missing values; a lag cannot be "
                "aligned to the time axis when the axis itself is incomplete"
            )
        # Stata's `tsset` requires strictly increasing periods; its `delta` is what names L1, L2.
        #
        # A DATETIME AXIS MUST NOT BECOME NUMBERS. `pd.to_numeric` on datetime64 yields nanoseconds
        # (~1e18), and the lookup below does `current - delta * lag` in float64, which loses the low
        # bits: every match misses and the sample comes out unusable. Measured on
        # macro_swatson.dta with `time="quarter"`: "varbasic: no usable observations" for a frame
        # that estimates fine WITHOUT `time`. Dates therefore take the positional branch, which is
        # also the right reading of a quarterly `tsset` -- "the previous PERIOD", not "-1 ns".
        _axis_is_calendar = bool(
            pd.api.types.is_datetime64_any_dtype(pd.Series(axis))
            or pd.api.types.is_timedelta64_dtype(pd.Series(axis))
        )
        numeric_axis = (
            pd.Series([], dtype=float)
            if _axis_is_calendar
            else pd.to_numeric(pd.Series(axis), errors="coerce")
        )
        if numeric_axis.notna().all() and not _axis_is_calendar:
            ordered_axis = np.sort(numeric_axis.to_numpy(dtype=float))
            steps = np.diff(ordered_axis)
            if steps.size and np.any(steps <= 0):
                raise ConsistencyError(
                    f"varbasic: `time` column {time!r} has duplicate periods, which `tsset` "
                    "refuses; the lag would be ambiguous"
                )
            delta = float(steps.min()) if steps.size else 1.0
        else:
            # A non-numeric axis (dates, strings) has no subtraction; fall back to ordering the
            # distinct values and stepping through their sorted positions.
            distinct = pd.unique(pd.Series(axis))
            rank = {value: index for index, value in enumerate(sorted(distinct))}
            numeric_axis = pd.Series([rank[value] for value in axis]).astype(float)
            delta = 1.0
        axis = numeric_axis.to_numpy(dtype=float)

    # Each retained lag contributes its own regressor block; the response is the current row.
    lag_blocks: list[np.ndarray] = []
    if axis is None:
        for lag in order:
            block = np.full_like(raw, np.nan)
            block[lag:, :] = raw[: n_rows - lag, :]
            lag_blocks.append(block)
    else:
        position_of = {float(value): index for index, value in enumerate(axis)}
        for lag in order:
            block = np.full_like(raw, np.nan)
            for index, current in enumerate(axis):
                source = position_of.get(current - delta * lag)
                if source is not None:
                    block[index, :] = raw[source, :]
            lag_blocks.append(block)
    response = raw
    if trend == "c":
        design = np.column_stack([np.ones((n_rows, 1))] + lag_blocks)
    else:
        design = np.column_stack(lag_blocks)

    # A row is usable when every retained lag and the response are present.
    usable = ~np.isnan(response).any(axis=1)
    for block in lag_blocks:
        usable &= ~np.isnan(block).any(axis=1)

    if not usable.any():
        raise ConsistencyError("varbasic: no usable observations")

    design = design[usable]
    response = response[usable]
    kept_positions = np.flatnonzero(usable)

    coefficients, residuals, sigma = _ols(design, response)

    n, k = response.shape
    rows = design.shape[1]
    # Stata counts each equation's parameters as the regressors of ONE equation.
    parms = tuple([trend == "c"] * 0 + [rows for _ in range(k)])

    dof = n - rows
    rmse, r2, chi2, chi2_p = [], [], [], []
    from scipy import stats as _stats

    # Stata's per-equation chi2 is the WALD test that the whole lag block is zero, not an r2
    # transformation. MEASURED: (b_L1/se_L1)^2 = 13.34017 in the one-regressor case, exactly the
    # printed value; the r2-based expression gave 0.40425.
    xtx_inv = np.linalg.pinv(design.T @ design)
    offset = 1 if trend == "c" else 0
    for index in range(k):
        resid = residuals[:, index]
        rss = float(resid @ resid)
        rmse.append(float(np.sqrt(rss / dof)))
        total = float(((response[:, index] - response[:, index].mean()) ** 2).sum())
        r2.append(1.0 - rss / total if total > 0 else float("nan"))

        # Coefficient covariance for this equation: sigma_jj / T, using Stata's /T normalisation.
        sigma_jj = rss / n
        cov = sigma_jj * xtx_inv
        lag_rows = list(range(offset, design.shape[1]))
        b_lags = coefficients[lag_rows, index]
        sub = cov[np.ix_(lag_rows, lag_rows)]
        statistic = float(b_lags @ np.linalg.pinv(sub) @ b_lags)
        df_m = len(lag_rows)
        chi2.append(statistic)
        chi2_p.append(
            float(1.0 - _stats.chi2.cdf(statistic, df_m)) if np.isfinite(statistic) else float("nan")
        )

    det_sigma = float(np.linalg.det(sigma))
    ll = float(-0.5 * n * (k * (1.0 + np.log(2.0 * np.pi)) + np.log(det_sigma)))

    # Information criteria. MEASURED against gdp_china.dta (N=33, one equation, one retained lag):
    #
    #     aic  -4.78552445   hqic -4.75500752   sbic -4.69482702   fpe .00048895
    #
    # All four use the count of the VAR's own coefficients -- TWO here (L1 and _cons) -- not the
    # equation count or a covariance-parameter correction. Applying `k*rows + k*(k+1)/2` gave AIC
    # -4.724918 instead, which is exactly what a wrong count of 3 produces.
    total_parms = float(coefficients.size)
    aic = float(-2.0 * ll / n + 2.0 * total_parms / n)
    hqic = float(-2.0 * ll / n + 2.0 * np.log(np.log(n)) * total_parms / n)
    sbic = float(-2.0 * ll / n + np.log(n) * total_parms / n)
    # MEASURED: FPE divides by (N - k), not N. `det * (N+k)/N` gives .00045935 against Stata's
    # .00048895; `det * (N+k)/(N-k)` reproduces it.
    fpe = float(det_sigma * (n + total_parms) / (n - total_parms))

    sigma_se = sigma * np.sqrt(2.0 / n)

    # THE SAMPLE RANGE COMES FROM THE ROWS THAT ACTUALLY ENTERED, not from "the first row after
    # `max(order)` in the frame". Those two agree on a gapless series and disagree exactly when it
    # matters: with an interior gap the shifted-by-position rule reported `Sample: 1980 thru 2013`
    # while the estimation sample skipped a year, so the label described a different sample than the
    # numbers below it.
    first = last = None
    has_gap = False
    if time is not None:
        # `kept_positions` indexes the FULL frame (that is what `raw` spans), so the labels must come
        # from the unfiltered time column -- dropping rows here first would shift every position.
        row_labels = data[time].to_numpy()
        sample_values = row_labels[kept_positions]
        if len(sample_values):
            ordered = pd.Series(sample_values).sort_values()
            first = ordered.iloc[0]
            last = ordered.iloc[-1]
            # MEASURED: Stata prints 'Sample: 1980 thru 2012', not '1980.0 thru 2012.0'. A year stored
            # as a float must be shown as the integer it is.
            #
            # A CALENDAR AXIS IS LABELLED THE WAY `tsset` LABELS IT -- '1961q2', not a Timestamp,
            # which `float()` cannot even accept. Measured: `varbasic dinf unem if
            # quarter<tq(1999q1), lags(1/2)` reports 'Sample: 1961q2 thru 1998q4'.
            if pd.api.types.is_datetime64_any_dtype(pd.Series(row_labels)):
                first, last = _stata_period_label(first), _stata_period_label(last)
            else:
                first = int(first) if float(first) == int(first) else first
                last = int(last) if float(last) == int(last) else last
            # MEASURED: `tsset` prints 'Time variable: year, 1978 to 2013, but with a gap' and
            # `varbasic` carries that suffix at the END of its Sample line, after the last period:
            #     Sample: 1980 thru 2013, but with a gap          Number of obs     =         32
            # It is a property of the TIME AXIS, not of the estimation sample -- the gap is visible
            # even when no row was dropped because of it.
            # A CALENDAR AXIS HAS UNEQUAL DAY COUNTS between adjacent periods -- quarters are 90, 91
            # or 92 days apart -- so comparing elapsed days invents a gap that is not there.
            # Measured: macro_swatson's quarterly axis gave step differences {90, 91, 92} and the
            # Sample line printed a spurious ', but with a gap'. Calendar axes are therefore
            # compared as PERIOD ORDINALS: consecutive quarters differ by exactly 1.
            if pd.api.types.is_datetime64_any_dtype(pd.Series(row_labels)):
                unique_ordinals = np.sort(np.unique(_period_ordinals(pd.Series(row_labels).dropna().to_numpy())))
                if unique_ordinals.size > 1 and np.any(np.diff(unique_ordinals) > 1):
                    has_gap = True
            else:
                axis_values = np.sort(pd.to_numeric(pd.Series(row_labels), errors="coerce").dropna().to_numpy(dtype=float))
                if axis_values.size > 1:
                    step = float(np.min(np.diff(axis_values)))
                    if step > 0 and np.any(np.diff(axis_values) > step * (1.0 + 1e-9)):
                        has_gap = True

    # Regressor names in the order the table prints them: every retained lag first (`L1.`, `L2.` ...),
    # the constant last. MEASURED from the capture: `L1.`, `L2.`, then `_cons`.
    order = _lag_order(lags)
    regressor_names: list[str] = []
    for lag in order:
        for _ in range(len(names)):
            regressor_names.append(f"L{lag}.")
    if trend == "c":
        regressor_names.append("_cons")

    return IrfResults(
        names=names,
        regressor_names=tuple(regressor_names),
        lags=len(order),
        lag_orders=lags if isinstance(lags, int) else tuple(order),
        nobs=n,
        sample_first=first,
        sample_last=last,
        sample_has_gap=has_gap,
        coefficients=coefficients,
        sigma=sigma,
        sigma_se=sigma_se,
        ll=ll,
        aic=aic,
        hqic=hqic,
        sbic=sbic,
        fpe=fpe,
        det_sigma=det_sigma,
        rmse=tuple(rmse),
        r2=tuple(r2),
        chi2=tuple(chi2),
        chi2_p=tuple(chi2_p),
        parms=parms,
        design=design,
    )

# ---------------------------------------------------------------- IRF
@dataclass(frozen=True)
class IrfTable(ResultSerialisation):
    """Stata's ``irf table irf`` block."""

    responses: "tuple[ImpulseResponse, ...]"
    ci_level: str = "95"

    def plot(
        self,
        *,
        ax: "Any | None" = None,
        figsize: "tuple[float, float] | None" = None,
        show_ci: bool = True,
        ci_alpha: float = 0.25,
        markers: bool = False,
        title: "str | None" = None,
        grid: bool = True,
        **plot_kwds: Any,
    ) -> "Any":
        """Draw the impulse responses, the figure Stata's ``irf graph`` produces.

        One panel per (impulse, response) pair; the band is the delta-method interval already stored
        on each :class:`ImpulseResponse`, filled at ``ci_alpha``.

        ``matplotlib`` is imported HERE rather than at module scope, the same way ``arch`` is: a user
        who never plots should not need it installed (R1.1 -- no forced dependency).

        Parameters
        ----------
        ax:
            An existing ``matplotlib`` axis, or a single one for the first pair. When omitted a
            figure is created sized to the number of pairs.
        figsize:
            Passed to ``plt.subplots``; defaults to ``(6 * columns, 4 * rows)``.
        show_ci:
            Draw the confidence band. Off leaves only the point path.
        ci_alpha:
            Band opacity.
        markers:
            Mark the points in addition to the line.
        title:
            Override the per-panel title (``"impulse -> response"``).
        grid:
            Draw a light grid.
        **plot_kwds:
            Forwarded to ``Axes.plot`` for the point path.

        Returns
        -------
        The matplotlib ``Figure``.
        """

        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "varbasic.plot() needs matplotlib: pip install 'westata[plot]'"
            ) from exc

        pairs = list(self.responses)
        if not pairs:
            raise ConsistencyError("irf_table: there is nothing to plot")

        if ax is not None:
            axes = [ax]
        else:
            columns = 1 if len(pairs) == 1 else 2
            rows = -(-len(pairs) // columns)
            if figsize is None:
                figsize = (6.0 * columns, 4.0 * rows)
            _figure, axes_flat = make_axes(rows, columns, figsize=figsize)
            axes = axes_flat
            for unused in axes[len(pairs):]:
                unused.set_visible(False)

        for axis, pair in zip(axes, pairs):
            steps = np.asarray(pair.steps, dtype=float)
            values = np.asarray(pair.values, dtype=float)
            lower = np.asarray(pair.lower, dtype=float)
            upper = np.asarray(pair.upper, dtype=float)

            if show_ci:
                axis.fill_between(
                    steps, lower, upper, alpha=ci_alpha, linewidth=0, label=f"{self.ci_level}% CI"
                )
            style = dict(plot_kwds)
            style.setdefault("linewidth", 1.6)
            style.setdefault("color", "tab:blue")
            if markers:
                style.setdefault("marker", "o")
                style.setdefault("markersize", 3.0)
            axis.plot(steps, values, label="IRF", **style)
            axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
            if grid:
                axis.grid(True, alpha=0.3, linewidth=0.6)
            if title is None:
                axis.set_title(f"{pair.impulse} → {pair.response}", fontsize=10)
            else:
                axis.set_title(title, fontsize=10)
            axis.set_xlabel("step")
            axis.set_ylabel("response")

        if ax is None:
            _figure.tight_layout()
            return _figure
        return ax.figure


    def __str__(self) -> str:
        """MEASURED from ``irf table irf`` on gdp_china.dta: 45 characters wide, the pipe at 9.

        One block per (impulse, response) pair; the header carries the pair's index in parentheses.
        """

        # MEASURED: the rule lines are 45 wide, the header rows 45, and the DATA rows 43 -- the
        # header carries two trailing spaces the data rows do not. Fields are right-aligned so they
        # END at columns 19, 31 and 43, and the step number ends at column 8.
        lines = ["", "Results from varbasic", ""]
        lines.append("-" * 45)
        lines.append("         |      (1)         (1)         (1)  ")
        lines.append("    Step |      irf       Lower       Upper  ")
        lines.append("-" * 9 + "+" + "-" * 35)
        first = self.responses[0]
        for step, value, lower, upper in zip(
            first.steps, first.values, first.lower, first.upper
        ):
            head = f"{step:>8} |"
            cells = _cell_irf(_irf_number(value), 19, 10)
            cells += _cell_irf(_irf_number(lower), 31, 19)
            cells += _cell_irf(_irf_number(upper), 43, 31)
            lines.append(head + cells)
        lines.append("-" * 45)
        lines.append(f"{self.ci_level}% lower and upper bounds reported.")
        lines.append(
            f"(1) irfname = varbasic, impulse = {first.impulse}, "
            f"and response = {first.response}."
        )
        # Stata closes the block with a blank line: the capture is 20 lines, the last empty.
        return "\n".join(lines + [""])


def _irf_number(value: float) -> str:
    """Stata's `irf table` number format.

    Derived from fifteen captured values rather than assumed: SIX decimals with trailing zeros
    trimmed, but the field may not exceed SEVEN characters (eight when negative, since the sign takes
    a column). Exceeding it costs a decimal place:

        .771162   .14597   -.23347   -.245543   .0707
        1.083528 -> '1.08353'        (8 chars would overflow, so it drops to five decimals)

    A plain `%.6g` prints `-.0332398` where Stata prints `-.03324`; `%.6f` alone cannot
    produce `1.08353`. The width fallback is what reconciles the two.
    """

    if value is None or not np.isfinite(value):
        return "."
    number = float(value)
    if number == 0.0:
        return "0"
    # An exact integer prints bare: step 0 is 1, not 1.0.
    if number == int(number):
        return str(int(number))
    # Below 1e-5 Stata switches to scientific notation: a VAR(2) step 8 prints 1.2e-06.
    if abs(number) < 1e-5:
        text = f"{number:.1e}"
        mantissa, _, exponent = text.partition("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        if mantissa.startswith("0."):
            mantissa = mantissa[1:]
        elif mantissa.startswith("-0."):
            mantissa = "-" + mantissa[2:]
        sign = "-" if exponent.startswith("-") else "+"
        digits = exponent.lstrip("+-").lstrip("0").rjust(2, "0")
        return f"{mantissa}e{sign}{digits}"
    for decimals in range(6, 0, -1):
        text = ("%.*f" % (decimals, number)).rstrip("0")
        if text.endswith("."):
            text += "0"
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
        limit = 8 if text.startswith("-") else 7
        if len(text) <= limit:
            return text
    return text


def _cell_irf(text: str, end: int, used: int) -> str:
    """Right-align ``text`` so the field ends at column ``end``, given ``used`` already emitted."""

    return " " * max(end - used - len(text), 1) + text


def irf_table(
    fitted: IrfResults,
    *,
    impulse: str | None = None,
    response: str | None = None,
    steps: int = 8,
    all: bool = False,
) -> IrfTable:
    """Stata's ``irf table irf``: impulse responses with delta-method bands.

    ``varbasic`` registers the IRF but prints no table; this is the second command that shows it.

    By default ONE pair is tabulated -- ``impulse`` and ``response`` default to the first variable,
    which is what ``irf table irf`` prints for a single-variable VAR and keeps the block
    byte-identical to Stata. Pass ``all=True`` to get every combination, which is what
    :meth:`IrFTable.plot` lays out in panels.

    The bands come from the coefficient covariance with Stata's ``/T`` normalisation, not
    ``/(T - k)``.
    """

    names = list(fitted.names)
    for role, value in (("impulse", impulse), ("response", response)):
        if value is not None and value not in names:
            raise ConsistencyError(f"irf_table: {role} {value!r} is not in the VAR")
    if steps < 0:
        raise ConsistencyError("irf_table: `steps` must not be negative")

    if all:
        pairs = [
            _response_pair(fitted, source, target, steps)
            for source in names
            for target in names
        ]
    else:
        pairs = [
            _response_pair(
                fitted, impulse or names[0], response or names[0], steps
            )
        ]
    return IrfTable(responses=tuple(pairs))


def _response_pair(
    fitted: IrfResults, impulse: str, response: str, steps: int
) -> ImpulseResponse:
    """One impulse-response path with its band."""

    names = list(fitted.names)
    coefficients = np.asarray(fitted.coefficients, dtype=float)
    k = len(names)
    offset = 1 if (coefficients.shape[0] > k and fitted.regressor_names[-1] == "_cons") else 0

    chol = np.tril(np.ones((k, k)))
    companion = _companion(coefficients, k, k, offset)
    impulse_index = names.index(impulse)
    response_index = names.index(response)

    design = np.asarray(fitted.design, dtype=float)
    z = 1.959963985

    steps_out, values, lower, upper = [], [], [], []
    single_lag = isinstance(fitted.lag_orders, int) and _lag_count(fitted) == 1
    for step in range(steps + 1):
        if single_lag and step % int(fitted.lag_orders) != 0:
            steps_out.append(step)
            values.append(0.0)
            lower.append(0.0)
            upper.append(0.0)
            continue
        effective = step // int(fitted.lag_orders) if single_lag else step
        theta = _theta(companion, chol, effective, k)
        point = float(theta[response_index, impulse_index])

        gradient = _gradient(
            coefficients, _lag_count(fitted), k, impulse_index, response_index,
            effective, chol, offset,
        )
        block = _coefficient_covariance(fitted, coefficients, design, response_index, offset)
        variance = float(gradient[offset:] @ block @ gradient[offset:]) if block.size else 0.0
        error = float(np.sqrt(max(variance, 0.0)))

        steps_out.append(step)
        values.append(point)
        lower.append(point - z * error)
        upper.append(point + z * error)

    return ImpulseResponse(
        impulse=impulse,
        response=response,
        steps=tuple(steps_out),
        values=tuple(values),
        lower=tuple(lower),
        upper=tuple(upper),
    )


def _coefficient_covariance(
    fitted: IrfResults,
    coefficients: np.ndarray,
    design: np.ndarray,
    equation: int,
    offset: int,
) -> np.ndarray:
    """Covariance of one equation's lag coefficients: sigma_jj * (X'X)^-1, lag block only.

    Stata normalises sigma by T, not (T - k) -- see the module docstring.
    """

    xtx_inv = np.linalg.pinv(design.T @ design)
    sigma = np.asarray(fitted.sigma, dtype=float)
    sigma_jj = float(sigma[equation, equation])
    lag_rows = list(range(offset, design.shape[1]))
    return sigma_jj * xtx_inv[np.ix_(lag_rows, lag_rows)]


def _lag_count(fitted: IrfResults) -> int:
    """How many lag blocks the fit retained, read from its coefficient matrix."""

    coefficients = np.asarray(fitted.coefficients, dtype=float)
    constant_rows = 0 if fitted.regressor_names[-1] == "_cons" else 1
    rows = coefficients.shape[0] - (0 if constant_rows else 1)
    return max(rows // max(coefficients.shape[1], 1), 1)


def _companion(
    coefficients: "np.ndarray", lags: int, k: int, offset: int
) -> "np.ndarray":
    """The companion matrix of a VAR(p), in (response, impulse) order.

    ``coefficients`` is laid out (constant, lag block 1, lag block 2, ...). Each block is
    k x k with rows = regressor and columns = equation, so it is TRANSPOSED into the
    (response, impulse) orientation the companion needs.

    For p > 1 the lower blocks are the identity shifts::

        [ A1  A2  ...  Ap ]
        [ I   0   ...  0  ]
        [ 0   I   ...  0  ]
        ...

    MISSING THESE SHIFTS WAS A REAL BUG. The previous version returned only ``A1.T``, which is a
    VAR(1) whatever p is; on a VAR(2) it produced ``.594687`` at step 2 where Stata reports
    ``.145969``. The recursion Stata follows is ``s[h] = A1 s[h-1] + A2 s[h-2] + ...``, and the
    companion reproduces exactly that:

        s1 = A1 = .771159
        s2 = A1*s1 + A2 = .594687 - .448718 = .145969
    """

    # Derive the block count from the coefficient matrix rather than trusting lags: the argument
    # arrives as the ORIGINAL user value (an int for a bare arbasic), while the matrix always has
    # one row per retained lag plus the constant. Trusting the argument built a companion of the
    # wrong width.
    total_rows = coefficients.shape[0]
    factor_rows = coefficients.shape[1]
    order = max(total_rows - offset, 1) // max(factor_rows, 1)
    blocks = []
    for index in range(order):
        block = coefficients[offset + index * k : offset + (index + 1) * k, :]
        blocks.append(block.T)

    top = np.hstack(blocks) if blocks else np.zeros((k, 0))
    if order == 1:
        return top

    lower = np.zeros((k * (order - 1), k * order))
    for index in range(order - 1):
        lower[index * k : (index + 1) * k, index * k : (index + 1) * k] = np.eye(k)
    return np.vstack([top, lower])


def _theta(
    companion: "np.ndarray", chol: "np.ndarray", step: int, k: int
) -> "np.ndarray":
    """The k x k response matrix at `step`.

    The companion is (k*lags) x (k*lags) while `chol` is k x k, so the top-left k rows must be
    restricted to the first k columns before multiplying -- otherwise the shapes disagree whenever
    `lags > 1`.
    """

    power = np.linalg.matrix_power(companion, step) if step else np.eye(companion.shape[0])
    return power[:k, :k] @ chol


def _gradient(
    coefficients: "np.ndarray",
    lags: int,
    k: int,
    impulse: int,
    response: int,
    step: int,
    chol: "np.ndarray",
    offset: int,
) -> "np.ndarray":
    """d(theta[response, impulse]) / d(each coefficient), by central differences.

    Analytic differentiation through the companion power is possible but error-prone for p > 1; the
    numerical gradient is exact enough here because the function is a polynomial in the coefficients,
    and it keeps the p = 1 and p > 1 cases on one code path.
    """

    base = coefficients.ravel().copy()
    gradient = np.zeros(base.size)
    epsilon = 1e-7
    for index in range(base.size):
        plus = base.copy()
        plus[index] += epsilon
        minus = base.copy()
        minus[index] -= epsilon
        f_plus = _theta(
            _companion(plus.reshape(coefficients.shape), lags, k, offset), chol, step, k
        )[response, impulse]
        f_minus = _theta(
            _companion(minus.reshape(coefficients.shape), lags, k, offset), chol, step, k
        )[response, impulse]
        gradient[index] = (f_plus - f_minus) / (2 * epsilon)
    return gradient

