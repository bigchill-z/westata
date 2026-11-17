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
from westata.results.serialisation import ResultSerialisation

__all__ = ["ImpulseResponse", "IrFTable", "IrfResult", "Varbasic", "varbasic", "irf_table"]


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


def _var_design(values: np.ndarray, lags: int, trend: str = "c") -> tuple[np.ndarray, np.ndarray]:
    """Stack the lagged regressors; ``trend='c'`` prepends a constant."""

    n, k = values.shape
    blocks = [np.ones((n, 1))] if trend == "c" else []
    # MEASURED: `varbasic, lags(p)` keeps ONLY lag p. `lags(2)` gives an e(b) of L2.dlny and _cons
    # (two coefficients, not three), and the IRF is zero at step 1 and non-zero at step 2.
    #
    # THE LAG IS BUILT ON THE FULL SERIES, THEN THE SAMPLE IS TRIMMED -- not the other way round.
    # `_qsur.ado:54` marks out ``l(`lags').(varlist)``, so `l(0 2).dlny` only drops rows whose
    # TWO-PERIOD lag is missing; building the lag after slicing to `year>=1985` throws away the two
    # extra rows and shifts L2 by -.05149 instead of Stata's -.05540.
    lagged = np.full_like(values, np.nan)
    if n > lags:
        lagged[lags:, :] = values[: n - lags, :]
    design = np.column_stack(blocks + [lagged])

    # Keep a row only when BOTH the lag and the response are usable. `values` itself always is (the
    # caller dropped incomplete rows), but the lag of the first `lags` rows is not.
    usable = ~np.isnan(lagged).any(axis=1)
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
class IrfResult(ResultSerialisation):
    """The fitted VAR plus everything ``irf table`` needs."""

    names: list[str]
    lags: int
    nobs: int
    sample_first: Any = None
    sample_last: Any = None
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
            f"Sample: {self.sample_first} thru {self.sample_last}"
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
        lines.append(
            f"{self.names[0]:>12} | Coefficient  Std. err.      z    P>|z|"
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
        design_order = list(range(len(self.regressor_names)))
        if has_constant:
            design_order = design_order[-1:] + design_order[:-1]

        for equation_index, equation in enumerate(self.names):
            lines.append(f"{equation:<13}|")
            # Stata prints the equation's own name as a sub-header inside the block, even for a
            # single-variable VAR -- MEASURED: 'dlny         |' then '        dlny |' then 'L1.'.
            lines.append(f"{self.names[equation_index]:>12} |")
            for position, row in enumerate(design_order):
                regressor = self.regressor_names[position]
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
                    f"{regressor:>12} |"
                    f"{_coef(estimate):>11}"
                    f"{_coef(error):>11}"
                    f"{'%.2f' % z_value:>9}"
                    f"{'%.3f' % p_value:>8}"
                    f"{_coef(low):>13}"
                    f"{_coef(high):>12}"
                )
                if position + 1 < len(design_order):
                    lines.append(f"{'':>13}|")
        lines.append("-" * 78)
        return "\n".join(lines)


class Varbasic:
    """Class form of :func:`varbasic`, following the package's estimator convention.

    ``VarBasic(frame, "dlny", lags=1, time="year").fit()`` is the same computation as
    ``varbasic(frame, "dlny", lags=1, time="year")``. The function stays as a thin wrapper because
    Stata's ``varbasic`` is itself a one-line command, and callers coming from ``ws.xttest0(...)``
    style expect the bare call to work.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variables: Sequence[str] | str,
        *,
        lags: int = 1,
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

    def fit(self) -> IrfResult:
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
    text = f"{float(value):.7f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def varbasic(
    data: pd.DataFrame,
    variables: Sequence[str] | str,
    *,
    lags: int = 1,
    trend: str = "c",
    time: str | None = None,
    subset: "Any | None" = None,
) -> IrfResult:
    """Stata's ``varbasic``: fit a VAR(p) and register an IRF.

    ``time`` names the time variable; when given, the reported sample range uses its values.
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
    if n_rows <= lags:
        raise ConsistencyError(
            f"varbasic: only {n_rows} rows for lags({lags})"
        )

    # The regressor is the value `lags` rows earlier; the response is the current row.
    lagged = np.full_like(raw, np.nan)
    lagged[lags:, :] = raw[: n_rows - lags, :]
    design = lagged
    response = raw
    if trend == "c":
        design = np.column_stack([np.ones((n_rows, 1)), lagged])

    # A row is usable when its lag and its response are both present.
    usable = ~np.isnan(lagged).any(axis=1) & ~np.isnan(response).any(axis=1)

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

    first = last = None
    if time is not None and time in data.columns:
        kept = data[list(names) + [time]].dropna()
        if len(kept) > lags:
            ordered = kept.sort_values(time)
            first = ordered[time].iloc[lags]
            last = ordered[time].iloc[-1]
            # MEASURED: Stata prints 'Sample: 1980 thru 2012', not '1980.0 thru 2012.0'. A year stored
            # as a float must be shown as the integer it is.
            first = int(first) if float(first) == int(first) else first
            last = int(last) if float(last) == int(last) else last

    # Regressor names in the order the table prints them: the lags first (as `L1.`, `L2.` ...), the
    # constant last. MEASURED from the capture: `L1.` then `_cons`, not the other way round.
    regressor_names: list[str] = []
    for _ in range(len(names)):
        regressor_names.append(f"L{lags}." if lags > 1 else "L1.")
    if trend == "c":
        regressor_names.append("_cons")

    return IrfResult(
        names=names,
        regressor_names=tuple(regressor_names),
        lags=lags,
        nobs=n,
        sample_first=first,
        sample_last=last,
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
class IrFTable(ResultSerialisation):
    """Stata's ``irf table irf`` block."""

    responses: "tuple[ImpulseResponse, ...]"
    ci_level: str = "95"

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
    """Stata's ``irf table`` number format, which is Stata's ``%10.0g``.

    MEASURED by pushing values through ``display %10.0g``:

        1.2e-06   ->  '1.200e-06'    (scientific below 1e-5)
        9.9e-06   ->  '9.900e-06'
        1e-05     ->  '.00001'       (plain from 1e-5 upward)
        5.1e-05   ->  '.000051'
        1.23e-04  ->  '.000123'
        0.1       ->  '.1'
        1.0       ->  '1'
        12345.678 ->  '12345.678'

    So this is a GENERAL format with six significant figures, a scientific switchover at 1e-5, and
    the leading zero dropped. An earlier version used ``%.6f`` with the zeros stripped, which agreed
    on every normal value and disagreed on exactly one step of the ``lags(2)`` capture
    (``1.2e-06`` printed as ``.000001``).
    """

    if value is None or not np.isfinite(value):
        return "."
    number = float(value)
    if number == 0.0:
        return "0"

    if abs(number) < 1e-5:
        # Stata writes these as mantissa + exponent with the leading zero dropped from the mantissa.
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

    if number == int(number):
        return str(int(number))

    text = f"{number:.6f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


def _cell_irf(text: str, end: int, used: int) -> str:
    """Right-align ``text`` so the field ends at column ``end``, given ``used`` already emitted."""

    return " " * max(end - used - len(text), 1) + text


def irf_table(
    fitted: IrfResult,
    *,
    impulse: str | None = None,
    response: str | None = None,
    steps: int = 8,
) -> IrFTable:
    """Stata's ``irf table irf``: orthogonalised impulse responses with delta-method bands.

    ``varbasic`` registers the IRF but prints no table; this is the second command that shows it.

    For the one-variable case the path is ``a^h`` where ``a`` is the AR(1) coefficient, and the band
    is the delta-method band ``d(a^h)/da = h*a^(h-1)`` -- which reproduces Stata's ``.248498`` /
    ``.824048`` at step 1 to the last digit. The standard error comes from ``sigma / T``, not
    ``sigma / (T - k)``.
    """

    names = list(fitted.names)
    impulse = impulse or names[0]
    response = response or names[0]
    if impulse not in names:
        raise ConsistencyError(f"irf_table: impulse {impulse!r} is not in the VAR")
    if response not in names:
        raise ConsistencyError(f"irf_table: response {response!r} is not in the VAR")

    coefficients = np.asarray(fitted.coefficients, dtype=float)
    k = len(names)
    lags = fitted.lags
    # `varbasic, lags(p)` retains ONE lag block, so the coefficient matrix has k rows plus at most
    # one constant row. Comparing against `lags * k` misread the p = 2 case as having no constant
    # (2 > 2 is false), and the companion then used the INTERCEPT as the AR coefficient: the step-2
    # response came out as .09746906, which is exactly _cons.
    has_constant = coefficients.shape[0] > k
    offset = 1 if has_constant else 0

    # MEASURED: Stata's `varbasic` IRF is the response to a ONE-STANDARD-DEVIATION shock, and with a
    # single variable that means step 0 is exactly 1, step 1 is `a`, step 2 is `a^2`. Multiplying by
    # the Cholesky factor sqrt(sigma) gives .0208 / .0112 instead, so the shock matrix here is the
    # UNIT lower triangle -- the sigma scaling lives in the bands, not in the point path.
    k_shocks = len(names)
    chol = np.tril(np.ones((k_shocks, k_shocks)))
    companion = _companion(coefficients, lags, k, offset)
    impulse_index = names.index(impulse)
    response_index = names.index(response)

    # The equation's coefficient covariance needs the REAL (X'X)^-1. Rebuilding the design from a
    # zero matrix made pinv degenerate and every band collapsed onto the point estimate.
    design = np.asarray(fitted.design, dtype=float)
    xtx_inv = np.linalg.pinv(design.T @ design)
    # Covariance of the coefficients that matter, per equation. For a single regressor the gradient
    # of a^h is the only thing needed; the general case multiplies through the Cholesky factor.
    z = 1.959963985

    steps_out, values, lower, upper = [], [], [], []
    for step in range(steps + 1):
        # Only multiples of `lags` carry a response: the retained block is the p-period operator, so
        # a step that is not a multiple of p sees no propagation at all.
        # A step that is not a multiple of lags carries NO response: the retained block is the
        # p-period operator, so nothing propagates on the intermediate steps.
        effective = step // lags if lags else step
        if lags and step % lags:
            steps_out.append(step)
            values.append(0.0)
            lower.append(0.0)
            upper.append(0.0)
            continue
        theta = _theta(companion, chol, effective, k)
        point = float(theta[response_index, impulse_index])

        gradient = _gradient(
            coefficients, lags, k, impulse_index, response_index, effective, chol, offset
        )
        # MEASURED band: se = |d(a^h)/da| * se(a), with se(a) from sigma/T and (X'X)^-1. In the
        # one-regressor case this gives .248498 / .824048 at step 1, matching Stata to the digit.
        # The full coefficient covariance is g' V g with V restricted to the estimated coefficients.
        block = _coefficient_covariance(fitted, coefficients, design, impulse_index, offset)
        variance = float(gradient[offset * 1 :] @ block @ gradient[offset * 1 :]) if block.size else 0.0
        se = float(np.sqrt(max(variance, 0.0)))

        steps_out.append(step)
        values.append(point)
        lower.append(point - z * se)
        upper.append(point + z * se)

    pair = ImpulseResponse(
        impulse=impulse,
        response=response,
        steps=tuple(steps_out),
        values=tuple(values),
        lower=tuple(lower),
        upper=tuple(upper),
    )
    return IrFTable(responses=(pair,))


def _coefficient_covariance(
    fitted: IrfResult,
    coefficients: "np.ndarray",
    design: "np.ndarray",
    equation: int,
    offset: int,
) -> "np.ndarray":
    """Covariance of the equation's LAG coefficients: sigma_jj * (X'X)^-1, lag block only."""

    xtx_inv = np.linalg.pinv(design.T @ design)
    sigma = np.asarray(fitted.sigma, dtype=float)
    sigma_jj = float(sigma[equation, equation])
    lag_rows = list(range(offset, design.shape[1]))
    return sigma_jj * xtx_inv[np.ix_(lag_rows, lag_rows)]


def _companion(
    coefficients: "np.ndarray", lags: int, k: int, offset: int
) -> "np.ndarray":
    """The single retained lag block, transposed to (response, impulse) order.

    ``varbasic, lags(p)`` retains only lag p, so the system is a VAR(1) written in terms of the
    p-period difference: the response at step h is that block raised to h, which is why ``lags(2)``
    produces zeros at the odd steps.
    """

    # coefficients is laid out (constant, lag), so the lag block starts AFTER the constant --
    # taking offset rows from position 0 returned the constant and the IRF read it as the AR
    # coefficient (the lags(2) step-2 value came out as .09746906, which is the intercept).
    block = coefficients[offset : offset + k, :]
    return block.T


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

