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
def _prepared(series: pd.DataFrame, variables: Sequence[str], lags: int) -> tuple[np.ndarray, int]:
    """The estimation sample after lags, as an (n, k) float array."""

    for name in variables:
        if name not in series.columns:
            raise ConsistencyError(
                f"varbasic: variable {name!r} not found; available columns: {list(series.columns)}"
            )
    frame = series[list(variables)].dropna()
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
            f"{'Log likelihood = ' + f'{self.ll:>10.5f}':<48}AIC               ={self.aic:>11.6f}"
        )
        lines.append(
            f"{'FPE            = ' + f'{self.fpe:>10.7f}':<48}HQIC              ={self.hqic:>11.6f}"
        )
        lines.append(
            f"{'Det(Sigma_ml)  = ' + f'{self.det_sigma:>10.7f}':<48}SBIC              ={self.sbic:>11.6f}"
        )
        lines.append("")
        lines.append("Equation           Parms      RMSE     R-sq      chi2     P>chi2")
        lines.append("-" * 64)
        for index, name in enumerate(self.names):
            lines.append(
                f"{name:<16}{self.parms[index]:>8}"
                f"{self.rmse[index]:>10.6f}"
                f"{self.r2[index]:>9.4f}"
                f"{self.chi2[index]:>11.5f}"
                f"{self.chi2_p[index]:>10.4f}"
            )
        lines.append("-" * 64)
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
        series: pd.DataFrame,
        variables: Sequence[str] | str,
        *,
        lags: int = 1,
        trend: str = "c",
        time: str | None = None,
        keep: "pd.Series | None" = None,
    ) -> None:
        self._series = series
        self._variables = variables
        self._lags = lags
        self._trend = trend
        self._time = time
        self._keep = keep

    def fit(self) -> IrfResult:
        """Estimate the VAR and register the IRF (the responses need :func:`irf_table`)."""

        return varbasic(
            self._series,
            self._variables,
            lags=self._lags,
            trend=self._trend,
            time=self._time,
            keep=self._keep,
        )


def varbasic(
    series: pd.DataFrame,
    variables: Sequence[str] | str,
    *,
    lags: int = 1,
    trend: str = "c",
    time: str | None = None,
    keep: "pd.Series | None" = None,
) -> IrfResult:
    """Stata's ``varbasic``: fit a VAR(p) and register an IRF.

    ``time`` names the time variable; when given, the reported sample range uses its values.
    ``varbasic`` itself prints only the estimation block -- the responses need :func:`irf_table`,
    which is also what Stata requires.

    ``keep`` is Stata's ``if`` condition as a boolean mask over ``series``. PASS THE FULL SERIES with
    ``keep`` rather than pre-filtering: Stata builds the lag on the complete time axis and only then
    restricts the sample (`_qsur.ado:54` marks out ``l(touse).(varlist)``). Pre-filtering changes the
    first retained observation's lag from a real value to a missing one and shifts every estimate --
    on this fixture, ``year>=1985`` gives ``.528017`` from the full series but ``.502390`` from a
    pre-filtered frame.
    """

    names = [variables] if isinstance(variables, str) else list(variables)
    if not names:
        raise ConsistencyError("varbasic: at least one variable is required")

    for name in names:
        if name not in series.columns:
            raise ConsistencyError(
                f"varbasic: variable {name!r} not found; available columns: {list(series.columns)}"
            )

    # Build the engine's view of the data over EVERY row, so a retained row's lag can reach back
    # before the `if` condition. `_qsur.ado:54` marks out l(lags).(varlist) on the full sample.
    raw = series[list(names)].to_numpy(dtype=float)
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

    # A row is usable when its lag and its response are both present ...
    usable = ~np.isnan(lagged).any(axis=1) & ~np.isnan(response).any(axis=1)
    # ... and, when `keep` is given, when the `if` condition accepts the RESPONSE row.
    if keep is not None:
        mask = pd.Series(keep, index=series.index).fillna(False).astype(bool).to_numpy()
        usable &= mask

    if not usable.any():
        raise ConsistencyError("varbasic: no usable observations after applying `keep`")

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
    if time is not None and time in series.columns:
        kept = series[list(names) + [time]].dropna()
        if len(kept) > lags:
            ordered = kept.sort_values(time)
            first = ordered[time].iloc[lags]
            last = ordered[time].iloc[-1]

    return IrfResult(
        names=names,
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

