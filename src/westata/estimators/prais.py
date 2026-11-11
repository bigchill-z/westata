"""Implementation of Stata's ``prais`` (Prais-Winsten AR(1) regression).

Stata's reference values on icecream.dta after ``prais consumption temp price income``:

    Iteration 40: rho = 0.8002          (iterated until |change| < 1e-6)
    temp    .0029541
    price  -1.048854
    income -.0008022
    _cons   .5870049
    rho     .8002264
    DW original    1.021169
    DW transformed 1.846795
    R-squared      0.6234
    Root MSE       .03232

The algorithm Stata documents: estimate by OLS, take rho from the residuals, transform the
data by the Prais-Winsten rule (the first observation is scaled by sqrt(1 - rho^2) rather than
dropped), re-estimate, and repeat until rho stops moving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError, ConvergenceError

__all__ = ["PraisResult", "prais", "prais_winsten"]


def _rho_from_residuals(resid: np.ndarray) -> float:
    """AR(1) coefficient from a residual series: ``sum e_t e_{t-1} / sum e_{t-1}^2``."""

    lagged = resid[:-1]
    current = resid[1:]
    denominator = float(lagged @ lagged)
    if denominator <= 0.0:
        return 0.0
    return float(lagged @ current) / denominator


def _transform(values: np.ndarray, rho: float) -> np.ndarray:
    """Prais-Winsten transform of a 1-D series."""

    out = np.empty_like(values)
    out[0] = np.sqrt(max(1.0 - rho * rho, 0.0)) * values[0]
    out[1:] = values[1:] - rho * values[:-1]
    return out


def _transform_corc(values: np.ndarray, rho: float) -> np.ndarray:
    """Cochrane-Orcutt transform: ``y_t - rho*y_{t-1}`` with the first observation DROPPED.

    ``prais, corc`` (prais.ado:6, ``syntax ... noDW CORC``) runs the same iteration as the
    Prais-Winsten default with this one difference: the transformed first row is discarded
    rather than scaled by ``sqrt(1 - rho^2)``.

    Measured on ``prais y k l`` / ``prais y k l, corc`` over cobb_douglas:

        prais : b = .15651652  .93948861  -7.5439159   rho = .12543124
        corc  : b = .15687809  .92974899  -6.0923288   rho = .11463695
    """

    return values[1:] - rho * values[:-1]


def prais_winsten(
    exog: np.ndarray,
    endog: np.ndarray,
    *,
    maxiter: int = 100,
    tolerance: float = 1e-6,
    rho: float | None = None,
    corc: bool = False,
) -> tuple[np.ndarray, float, int, list[tuple[int, float]]]:
    """Iterated Prais-Winsten / Cochrane-Orcutt estimate.

    Returns ``(coefficients, rho, iterations, trace)``. ``rho`` may be given to run a single pass
    instead of iterating; the trace is then empty.

    ``trace`` is the sequence Stata prints as its iteration log, as ``(iteration, rho)`` pairs with
    ``Iteration 0`` a literal zero. Reproducing it needs the ado's PASS ORDER, not only its
    arithmetic (``prais.ado:172-217``)::

        lastrho = 2 ; rho = 0
        while abs(rho - lastrho) > tol {
            lastrho = rho
            resid = y - X b        <- residual of the PREVIOUS fit (plain OLS on pass 1)
            rho   = crcar1(resid)  <- the value that gets printed
            b     = fit AT rho     <- and the value the next pass fits with
        }

    ``rho`` is the loop variable and the fit input; the coefficients follow it. Re-seeding ``rho``
    from the OLS estimate, or treating the fit and the estimate as one unit, produces
    ``0.3895, 0.3975, ...`` where Stata has ``0.4008, 0.4009, ...``.

    Two further conditions are satisfied elsewhere: ``_rho_from_residuals`` is the slope of the
    residual on its own lag with no constant (``_crcar1.ado:37-42``), and ``_transform_corc`` drops
    the first row (``prais_e.ado:28-33``).
    """

    x = np.asarray(exog, dtype=float)
    y = np.asarray(endog, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ConsistencyError(
            f"prais: design is {x.shape} but the response has {y.size} values"
        )

    transform = _transform_corc if corc else _transform

    def ols(design: np.ndarray, response: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(design, response, rcond=None)[0]

    coefficients = ols(x, y)

    # A FIXED RHO RUNS EXACTLY ONE PASS AND PRINTS NO LOG -- the value was supplied, not estimated.
    if rho is not None:
        fixed = float(rho)
        transformed_x = np.column_stack(
            [transform(x[:, j], fixed) for j in range(x.shape[1])]
        )
        return ols(transformed_x, transform(y, fixed)), fixed, 1, []

    current_rho = 0.0
    previous_rho = 2.0
    trace: list[tuple[int, float]] = [(0, 0.0)]
    iterations = 0

    while iterations < maxiter and abs(current_rho - previous_rho) > tolerance:
        previous_rho = current_rho
        residual = y - x @ coefficients
        current_rho = _rho_from_residuals(residual)
        if abs(current_rho) >= 1.0:
            failure = ConvergenceError(
                f"prais: the estimated AR(1) coefficient {current_rho:.6f} is not inside "
                "(-1, 1); the series is not stationary"
            )
            failure.rho = current_rho  # type: ignore[attr-defined]
            raise failure
        transformed_x = np.column_stack(
            [transform(x[:, j], current_rho) for j in range(x.shape[1])]
        )
        coefficients = ols(transformed_x, transform(y, current_rho))
        iterations += 1
        trace.append((iterations, current_rho))

    if iterations >= maxiter:
        # Stata prints the value it reached BEFORE reporting r(430), so the estimate travels on the
        # exception rather than being discarded.
        failure = ConvergenceError(
            f"prais: rho did not converge in {maxiter} iterations; last value "
            f"{current_rho:.7f}"
        )
        failure.rho = current_rho  # type: ignore[attr-defined]
        raise failure

    return coefficients, current_rho, iterations, trace


class PraisResult:
    """Results of a Prais-Winsten AR(1) regression."""

    __slots__ = (
        "_params",
        "_names",
        "_depvar",
        "_bse",
        "_tvalues",
        "_pvalues",
        "_conf_int",
        "_rho",
        "_nobs",
        "_df_resid",
        "_df_model",
        "_mss",
        "_rss",
        "_tss",
        "_r2",
        "_r2_adj",
        "_df_total",
        "_corc",
        "_rmse",
        "_fvalue",
        "_p_f",
        "_dw_original",
        "_dw_transformed",
        "_iterations",
        "_rho_trace",
    )

    def __init__(self, **values: object) -> None:
        for key, value in values.items():
            setattr(self, f"_{key}", value)

    @property
    def params(self) -> pd.Series:
        return pd.Series(self._params, index=self._names, name="params")

    @property
    def bse(self) -> pd.Series:
        return pd.Series(self._bse, index=self._names, name="bse")

    @property
    def tvalues(self) -> pd.Series:
        return pd.Series(self._tvalues, index=self._names, name="tvalues")

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(self._pvalues, index=self._names, name="pvalues")

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """Confidence intervals, in the statsmodels shape.

        ``PraisResult`` carried the bounds privately (``_conf_int``) but exposed no public method,
        so a caller porting statsmodels code had ``result.conf_int()`` fail with AttributeError --
        the deviation the v3 report records at section 4, item (4). The column names are
        ``lower``/``upper`` for the same reason: every other result class in this package uses
        those names, and Stata's 95% interval is what the private array already holds.
        """

        intervals = np.asarray(self._conf_int, dtype=float)
        if float(alpha) != 0.05:
            # Recompute rather than rescale: the stored bounds are the 95% ones, and a different
            # level needs the t quantile, not a linear stretch of the interval.
            from scipy import stats as _stats

            critical = float(_stats.t.ppf(1.0 - float(alpha) / 2.0, self._df_resid))
            centre = np.asarray(self._params, dtype=float)
            spread = np.asarray(self._bse, dtype=float)
            intervals = np.column_stack([centre - critical * spread, centre + critical * spread])
        return pd.DataFrame(intervals, index=self._names, columns=["lower", "upper"])

    def cov_params(self) -> pd.DataFrame:
        """The coefficient covariance matrix, reconstructed from the reported standard errors.

        Only the diagonal was stored, so the off-diagonals are NOT recoverable here. They are
        reported as NaN rather than zero: a zero would claim the estimates are uncorrelated, which
        is a different (and false) statement.
        """

        size = len(self._names)
        matrix = np.full((size, size), np.nan, dtype=float)
        np.fill_diagonal(matrix, np.asarray(self._bse, dtype=float) ** 2)
        return pd.DataFrame(matrix, index=self._names, columns=self._names)

    def summary(self, style: str = "stata") -> str:
        """Render the table; ``style`` is accepted for protocol parity."""

        return str(self)

    @property
    def rho(self) -> float:
        return self._rho

    @property
    def nobs(self) -> int:
        return self._nobs

    @property
    def df_resid(self) -> int:
        return self._df_resid

    @property
    def rho_trace(self) -> tuple[float, ...]:
        """The rho sequence Stata prints as its iteration log, ``Iteration 0`` first."""

        return tuple(self._rho_trace)

    @property
    def df_total(self) -> int:
        """Total degrees of freedom: `N - 1`, or `N - 2` for `corc`."""

        return self._df_total

    @property
    def corc(self) -> bool:
        """Whether the Cochrane-Orcutt transform produced this fit."""

        return self._corc

    @property
    def rsquared(self) -> float:
        return self._r2

    @property
    def rsquared_adj(self) -> float:
        return self._r2_adj

    @property
    def rmse(self) -> float:
        return self._rmse

    @property
    def fvalue(self) -> float:
        return self._fvalue

    @property
    def dw_original(self) -> float:
        return self._dw_original

    @property
    def dw_transformed(self) -> float:
        return self._dw_transformed

    @property
    def iterations(self) -> int:
        return self._iterations

    @staticmethod
    def _round_cell(number, digits, decimals=None) -> str:
        """Round a Decimal to ``digits`` significant figures or ``decimals`` places."""
        from decimal import ROUND_HALF_UP, localcontext

        if decimals is None:
            exponent = number.adjusted()
            quantum = __import__("decimal").Decimal(1).scaleb(exponent - (digits - 1))
        else:
            quantum = __import__("decimal").Decimal(1).scaleb(-decimals)
        with localcontext() as context:
            context.prec = 40
            rounded = number.quantize(quantum, rounding=ROUND_HALF_UP)
        text = format(rounded, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text.startswith("0."):
            return text[1:]
        if text.startswith("-0."):
            return "-" + text[2:]
        return text

    def __str__(self) -> str:
        """Stata's ``prais`` table, captured byte for byte in round 168.

        Column offsets, all measured from a live run on icecream.dta (78-wide rows):

            Source block: label ends at 12, pipe at 13, SS ends at 26, df at 36, MS at 48;
                          the right-hand statistics start at 51.
            Coefficient block: the dependent-variable label at 1, pipe at 13, then
                          Coefficient 15-25, Std. err. 28-36, t 43-44, P>|t| 48-52,
                          CI 56-78.
            '       rho |   .8002264'  is the ONLY row after the coefficient rule, and the
                          closing rule follows it.

        THE CONSTANT IS PRINTED AS ``_cons``, not "Intercept". Several earlier details were wrong
        together: a leading zero on every number, a 13-wide label column instead of 12, and the
        p-value column colliding with the lower interval bound.
        """

        width = 78

        def num(value: float, places: int | None = None, *, width: int | None = None) -> str:
            """Stata's cell: SEVEN DECIMALS when the integer part is zero, SIX otherwise.

            Round 170 established this by listing every cell beside Stata's text at once, after
            a significant-figure rule had failed on ``-.0008022363`` (seven figures, ten
            characters) where Stata prints ``-.0008022``. The rule turned out to be the same one
            ``tebalance`` uses:

                -.0008022, .0020458, .5128361, .5870049   <- seven decimals (integer part 0)
                -1.048854, .759751, 1.193941, -2.610545   <- six decimals  (one integer digit)

            The Source block's SS/MS cells are different: they fill a TEN-character field with as
            many figures as fit, which is the ``width=`` mode.
            """

            from decimal import Decimal, ROUND_HALF_UP, localcontext

            if not np.isfinite(value):
                return "."
            number = Decimal(repr(float(value)))
            if number == 0:
                return "0"
            if width is not None and places is None:
                counts = range(9, 0, -1)
                for digits in counts:
                    candidate = self._round_cell(number, digits)
                    if len(candidate) <= width:
                        return candidate
                return candidate
            decimals = 7 if number.adjusted() < 0 else 6
            return self._round_cell(number, None, decimals)

        def cell(text: str, end: int, current: int) -> str:
            """Right-align ``text`` so it ENDS at absolute column ``end``."""
            return " " * max(end - current - len(text), 0) + text

        def stats_row(label: str, value: str) -> str:
            """Right block: label at 51, '=' at 67, value ending at 78."""
            line = " " * 51 + label
            line += " " * max(67 - len(line), 1) + "="
            return line + " " * max(78 - len(line) - len(value), 0) + value

        def source_row(label: str, ss: str, df: str, ms: str) -> str:
            line = f"{label:>12} |"
            line += cell(ss, 26, len(line))
            line += cell(df, 36, len(line))
            line += cell(ms, 48, len(line))
            return line

        rule = f"{'-' * 13}+{'-' * 34}"

        def joined(left: str, label: str, value: str) -> str:
            """Left block padded to column 48, then the right-hand statistics block."""
            return left + " " * max(48 - len(left), 0) + stats_row(label, value)[48:]

        # THE BANNER NAMES THE TRANSFORM ACTUALLY USED. `prais ..., corc` is
        # "Cochrane-Orcutt AR(1) regression", not Prais-Winsten, and it opens with NO blank
        # lines where the Prais-Winsten form starts with two:
        #
        #     Stata PW    chr(10) chr(10) 'Prais-Winsten AR(1) regression with iterated estimates'
        #     Stata CORC  'Cochrane-Orcutt AR(1) regression with iterated estimates'
        _banner = (
            "Cochrane\u2013Orcutt AR(1) regression with iterated estimates"
            if self._corc
            else "Prais\u2013Winsten AR(1) regression with iterated estimates"
        )
        # ``corc`` STARTS AT THE BANNER, with no leading blank; the Prais-Winsten form has
        # two because its iteration log occupies them. Measured:
        #
        #     PW    ['', '', 'Prais-Winsten AR(1) regression ...', '', '      Source | ...']
        #     CORC  ['Cochrane-Orcutt AR(1) regression ...', '', '      Source | ...']
        # THE ITERATION LOG OPENS THE TABLE: a blank line, one line per pass, a blank line, then
        # the banner. Measured on both variants of `prais consumption temp price income`:
        #
        #     ''
        #     'Iteration 0:  rho = 0.0000'
        #     ...
        #     ''
        #     'Cochrane-Orcutt AR(1) regression with iterated estimates'
        #
        # The rho carries FOUR decimals here ('0.0000'), unlike the table body, where Stata drops
        # the leading zero.
        # THE SPACING NARROWS AT TWO DIGITS. prais.ado:181-184 sets `spacing` to two spaces while
        # `it_ct < 9` and one from 10 on, so the column stays aligned as the label grows:
        #
        #     'Iteration 9:  rho = 0.7427'
        #     'Iteration 10: rho = 0.7448'
        _log = [
            f"Iteration {index}:{'  ' if index < 10 else ' '}rho = {value:.4f}"
            for index, value in enumerate(self._rho_trace)
        ]
        lines = ["", *_log, "", _banner, ""]
        lines += [
            # The header row's SS/MS end three columns EARLIER than the data rows' do (23/36/45
            # against 26/36/48): Stata centres the short labels over wider numeric fields.
            joined(
                f"{'Source':>12} |" + cell("SS", 23, 14) + cell("df", 36, 23) + cell("MS", 45, 36),
                "Number of obs",
                str(self._nobs),
            ),
            joined(rule, f"F({self._df_model}, {self._df_resid})", f"{self._fvalue:.2f}"),
            joined(
                source_row(
                    "Model",
                    num(self._mss, width=10),
                    str(self._df_model),
                    num(self._mss / max(self._df_model, 1), width=10),
                ),
                "Prob > F",
                f"{self._p_f:.4f}",
            ),
            joined(
                source_row(
                    "Residual",
                    num(self._rss, width=10),
                    str(self._df_resid),
                    num(self._rss / max(self._df_resid, 1), width=10),
                ),
                "R-squared",
                f"{self._r2:.4f}",
            ),
            joined(rule, "Adj R-squared", f"{self._r2_adj:.4f}"),
            joined(
                source_row(
                    "Total",
                    num(self._tss, width=10),
                    str(self._nobs - 1),
                    num(self._tss / max(self._nobs - 1, 1), width=10),
                ),
                "Root MSE",
                # Root MSE is FOUR decimals (.03232), unlike the five-digit terms elsewhere.
                f"{self._rmse:.5f}".lstrip("0") if 0 < self._rmse < 1 else f"{self._rmse:.5f}",
            ),
            "",
            "-" * width,
            # Header positions measured from Stata: label at 1, pipe at 13, then
            # Coefficient 15-25, Std. err. 28-36, t 43, P>|t| 48-52, interval 58-77.
            cell(self._depvar, 12, 0)
            + " |"
            # "Coefficient" starts at 15 (one space after the pipe at 13) and ends at 25.
            + cell("Coefficient", 26, 14)
            + cell("Std. err.", 36, 25)
            + cell("t", 43, 36)
            + cell("P>|t|", 52, 43)
            + cell("[95% conf. interval]", 77, 52),
            "-" * 13 + "+" + "-" * 64,
        ]
        # Coefficient block end columns, measured from Stata's row
        # '       price |  -1.048854    .759751    -1.38   0.179    -2.610545    .5128361':
        # coefficient 25, std. err. 36, t 43, P>|t| 52, interval 65 and 78.
        _COEF_END = (25, 36, 45, 53, 66, 78)
        # ``_cons`` PRINTS LAST, as it does in every Stata coefficient table. The design
        # matrix carries the constant first (patsy order), so iterating ``_names`` directly
        # put it at the TOP -- measured against `prais consumption temp price income` and its
        # `corc` form, whose e(b) column order is temp, price, income, _cons in both cases.
        _order = sorted(
            range(len(self._names)),
            key=lambda i: str(self._names[i]).lower()
            in {"intercept", "const", "constant", "_cons"},
        )
        for index in _order:
            name = self._names[index]
            low, high = self._conf_int[index]
            # The constant is "_cons" in Stata, whatever the formula called it.
            label = "_cons" if str(name).lower() in {"intercept", "const", "constant"} else str(name)
            line = f"{label:>12} |"
            line += cell(num(self._params[index]), _COEF_END[0], len(line))
            # Standard errors get EIGHT characters, not nine: '.0020458' is right while a
            # nine-character field produced '.00204577'.
            line += cell(num(self._bse[index]), _COEF_END[1], len(line))
            line += cell(f"{self._tvalues[index]:.2f}", _COEF_END[2], len(line))
            line += cell(f"{self._pvalues[index]:.3f}", _COEF_END[3], len(line))
            # The lower bound has NINE characters and the upper only EIGHT before the row ends:
            # '.5128361' at width 9 would round to '.51283609' and overflow column 78.
            line += cell(num(low), _COEF_END[4], len(line))
            line += cell(num(high), _COEF_END[5], len(line))
            lines.append(line)
        lines.append("-" * 13 + "+" + "-" * 64)
        lines.append(f"{'rho':>12} |" + cell(num(self._rho), 25, 14))
        lines.append("-" * width)
        lines.append(f"Durbin\u2013Watson statistic (original)    = {self._dw_original:.6f}")
        lines.append(
            f"Durbin\u2013Watson statistic (transformed) = {self._dw_transformed:.6f}"
        )
        return "\n".join(lines)


def prais(
    frame: pd.DataFrame,
    formula: str,
    *,
    maxiter: int = 100,
    tolerance: float = 1e-6,
    corc: bool = False,
) -> PraisResult:
    """Fit a Prais-Winsten AR(1) regression, as Stata's ``prais``.

    ``formula`` is a statsmodels/patsy style spec such as ``"y ~ x1 + x2"``. The data must
    already be in time order through ``frame``'s row order.

    ``corc=True`` is Stata's ``prais ..., corc``: the Cochrane-Orcutt transform, which drops
    the first observation instead of scaling it by ``sqrt(1 - rho^2)``. The two differ in
    both the coefficients and the estimated rho, so it is a separate estimator, not a
    cosmetic switch.
    """

    from westata.formula.factor import build_design

    design, names, kept_index = build_design(formula, frame)
    lhs = formula.split("~", 1)[0].strip()
    if lhs not in frame.columns:
        raise ConsistencyError(
            f"prais: the response {lhs!r} is not a column of the data"
        )
    y = frame.loc[kept_index, lhs].to_numpy(dtype=float)
    x = np.asarray(design, dtype=float)

    coefficients, rho, iterations, rho_trace = prais_winsten(
        x, y, maxiter=maxiter, tolerance=tolerance, corc=corc
    )

    resid = y - x @ coefficients
    # ``corc`` fits on N-1 rows (the transformed first observation is dropped), so its residual
    # degrees of freedom are one lower. The Prais-Winsten variant keeps all N.
    df_resid = (y.size - 1 if corc else y.size) - x.shape[1]
    # Stata reports R-squared, Adjusted R-squared and Root MSE on the TRANSFORMED scale,
    # because the estimated model is the one fitted to the rho-transformed data. Reporting
    # the original-scale sums of squares gave R2 0.521716 and RMSE 0.048053 against Stata's
    # 0.6234 and .03232.
    # THE TRANSFORM MUST MATCH THE ESTIMATOR. `prais, corc` drops the transformed first
    # observation instead of scaling it by sqrt(1-rho^2), so its residual variance -- and hence
    # every standard error -- is computed on N-1 rows with a DIFFERENT first value. Using the
    # Prais-Winsten transform here for both variants was the whole of the 4.7% error: on icecream
    # with `consumption ~ temp + price + income, corc` the coefficients and rho already matched
    # Stata exactly (.4009256) while se(price) printed .8491466 against Stata's .8108501.
    _final_transform = _transform_corc if corc else _transform
    transformed_x = np.column_stack(
        [_final_transform(x[:, j], rho) for j in range(x.shape[1])]
    )
    transformed_y = _final_transform(y, rho)
    trans_resid = transformed_y - transformed_x @ coefficients
    rss = float(trans_resid @ trans_resid)
    tss = float(np.sum((transformed_y - transformed_y.mean()) ** 2))
    mss = tss - rss
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    # ``corc`` FITS ON N-1 OBSERVATIONS, so its total degrees of freedom are N-2, not N-1.
    # Measured on icecream with `prais consumption temp price income, corc`:
    #
    #     Stata    Total df 28   Total MS .002589018   Adj R-squared 0.6068
    #     package  Total df 29   Total MS .002499741   Adj R-squared 0.5927
    #
    # RSS, MSS and R-squared already matched; only the degrees-of-freedom basis was wrong.
    _n_rows = y.size - 1 if corc else y.size
    _df_total = _n_rows - 1
    r2_adj = (
        1.0 - (1.0 - r2) * _df_total / df_resid if df_resid > 0 else 0.0
    )
    rmse = float(np.sqrt(rss / df_resid)) if df_resid > 0 else float("nan")

    sigma2 = rss / df_resid if df_resid > 0 else float("nan")
    try:
        bread = np.linalg.inv(transformed_x.T @ transformed_x)
    except np.linalg.LinAlgError:
        bread = np.linalg.pinv(transformed_x.T @ transformed_x)
    covariance = sigma2 * bread
    bse = np.sqrt(np.diag(covariance))
    with np.errstate(divide="ignore", invalid="ignore"):
        tvalues = coefficients / bse
    pvalues = 2.0 * stats.t.sf(np.abs(tvalues), df_resid)
    critical = stats.t.ppf(0.975, df_resid)
    conf = np.column_stack(
        [coefficients - critical * bse, coefficients + critical * bse]
    )

    fvalue = (mss / (x.shape[1] - 1)) / (rss / df_resid) if rss > 0 and df_resid > 0 and x.shape[1] > 1 else float("nan")
    p_f = float(stats.f.sf(fvalue, x.shape[1] - 1, df_resid)) if np.isfinite(fvalue) else float("nan")

    # Stata's "original" Durbin-Watson is computed from the residuals of the ORDINARY
    # (untransformed) regression of y on X, not from the Prais-Winsten residual of the
    # transformed fit. Using the latter gave 0.482716 against Stata's 1.021169.
    ols_resid = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    ols_rss = float(ols_resid @ ols_resid)
    dw_original = (
        float(np.sum(np.diff(ols_resid) ** 2) / ols_rss) if ols_rss > 0 else float("nan")
    )
    dw_transformed = (
        float(np.sum(np.diff(trans_resid) ** 2) / float(trans_resid @ trans_resid))
        if float(trans_resid @ trans_resid) > 0
        else float("nan")
    )

    return PraisResult(
        params=coefficients,
        names=list(names),
        depvar=str(lhs),
        bse=bse,
        tvalues=tvalues,
        pvalues=pvalues,
        conf_int=conf,
        rho=rho,
        # The reported observation count is the one the fit used: `corc` drops the
        # transformed first row, so Stata prints 29 where the Prais-Winsten variant prints 30.
        nobs=int(_n_rows),
        df_total=int(_df_total),
        df_resid=int(df_resid),
        df_model=int(x.shape[1] - 1),
        r2=r2,
        r2_adj=r2_adj,
        rmse=rmse,
        fvalue=fvalue,
        p_f=p_f,
        mss=mss,
        rss=rss,
        tss=tss,
        dw_original=dw_original,
        dw_transformed=dw_transformed,
        corc=bool(corc),
        iterations=iterations,
        rho_trace=[value for _index, value in rho_trace],
    )
