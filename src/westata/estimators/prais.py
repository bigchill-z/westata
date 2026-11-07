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
) -> tuple[np.ndarray, float, int]:
    """Iterated Prais-Winsten estimate.

    Returns ``(coefficients, rho, iterations)``. ``rho`` may be given to run a single
    Cochrane-Orcutt style pass instead of iterating.
    """

    x = np.asarray(exog, dtype=float)
    y = np.asarray(endog, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[0] != y.size:
        raise ConsistencyError(
            f"prais: design is {x.shape} but the response has {y.size} values"
        )

    def ols(design: np.ndarray, response: np.ndarray) -> np.ndarray:
        return np.linalg.lstsq(design, response, rcond=None)[0]

    coefficients = ols(x, y)
    current_rho = _rho_from_residuals(y - x @ coefficients) if rho is None else float(rho)

    iterations = 0
    for iterations in range(1, maxiter + 1):
        if abs(current_rho) >= 1.0:
            raise ConvergenceError(
                f"prais: the estimated AR(1) coefficient {current_rho:.6f} is not inside "
                "(-1, 1); the series is not stationary"
            )
        transformer = _transform_corc if corc else _transform
        transformed_x = np.column_stack(
            [transformer(x[:, j], current_rho) for j in range(x.shape[1])]
        )
        transformed_y = transformer(y, current_rho)
        updated = ols(transformed_x, transformed_y)
        residual = y - x @ updated
        next_rho = _rho_from_residuals(residual)
        if rho is not None or abs(next_rho - current_rho) < tolerance:
            # Stata reports the rho it stopped AT, but the coefficients refitted AT that rho --
            # not the ones from the previous pass. Measured on icecream.dta: stopping at
            # rho .8002264 with the previous coefficients gives price -1.0488550 and
            # _cons .5870043, while refitting once more gives -1.0488545 ... .5870049, which is
            # what Stata prints (-1.048854, .5870049).
            final_rho = next_rho if rho is None else current_rho
            if rho is None:
                transformer = _transform_corc if corc else _transform
                final_x = np.column_stack(
                    [transformer(x[:, j], final_rho) for j in range(x.shape[1])]
                )
                final_y = transformer(y, final_rho)
                coefficients = ols(final_x, final_y)
            else:
                coefficients = updated
            current_rho = final_rho
            break
        coefficients = updated
        current_rho = next_rho
    else:
        # Stata prints the value it reached BEFORE reporting r(430):
        #
        #   rho |   .9724118
        #   convergence not achieved
        #   r(430);
        #
        # so the estimate travels on the exception rather than being discarded. Without it a
        # caller cannot tell a rho of .97 that is still creeping from one that is oscillating
        # wildly, and has to re-run with a raised tolerance to find out.
        failure = ConvergenceError(
            f"prais: rho did not converge in {maxiter} iterations; last value "
            f"{current_rho:.7f}"
        )
        failure.rho = current_rho  # type: ignore[attr-defined]
        raise failure

    return coefficients, current_rho, iterations


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
        "_rmse",
        "_fvalue",
        "_p_f",
        "_dw_original",
        "_dw_transformed",
        "_iterations",
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

        lines = [
            "",
            "",
            "Prais\u2013Winsten AR(1) regression with iterated estimates",
            "",
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
        for index, name in enumerate(self._names):
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

    coefficients, rho, iterations = prais_winsten(
        x, y, maxiter=maxiter, tolerance=tolerance, corc=corc
    )

    resid = y - x @ coefficients
    df_resid = y.size - x.shape[1]
    # Stata reports R-squared, Adjusted R-squared and Root MSE on the TRANSFORMED scale,
    # because the estimated model is the one fitted to the rho-transformed data. Reporting
    # the original-scale sums of squares gave R2 0.521716 and RMSE 0.048053 against Stata's
    # 0.6234 and .03232.
    transformed_x = np.column_stack([_transform(x[:, j], rho) for j in range(x.shape[1])])
    transformed_y = _transform(y, rho)
    trans_resid = transformed_y - transformed_x @ coefficients
    rss = float(trans_resid @ trans_resid)
    tss = float(np.sum((transformed_y - transformed_y.mean()) ** 2))
    mss = tss - rss
    r2 = 1.0 - rss / tss if tss > 0 else 0.0
    r2_adj = 1.0 - (1.0 - r2) * (y.size - 1) / df_resid if df_resid > 0 else 0.0
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
        nobs=int(y.size),
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
        iterations=iterations,
    )
