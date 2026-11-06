"""Stata's ``ivregress ..., first``: the first-stage regression table.

Report case C10-01 said the first-stage regression TABLE is missing. It was: the package had
``estat firststage`` (the diagnostic summary) but nothing that printed the first-stage
equation's coefficients the way ``, first`` does.

Reference, ``ivregress 2sls lnw s expr (tenure = rns smsa), first`` on grilic (n = 758)::

    First-stage regressions
    -----------------------

                                                            Number of obs =    758
                                                            F(4, 753)     =  11.15
                                                            Prob > F      = 0.0000
                                                            R-squared     = 0.0559
                                                            Adj R-squared = 0.0509
                                                            Root MSE      = 1.6305

    ------------------------------------------------------------------------------
          tenure | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
    -------------+----------------------------------------------------------------
               s |   .0006668   .0275293     0.02   0.981    -.0533765    .0547102
            expr |   .1846684   .0290088     6.37   0.000     .1277207    .2416161
             rns |  -.1210981   .1354721    -0.89   0.372    -.3870459    .1448497
            smsa |   .1304456   .1320924     0.99   0.324    -.1288676    .3897587
           _cons |   1.442411   .3970731     3.63   0.000     .6629091    2.221913
    ------------------------------------------------------------------------------

The dependent variable of each first-stage equation is the ENDOGENOUS regressor being
instrumented, regressed on the included exogenous variables plus ALL the excluded instruments.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy import stats


def _stata_number(value: float, width: int = 9) -> str:
    """Stata's ``%9.0g``, delegated to the one implementation already verified against it.

    ``westata.results.summary._stata_number`` was pinned earlier against a live
    ``display %9.0g``: it picks the widest representation that fits the field rather than a
    fixed significant-figure count, which is why ``.0006668274`` prints as ``.0006668``
    while ``.1236468764`` prints as ``.1236469``. Re-deriving that here produced
    ``.0006668274`` -- ten characters that overflow the column.
    """

    from westata.results.summary import _stata_number as _formatter

    return _formatter(value, width)


def _row(label: str, right: str, left: str = "") -> str:
    """One ``right = value`` line of the statistics block."""

    label = f"{right} ="
    return f"{left:>56}{label:>15}{value:>12}"


def first_stage_table(
    design: np.ndarray,
    response: np.ndarray,
    names: Sequence[str],
    depvar: str,
) -> str:
    """Render one first-stage equation as Stata's ``, first`` block.

    ``design`` already contains the constant; ``names`` labels its columns.
    """

    x = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float).reshape(-1)
    names = [str(name) for name in names]
    nobs, k = x.shape

    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coefficients
    dof = nobs - k
    sigma2 = float(residual @ residual) / dof
    try:
        bread = np.linalg.inv(x.T @ x)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        bread = np.linalg.pinv(x.T @ x)
    standard_errors = np.sqrt(np.maximum(np.diag(bread) * sigma2, 0.0))
    tvalues = np.divide(
        coefficients, standard_errors, out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    pvalues = np.array([2.0 * stats.t.sf(abs(t), dof) for t in tvalues])
    critical = float(stats.t.ppf(0.975, dof))
    lower = coefficients - critical * standard_errors
    upper = coefficients + critical * standard_errors

    total_ss = float(((y - y.mean()) ** 2).sum())
    rss = float(residual @ residual)
    r2 = 1.0 - rss / total_ss if total_ss > 0 else float("nan")
    r2_adj = 1.0 - (1.0 - r2) * (nobs - 1) / dof if dof > 0 else float("nan")
    root_mse = float(np.sqrt(sigma2))

    # Overall F: all slopes zero. df_num = k - 1 (the constant is not tested).
    df_model = k - 1
    if df_model > 0 and r2 < 1.0:
        fvalue = (r2 / df_model) / ((1.0 - r2) / dof)
        p_f = float(stats.f.sf(fvalue, df_model, dof))
    else:  # pragma: no cover - degenerate designs
        fvalue = float("nan")
        p_f = float("nan")

    # Column ends measured with re.finditer on Stata's own rows, not derived:
    #   header: "|" at 13, Coefficient ends 26, Std. err. 37, t 45, P>|t| 53, CI 78
    #   row   : estimate ends 25, se 36, t 45, p 53, lo 66, hi 78
    #
    # The statistics block puts the LABEL at column 56 and the value right after " = ",
    # which is a different alignment from the coefficient table -- and R-squared/Adj
    # R-squared print to FOUR decimals there, unlike the %9.0g used in the table body.
    rule = "-" * 78

    # "=" sits at column 70 on every one of these lines; the value is right-aligned in
    # the 8 columns after it (78 total). Measured with str.index on Stata's own output.
    def _stat(label: str, value: str) -> str:
        # The label STARTS at column 56 for every row (measured with str.index), and "="
        # sits at column 70, so the padding between them depends on the label's length.
        return " " * 56 + label + " " * (70 - 56 - len(label)) + "=" + f"{value:>7}"

    lines = [
        "First-stage regressions",
        "-" * 23,
        "",
        _stat("Number of obs", f"{nobs:,.0f}"),
    ]
    if np.isfinite(fvalue):
        lines.append(_stat(f"F({df_model}, {dof})", f"{fvalue:.2f}"))
        lines.append(_stat("Prob > F", f"{p_f:.4f}"))
    # Four decimals with the leading zero dropped, as Stata prints them: .0559 / 1.6305.
    def _four(value: float) -> str:
        # These keep the LEADING ZERO ("0.0559"), unlike the coefficient cells above.
        return f"{value:.4f}"

    lines.append(_stat("R-squared", _four(r2)))
    lines.append(_stat("Adj R-squared", _four(r2_adj)))
    lines.append(_stat("Root MSE", _four(root_mse)))
    header_pieces: list[str] = [f"{depvar:>12} |"]
    for word, end in (
        ("Coefficient", 26),
        ("Std. err.", 37),
        ("t", 44),
        ("P>|t|", 53),
        ("[95% conf. interval]", 78),
    ):
        used = sum(len(piece) for piece in header_pieces)
        header_pieces.append(" " * max(end - used - len(word), 0) + word)
    lines.extend(
        [
            "",
            rule,
            "".join(header_pieces),
            "-" * 13 + "+" + "-" * 64,
        ]
    )

    def _emit(text: str, end: int, pieces: list[str]) -> None:
        used = sum(len(piece) for piece in pieces)
        pieces.append(" " * max(end - used - len(text), 0) + text)

    for index, name in enumerate(names):
        label = "_cons" if str(name) in ("Intercept", "_cons", "const") else str(name)
        pieces: list[str] = [f"{label:>12} |"]
        _emit(_stata_number(coefficients[index], 9).strip(), 25, pieces)
        _emit(_stata_number(standard_errors[index]).strip(), 36, pieces)
        _emit(f"{tvalues[index]:.2f}", 45, pieces)
        _emit(f"{pvalues[index]:.3f}", 53, pieces)
        _emit(_stata_number(lower[index]).strip(), 66, pieces)
        _emit(_stata_number(upper[index]).strip(), 78, pieces)
        lines.append("".join(pieces))
    lines.append(rule)
    return "\n".join(lines)


def render_first_stage(result: Any) -> str:
    """First-stage table(s) for an IV result, one block per endogenous regressor.

    Reads ``result.iv_structure``, which ``IV2SLS.fit`` attaches precisely so the
    auxiliary regressions can be rebuilt (``, first`` and ``estat firststage`` need the
    same pieces).
    """

    structure = getattr(result, "iv_structure", None)
    if not structure:
        raise AttributeError(
            "first_stage: this result carries no iv_structure; it was not produced by an "
            "IV estimator"
        )
    exog = np.asarray(structure["exog"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    instruments = np.asarray(structure["instruments"], dtype=float)
    exog_names = [str(name) for name in structure["exog_names"]]
    instr_names = [str(name) for name in structure["instrument_names"]]
    endog_names = [str(name) for name in structure["endog_names"]]
    design = np.column_stack([exog, instruments])
    names = exog_names + instr_names
    blocks = [
        first_stage_table(design, endog[:, column], names, endog_names[column])
        for column in range(endog.shape[1])
    ]
    separator = chr(10) * 2
    return separator.join(blocks)
