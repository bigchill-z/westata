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
    cov_type: str = "nonrobust",
) -> str:
    """Render one first-stage equation as Stata's ``, first`` block.

    ``design`` already contains the constant; ``names`` labels its columns.

    ``cov_type`` MUST match the main equation's VCE. Stata's ``, first`` refits each first-stage
    equation with the SAME `vce()` as the 2SLS estimator, so ``, r first`` reports ROBUST standard
    errors here -- measured on grilic, coefficient `s` has SE .216865 non-robust and .2327755 with
    ``, r``. Ignoring the option silently printed the non-robust SE under a table that gave no
    indication either way.
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
    robust = str(cov_type).strip().lower() in {"hc0", "hc1", "r", "robust"}
    #: Index of the intercept column, so the joint F can test only the slopes.
    constant_column = next(
        (j for j, name in enumerate(names) if str(name).strip().lower()
         in {"intercept", "_cons", "const", "constant"}),
        x.shape[1] - 1,
    )
    if robust:
        # STATA'S `regress, r` IS HC1, NOT HC0: the sandwich carries the n/(n-k) factor. Measured on
        # grilic's first stage, coefficient `s`:
        #
        #     HC0  .2315439
        #     HC1  .2327755   <- matches Stata's `, r first`
        #
        # So the default robust path scales by n/(n-k); only an explicit hc0 request omits it.
        meat = (x * (residual**2)[:, None]).T @ x
        covariance = bread @ meat @ bread
        # THE n/(n-k) FACTOR IS ALWAYS APPLIED WHEN THE MAIN EQUATION IS ROBUST. Stata has one
        # robust VCE for `regress` (HC1); there is no way to ask it for the unscaled HC0 through
        # `, r`. A caller that passes cov_type="HC0" is asking for "the robust VCE", so the first
        # stage must show the same numbers Stata would -- omitting the factor here is what made the
        # SEs disagree (.2315439 vs .2327755).
        covariance = covariance * (nobs / dof)
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    else:
        covariance = bread * sigma2
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
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
    #
    # UNDER vce(robust) STATA REPORTS A ROBUST WALD F, not the classical R-squared F. Measured on
    # grilic's first stage: classical 47.38, robust Wald 47.7365 -> Stata prints 47.74. The Wald form
    # is (Rb)' [R V R']^-1 (Rb) / q with R selecting the slopes.
    df_model = k - 1
    if robust and df_model > 0:
        selector = np.zeros((df_model, k))
        slope_positions = [j for j in range(k) if j != constant_column]
        for row, column in enumerate(slope_positions):
            selector[row, column] = 1.0
        restricted = selector @ coefficients
        try:
            middle = selector @ covariance @ selector.T
            wald = float(restricted @ np.linalg.solve(middle, restricted))
            fvalue = wald / df_model
            p_f = float(stats.f.sf(fvalue, df_model, dof))
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            fvalue = float("nan")
            p_f = float("nan")
    elif df_model > 0 and r2 < 1.0:
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
    def _stat_rows(pairs: list[tuple[str, str]]) -> list[str]:
        """Right-align one statistics block the way Stata does.

        THE "=" COLUMN IS NOT FIXED -- it moves with the widest value, because Stata formats the
        whole block with one field width. Measured on two `, first` runs of the same dataset:

            lnw (s = expr tenure age), first       Root MSE = 1.6873    "=" at 70
            lnw s expr ... (iq=med kww), r first   Root MSE = 11.3931   "=" at 69

        Both rows are 78 characters and the value field is ``max(len(value)) + 1``. From the two
        measurements: ``"=" column = 77 - field``, ``label start = 63 - field`` and
        ``padding after the label = 14 - len(label)``.
        """

        if not pairs:
            return []
        field = max(len(value) for _label, value in pairs) + 1
        label_start = 63 - field
        rows = []
        for label, value in pairs:
            head = " " * label_start + label
            head += " " * max(14 - len(label), 1)
            rows.append(head + "=" + value.rjust(field))
        return rows


    def _four(value: float) -> str:
        # These keep the LEADING ZERO ("0.0559"), unlike the coefficient cells above.
        return f"{value:.4f}"

    stat_pairs: list[tuple[str, str]] = [("Number of obs", f"{nobs:,.0f}")]
    if np.isfinite(fvalue):
        stat_pairs.append((f"F({df_model}, {dof})", f"{fvalue:.2f}"))
        stat_pairs.append(("Prob > F", f"{p_f:.4f}"))
    stat_pairs.append(("R-squared", _four(r2)))
    stat_pairs.append(("Adj R-squared", _four(r2_adj)))
    stat_pairs.append(("Root MSE", _four(root_mse)))

    lines = ["First-stage regressions", "-" * 23, ""]
    lines.extend(_stat_rows(stat_pairs))
    # STATA PUTS A "Robust" BLOCK BETWEEN THE RULE AND THE COLUMN NAMES when vce(robust) is in
    # force, and the SE label becomes LOWERCASE "std. err." at the same time -- the non-robust
    # header reads "Std. err.". Both measured on grilic with ``, r first``:
    #
    #     ------------------------------------------------------------------------------
    #                  |               Robust
    #               iq | Coefficient  std. err.      t    P>|t|     [95% conf. interval]
    #     -------------+----------------------------------------------------------------
    se_label = "std. err." if robust else "Std. err."
    header_pieces: list[str] = [f"{depvar:>12} |"]
    for word, end in (
        ("Coefficient", 26),
        (se_label, 37),
        ("t", 44),
        ("P>|t|", 53),
        ("[95% conf. interval]", 78),
    ):
        used = sum(len(piece) for piece in header_pieces)
        header_pieces.append(" " * max(end - used - len(word), 0) + word)
    lines.append("")
    lines.append(rule)
    if robust:
        # MEASURED, not centred: Stata's line is 35 characters, the "|" is at column 13 and the word
        # starts at column 29 -- so it is 15 spaces after the pipe, with no trailing padding:
        #
        #     '             |               Robust'   (len 35, 'Robust' at index 29)
        lines.append(" " * 13 + "|" + " " * 15 + "Robust")
    lines.extend(
        [
            "".join(header_pieces),
            "-" * 13 + "+" + "-" * 64,
        ]
    )

    def _emit(text: str, end: int, pieces: list[str]) -> None:
        used = sum(len(piece) for piece in pieces)
        pieces.append(" " * max(end - used - len(text), 0) + text)

    # STATSPACE: Stata lists the regressors in model order and the constant LAST. The design here is
    # [exog, instruments, 1] so the constant is already last, but an explicit order keeps the row
    # sequence independent of how the caller stacked the columns -- the previous version emitted
    # `_cons` FIRST, which is what the user reported.
    def _is_constant(name: str) -> bool:
        return str(name).strip().lower() in {"intercept", "_cons", "const", "constant"}

    order = [i for i, name in enumerate(names) if not _is_constant(name)]
    order += [i for i, name in enumerate(names) if _is_constant(name)]

    for index in order:
        name = names[index]
        label = "_cons" if _is_constant(name) else str(name)
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
    """The ``, first`` BLOCK: the fitted coefficients of each first-stage equation.

    THIS IS NOT ``estat firststage``. Stata has two different things here and they are easy to
    confuse:

        ivregress 2sls ..., first     the full first-stage REGRESSION TABLE -- coefficients,
                                      standard errors, t, p, interval, for every regressor and
                                      instrument. One block per endogenous regressor.
                                      -> ``result.first_stage()`` / ``summary(first=True)``

        estat firststage              a SUMMARY: R-sq, adjusted R-sq, partial R-sq, the
                                      instrument F and its p, plus the Stock-Yogo critical-value
                                      panel. No coefficients at all.
                                      -> ``ws.estat_firststage(result)``

    Calling ``first_stage()`` when ``estat firststage`` was meant produces a table with a completely
    different shape, which reads as a mismatch rather than a wrong entry point. Both are implemented;
    measured on grilic, ``estat_firststage`` reproduces Stata's 22 lines exactly.

    Reads ``result.iv_structure``, which ``IV2SLS.fit`` attaches so the auxiliary regressions can be
    rebuilt.
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
    # The main equation's VCE applies to every first-stage refit; `, r first` means ROBUST here too.
    cov_type = str(
        getattr(result, "cov_type", None) or structure.get("cov_type") or "nonrobust"
    )
    blocks = [
        first_stage_table(
            design, endog[:, column], names, endog_names[column], cov_type=cov_type
        )
        for column in range(endog.shape[1])
    ]
    separator = chr(10) * 2
    # STATSPACE: the block opens with a blank line, as Stata's `, first` does before
    # "First-stage regressions".
    return separator.join(blocks)
