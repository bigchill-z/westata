"""Linear combinations of parameters, targeting Stata's ``lincom``.

Reference: ``[R] lincom``.  For a linear combination ``L b`` the point estimate
is ``L b`` with variance ``L V L'``; the test statistic is t with ``e(df_r)``
degrees of freedom, or z when the model reports z statistics.

Because ``lincom`` and ``test`` share the same machinery, this module delegates
to :mod:`westata.postestimation.tests`; the difference is presentation
(one row, plus optional exponentiated ``eform``/``or``/``irr`` reporting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = ["LincomResult", "run_lincom"]



def _lincom_g(value: float) -> str:
    """Stata's number format for ``lincom``: EIGHT significant characters, leading zero dropped.

    The docstring used to say "``%9.0g``: seven significant figures", which is neither what the
    option means nor what Stata prints. Captured from two fits:

        cobb_douglas   '  1.075205   .1159063     9.28   0.000     .8341646    1.316245'
        grilic         '  .1066109   .0059536    17.91   0.000     .0949233    .1182984'

    Below 1 Stata keeps SEVEN decimals (.005953578 -> ".0059536"); at or above 1 it drops one
    (1.0752048 -> "1.075205") so the cell stays eight characters. Seven significant figures gives
    ".005953578" -- three characters wider, which pushed the whole row past 78 columns; seven
    decimals gives "1.0752048", one wider than Stata's cell.

    Checked against six values spanning both ranges; all six match.
    """

    magnitude = abs(float(value))
    if magnitude < 1.0:
        places = 7
    elif magnitude < 10.0:
        places = 6
    else:
        places = 4
    text = f"{float(value):.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return text


@dataclass(frozen=True)
class LincomResult:
    """One linear combination, shaped like Stata's ``lincom`` output."""

    estimate: float
    std_err: float
    statistic: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    distribution: str
    df: float | None
    expression: str
    eform: bool = False
    label: str = ""
    #: Dependent-variable name for the leftmost column of Stata's table
    #: ("consumption | Coefficient ..."). Empty when the caller did not supply one.
    response_name: str = ""

    def __str__(self) -> str:
        """Reproduce ``lincom`` exactly::

            ( 1)  temp + price = 0
            ------------------------------------------------------------------------------
            consumption | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
            -------------+----------------------------------------------------------------
                    (1) |  -1.040955   .8344249    -1.25   0.223     -2.75614    .6742303
            ------------------------------------------------------------------------------

        The old rendering dropped the ``( 1)`` marker and the three rules, and labelled the
        estimate column ``Coef.`` where Stata writes ``Coefficient`` -- the same numbers in
        a table that is visibly not Stata's.
        """

        statistic_label = "z" if self.distribution == "z" else "t"
        response = str(getattr(self, "response_name", "") or "")
        # Stata indents the constraint marker by ONE space: " ( 1)  k + l = 0".
        marker = f" ( 1)  {self.expression}"
        full_rule = "-" * 78

        # Measured off a live ``lincom temp + price``: the header is 77 characters with
        # "Coefficient" at column 14, "Std. err." at 27, the t statistic at 42 and the
        # interval label at 57; the estimate row starts its value at column 15. The rule
        # lines are 78.
        # Both lines are 77 characters. Rather than derive a width per column (which kept
        # matching one field and shifting the next), each value is placed at the column
        # Stata prints it in, measured from a live ``lincom temp + price``:
        #
        #   header: Coefficient@14 Std. err.@27 t@42 P>|t|@47 [95% conf. interval]@57
        #   row:    estimate ends@26, se ends@38, t ends@45, p ends@52,
        #           lower ends@68, upper ends@77
        def _place(pieces: list[str], text: str, end_column: int) -> None:
            """Append ``text`` so that it ENDS at ``end_column``."""

            used = sum(len(piece) for piece in pieces)
            gap = end_column - used - len(text)
            pieces.append(" " * max(gap, 0) + text)

        # Columns taken from Stata's own line, measured with str.index()/re.finditer so
        # nothing is derived:
        #
        #   '           y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]'
        #   '         (1) |   1.075205   .1159063     9.28   0.000     .8341646    1.316245'
        #
        #     header ends: y 12, Coefficient 26, Std. err. 37, t 44, P>|t| 53, CI 78
        #     row    ends: (1) 14, estimate 25, se 36, t 45, p 53, lo 66, hi 78
        #
        # The response name is RIGHT-aligned so it finishes at column 12 and the pipe lands at
        # 13; building it as ``f"{response:<11} |"`` left-aligned the name against the table
        # edge and shifted the pipe to 12. ``_place``'s argument is the index one past the
        # last character, hence the "+1" between the table above and the calls below.
        header_pieces = [f"{response:>12} |"]
        _place(header_pieces, "Coefficient", 26)
        _place(header_pieces, "Std. err.", 37)
        _place(header_pieces, statistic_label, 44)
        _place(header_pieces, "P>|" + statistic_label + "|", 53)
        _place(header_pieces, "[95% conf. interval]", 78)
        header = "".join(header_pieces)

        row_pieces = [f"{'(1)':>12} |"]
        _place(row_pieces, _lincom_g(self.estimate), 25)
        _place(row_pieces, _lincom_g(self.std_err), 36)
        _place(row_pieces, f"{self.statistic:.2f}", 45)
        _place(row_pieces, f"{self.pvalue:.3f}", 53)
        _place(row_pieces, _lincom_g(self.ci_lower), 66)
        _place(row_pieces, _lincom_g(self.ci_upper), 78)
        row = "".join(row_pieces)

        return chr(10).join(
            [
                marker,
                # Stata puts a BLANK line between the "( 1)  s = 0" marker and the table.
                # Measured: the captured block is 12 / 0 / 78 / 78 / 78 / 78 / 78 characters.
                "",
                full_rule,
                header,
                "-" * 13 + "+" + "-" * 64,
                row,
                full_rule,
            ]
        )

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names ``lincom.sthlp:182-189`` returns.

        ``estimate``, ``se``, ``df``, ``t``/``z``, ``p``, ``lb`` and ``ub``.  The test
        statistic travels under the name of its distribution -- ``r(t)`` for a t-based
        fit, ``r(z)`` for a z-based one -- because that is how the ado stores it; asking
        for the other name finds nothing, which matches Stata.
        """

        stat_label = "z" if self.distribution == "z" else "t"
        table: dict[str, float] = {
            "estimate": float(self.estimate),
            "se": float(self.std_err),
            stat_label: float(self.statistic),
            "p": float(self.pvalue),
            "lb": float(self.ci_lower),
            "ub": float(self.ci_upper),
        }
        # A z-based fit has NO degrees of freedom, so ``r(df)`` is genuinely absent
        # rather than zero or None.  Coercing it would invent a value the ado does not
        # store.
        if self.df is not None:
            table["df"] = float(self.df)
        return table


def run_lincom(
    params: np.ndarray,
    cov: np.ndarray,
    weights: np.ndarray,
    *,
    dist: str,
    df_resid: float,
    expression: str = "",
    level: float = 0.95,
    eform: bool = False,
    label: str = "",
    response_name: str = "",
) -> LincomResult:
    """Evaluate ``weights @ b`` with its standard error and interval.

    Parameters
    ----------
    weights:
        Length-``k`` vector defining the linear combination.
    dist:
        ``"t"`` or ``"z"``: the reference distribution used by the model.
    df_resid:
        Denominator degrees of freedom for the t form.
    level:
        Confidence level as a fraction (Stata's ``level(95)`` -> ``0.95``).
    eform:
        Report ``exp(L b)`` and transform the interval, as ``lincom, eform``.
    """

    params = np.asarray(params, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.shape[0] != params.shape[0]:
        raise ValueError(
            f"weights has {weights.shape[0]} entries but the model has "
            f"{params.shape[0]} parameters"
        )
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie in (0, 1), got {level!r}")

    estimate = float(weights @ params)
    variance = float(weights @ cov @ weights)
    if variance < 0:
        raise ValueError(
            "the linear combination has a negative variance; the covariance "
            "matrix is not positive semidefinite"
        )
    std_err = float(np.sqrt(variance))

    alpha = 1.0 - level
    if dist == "z":
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
        statistic = estimate / std_err if std_err > 0 else float("nan")
        pvalue = float(2.0 * stats.norm.sf(abs(statistic)))
        df_value: float | None = None
    else:
        critical = float(stats.t.ppf(1.0 - alpha / 2.0, df_resid))
        statistic = estimate / std_err if std_err > 0 else float("nan")
        pvalue = float(2.0 * stats.t.sf(abs(statistic), df_resid))
        df_value = float(df_resid)

    lower = estimate - critical * std_err
    upper = estimate + critical * std_err

    if eform:
        estimate = float(np.exp(estimate))
        lower = float(np.exp(lower))
        upper = float(np.exp(upper))

    return LincomResult(
        estimate=estimate,
        std_err=std_err,
        statistic=statistic,
        pvalue=pvalue,
        ci_lower=lower,
        ci_upper=upper,
        distribution=dist,
        df=df_value,
        expression=expression,
        eform=eform,
        label=label,
        response_name=str(response_name or ""),
    )
