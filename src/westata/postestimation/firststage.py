"""First-stage diagnostics for instrumental variables: Stata's ``estat firststage``.

Transcribed from ``ivregress_estat.ado`` (StataNow 19.5), program ``FirstStage``.
For each endogenous regressor the report columns are

    Variable | R-sq.  Adjusted R-sq.  Partial R-sq.  F(  #,  #)  Prob > F

built as follows (``ivregress_estat.ado:645-680``):

.. code-block:: stata

    * R-sq and adjusted R-sq of endog on ALL exogenous variables
    ivregress 2sls `endotmp' `allexog', ... small
    scalar `r2'  = e(r2)
    scalar `ar2' = e(r2_a)

    * partial F: joint test of the EXCLUDED instruments
    test `exclexog'
    scalar `F' = r(F) ; `Fn' = r(df) ; `Fd' = r(df_r) ; `p' = Ftail(...)

    * partial R-sq: residualise endog on the INCLUDED exog, residualise the
      excluded instruments on the included exog, then regress the two residuals
    _regress `endotmp' `inclexog'
    predict `Mxy', residuals
    _regress `Mxinst' `inclexog'   (per instrument, residuals saved as `Mxinst')
    _regress `Mxy' `Mxinst'
    partial R2 = e(r2)

Shea's partial R-squared (``ivregress_estat.ado:700-736``) removes the
correlation among the endogenous regressors before measuring the instruments'
explanatory power:

.. code-block:: stata

    * vhat_i = fitted value of endog_i on all exogenous variables
    * x1tilde = residual of endog_i on the OTHER endogenous + included exog
    * x1bar   = residual of vhat_i on the other vhats + included exog
    * Shea_R2_i = R2 of x1tilde on x1bar

with ``adj = 1 - ((N-1)/(N-nexog))*(1 - Shea_R2)``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.postestimation.stock_yogo import mineig_critical_values

__all__ = ["FirstStageResult", "estat_firststage"]

_CONSTANT_NAMES = frozenset({"intercept", "intercept_", "_cons", "const"})


def _is_constant(name: str) -> bool:
    return str(name).strip().lower() in _CONSTANT_NAMES


def _ols(y: np.ndarray, design: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS fitted values and R-squared (with an intercept added if absent)."""

    if design.size == 0 or design.shape[1] == 0:
        return np.zeros_like(y), float("nan")
    design = np.asarray(design, dtype=float)
    if not any(np.ptp(design[:, j]) == 0.0 for j in range(design.shape[1])):
        design = np.column_stack([np.ones(y.size), design])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    resid = y - fitted
    centred = y - y.mean()
    tss = float(centred @ centred)
    rss = float(resid @ resid)
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    return fitted, r2


def _residualise(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Residual of ``y`` on ``design`` (with an intercept).

    An EMPTY design still demeans.  ``_ols`` short-circuits to a zero fit when
    ``design.size == 0``, so without this the columns stayed uncentred whenever a
    model had no included exogenous regressors -- which shifted the partial
    R-squared and made the Cragg-Donald eigenvalue disagree with it at
    (1 endog, 1 instrument), where the two must coincide.
    """

    design = np.asarray(design, dtype=float)
    if design.size == 0 or design.shape[1] == 0:
        return np.asarray(y, dtype=float) - np.asarray(y, dtype=float).mean()
    fitted, _ = _ols(y, design)
    return y - fitted


class FirstStageResult:
    """One row per endogenous regressor, plus the headline F.

    Attributes
    ----------
    endog:
        Names of the endogenous regressors, in reported order.
    r2, adj_r2:
        R-squared of the endogenous regressor on **all** exogenous variables.
    partial_r2:
        R-squared of the residualised endogenous regressor on the residualised
        excluded instruments.
    f, df_num, df_denom, pvalue:
        The joint test of the excluded instruments in that first stage.
    shea, shea_adj:
        Shea's partial R-squared and its degrees-of-freedom adjustment.  Only
        computed when there is more than one endogenous regressor, because with
        one the two coincide with ``partial_r2``.
    """

    __slots__ = (
        "endog",
        "r2",
        "adj_r2",
        "partial_r2",
        "f",
        "df_num",
        "df_denom",
        "pvalue",
        "shea",
        "shea_adj",
        "mineig",
        "mineigcv",
        "show_all",
        "force_nonrobust",
        "robust_vce",
    )

    def __init__(
        self,
        *,
        endog: Sequence[str],
        r2: Sequence[float],
        adj_r2: Sequence[float],
        partial_r2: Sequence[float],
        f: Sequence[float],
        df_num: Sequence[float],
        df_denom: Sequence[float],
        pvalue: Sequence[float],
        shea: Sequence[float] | None = None,
        shea_adj: Sequence[float] | None = None,
        mineig: float | None = None,
        mineigcv: Any = None,
        show_all: bool = False,
        force_nonrobust: bool = False,
        robust_vce: bool = False,
    ) -> None:
        self.endog = list(endog)
        self.r2 = np.asarray(r2, dtype=float)
        self.adj_r2 = np.asarray(adj_r2, dtype=float)
        self.partial_r2 = np.asarray(partial_r2, dtype=float)
        self.f = np.asarray(f, dtype=float)
        self.df_num = np.asarray(df_num, dtype=float)
        self.df_denom = np.asarray(df_denom, dtype=float)
        self.pvalue = np.asarray(pvalue, dtype=float)
        self.shea = None if shea is None else np.asarray(shea, dtype=float)
        self.shea_adj = None if shea_adj is None else np.asarray(shea_adj, dtype=float)
        #: ``r(mineig)``: the Cragg-Donald minimum eigenvalue.  ``None`` when the
        #: blocks are singular and no eigenvalue is available.
        self.mineig = None if mineig is None else float(mineig)
        #: ``r(mineigcv)``: the 3x4 Stock-Yogo critical-value matrix.  Row 0 is
        #: relative bias (5/10/20/30%), rows 1 and 2 the TSLS and LIML worst-case
        #: size (10/15/20/25%).  Entries Stata leaves as ``.`` are ``nan``.
        self.mineigcv = None if mineigcv is None else np.asarray(mineigcv, dtype=float)
        #: Stata's ``all`` option (``ivregress_postestimation.sthlp:408-412``): report the
        #: R-squared block even when there is more than one endogenous regressor.
        self.show_all = bool(show_all)
        #: Stata's ``forcenonrobust`` option (``sthlp:416-422``): report the minimum
        #: eigenvalue statistic and its critical values even though a robust VCE was used
        #: at estimation time.  Without it those numbers are SUPPRESSED under a robust
        #: VCE, because the critical values assume i.i.d. normal errors.
        self.force_nonrobust = bool(force_nonrobust)
        #: Whether the fit used a robust/clustered VCE -- the condition the option exists
        #: to override.
        self.robust_vce = bool(robust_vce)

    def __repr__(self) -> str:
        return (
            f"<FirstStageResult {len(self.endog)} endogenous, "
            f"min partial R2 {np.nanmin(self.partial_r2):.4f}>"
        )

    @property
    def reports_mineig(self) -> bool:
        """Whether the minimum-eigenvalue statistic should be reported.

        ``ivregress_postestimation.sthlp:416-422``: under a robust VCE Stata suppresses
        it, because the Stock-Yogo critical values assume i.i.d. normal errors; the
        ``forcenonrobust`` option is the explicit override.  Exposed as a property so a
        caller can tell "not available" from "not applicable", which a bare ``None`` in
        ``mineig`` cannot distinguish.
        """

        return (not self.robust_vce) or self.force_nonrobust

    def __str__(self) -> str:
        # ``ivregress_postestimation.sthlp:408-414``: "By default, if the model contains
        # one endogenous regressor, then the first-stage R-squared, adjusted R-squared,
        # partial R-squared, and F statistics are reported, whereas if the model contains
        # multiple endogenous regressors, then Shea's partial R-squared and adjusted
        # partial R-squared are reported instead."
        #
        # So the DEFAULT table depends on how many endogenous regressors there are.  Our
        # version always drew the R-squared block, which meant a multi-endogenous model
        # showed a table Stata does not print -- and the Shea statistics that Stata DOES
        # print were computed but never rendered.
        #
        # ``all`` (``sthlp:408-412``) asks for the R-squared block "regardless of whether
        # the model contains one or more endogenous regressors", so it wins.
        if self.show_all:
            return self._render_r2_table() + self._render_mineig_block()
        if len(self.endog) > 1 and self.shea is not None:
            return self._render_shea_table()
        return self._render_r2_table() + self._render_mineig_block()

    def _render_mineig_block(self) -> str:
        """The Stock-Yogo block Stata prints UNDER the summary table::

            Minimum eigenvalue statistic = 332.355

            Critical Values                      # of endogenous regressors:    1
            H0: Instruments are weak             # of excluded instruments:     2
            ---------------------------------------------------------------------
                                                 |    5%     10%     20%     30%
            2SLS relative bias                   |         (not available)
            -------------------------------------+---------------------------------
                                                 |   10%     15%     20%     25%
            2SLS size of nominal 5% Wald test     |  19.93   11.59    8.75    7.25
            LIML size of nominal 5% Wald test     |   8.68    5.33    4.42    3.92
            ---------------------------------------------------------------------

        The numbers were already computed and stored -- ``r(mineigcv)`` reproduces Stata's
        19.93 / 11.59 / 8.75 / 7.25 exactly -- but nothing rendered them, so
        ``estat firststage`` stopped after the summary table and never reported whether the
        instruments are weak, which is the whole point of running it.
        """

        if self.mineig is None or self.mineigcv is None:
            return ""

        matrix = np.atleast_2d(self.mineigcv)
        n_endog = len(self.endog)
        # ``df_num`` for the first stage is the number of excluded instruments: the joint F
        # tests exactly those.
        n_excluded = int(np.atleast_1d(self.df_num)[0])

        # Layout measured from Stata's block: the label column is 37 characters wide
        # including the two-space indent, the pipe sits at column 37, and each critical
        # value occupies 8 columns (giving a 69-character rule).
        def _cell(value: float) -> str:
            return f"{value:>8.2f}" if np.isfinite(value) else ".".rjust(8)

        # Every row below was measured from Stata's output: the pipe sits at column 37,
        # the header row is 68 characters, the value rows 69, and the separator 71 with
        # its "+" at column 37. The padding is therefore written out literally rather than
        # derived -- deriving it kept landing one column off.
        rule = "  " + "-" * 69
        separator = "  " + "-" * 35 + "+" + "-" * 33
        lines = [
            "",
            # TWO blanks before the eigenvalue line, not one: measured on a live
            # `estat firststage`, the rule sits at index 7, a blank at 8, ANOTHER blank at 9,
            # then "Minimum eigenvalue statistic" at 10.
            "",
            # The line is padded to a FIXED total width of 45, not to a fixed number of trailing
            # spaces: '190.149' needs five and '44.963' needs six to reach the same 45. Hard-coding
            # either count broke the other case -- the two reference fits in the suite disagree by
            # exactly one space, which is how this was found (round 188).
            # SIX SIGNIFICANT FIGURES, not a fixed three decimals. Stata prints r(mineig) with its
            # default %g style, so 332.355 stays three decimals while 20.0861 needs four. A
            # hardcoded ``:.3f`` truncated the second case to "20.086" -- measured on a live
            # `estat firststage` where r(mineig) = 20.0861029023 and Stata prints 20.0861.
            f"  Minimum eigenvalue statistic = {self.mineig:.6g}".ljust(45),
            "",
            f"  Critical Values                      # of endogenous regressors: {n_endog:>4}",
            f"  H0: Instruments are weak             # of excluded instruments: {n_excluded:>5}",
            rule,
            "                                     |    5%     10%     20%     30%",
            # The RELATIVE BIAS row comes from matrix[0]. When every cell is ``nan`` Stata prints
            # a single "(not available)" instead of four dots -- which is what this package did
            # unconditionally, hiding the real critical values (13.91 9.08 6.46 5.39 on
            # grilic.dta). Round 188: render the numbers when they exist, and keep the phrase
            # only for the genuinely-absent case.
            "  2SLS relative bias                 |"
            + (
                "         (not available)"
                if not np.isfinite(np.asarray(matrix[0], dtype=float)).any()
                else "".join(
                    f"{value:>{7 if position == 0 else 8}.2f}"
                    if np.isfinite(value)
                    else ".".rjust(7 if position == 0 else 8)
                    for position, value in enumerate(matrix[0])
                )
            ),
            separator,
            "                                     |   10%     15%     20%     25%",
        ]
        labels = (
            "2SLS size of nominal 5% Wald test",
            "LIML size of nominal 5% Wald test",
        )
        for offset, label in enumerate(labels, start=1):
            if offset >= matrix.shape[0]:
                continue
            # The first cell is SEVEN wide and the rest eight; that asymmetry is
            # Stata's, measured off its own rows ("  19.93   11.59    8.75    7.25").
            cells = ""
            for position, value in enumerate(matrix[offset]):
                width = 7 if position == 0 else 8
                cells += (
                    f"{value:>{width}.2f}" if np.isfinite(value) else ".".rjust(width)
                )
            lines.append(f"  {label:<35}|{cells}")
        lines.append(rule)
        return chr(10).join(lines)

    def _render_r2_table(self) -> str:
        """Reproduce Stata's R-squared block exactly::

            First-stage regression summary statistics
            --------------------------------------------------------------------------
                         |            Adjusted      Partial
                Variable |   R-sq.       R-sq.        R-sq.      F(2,396)   Prob > F
            -------------+------------------------------------------------------------
                   endog |  0.6288      0.6260       0.6267       332.355    0.0000
            --------------------------------------------------------------------------

        Stata prints a two-line header (the "Adjusted / Partial" words sit on the line
        ABOVE "R-sq." and "R-sq."), the F column is labelled with its degrees of freedom
        ("F(2,396)"), and the F statistic carries three decimals where the R-squared
        columns carry four. The old single-line header could not express any of that.
        """

        # ``df_num``/``df_denom`` are per-equation arrays; the heading carries the first
        # equation's degrees of freedom, which is what Stata prints for the block.
        first_df = np.atleast_1d(self.df_num)[0]
        first_denom = np.atleast_1d(self.df_denom)[0]
        f_label = f"F({first_df:g},{first_denom:g})"

        def _place(pieces: list[str], text: str, end_column: int) -> None:
            """Append ``text`` so its last character sits at ``end_column - 1``."""

            used = sum(len(piece) for piece in pieces)
            pieces.append(" " * max(end_column - used - len(text), 0) + text)

        # Every start/end column below was measured from Stata's own output:
        #
        #   top   "|" 13-14, "Adjusted" 26-34, "Partial" 40-47                 (47 wide)
        #   hdr   "Variable" 4-12, "|" 13, "R-sq." 17-22, ... "Prob > F" 64-72 (72 wide)
        #   row   name ends 12, "|" at 13, r2 16-22, adj 28-34,
        #         partial 41-47, F 54-61, p 65-71                              (71 wide)
        #   rules 74 wide
        #
        # Stata's header rows are FIXED text apart from the F label, so they are written
        # out literally; only the data rows need placement.
        #
        # EVERY line in this block starts with TWO spaces that the columns recorded above did not
        # account for, which is why the block sat two columns left of Stata's. Captured lengths:
        # title 43, rule 76, header_top 49, header_bottom 74, separator 76.
        rule = "  " + "-" * 74
        header_top = "  " + " " * 13 + "|" + " " * 12 + "Adjusted" + " " * 6 + "Partial"
        # Built by placing each field at its measured column: the pipe at 13, the three
        # "R-sq." headings starting at 17 / 30 / 43, the F label ending at 61 and
        # "Prob > F" ending at 72. Writing the padding out by hand kept drifting by one
        # or three columns depending on the F label's width.
        def _row(pieces: list[tuple[str, int]]) -> str:
            text = ""
            for value, end_column in pieces:
                text += " " * max(end_column - len(text) - len(value), 0) + value
            return text

        header_bottom = "  " + _row(
            [
                ("Variable", 12),
                ("|", 14),
                ("R-sq.", 22),
                ("R-sq.", 34),
                ("R-sq.", 47),
                (f_label, 61),
                ("Prob > F", 72),
            ]
        )
        # A DUPLICATED ASSIGNMENT USED TO SIT HERE. The block was immediately overwritten
        # by the one below, so it changed nothing while looking authoritative -- and the
        # two differed by one leading space in the rule, which is exactly the kind of
        # difference a reader would assume was load-bearing. Removed in round 10.
        lines = [
            "",
            "  First-stage regression summary statistics",
            rule,
            header_top,
            header_bottom,
            "  " + "-" * 13 + "+" + "-" * 60,
        ]
        for i, name in enumerate(self.endog):
            # The name is LEFT-aligned at column 10 and the pipe follows it, so the pipe lands at
            # 15 for a 4-character name and 16 for a 5-character one. Every numeric field after
            # the pipe is right-aligned at a fixed offset FROM THE PIPE -- measured, and identical
            # for both reference rows:
            #
            #   '          expr |  0.1064 ...'   pipe 15, fields end at +9 +21 +34 +48 +58
            #   '          endog |  0.6288 ...'  pipe 16, same offsets
            #
            # Absolute columns cannot express this: six position sets were tried, each matching
            # one row and breaking the other.
            # The name is RIGHT-aligned so the pipe always lands at column 15, which is what
            # Stata does: '             s |' and '          endog |' both put it there. A fixed
            # ten-space prefix left the pipe at 12 for a one-character name, which is why the
            # short-name rows sat one column left. Measured in round 188.
            prefix = "  " + name.rjust(12) + " "
            row = prefix + "|"
            for value, offset in (
                (f"{self.r2[i]:.4f}", 9),
                (f"{self.adj_r2[i]:.4f}", 21),
                (f"{self.partial_r2[i]:.4f}", 34),
                # The F column is VARIABLE width too: Stata prints 20.0861 here and 332.355
                # elsewhere, so a fixed .3f truncated this one. Same %g rule as the eigenvalue.
                (f"{self.f[i]:.6g}", 48),
                (f"{self.pvalue[i]:.4f}", 58),
            ):
                target = len(prefix) + max(14 - len(prefix), 0) + offset
                row += " " * max(target - len(row) - len(value), 0) + value
            lines.append(row)
        lines.append(rule)
        # A BLANK line closes the summary table before the Stock-Yogo block begins.
        # Stata prints it: the rule, then a blank, then "Minimum eigenvalue statistic".
        # Measured on a live `estat firststage` in round 9 of the v3 work.
        lines.append("")
        return chr(10).join(lines)

    def _render_shea_table(self) -> str:
        """The multi-endogenous default: Shea's partial R-squared.

        ``ivregress_postestimation.sthlp:413-414``.  The ``all`` option is what asks for
        the R-squared block in this case, which is why the renderer has to know the
        count rather than always drawing one shape.
        """

        header = (
            f"  {'Variable':<14}|{'Shea':>10}{'Adj. Shea':>12}"
            f"{'Partial':>10}{'F':>10}{'Prob > F':>10}"
        )
        lines = [
            "",
            "First-stage regression summary statistics",
            "",
            header,
            "-" * 14 + "+" + "-" * 54,
        ]
        for i, name in enumerate(self.endog):
            adjusted = (
                self.shea_adj[i]
                if self.shea_adj is not None and i < len(self.shea_adj)
                else float("nan")
            )
            adjusted_text = (
                f"{adjusted:>12.4f}" if np.isfinite(adjusted) else f"{'.':>12}"
            )
            lines.append(
                f"  {name:<14}|{self.shea[i]:>10.4f}{adjusted_text}"
                f"{self.partial_r2[i]:>10.4f}{self.f[i]:>10.4g}"
                f"{self.pvalue[i]:>10.4f}"
            )
        lines.append("-" * 14 + "+" + "-" * 54)
        return "\n".join(lines)

    def to_frame(self) -> pd.DataFrame:
        """The table as a DataFrame indexed by endogenous regressor."""

        payload: dict[str, np.ndarray] = {
            "R-sq.": self.r2,
            "Adjusted R-sq.": self.adj_r2,
            "Partial R-sq.": self.partial_r2,
            "F": self.f,
            "df_num": self.df_num,
            "df_denom": self.df_denom,
            "Prob > F": self.pvalue,
        }
        if self.shea is not None:
            payload["Shea's partial R-sq."] = self.shea
        if self.shea_adj is not None:
            payload["Shea's adj. partial R-sq."] = self.shea_adj
        return pd.DataFrame(payload, index=pd.Index(self.endog, name="Variable"))

    def to_dict(self) -> dict[str, Any]:
        """The ``e()``/``r()`` style summary, keyed by endogenous name."""

        return {
            name: {
                "r2": float(self.r2[i]),
                "adj_r2": float(self.adj_r2[i]),
                "partial_r2": float(self.partial_r2[i]),
                "F": float(self.f[i]),
                "df_num": float(self.df_num[i]),
                "df_denom": float(self.df_denom[i]),
                "p": float(self.pvalue[i]),
            }
            for i, name in enumerate(self.endog)
        }


def _has_robust_vce(result: Any) -> bool:
    """Whether the fit used a robust or clustered covariance.

    ``ivregress_postestimation.sthlp:417-422`` makes the minimum-eigenvalue statistic
    conditional on this: its critical values assume i.i.d. normal errors, so Stata hides
    them under a robust VCE unless ``forcenonrobust`` is given.

    The covariance type is read from the estimated result rather than guessed from the
    presence of clusters: a fit can be robust WITHOUT clustering.
    """

    for attribute in ("cov_type", "covariance_type"):
        value = getattr(result, attribute, None)
        if isinstance(value, str) and value:
            normalised = value.lower()
            return not normalised.startswith("nonrobust") and normalised not in (
                "unadjusted",
                "homoskedastic",
                "iid",
            )
    # ``weight_spec.implies_robust`` covers the analytic-weight cases, where the VCE is
    # robust as a consequence of the weighting rather than an explicit request.
    spec = getattr(result, "weight_spec", None)
    if spec is not None and getattr(spec, "implies_robust", False):
        return True
    return False


def _require_iv(result: Any) -> dict[str, Any]:
    structure = getattr(result, "iv_structure", None)
    if structure is None:
        raise NotImplementedError(
            "estat firststage is only defined after an instrumental-variables "
            "fit: Stata reaches it through ivregress_estat, and it needs the "
            "split between included exogenous regressors, endogenous "
            "regressors and excluded instruments; got "
            f"{getattr(result, 'model_name', 'an unnamed model')!r} without that "
            "structure"
        )
    if structure["endog"].shape[1] == 0:
        raise ValueError(
            "estat firststage: model contains no endogenous variables"
        )
    return structure


def estat_firststage(
    result: Any, *, all: bool = False, forcenonrobust: bool = False
) -> FirstStageResult:
    """Stata's ``estat firststage`` for an IV fit.

    Parameters
    ----------
    all:
        Stata's ``all`` option: report the R-squared block even when the model has MORE
        than one endogenous regressor.  ``ivregress_postestimation.sthlp:408-414`` says
        the default switches to Shea's partial R-squared in that case, and that ``all``
        asks for every first-stage statistic "regardless of whether the model contains
        one or more endogenous regressors".
    forcenonrobust:
        Stata's ``forcenonrobust`` option (``sthlp:416-422``): report the minimum
        eigenvalue statistic and its critical values even though a robust VCE was used.
        Without it those numbers are suppressed under a robust VCE, because the critical
        values assume independent, identically distributed normal errors.

    Raises
    ------
    NotImplementedError
        When the result carries no IV structure.
    ValueError
        When the model has no endogenous regressor, or no excluded instrument
        (in which case the first stage is not identified).
    """

    structure = _require_iv(result)

    exog = np.asarray(structure["exog"], dtype=float)
    endog = np.asarray(structure["endog"], dtype=float)
    instruments = np.asarray(structure["instruments"], dtype=float)
    y = np.asarray(structure["response"], dtype=float).reshape(-1)
    exog_names = list(structure["exog_names"])
    endog_names = list(structure["endog_names"])

    if instruments.shape[1] == 0:
        raise ValueError(
            "estat firststage needs at least one excluded instrument; this "
            "model has none, so no first stage exists"
        )

    # Included exogenous regressors: everything in ``exog`` except the constant,
    # which ``_regress`` supplies itself.
    included = [j for j, name in enumerate(exog_names) if not _is_constant(name)]
    included_matrix = exog[:, included] if included else np.empty((y.size, 0))

    nobs = y.size
    k_included = len(included) + 1  # + intercept

    # Residualise the excluded instruments on the included exogenous regressors
    # once: ``ivregress_estat.ado`` computes one such residual per instrument.
    #
    # When there are no included regressors the residualisation must still
    # demean: ``_ols`` returns a zero fit for an empty design, so passing the
    # empty matrix through would leave the columns uncentred and shift every
    # statistic that uses them.
    def _on_included(column: np.ndarray) -> np.ndarray:
        return _residualise(column, included_matrix)

    instrument_residuals = np.column_stack(
        [_residualise(instruments[:, j], included_matrix) for j in range(instruments.shape[1])]
    )

    rows: list[dict[str, float]] = []
    for i in range(endog.shape[1]):
        target = endog[:, i]

        # R-sq / adjusted R-sq of the endogenous regressor on ALL exogenous
        # variables (included + excluded).
        all_exog = np.column_stack([included_matrix, instruments])
        _, r2 = _ols(target, all_exog)
        df_resid_all = nobs - all_exog.shape[1] - 1
        adj_r2 = (
            1.0 - (1.0 - r2) * (nobs - 1) / df_resid_all
            if np.isfinite(r2) and df_resid_all > 0
            else float("nan")
        )

        # Partial R-sq: residualise the endogenous regressor on the included
        # exog, then regress those residuals on the residualised instruments.
        target_residual = _residualise(target, included_matrix)
        _, partial_r2 = _ols(target_residual, instrument_residuals)

        # Partial F: joint test of the excluded instruments.  With the
        # residualised construction this is the F of the second auxiliary
        # regression on all instrument residuals.
        design = instrument_residuals
        if not any(np.ptp(design[:, j]) == 0.0 for j in range(design.shape[1])):
            design_aug = np.column_stack([np.ones(nobs), design])
        else:
            design_aug = design
        df_num = float(instruments.shape[1])
        # ``ivregress_estat.ado:581`` defines the F denominator as
        #
        #     local dfdenom = e(N) - `rank' - `df_a' - `nendog'
        #
        # which for one endogenous regressor and no absorbed terms is
        # ``N - rank - 1``.  Using the auxiliary regression's own ``df_resid`` instead
        # gives ``N - (1 + n_instruments)``, which is LARGER by
        # ``rank + 1 - 1 - n_instruments`` -- measured as 398 against Stata's 396 on a
        # 400-row fit with one instrument, i.e. the F was 0.25% too small.
        #
        # ``rank`` is the IV model's own ``e(rank)``; ``df_a`` is 0 without absorbed
        # terms and ``nendog`` is 1 per endogenous regressor, so the correction is
        # ``N - rank - 1``.
        model_rank = float(getattr(result, "e", {}).get("rank", np.nan) or np.nan)
        absorbed = float(getattr(result, "e", {}).get("df_a", 0.0) or 0.0)
        # The denominator depends on the number of EXCLUDED INSTRUMENTS, which two live
        # fits pin down:
        #
        #   ivregress 2sls y l (k = z)        n=24,  e(rank)=3, 1 instrument -> F(1,21)
        #   ivregress 2sls y x (e = z1 z2)    n=400, e(rank)=3, 2 instruments -> F(2,396)
        #
        # i.e. ``N - rank - (n_instr - 1)``: 24-3-0 = 21 and 400-3-1 = 396. Neither the ado
        # line's ``- nendog`` (which gives 20 and 395) nor plain ``N - rank`` (397 for the
        # second) fits both. With one instrument the two candidates coincide, which is why
        # this went unnoticed: the earlier code matched single-instrument fits and was 1 low
        # on the F for many-instrument ones.
        # ``df_num`` is already the number of excluded instruments (set above from
        # ``instruments.shape[1]``), so it carries exactly the count this needs.
        n_excluded = int(df_num) if np.isfinite(df_num) else 1
        instrument_adjustment = max(n_excluded - 1, 0)
        if np.isfinite(model_rank):
            df_denom = float(nobs - model_rank - absorbed - instrument_adjustment)
        else:  # pragma: no cover - defensive
            df_denom = float(nobs - design_aug.shape[1])
        if df_denom <= 0 or not np.isfinite(partial_r2) or partial_r2 >= 1.0:
            f_stat = float("nan")
            pvalue = float("nan")
        else:
            # F = (R2/df_num) / ((1-R2)/df_denom), Stata's ``test`` after the
            # auxiliary regression reduces to the same quantity.
            f_stat = (partial_r2 / df_num) / ((1.0 - partial_r2) / df_denom)
            pvalue = float(stats.f.sf(f_stat, df_num, df_denom))

        rows.append(
            {
                "r2": r2,
                "adj_r2": adj_r2,
                "partial_r2": partial_r2,
                "f": f_stat,
                "df_num": df_num,
                "df_denom": df_denom,
                "p": pvalue,
            }
        )

    # ---------------------------------------------------------------- Shea
    shea = None
    shea_adj = None
    if endog.shape[1] > 1:
        all_exog = np.column_stack([included_matrix, instruments])
        vhats = np.column_stack([_ols(endog[:, i], all_exog)[0] for i in range(endog.shape[1])])
        shea_values: list[float] = []
        shea_adjusted: list[float] = []
        nexog = included_matrix.shape[1] + 2 * instruments.shape[1]
        for i in range(endog.shape[1]):
            others = [j for j in range(endog.shape[1]) if j != i]
            # x1tilde: residual of endog_i on the other endogenous regressors
            # and the included exogenous ones.
            other_endog = endog[:, others] if others else np.empty((nobs, 0))
            x1tilde = _residualise(
                endog[:, i], np.column_stack([other_endog, included_matrix])
            )
            # x1bar: residual of vhat_i on the other vhats and included exog.
            other_vhats = vhats[:, others] if others else np.empty((nobs, 0))
            x1bar = _residualise(
                vhats[:, i], np.column_stack([other_vhats, included_matrix])
            )
            _, shea_r2 = _ols(x1tilde, x1bar.reshape(-1, 1))
            shea_values.append(shea_r2)
            if np.isfinite(shea_r2) and nobs > nexog:
                shea_adjusted.append(
                    1.0 - ((nobs - 1) / (nobs - nexog)) * (1.0 - shea_r2)
                )
            else:
                shea_adjusted.append(float("nan"))
        shea = shea_values
        shea_adj = shea_adjusted

    # ------------------------------------------------------ Cragg-Donald mineig
    #
    # ``FirstStage`` returns ``r(mineig)`` -- the minimum eigenvalue of the
    # canonical-correlation matrix between the endogenous regressors and the
    # residualised excluded instruments (``ivregress_estat.ado:777-778`` calls the
    # Mata builtin ``_iv_craggdonald_wrk``).
    #
    # It is the squared canonical correlations, so its eigenvalues come from
    # ``(E'E)^-1 (E'Z)(Z'Z)^-1 (Z'E)`` where E is the endogenous block
    # residualised on the included exogenous regressors and Z the residualised
    # instruments -- the same residualisation the partial R-squared above uses.
    #
    # Why this is checkable rather than guessed: with ONE endogenous regressor
    # the matrix is 1x1, so the only eigenvalue IS the partial R-squared, and the
    # partial-F identity ``partial_r2 = F/(F + N - k)`` holds numerically (verified
    # to 1e-10).  With more regressors it becomes a genuine eigenvalue problem and
    # no self-check exists -- that part still has no Stata reference here.
    mineig: float | None = None
    if instruments.shape[1] > 0:
        # Residualise the endogenous block on the INCLUDED exogenous regressors,
        # always including an intercept.  ``_ols`` returns a zero fit when the
        # design is empty (``design.size == 0``), so with no included regressors
        # the raw column would stay uncentred; that made the 1x1 eigenvalue come
        # out at 0.21434 instead of the partial R-squared 0.21110.
        def _centre_on_included(column: np.ndarray) -> np.ndarray:
            if included_matrix.shape[1] == 0:
                return column - column.mean()
            return _residualise(column, included_matrix)

        endog_r = np.column_stack(
            [_centre_on_included(endog[:, i]) for i in range(endog.shape[1])]
        )
        s_ee = endog_r.T @ endog_r
        s_zz = instrument_residuals.T @ instrument_residuals
        s_ez = endog_r.T @ instrument_residuals
        try:
            middle = np.linalg.solve(s_ee, s_ez) @ np.linalg.solve(s_zz, s_ez.T)
            eigenvalues = np.real(np.linalg.eigvals(middle))
            finite = eigenvalues[np.isfinite(eigenvalues)]
            if finite.size:
                # ``r(mineig)`` is the Cragg-Donald STATISTIC, not the raw eigenvalue. With
                # one endogenous regressor the 1x1 eigenvalue is the partial R-squared
                # (0.6267 here) while Stata prints 332.355 -- the same number as the
                # first-stage F. The two are related by
                #     F = eig/(1-eig) * df_denom/df_num
                # which reproduces 332.3552767 from 0.6266654, confirming the identity on
                # this sample rather than assuming it. Dividing by (1 - eig) and rescaling
                # is therefore the correct conversion.
                smallest = float(np.min(finite))
                remainder = 1.0 - smallest
                if remainder > 0.0:
                    numerator_dof = float(rows[0]["df_num"])
                    denominator_dof = float(rows[0]["df_denom"])
                    mineig = smallest / remainder * denominator_dof / numerator_dof
                else:
                    mineig = float("nan")
        except np.linalg.LinAlgError:  # pragma: no cover - singular blocks
            mineig = None

    return FirstStageResult(
        endog=endog_names,
        r2=[row["r2"] for row in rows],
        adj_r2=[row["adj_r2"] for row in rows],
        partial_r2=[row["partial_r2"] for row in rows],
        f=[row["f"] for row in rows],
        df_num=[row["df_num"] for row in rows],
        df_denom=[row["df_denom"] for row in rows],
        pvalue=[row["p"] for row in rows],
        shea=shea,
        shea_adj=shea_adj,
        mineig=mineig,
        mineigcv=mineig_critical_values(
            n_endog=endog.shape[1], n_instruments=instruments.shape[1]
        ),
        show_all=all,
        force_nonrobust=forcenonrobust,
        robust_vce=_has_robust_vce(result),
    )
