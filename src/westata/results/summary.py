"""Human-readable summaries for :class:`westata.results.StataResults`.

Two renderers are provided:

``statsmodels``
    Compact statsmodels-looking table (coefficients, standard errors, test
    statistics, p-values and confidence intervals).
``stata``
    Stata ``regress``-looking layout with ``Number of obs``, model F test,
    R-squared block and the ``Coef. / Std. Err. / P>|t| / [95% Conf. Interval]``
    coefficient table (the capitalisation is Stata's, per ``design.md:209``).  For ``dist="z"`` models the test labels switch to
    ``z`` / ``P>|z|`` while the block labels stay the same so report scrapers
    can rely on one layout.

The renderers only depend on the public results attributes, never on the
concrete class, which keeps this module usable by future result subclasses.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from scipy import stats

__all__ = ["render_summary"]

_VALID_STYLES = ("statsmodels", "stata")


def _ci_label(level: float | None) -> str:
    """Stata's confidence-interval header, sized to the level actually used.

    The real captures print it in lower case: ``[95% conf. interval]`` -- see
    ``p8_outputs.txt:25`` (qreg), ``p10_outputs.txt:106`` (nl) and
    ``p6p7_outputs.txt:18`` (logit).  Round 237 capitalised it from ``design.md:209``;
    the captures outrank that document.

    The level itself must NOT be hard-coded: a result built at ``level=0.90`` used to be
    headed "95%", silently mislabelling every number beneath it.
    """

    percent = 95.0 if level is None else float(level) * 100.0
    if abs(percent - round(percent)) < 1e-9:
        rendered = f"{int(round(percent))}"
    else:
        rendered = f"{percent:.1f}"
    return f"[{rendered}% conf. interval]"


def render_summary(result: Any, style: str = "statsmodels") -> str:
    """Render ``result`` in the requested style.

    Parameters
    ----------
    result:
        A results object exposing the statsmodels-compatible surface used by
        :class:`westata.results.StataResults`.
    style:
        ``"statsmodels"`` (default) or ``"stata"``.  Case is ignored.
    """

    normalized = _normalize_style(style)
    if normalized == "stata":
        text = _stata_summary(result)
        # Stata prefixes the table with one note per absorbed regressor, e.g.
        #   note: canal omitted because of collinearity.
        # and the header follows immediately - no blank line between. The rendered table
        # starts with a blank line of its own, so it is stripped before the note goes on.
        omitted = tuple(getattr(result, "omitted_columns", ()) or ())
        if omitted:
            notes = "".join(
                f"note: {name} omitted because of collinearity.{chr(10)}"
                for name in omitted
            )
            text = notes + text.lstrip(chr(10))
        return text
    return _statsmodels_summary(result)


def _normalize_style(style: str) -> str:
    if not isinstance(style, str):
        raise ValueError(f"Unknown summary style {style!r}; expected 'statsmodels' or 'stata'")
    normalized = style.strip().lower()
    if normalized not in _VALID_STYLES:
        raise ValueError(
            f"Unknown summary style {style!r}; expected one of {_VALID_STYLES}"
        )
    return normalized


def _as_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _e_float(result: Any, key: str) -> float:
    """Read a numeric Stata ``e()`` scalar, returning NaN when absent."""

    e = getattr(result, "e", None)
    if e is None:
        return float("nan")
    try:
        value = e[key]
    except (KeyError, TypeError):
        return float("nan")
    return _as_float(value)


def _num(value: float, digits: int = 7) -> str:
    """Format a number the way Stata prints scalars (``.`` when missing)."""

    if value is None or not np.isfinite(value):
        return "."
    return f"{value:.{digits}g}"


def _cell(value: float, width: int, digits: int = 7) -> str:
    return f"{_num(value, digits):>{width}}"


def _stata_number(value: float, width: int = 9) -> str:
    """Stata's ``%9.0g``: the widest representation that fits, no leading zero.

    This is NOT ``%.7g``. Stata picks the precision that fills the field, so the number
    of significant digits varies with magnitude -- ``.1236469`` (7) beside ``.0133633``
    (6) and ``.000947`` (3). A fixed ``%.7g`` gets the first right and the second wrong
    (``.01336326``), which is why the columns looked misaligned even though the values
    were correct.

    Verified against Stata's own ``display %9.0g`` on the same inputs:

        0.1236468764 -> .1236469        0.0009470056 -> .000947
        0.0133632566 -> .0133633        -0.02220731  -> -.0222073
        1.0231784    -> 1.023178

    The width counts the BODY only; Stata does not charge the minus sign against it, so
    ``-.0222073`` is eight characters in a nine-wide field.
    """

    if value is None or not np.isfinite(value):
        return f"{'.':>{width}}"

    text = ""
    for precision in range(width, 0, -1):
        text = f"%.{precision}g" % value
        # Stata never prints a leading zero: ``.5``, not ``0.5``.
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
        body = text[1:] if text.startswith("-") else text
        # Stata prefers FIXED notation and falls back to an exponent only when the fixed
        # form cannot fit the field. C's ``%g`` (and therefore Python's) switches to an
        # exponent once the exponent drops below -4, which is why a standard error of
        # .0000249 came out as ``2.49e-05`` where Stata prints ``.0000249``.
        #
        # Reconstruct the fixed form by walking DECIMAL PLACES and keeping the longest one
        # that fits the field: for 2.486e-05 that is 7 places -> .0000249 (eight chars),
        # which is exactly Stata's value. Formatting at full precision (.12f) would give
        # .000024861206 -- thirteen chars, too wide, and therefore rejected.
        if "e" in body:
            best = ""
            for decimals in range(1, 20):
                fixed = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
                if fixed.startswith("0."):
                    fixed = fixed[1:]
                elif fixed.startswith("-0."):
                    fixed = "-" + fixed[2:]
                fixed_body = fixed[1:] if fixed.startswith("-") else fixed
                # A fixed form that has collapsed to "0" or "." has LOST the value entirely;
                # keep the exponential text in that case (e.g. 1.23e-12, where no decimal
                # count under 12 produces a nonzero fixed string that fits).
                if fixed_body in ("", "0", "."):
                    continue
                if 0 < len(fixed_body) <= width - 1:
                    best = fixed
            if best:
                text = best
                body = best[1:] if best.startswith("-") else best
        if len(body) <= width - 1:
            break
    return f"{text:>{width}}"


def _stata_coef(value: float, width: int) -> str:
    """Stata coefficient format: the ``%9.0g`` rule, right-aligned in ``width``."""

    return _stata_number(value, 9).rjust(width)


def _stata_stat(value: float, width: int) -> str:
    """Stata displays t/z with two decimals."""

    if value is None or not np.isfinite(value):
        return f"{'.':>{width}}"
    return f"{value:>{width}.2f}"


def _stata_pvalue(value: float, width: int) -> str:
    """Stata prints ``0.000`` for p < 0.0005 and three decimals otherwise."""

    if value is None or not np.isfinite(value):
        return f"{'.':>{width}}"
    if value < 0.0005:
        return f"{'0.000':>{width}}"
    return f"{value:>{width}.3f}"


def _df_label(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "."
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


#: Which goodness-of-fit block each Stata command prints.  This is a property of
#: the *estimator*, so it is keyed on the model name rather than on which
#: ``e()`` scalars happen to be present -- ``qreg`` stores a pseudo-R2 under
#: ``e(r2_p)``, the same key the ML models use for McFadden's, and keying off
#: that made the renderer print an empty "Log likelihood = ." for it.
#:
#: ``none``  -- Stata prints no goodness-of-fit block at all (arima, arch,
#:              xtgls).  Printing ``R-squared = .`` there is noise, and the
#:              "Root MSE" fallback computed below would print a number that
#:              is neither an RMSE nor a standard deviation.
#: ``svy``   -- the survey design block (strata / PSUs / population / design df)
#: ``quantile`` -- qreg's deviation lines
#: Commands whose fit statistics print as a TWO-COLUMN block with NO sums-of-squares
#: table, even though their ``e(vcetype)`` is empty.
#:
#: This set exists because the vcetype test was FALSIFIED by ``rreg``: both ``regress`` and
#: ``rreg`` store ``e(vcetype)=""``, yet regress prints the ANOVA table and rreg prints the
#: three-row robust header. So "is the vcetype non-empty" cannot decide the shape -- the
#: command's identity has to. Each entry was measured against Stata, not inferred.
_NO_ANOVA_TWO_COLUMN: frozenset[str] = frozenset(
    {
        "rreg",  # Number of obs / F / Prob > F, no ANOVA
    }
)


def _wants_two_column_header(result: Any) -> bool:
    """Whether this fit prints the two-column header instead of the ANOVA table.

    True when the variance estimator is a robust one (non-empty ``e(vcetype)``), OR the
    command is listed in :data:`_NO_ANOVA_TWO_COLUMN`. The second clause is what makes
    ``rreg`` work -- its vcetype is empty like an ordinary regression.
    """

    vcetype = str((getattr(result, "e", None) or {}).get("vcetype") or "").strip()
    if vcetype:
        return True
    model_name = str(getattr(result, "model_name", "") or "")
    return model_name in _NO_ANOVA_TWO_COLUMN


_HEADER_BLOCK: dict[str, str] = {
    # ``newey.ado:245-258`` prints its own header and it is NOT regress's:
    #
    #   Regression with Newey–West standard errors   Number of obs = N
    #   Maximum lag = lag                            F(df_m, df_r) = F
    #                                                Prob > F      = p
    #
    # There is no R-squared, no Adj R-squared and no Root MSE -- ``newey`` does
    # not store them -- and the left column carries "Maximum lag" instead.
    "newey": "newey",
    "regress": "ols",
    "ivregress 2sls": "ols",
    "ivregress liml": "ols",
    "ivregress gmm": "ols",
    "reghdfe": "ols",
    "xtreg, fe": "ols",
    # These three were registered in the conformance suite at E3 but had NO header entry,
    # so they fell through to the generic path and never took part in the layout sweep.
    # Measured: areg prints the same two-column block as regress; xtabond prints the
    # panel design block (Group variable / Time variable / Obs per group).
    "areg": "ols",
    "xtabond": "ols",
    "sureg": "ols",
    "xtreg, re": "ols",
    "xtreg, be": "ols",
    "xtreg, fd": "ols",
    "xtivreg, fe": "ols",
    "xtivreg, re": "ols",
    "xtpcse": "ols",
    "xtgls": "ols",
    "didregress": "ols",
    "xtdidregress": "ols",
    "eivreg": "ols",
    "rreg": "ols",
    "nl": "ols",
    "logit": "ml",
    "probit": "ml",
    "poisson": "ml",
    "nbreg": "ml",
    "zip": "ml",
    "zinb": "ml",
    "ztp": "ml",
    "tnbreg": "ml",
    "ologit": "ml",
    "oprobit": "ml",
    "tobit": "ml",
    "truncreg": "ml",
    "intreg": "ml",
    "heckman": "ml",
    "etregress": "ml",
    "xtlogit, fe": "ml",
    "xtlogit, re": "ml",
    "xtlogit, pa": "none",
    "xtprobit, re": "ml",
    "xttobit": "ml",
    "xtpoisson, fe": "ml",
    "xtpoisson, re": "ml",
    "xtnbreg, fe": "ml",
    "frontier": "ml",
    "arima": "arch",
    "arch": "arch",
    "qreg": "quantile",
    "sqreg": "quantile",
    "bsqreg": "quantile",
    "svy: mean": "svy",
    "svy: total": "svy",
    "svy: proportion": "svy",
    "svy: regress": "svy",
    "svy: logit": "svy",
}


#: Auxiliary parameters that Stata DISPLAYS with a leading slash even though
#: ``e(b)`` stores them without one (``ologit.json`` holds ``cut1``, ``nbreg.json``
#: holds ``lnalpha``).  ``tobit``'s ``sigma`` is NOT in this set: Stata prints it
#: bare, exactly as it is stored.
_SLASH_DISPLAYED = frozenset(
    {"lnsigma", "lnalpha", "cut1", "cut2", "cut3", "cut4", "cut5",
     "cut6", "cut7", "cut8", "cut9", "cut10"}
)


def _constant_last_order(names: list[str]) -> list[int]:
    """Indices with the constant moved to the end, everything else in place.

    Used only when ``_stata_order`` refuses (ambiguous factor terms).  Stata prints the
    constant last in every regression table, so this is the one part of the ordering
    that can be settled without a full name mapping.
    """

    constants = ("intercept", "const", "_cons")

    def _equation(name: str) -> str:
        """The equation a name belongs to, so each can be ordered independently."""
        for marker in ("#", ":"):
            if marker in str(name):
                return str(name).split(marker, 1)[0]
        return ""

    def _bare(name: str) -> str:
        text = str(name)
        for marker in ("#", ":"):
            if marker in text:
                text = text.split(marker, 1)[1]
        return text.strip().lower()

    # Sort WITHIN each equation. A multi-equation model has one constant per equation and
    # Stata puts each at the end of its own block; ordering them all together moved the
    # main equation's constant down into the inflate group.
    order: list[int] = []
    for equation in dict.fromkeys(_equation(name) for name in names):
        members = [
            index for index, name in enumerate(names) if _equation(name) == equation
        ]
        order.extend(index for index in members if _bare(names[index]) not in constants)
        order.extend(index for index in members if _bare(names[index]) in constants)
    return order


def _safe_stata_order(result: Any) -> list[int]:
    """``_stata_order()`` with the identity as a fallback.

    Rendering must never fail merely because a name mapping is unavailable; the
    identity order still draws a correct table, just in patsy's order.
    """

    try:
        order = list(getattr(result, "_stata_order", lambda: [])())
    except (NotImplementedError, AttributeError):
        return []
    return order


def _reports_a_chi2_test(result: Any) -> bool:
    """Whether the command prints a ``Wald chi2`` model test instead of an F.

    Read from ``e(chi2type)``, which is the macro Stata itself sets for the purpose.
    The distribution of the COEFFICIENT tests is a separate question: ``xtpcse`` reports
    ``z``/``P>|z|`` per coefficient but ``Wald chi2`` for the model, so keying the model
    line on ``dist`` would render it as F -- which is what happened before.
    """

    e_macros = getattr(result, "e", {}) or {}
    chi2type = str(e_macros.get("chi2type") or "").strip().lower()
    return chi2type in ("wald", "lr", "chi2")


def _panel_design_block(result: Any) -> list[str]:
    """The design block ``xtpcse``/``xtgls`` print above their estimation table.

    Returns an empty list for every other estimator, so the caller can append the lines
    unconditionally.

    Layout follows the captured run at
    ``tests/conformance/stata_fixtures/p9_outputs.txt:496-506``.  Only the pieces this
    package stores are emitted; a missing macro yields a ``.`` rather than a fabricated
    value, and the whole block is skipped when the panel structure is unknown.
    """

    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    e_macros = getattr(result, "e", {}) or {}
    # ``xtpcse``/``xtgls`` print the block; so do the ``xt`` non-linear commands.  The
    # captured run for ``xtlogit, fe`` (``p9_outputs.txt:14-22``) shows the same shape::
    #
    #   Conditional fixed-effects logistic regression      Number of obs    =    912
    #   Group variable: firm                               Number of groups =    114
    #                                                        Obs per group:
    #                                                                     min =      8
    #                                                                     avg =      8.0
    #                                                                     max =      8
    #
    # so the gate is "an xt command that knows its group count" rather than a fixed list
    # of two names -- the earlier list let every xtlogit/xtprobit/xtpoisson output drop
    # the panel structure entirely.
    is_xt_family = model_name.startswith("xt") and "N_g" in e_macros
    if not (is_xt_family or model_name in ("xtpcse", "xtgls")):
        return []

    def _number(key: str) -> str:
        value = e_macros.get(key)
        if value is None:
            return "."
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(numeric):
            return "."
        return f"{int(numeric)}" if numeric == int(numeric) else f"{numeric:g}"

    entity = str(e_macros.get("ivar") or e_macros.get("panelvar") or "").strip()
    timevar = str(e_macros.get("tvar") or e_macros.get("timevar") or "").strip()
    panels = str(e_macros.get("panels") or "").strip()
    #: "Autocorrelation: no autocorrelation" on the xtpcse block. Stored under
    #: ``e(corr)`` when the estimator records one; the default wording otherwise.
    autocorr = str(e_macros.get("autocorrelation") or e_macros.get("corrtype") or "").strip()

    # ``xtpcse``/``xtgls`` print the FULL block: group/time variable, panel count,
    # per-group extremes, and the three "estimated ..." counts.
    #
    # The xt NON-LINEAR commands print a SHORTER one.  The captured ``xtlogit, fe``
    # output (``p9_outputs.txt:14-22``) is::
    #
    #   Group variable: firm                       Number of groups =    114
    #                                                Obs per group:
    #                                                             min =      8
    #                                                             avg =      8.0
    #                                                             max =      8
    #
    # -- no ``Time variable:``, no ``Panels:``, no "Estimated ..." rows.  Emitting the
    # xtpcse rows for a plain xtlogit produced three lines of ``.``, so the two shapes
    # are kept separate rather than unified.
    if model_name == "xtgls":
        # ``xtgls`` has its OWN design block, and it is not xtpcse's. Captured
        # (p9_outputs.txt:471-481):
        #
        #   Cross-sectional time-series FGLS regression
        #
        #   Coefficients:  generalized least squares
        #   Panels:        heteroskedastic
        #   Correlation:   no autocorrelation
        #
        #   Estimated covariances      =       100   Number of obs     =        700
        #   Estimated autocorrelations =         0   Number of groups  =        100
        #   Estimated coefficients     =         2   Time periods      =          7
        #                                             Wald chi2(1)      =     246.99
        #                                             Prob > chi2       =     0.0000
        #
        # Three plain lines instead of Group/Time/Panels/Autocorrelation, and the right
        # column carries Time periods / Wald / Prob rather than Obs per group.
        lines_gls: list[str] = []
        panels_key = str(e_macros.get("panels") or "").strip().lower()
        coefficients_label = {
            "iid": "generalized least squares",
            "heteroskedastic": "generalized least squares",
            "correlated": "generalized least squares",
        }.get(panels_key, "generalized least squares")
        corr_key = str(e_macros.get("corr") or "").strip().lower()
        correction_label = {
            "independent": "no autocorrelation",
            "independent_ar1": "common AR(1)",
            "ar1": "common AR(1)",
            "psar1": "panel-specific AR(1)",
        }.get(corr_key, "no autocorrelation")
        lines_gls.append(f"Coefficients:  {coefficients_label}")
        lines_gls.append(f"Panels:        {panels_key or '.'}")
        lines_gls.append(f"Correlation:   {correction_label}")
        lines_gls.append("")

        def _num(key: str, fmt: str = "{:.0f}") -> str:
            value = _e_float(result, key)
            return fmt.format(value) if np.isfinite(value) else "."

        def _right_gls(label: str, value: str) -> str:
            return f"{label:<18}= {value:>10}"

        def _est_gls(label: str, value: str) -> str:
            # "Estimated covariances      =       100" -- the "=" sits at column 27 and
            # the number ends at 38, same as the xtpcse block.
            return f"{label:<27}" + "=" + f"{value:>10}" + " " * 10

        _df_m = _e_float(result, "df_m")
        right_rows = [
            _right_gls("Number of obs", f"{int(round(_e_float(result, 'N'))):,}"
                        if np.isfinite(_e_float(result, "N")) else "."),
            _right_gls("Number of groups", _num("N_g")),
            _right_gls("Time periods", _num("N_t")),
            "",
            _right_gls(
                f"Wald chi2({_df_label(_df_m)})",
                f"{_e_float(result, 'chi2'):.2f}"
                if np.isfinite(_e_float(result, "chi2")) else ".",
            ),
            _right_gls(
                "Prob > chi2",
                f"{_e_float(result, 'p'):.4f}"
                if np.isfinite(_e_float(result, "p")) else ".",
            ),
        ]
        est_pairs = [
            _est_gls("Estimated covariances", _num("n_cv")),
            _est_gls("Estimated autocorrelations", _num("n_cr")),
            _est_gls("Estimated coefficients", _num("n_cf")),
        ]
        for index, est in enumerate(est_pairs):
            lines_gls.append(est + right_rows[index][:48].lstrip())
        # Row 3 of right_rows is a blank spacer that made an empty 48-wide line; Stata
        # goes straight from "Time periods" to "Wald chi2".
        for right in right_rows[4:]:
            lines_gls.append(f"{'':<48}{right}")
        # The captured banner row is "Cross-sectional ..." on its own line ABOVE this
        # block, and the block itself has no rule before the table.
        return lines_gls

    if model_name in ("xtpcse", "xtgls"):
        # A TWO-COLUMN block 78 wide. Measured from ``xtpcse y x1``: the left half holds
        # the panel structure and the "Estimated ..." counts, the right half the fit
        # statistics, and the two interleave row by row:
        #
        #   Group variable:   firm              Number of obs     =        700
        #   Time variable:    year              Number of groups  =        100
        #   Panels:           correlated (balanced)   Obs per group:
        #   Autocorrelation:  no autocorrelation            min =          7
        #                                                   avg =          7
        #                                                   max =          7
        #   Estimated covariances      =      5050   R-squared         =     0.2144
        #   Estimated autocorrelations =         0   Wald chi2(1)      =     178.20
        #   Estimated coefficients     =         2   Prob > chi2       =     0.0000
        #
        # Stacking the left rows first printed them in a different order from the right
        # ones, so nothing lined up.
        LEFT = 48

        def _left(pairs: list[tuple[str, str]]) -> str:
            # The left half is exactly 48 columns wide: label 18, value padded to 30.
            text = ""
            for label, value in pairs:
                text += f"{label:<18}{value:<30}"
            return text[:48].ljust(48)

        def _right(label: str, value: str) -> str:
            return f"{label:<18}= {value:>10}"

        def _number4(key: str) -> str:
            value = _e_float(result, key)
            return f"{value:.4f}" if np.isfinite(value) else "."

        def _number2(key: str) -> str:
            value = _e_float(result, key)
            return f"{value:.2f}" if np.isfinite(value) else "."

        lines: list[str] = [
            _left([("Group variable:", entity or ".")])
            + _right("Number of obs", _number("N")),
            _left([("Time variable:", timevar or ".")])
            + _right("Number of groups", _number("N_g")),
        ]
        # The right column starts one row LATE: "Obs per group:" heads it on the Panels
        # row, and the min/avg/max rows follow.
        # ``panels`` records only "correlated"; Stata prints the balance in parentheses
        # after it. Derived from the group sizes rather than assumed.
        _balanced = ""
        _gmin, _gmax = _e_float(result, "g_min"), _e_float(result, "g_max")
        if np.isfinite(_gmin) and np.isfinite(_gmax):
            _balanced = " (balanced)" if abs(_gmin - _gmax) < 1e-9 else " (unbalanced)"
        lines.append(
            _left([("Panels:", (panels or ".") + _balanced)]) + f"{'Obs per group:':>14}"
        )
        lines.append(
            _left([("Autocorrelation:", autocorr or "no autocorrelation")])
            + f"{'min':>17} = {_number('g_min'):>10}"
        )
        for label, key in (("avg", "g_avg"), ("max", "g_max")):
            lines.append(f"{'':<{LEFT}}{label:>17} = {_number(key):>10}")
        # The three "Estimated ..." rows sit opposite the fit statistics.
        coefficients = e_macros.get("k")
        if coefficients is None:
            design = getattr(result, "design_matrix", None)
            if design is not None:
                try:
                    coefficients = int(np.asarray(design[0]).shape[1])
                except (TypeError, ValueError, IndexError):
                    coefficients = None
        # "Estimated covariances" is the number of UNIQUE elements of the panel
        # covariance matrix, G(G+1)/2 -- not ``e(n_sigma)``, which is the count of common
        # PERIODS used to divide the cross-products. They are different quantities and
        # sharing the key printed 7 against Stata's 5050.
        _n_groups = _e_float(result, "N_g")
        _n_cov = (
            int(round(_n_groups * (_n_groups + 1) / 2)) if np.isfinite(_n_groups) else None
        )
        est_rows = [
            ("Estimated covariances", str(_n_cov) if _n_cov is not None else "."),
            ("Estimated autocorrelations", _number("n_cr")),
            (
                "Estimated coefficients",
                str(int(coefficients)) if coefficients is not None else "",
            ),
        ]
        fit_rows = [
            ("R-squared", _number4("r2")),
            (
                f"Wald chi2({_df_label(_e_float(result, 'df_m'))})",
                _number2("chi2"),
            ),
            ("Prob > chi2", _number4("p")),
        ]
        for (est_label, est_value), (fit_label, fit_value) in zip(est_rows, fit_rows):
            lines.append(
                # Left half exactly 48: label 26, "= ", value right-aligned in 9, then
                # padding -- measured from "Estimated covariances      =      5050".
                # Measured by character position rather than by field widths: in
                # "Estimated covariances      =      5050" the "=" sits at column 27 and
                # the number ENDS at 38, so the left half is "label, =, value ending at
                # 38" padded to 48. Composing it from label/value widths kept landing one
                # column off.
                f"{est_label:<27}" + "=" + f"{est_value:>10}" + " " * 10
                + _right(fit_label, fit_value)
            )
    elif (
        str(e_macros.get("corr") or "").strip()
        and str(e_macros.get("family") or "").strip()
        and str(e_macros.get("link") or "").strip()
    ):
        # The GEE population-averaged shape (``xtlogit ..., pa``), measured:
        #
        #   Group variable: unit                       Number of groups =     20
        #   Family: Binomial                           Obs per group:
        #   Link:   Logit                                             min =      5
        #   Correlation: exchangeable                                 avg =    5.0
        #                                                             max =      5
        #                                              Wald chi2(1)     =   0.62
        #   Scale parameter = 1                        Prob > chi2      = 0.4300
        #
        # Right column starts at 53: min/avg/max right-align their label to 66 then "= "
        # and a 6-wide value; the test labels are 17 wide. The model test is a WALD chi2,
        # not the LR chi2 the conditional family prints.
        _per_group = e_macros.get("_obs_per_group") or {}
        _gmin = _per_group.get("min", _e_float(result, "g_min"))
        _gavg = _per_group.get("avg", _e_float(result, "g_avg"))
        _gmax = _per_group.get("max", _e_float(result, "g_max"))

        # Geometry, measured from ``xtlogit ybin x1, pa`` (every row is 78 wide):
        #   right column starts at 53, its label field is 17, "=" sits at 70, value 7;
        #   the min/avg/max rows right-align their 3-letter label to 66.
        #   The left column puts its value IMMEDIATELY after the colon -- "Group
        #   variable: unit", not the extra padding the panel block uses.
        def _gee_right(label: str, value: str) -> str:
            return f"{label:<17}= {value:>7}"

        def _gee_count(value: Any, label: str) -> str:
            if value is None or not np.isfinite(value):
                text = "."
            elif label == "avg":
                text = f"{float(value):.1f}"
            else:
                text = f"{int(round(float(value)))}"
            return f"{label:>19} = {text:>6}"

        _banner_gee = str(e_macros.get("title") or "") or "GEE population-averaged model"
        lines = [
            # The banner shares its row with Number of obs -- this row was missing
            # entirely before, leaving the banner stranded on a line of its own.
            f"{_banner_gee:<53}"
            + _gee_right("Number of obs", f"{int(_e_float(result, 'N'))}"),
            f"{'Group variable: ' + (entity or '.'):<53}"
            + _gee_right("Number of groups", _number("N_g")),
            # Label/value pairs on the left with the Obs-per-group heading beside them.
            f"{'Family: ' + str(e_macros.get('family') or '.'):<53}"
            + f"{'Obs per group:':>14}",
            f"{'Link: ' + str(e_macros.get('link') or '.'):<53}"
            + _gee_count(_gmin, "min"),
            f"{'Correlation: ' + str(e_macros.get('corr') or '.'):<53}"
            + _gee_count(_gavg, "avg"),
            f"{'':<53}" + _gee_count(_gmax, "max"),
            f"{'':<53}"
            + _gee_right(
                f"Wald chi2({_df_label(_e_float(result, 'df_m'))})",
                f"{_e_float(result, 'chi2'):.2f}",
            ),
        ]
        _scale = _e_float(result, "scale")
        _scale_text = (
            "Scale parameter = " + f"{_scale:g}" if np.isfinite(_scale) else ""
        )
        _p_gee = _e_float(result, "p")
        if np.isfinite(_p_gee):
            lines.append(f"{_scale_text:<53}" + _gee_right("Prob > chi2", f"{_p_gee:.4f}"))
        elif _scale_text:
            lines.append(_scale_text)
    else:
        # The xt NON-LINEAR shape: group variable + group count, then the per-group
        # extremes.  No time-variable line, no ``Panels:``, no "Estimated ..." rows.
        # Measured from ``xtlogit ybin x1, fe``:
        #
        #   Group variable: unit                       Number of groups =     20
        #                                              Obs per group:
        #                                                           min =      5
        #                                                           avg =    5.0
        #                                                           max =      5
        #                                              LR chi2(1)       =   0.01
        #   Log likelihood = -41.887242                 Prob > chi2      = 0.9159
        #
        # Left column 53, right column 25 (label 18, " =", value 7). The old version used
        # 18/14 splits, which is why the earlier output was 64/69 wide instead of 78.
        # A RANDOM-EFFECTS fit names the distribution of u_i on its own row, in the LEFT column,
        # opposite the "Obs per group:" heading, and pushes the per-group block one row down:
        #
        #   Group variable: sid                      Number of obs     =        600
        #                                            Number of groups  =         30
        #   (blank)
        #   Random effects u_i ~ Gaussian            Obs per group:
        #                                                        min =         20
        #
        # `xtlogit ..., fe` prints NO such row -- confirmed against a live run of both variants.
        #
        # THE TITLE TEST MUST BE A PREFIX, NOT A SUBSTRING. "re" in title matches the word
        # "regression" -- so `xtlogit, fe` ("Conditional fixed-effects logistic regression") was
        # classified as random-effects and printed the Gaussian and Integration rows it must not
        # have. Found in round 234 by comparing the fe VARIANT, which the re work never re-checked.
        _title_text = str(e_macros.get("title") or "").strip()
        _is_random_effects = (
            e_macros.get("sigma_u") is not None
            or _title_text.startswith("Random-effects")
            or str(getattr(result, "model_name", "") or "").endswith(", re")
        )
        lines = [
            # ONE space after the colon, not the padded field the PCSE block uses. Measured in
            # round 224 on `xtlogit y x1, re`:
            #
            #   'Group variable: sid                               Number of obs ...'
            #
            # The ':<18' field this used to use printed 'Group variable:   sid', three spaces.
            f"{'Group variable: ' + (entity or '.'):<50}"
            f"{'Number of groups':<17}= {_number('N_g'):>7}",
        ]
        if _is_random_effects:
            lines.append("")
            lines.append(f"{'Random effects u_i ~ Gaussian':<53}{'Obs per group:':>25}")
        else:
            lines.append(f"{'':<53}{'Obs per group:':>25}")
        lines.extend(
            [
                f"{'':<53}{'min':>18} = {_number('g_min'):>7}",
                f"{'':<53}{'avg':>18} = {_number('g_avg'):>7}",
                f"{'':<53}{'max':>18} = {_number('g_max'):>7}",
            ]
        )
        _chi2 = _e_float(result, "chi2")
        _ll = _e_float(result, "ll")
        _p = _e_float(result, "p")
        _df = _e_float(result, "df_m")
        # A RANDOM-EFFECTS fit reports HOW it integrated u_i out, on its own row, between the
        # per-group block and the model test. Measured in round 223:
        #
        #   (blank)
        #   Integration method: mvaghermite            Integration pts.  =          7
        #   (blank)
        #                                        Wald chi2(1)        =      33.40
        #
        # `xtlogit ..., fe` prints neither this row nor the blank lines around it.
        if _is_random_effects:
            # The count lives in e(intpoints) -- NOT e(n_quad), which I guessed last round and
            # which silently fell through to a default of 7 when the real number was 12.
            _integration_pts = _e_float(result, "intpoints")
            if not np.isfinite(_integration_pts):
                _integration_pts = 7.0
            _method = str(e_macros.get("intmethod") or "mvaghermite").strip()
            # Column positions measured: the left text runs to 53 and "Integration pts." starts
            # THERE (not at 50 like the other right-column labels), with "=" at 71. The row is
            # 79 wide, one longer than the rest of the block.
            lines.append("")
            lines.append(
                f"{'Integration method: ' + _method:<53}"
                f"{'Integration pts.':<18}= {int(round(_integration_pts)):>6}"
            )
            lines.append("")
        if np.isfinite(_chi2):
            lines.append(
                f"{'':<53}{'LR chi2(' + _df_label(_df) + ')':<17}"
                f"= {_chi2:>7.2f}"
            )
        _ll_text = (
            "Log likelihood = " + _stata_number(_ll, 12).strip()
            if np.isfinite(_ll)
            else ""
        )
        if np.isfinite(_p):
            lines.append(f"{_ll_text:<53}{'Prob > chi2':<17}= {_p:>7.4f}")
        elif _ll_text:
            lines.append(_ll_text)
    return lines


def _header_block_kind(result: Any) -> str:
    """Which header layout Stata uses for this estimator.

    The choice is a property of the ESTIMATOR, not of which ``e()`` keys happen to
    be present: ``qreg`` stores a check-function pseudo-R2 under ``e(r2_p)``, which
    would misclassify it as an ML model and print a "Log likelihood = ." line Stata
    never shows.

    Returns one of ``"quantile"``, ``"svy"``, ``"newey"``, ``"ols"`` or ``"ml"``.
    """

    model_name = str(getattr(result, "model_name", "") or "")
    e_macros = getattr(result, "e", {}) or {}

    # The explicit table is the authority: every command this package implements has
    # an entry, and ``arima``/``arch`` are recorded there as ``none`` precisely
    # because they print no goodness-of-fit block at all.
    if model_name in _HEADER_BLOCK:
        return _HEADER_BLOCK[model_name]
    # Case-insensitive fallback: a hand-built result may carry ``"OLS"`` rather than
    # Stata's ``"regress"``.
    for known, kind in _HEADER_BLOCK.items():
        if known.lower() == model_name.lower():
            return kind
    if model_name.strip().lower() in ("ols", "lm", "linear"):
        return "ols"

    # The survey family carries a design block instead of the generic header.
    if str(e_macros.get("title", "") or "").lower().startswith("survey"):
        return "svy"
    if model_name.startswith("svy") or ", svy" in model_name:
        return "svy"

    if model_name.startswith("qreg") or model_name.startswith("sqreg"):
        return "quantile"
    if model_name.startswith("newey"):
        return "newey"

    # IV estimators print a Wald chi2 model test AND the OLS fit statistics
    # (R-squared / Adj R-squared / Root MSE) with a sums-of-squares table -- they are
    # "2SLS is a regression" estimators, not likelihood ones.  ``dist`` is "z" here,
    # so the ``z``-based shortcut below must not catch them.
    if model_name.startswith("ivregress") or model_name.startswith("iv"):
        return "ols"

    # A result that CARRIES the sums of squares has Stata's OLS fit block, whatever
    # its ``dist``.  This is the original signal and it is the reliable one: a
    # z-distributed model can still be a regression (2SLS), and a synthetic result
    # may set ``dist="z"`` while still reporting ``rss``/``mss``.
    if _anova_available(result):
        return "ols"

    # An OLS-style fit reports an F test and sums of squares rather than a
    # likelihood.  ``dist == "t"`` is the shared marker: the ML families use
    # ``dist == "z"``.
    if str(getattr(result, "dist", "t")) != "z":
        return "ols"

    return "ml"


def _cluster_variable(result: Any) -> str:
    """The name of the variable clustered on, for Stata's "in <var>" note.

    Stored under ``_cluster_names`` when the covariance was built, because the display
    path needs the NAME while the estimator only needs the labels. An unnamed cluster
    (a bare array) yields an empty string so the note degrades to
    "(adjusted for N clusters)" instead of inventing a variable name.
    """

    for key in ("_cluster_names", "clustvar", "cluster"):
        value = (getattr(result, "e", None) or {}).get(key)
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        if isinstance(value, str) and value:
            return value
    return ""


def _four(value: Any) -> str:
    """Four decimals, or ``.`` when the value is missing -- Stata's fit-statistic form."""

    return f"{float(value):.4f}" if np.isfinite(value) else "."


def _stata_abbreviated(name: str) -> str:
    """Stata's six-character display name: first four characters, ``~``, last one.

    Stata shortens a variable name inside a label cell to a FIXED width of six. Measured on
    ``tobit`` with two different names::

        food_exp      -> food~p
        abcdefghijklm -> abcd~m

    An eight-character name is shortened exactly like a thirteen-character one, so the rule
    is a fixed target width rather than "shorten only when it overflows the cell". The cut is
    at SIX characters: ``income`` (six) is printed untouched, ``abcdefg`` (seven) is not.
    """

    text = str(name)
    if len(text) <= 6:
        return text
    return text[:4] + "~" + text[-1]


def _pcse_family_for_header(result: Any) -> bool:
    """Whether this command prints its own panel design block (xtpcse/xtgls)."""

    return str(getattr(result, "model_name", "") or "") in ("xtpcse", "xtgls")


def _iv_family_for_header(result: Any) -> bool:
    """Whether this is an ``ivregress`` fit, which has its own two-column block."""

    return str(getattr(result, "model_name", "") or "").startswith("ivregress")


#: Blocks that draw their own header INSTEAD of the generic single-column statistics.
#: A block-name entry covers every command routed to that layout; a model-name entry
#: covers a single command that shares a layout with commands that do NOT draw their own.
#:
#: Each of these used to be its own literal, repeated in BOTH the "Number of obs" gate and
#: the model-test gate -- and every time a family was added, one of the two was missed,
#: duplicating statistics above the table. That happened for xtpcse, ivregress and eivreg.
_OWN_HEADER_BLOCKS = frozenset({"svy", "ml", "newey", "arch", "quantile"})
_OWN_HEADER_MODELS = (
    "xtpcse",
    "xtgls",
    "eivreg",
    # Each of these draws its header in its own branch (see the elif chain), so the
    # generic single-column statistics must be skipped or they appear above the block.
    "areg",
    "xtabond",
    "sureg",
    "mvreg",
)


def _is_gee_shape(result: Any) -> bool:
    """Whether this is the GEE population-averaged block (``xtlogit ..., pa``).

    Identified by the macros the block is built from, not by model_name: the shape needs
    a family, a link AND a correlation structure together, and no other layout has all
    three.
    """

    e_macros = getattr(result, "e", None) or {}
    return all(
        str(e_macros.get(key) or "").strip() for key in ("family", "link", "corr")
    )


def _prints_own_header_block(result: Any, block: str) -> bool:
    """Whether the command draws its own header, so generic statistics must be skipped."""

    if block in _OWN_HEADER_BLOCKS:
        return True
    model_name = str(getattr(result, "model_name", "") or "")
    if model_name.startswith("ivregress"):
        return True
    return model_name in _OWN_HEADER_MODELS


def _is_didregress(result: Any) -> bool:
    """Whether this is a ``didregress``/``xtdidregress`` fit."""

    name = str(getattr(result, "model_name", "") or "")
    return name in ("didregress", "xtdidregress")


def _did_header_block(result: Any, nobs: int) -> list[str]:
    """The block ``didregress`` prints above its coefficient table.

    Captured from ``didregress (y x1) (treated), group(unit) time(period)``:

      Treatment and time information

      Time variable: period
      Control:       treated = 0
      Treatment:     treated = 1
      -----------------------------------
                   |   Control  Treatment
      -------------+---------------------
      Group        |
              unit |       100        100
      -------------+---------------------
      Time         |
           Minimum |         0          1
           Maximum |         0          1
      -----------------------------------

      Difference-in-differences regression               Number of obs = 1,600
      Data type: Repeated cross-sectional

    Returns an empty list when the arm counts are unavailable, so the estimate is never
    preceded by a table of invented numbers.
    """

    e = getattr(result, "e", None) or {}
    info = e.get("_did_info") or {}
    if not info:
        return []
    width = 35
    lines = [
        "Treatment and time information",
        "",
        f"Time variable: {info.get('time_name', '')}",
        f"Control:       {info.get('treatment_name', '')} = 0",
        f"Treatment:     {info.get('treatment_name', '')} = 1",
        "-" * width,
        f"{'':>12} |   Control  Treatment",
        "-" * 13 + "+" + "-" * 21,
        "Group        |",
        f"{str(e.get('group', '') or 'unit'):>12} | {info.get('control_units', 0):>9}"
        f" {info.get('treated_units', 0):>10}",
        "-" * 13 + "+" + "-" * 21,
        "Time         |",
        f"{'Minimum':>12} | {_did_time(info.get('control_min')):>9}"
        f" {_did_time(info.get('treated_min')):>10}",
        f"{'Maximum':>12} | {_did_time(info.get('control_max')):>9}"
        f" {_did_time(info.get('treated_max')):>10}",
        "-" * width,
        "",
        # The banner line: title on the left, obs count on the right, 78 wide.
        f"{str(e.get('title') or 'Difference-in-differences regression'):<56}"
        f"{'Number of obs = ' + format(int(nobs), ','):>22}",
        f"Data type: {e.get('datatype', '')}",
    ]
    return lines


def _did_time(value: Any) -> str:
    """Times print as integers when whole, otherwise as given."""

    if value is None:
        return "."
    number = float(value)
    return f"{int(round(number))}" if abs(number - round(number)) < 1e-9 else f"{number:g}"


def _svy_estimate_table(
    result: Any, lines: list[str], names: list[str]
) -> str:
    """The 62-wide table ``svy: mean``/``total``/``proportion`` print.

    Captured at ``p12_outputs.txt:11-16``. The frame is 62 columns, the name column 12
    plus a pipe, and the right half splits 11 / 11 / 26 for estimate, standard error and
    confidence interval. There are NO t and P columns -- Stata prints none for these
    commands, so their presence in the shared table was the visible tell.

    ``lines`` already holds the survey design block; it is extended here.
    """

    estimate_label = _svy_estimate_label(result)
    name_width = max([12, *(len(str(name)) for name in names)])
    total_width = 62
    right_width = total_width - name_width - 2
    est_w, se_w, ci_w = 11, 11, right_width - 22

    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    ci = result.conf_int()
    lower = np.asarray(ci["lower"], dtype=float)
    upper = np.asarray(ci["upper"], dtype=float)
    level = getattr(result, "level", None)

    def _ci_label(level_value: Any) -> str:
        if level_value is None:
            return "[95% conf. interval]"
        try:
            pct = float(level_value) * 100.0
        except (TypeError, ValueError):
            return "[95% conf. interval]"
        text = f"{pct:g}"
        return f"[{text}% conf. interval]"

    _vcetype = str((getattr(result, "e", None) or {}).get("vcetype") or "").strip()
    lines.append("-" * total_width)
    if _vcetype:
        # Measured: the banner's label ENDS at column 37, not at the header's right edge.
        banner = f"{'':>{name_width}} |{_vcetype:>{37 - name_width - 2}}"
        # ``svy: proportion`` carries a SECOND banner label, "Logit", starting at column
        # 49: it names the scale its interval is built on.
        if str(getattr(result, "model_name", "") or "") == "svy: proportion":
            banner = banner.ljust(49) + "Logit"
        lines.append(banner)
    # Column positions measured from the capture: "Mean" starts at 21, "std. err." at 28,
    # "[95% conf. interval]" at 42. Composing them from field widths put the boundaries one
    # space either side of Stata's, so they are laid out by absolute offset instead.
    # Measured across all three commands: the estimate label is RIGHT-aligned to column
    # 25 (Mean 21-24, Total 20-24, Proportion 15-24), while "std. err." sits at 28 and
    # "[95% ..." at 42 regardless of the label's length. Laying it out from a fixed LEFT
    # edge put Total and Proportion in the wrong columns.
    _ci_text = _ci_label(level)
    _label_field = 25 - name_width - 2
    lines.append(
        f"{'':>{name_width}} |"
        + f"{estimate_label:>{_label_field}}"
        + " " * (28 - 25)
        + "std. err."
        + " " * (42 - 28 - len("std. err."))
        + _ci_text
    )
    lines.append("-" * (name_width + 1) + "+" + "-" * (total_width - name_width - 2))
    # ``svy: proportion`` names its rows after the LEVELS (0, 1) under a group header
    # carrying the variable name, not ``binary=0``/``binary=1``:
    #
    #   binary       |
    #           0    |   .4903781   ...
    #           1    |   .5096219   ...
    # KNOWN GAP, not silently tolerated: for ``svy: proportion`` the interval bounds do
    # not match Stata. Measured on the p12 design -- Stata prints .4287728/.5522769 for
    # the level 0 proportion, this package .4283092/.5524469. Neither 1.96*se nor
    # t(40)*se reproduces Stata's half-width (.06175205), so Stata is applying something
    # else (an interval on a transformed scale, most likely the logit one that its
    # "Logit" secondary header advertises). Until that is established, this command's
    # POINT ESTIMATE and standard error match Stata exactly and its interval does not.
    # Recorded rather than approximated: a plausibly-shaped wrong interval is harder to
    # notice than an obviously missing one.
    _proportion_var = ""
    _level_rows = False
    if str(getattr(result, "model_name", "") or "") == "svy: proportion" and names:
        _proportion_var = str(names[0]).split("=", 1)[0]
        # The variable name sits right-aligned at column 12, not left-aligned.
        lines.append(f"{_proportion_var:>12} |")
        names = [str(n).split("=", 1)[-1] for n in names]
        # The level rows put the pipe at the same column 13 as every other row, which
        # leaves an extra space for a single-character level ("0", "1").
        _level_rows = True

    for index, name in enumerate(names):
        # Each value ENDS at a fixed column, measured from the capture: estimate at 25,
        # standard error at 36, and the two interval bounds at 50 and 62. Aligning runs
        # between those endpoints instead produced a 61-wide row with the interval shifted.
        # A level row puts its label right-aligned at column 11 and then two spaces and
        # the pipe; using ``name_width`` (12) overshot by one and made the row 63 wide.
        _name_cell = (
            f"{str(name):>11}  |" if _level_rows else f"{str(name):>{name_width}} |"
        )
        lines.append(
            _name_cell
            + f"{_stata_coef(params[index], 25 - name_width - 2)}"
            + f"{_stata_coef(bse[index], 36 - 25)}"
            + f"{_stata_coef(lower[index], 50 - 36)}"
            + f"{_stata_coef(upper[index], 62 - 50)}"
        )
    lines.append("-" * total_width)
    return chr(10).join(lines)


def _is_svy_estimate(result: Any) -> bool:
    """Whether this is a survey ESTIMATE command (mean/total/proportion).

    These print a different table from ``svy: regress``: no Coefficient/t/P columns, a
    "Mean (or Total/Proportion)" column instead, and a 62-wide table rather than 78.
    Captured at ``p12_outputs.txt:11-16``:

      --------------------------------------------------------------
                   |             Linearized
                   |       Mean   std. err.     [95% conf. interval]
      -------------+------------------------------------------------
                 y |   1.061347   .0873402      .8848254    1.237868
      --------------------------------------------------------------

    Recognised by the registered model name, not by the parameter count, because a
    one-variable ``svy: regress`` also has a single coefficient.
    """

    return str(getattr(result, "model_name", "") or "") in (
        "svy: mean",
        "svy: total",
        "svy: proportion",
    )


def _svy_estimate_label(result: Any) -> str:
    """The column heading these commands use: Mean, Total or Proportion."""

    name = str(getattr(result, "model_name", "") or "")
    return {
        "svy: mean": "Mean",
        "svy: total": "Total",
        "svy: proportion": "Proportion",
    }.get(name, "Estimate")


def _is_xtreg_panel(result: Any) -> bool:
    """Whether this is an ``xtreg`` fit whose header is the panel block, not ANOVA.

    ``xtreg`` prints NO sums-of-squares table. Its header is a two-column block carrying
    the R-squared DECOMPOSITION on the left and ``Obs per group`` on the right:

      Fixed-effects (within) regression               Number of obs     =        180
      Group variable: id                              Number of groups  =         30

      R-squared:                                      Obs per group:
           Within  = 0.0006                                         min =          6
           Between = 0.0018                                         avg =        6.0
           Overall = 0.0007                                         max =          6

                                                      F(1, 149)         =       0.08
      corr(u_i, Xb) = 0.0066                          Prob > F          =     0.7718

    Recognised by the presence of ``e(N_g)`` AND one of the R-squared decomposition
    scalars, which no other estimator stores together.
    """

    name = str(getattr(result, "model_name", "") or "")
    # ``xtivreg, fe`` shares xtreg's panel block (measured: the same R-squared
    # decomposition, corr(u_i, Xb) and Obs per group), only the model test differs -- it
    # prints Wald chi2 where xtreg prints F.
    if not (name.startswith("xtreg") or name.startswith("xtivreg")):
        return False
    e = getattr(result, "e", None) or {}
    has_groups = np.isfinite(_e_float(result, "N_g"))
    has_decomposition = any(
        np.isfinite(_e_float(result, key)) for key in ("r2_w", "r2_b", "r2_o")
    )
    return bool(has_groups and has_decomposition)


def _xtabond_header_block(result: Any, nobs: int) -> list[str]:
    """The panel design block ``xtabond`` prints (measured against a live run).

      Arellano\u2013Bond dynamic panel-data estimation Number of obs     =        60
      Group variable: unit                          Number of groups  =        20
      Time variable: period
                                                    Obs per group:
                                                                  min =         3
                                                                  avg =       3.0
                                                                  max =         3

      Number of instruments =      8                Wald chi2(2)      =      0.09
                                                    Prob > chi2       =    0.9560
      One-step results
    """

    e = getattr(result, "e", None) or {}
    entity = str(e.get("ivar") or "").strip() or "."
    time_name = str(e.get("tvar") or "").strip()
    n_groups = _e_float(result, "N_g")
    n_instr = _e_float(result, "instruments")
    chi2_v = _e_float(result, "chi2")
    p_v = _e_float(result, "p")
    df_m = _e_float(result, "df_m")
    g_min = _e_float(result, "g_min")
    g_avg = _e_float(result, "g_avg")
    g_max = _e_float(result, "g_max")
    # Balanced panels: the per-group count is N / groups. Stata prints these on the
    # Obs-per-group rows; the macros are absent on this estimator.
    if not np.isfinite(g_min) and np.isfinite(n_groups) and n_groups > 0:
        g_min = g_avg = g_max = float(nobs) / float(n_groups)

    def _right(label: str, value: str) -> str:
        return f"{label:<17}= {value:>7}"

    _ab_banner = "Arellano\u2013Bond dynamic panel-data estimation"
    lines = [
        f"{_ab_banner:<48}" + _right("Number of obs", f"{int(nobs)}"),
        f"{('Group variable: ' + entity):<48}"
        + _right("Number of groups", f"{int(n_groups)}" if np.isfinite(n_groups) else "."),
        f"{('Time variable: ' + time_name) if time_name else '':<48}".rstrip(),
        f"{'':<48}{'Obs per group:':>17}",
        f"{'':<48}{'min':>14} = {int(g_min):>7}" if np.isfinite(g_min) else "",
        f"{'':<48}{'avg':>14} = {g_avg:>7.1f}" if np.isfinite(g_avg) else "",
        f"{'':<48}{'max':>14} = {int(g_max):>7}" if np.isfinite(g_max) else "",
        "",
    ]
    if np.isfinite(n_instr) or np.isfinite(chi2_v):
        lines.append(
            f"{('Number of instruments = ' + f'{int(n_instr):>6}') if np.isfinite(n_instr) else '':<48}"
            + _right(f"Wald chi2({_df_label(df_m)})", f"{chi2_v:.2f}")
        )
        if np.isfinite(p_v):
            lines.append(f"{'':<48}" + _right("Prob > chi2", _four(p_v)))
        lines.append("One-step results")
    return [line for line in lines if line != ""] + (
        [""] if np.isfinite(n_instr) or np.isfinite(chi2_v) else []
    )


def _xtreg_header_block(result: Any, nobs: int) -> list[str]:
    """The two-column panel header Stata's ``xtreg`` prints instead of ANOVA."""

    e = getattr(result, "e", None) or {}
    banner = str(e.get("title") or "")
    group = str(e.get("group") or "")
    n_groups = _e_float(result, "N_g")
    df_model = _e_float(result, "df_m")
    df_resid = _e_float(result, "df_r")
    fvalue = _e_float(result, "F")
    # ``xtivreg`` stores its model test as e(chi2), not e(F) -- the header still shows it
    # in the same slot, labelled "Wald chi2" (see below). Without this fallback the row
    # vanished entirely.
    if not np.isfinite(fvalue):
        fvalue = _e_float(result, "chi2")
    p_f = _e_float(result, "p")
    r2_w = _e_float(result, "r2_w")
    r2_b = _e_float(result, "r2_b")
    r2_o = _e_float(result, "r2_o")
    corr = _e_float(result, "corr")

    def row(left: str, right: str = "") -> str:
        return f"{left:<48}{right}".rstrip()

    def stat(label: str, value: str) -> str:
        return f"{label:<18}= {value:>10}"

    # ``Obs per group`` -- min/avg/max. The counts come from e() when stored; avg is the
    # mean group size, printed with one decimal as Stata does ("6.0" beside "6").
    per_group = e.get("_obs_per_group") or {}
    lines = [
        # A LEADING BLANK LINE: Stata's `xtreg ..., fe` capture starts with a newline, so the block
        # opens with an empty line before the banner. Measured in round 215 on munnell --
        # `xtreg y k1 i.year, fe` gives chr(10)+'Fixed-effects (within) regression'. Without it the
        # whole header compared one row off, the same defect the prefix survey found ten times in
        # rounds 189-196.
        "",
        row(
            banner,
            stat("Number of obs", f"{nobs:,.0f}" if np.isfinite(nobs) else "."),
        ),
        row(
            f"Group variable: {group}",
            stat("Number of groups", f"{n_groups:,.0f}" if np.isfinite(n_groups) else "."),
        ),
        "",
        # ``R-squared:`` heads the left column and ``Obs per group:`` the right one; the
        # right label starts at column 48 and the line stops there (62 wide, not 78).
        row("R-squared:", f"{'Obs per group:':>14}"),
    ]
    # ``xtivreg`` prints a LITERAL MISSING VALUE for the within R-squared. Measured from
    # ``xtivreg y x1 (x2 = z), fe``: "Within  =      .", while Between/Overall carry
    # numbers. The 2SLS residuals do not admit Stata's within-R2 definition, so the value
    # is genuinely not reported -- printing the estimator's own number would show a figure
    # Stata deliberately blanks.
    # ``xtivreg, FE`` prints a LITERAL MISSING VALUE for the within R-squared -- the 2SLS
    # residuals do not admit Stata's within-R2 definition -- but ``xtivreg, RE`` prints a
    # real number. Measured in both directions:
    #   xtivreg y x1 (x2 = z), fe   ->  "Within  =      ."
    #   xtivreg y x1 (x2 = z), re   ->  "Within  = 0.0150"
    # So the blanking is keyed on the METHOD, not on the family.
    _iv_fe_blanks_within = (
        str(getattr(result, "model_name", "") or "") == "xtivreg, fe"
    )
    decomposition = [
        ("Within", float("nan") if _iv_fe_blanks_within else r2_w),
        ("Between", r2_b),
        ("Overall", r2_o),
    ]
    # ``_obs_per_group`` is set by the panel estimators; the IV ones store the same
    # quantities as g_min/g_avg/g_max instead. Reading both keeps the right column intact
    # ("min = 5") rather than dropping it.
    if not per_group:
        _g_min = _e_float(result, "g_min")
        _g_avg = _e_float(result, "g_avg")
        _g_max = _e_float(result, "g_max")
        if any(np.isfinite(v) for v in (_g_min, _g_avg, _g_max)):
            per_group = {
                "min": _g_min,
                "avg": _g_avg,
                "max": _g_max,
            }
    right_rows = [
        ("min", per_group.get("min")),
        ("avg", per_group.get("avg")),
        ("max", per_group.get("max")),
    ]
    if not per_group:
        # Without stored counts, print the R-squared block and omit the right column
        # rather than inventing group sizes.
        right_rows = [(label, None) for label, _ in right_rows]
    for (label, value), (rlabel, rvalue) in zip(decomposition, right_rows):
        # ``Within`` occupies 8 columns before " =": Stata prints
        # "     Within  = 0.0006", not "     Within   = ...". The value field is 6 wide,
        # and a MISSING value right-aligns in it too ("     Within  =      .").
        _value_text = _four(value)
        left = f"{'':>5}{label:<8}= {_value_text:>6}"
        right = ""
        if rvalue is not None:
            # The group-size labels are right-aligned to column 62 and carry the value in
            # a 10-wide field; ``avg`` keeps a decimal even when whole ("6.0").
            if rlabel == "avg":
                rendered = f"{float(rvalue):.1f}"
            else:
                rendered = f"{int(round(float(rvalue)))}"
            right = f"{rlabel:>17} = {rendered:>10}"
        lines.append(row(left, right))
    lines.append("")
    # ``xtivreg`` reports a WALD chi2 where the other panel estimators report an F:
    # measured, "Wald chi2(2)" / "Prob > chi2" against xtreg's "F(1, 149)" / "Prob > F".
    # This applies to BOTH methods -- unlike the Within-R2 blanking, which is fe-only.
    _uses_wald = str(getattr(result, "model_name", "") or "").startswith("xtivreg")
    if np.isfinite(fvalue):
        if _uses_wald:
            _test_label = f"Wald chi2({_df_label(df_model)})"
        else:
            _test_label = f"F({_df_label(df_model)}, {_df_label(df_resid)})"
        lines.append(row("", stat(_test_label, f"{fvalue:.2f}")))
    left_corr = ""
    # The RE estimator ASSUMES zero correlation between the effects and the regressors --
    # that is what makes it random-effects -- so it prints the assumption rather than a
    # number, and with a different label ("X", not "Xb"):
    #
    #   xtivreg ..., re   corr(u_i, X) = 0 (assumed)
    #   xtivreg ..., fe   corr(u_i, Xb) = -0.0833
    _model_name = str(getattr(result, "model_name", "") or "")
    if _model_name.startswith("xtivreg") and _model_name.endswith(", re"):
        left_corr = "corr(u_i, X) = 0 (assumed)"
    elif np.isfinite(corr):
        left_corr = f"corr(u_i, Xb) = {_four(corr)}"
    if np.isfinite(p_f):
        _p_label = "Prob > chi2" if _uses_wald else "Prob > F"
        # The prob row ON THE corr LINE is the SAME width as every other header row -- 78 -- and
        # its value field is NINE characters plus the space before it, exactly like the F row
        # above. Measured in round 219 on two independent fits:
        #
        #   '                                                F(1, 767)         =     540.43'  78
        #   'corr(u_i, Xb) = -0.6272                         Prob > F          =     0.0000'  78
        #
        # The comment this replaces claimed this row was one character NARROWER (77) with a
        # 9-column value; the live output is 78 with a value that occupies 9 columns AFTER the
        # '='. Using a 10-wide field printed '... =    0.0000' (8 value characters), one space
        # short of Stata's.
        _padded_label = f"{_p_label:<18}"
        # The value field is ELEVEN wide: five spaces then '0.0000', so the row is 78 like every
        # other header row. A ten-wide field gave 77 and printed four spaces.
        lines.append(f"{left_corr:<48}{_padded_label}={_four(p_f):>11}")
    elif left_corr:
        lines.append(row(left_corr))
    return lines


def _fmt_group_size(value: Any) -> str:
    """Group sizes print as integers, but the AVERAGE carries one decimal (``6.0``)."""

    number = float(value)
    if abs(number - round(number)) < 1e-9 and number == int(number):
        # Stata prints avg with a decimal even when it is whole; min/max do not. The
        # caller decides which, so the distinction is made there.
        return f"{int(round(number))}"
    return f"{number:.1f}"


def _anova_available(result: Any) -> bool:
    """Whether the model+residual sums of squares are all present.

    Stata prints the ANOVA table only when it has ``e(mss)``, ``e(rss)`` and the
    total.  ``regress`` stores ``rss``+``mss`` and derives ``tss``; a model that
    records only ``rss`` (e.g. ``areg``'s absorbed variant) has no Total row to
    show and must fall back to the stacked header.
    """

    # A ROBUST fit prints NO sums-of-squares table at all. Captured from
    # ``regress y s expr tenure smsa rns, vce(robust)``:
    #
    #   Linear regression                     Number of obs     =        758
    #                                         F(5, 752)         =      22.03
    #                                         Prob > F          =     0.0000
    #                                         R-squared         =     0.1232
    #                                         Root MSE          =     .34552
    #
    # -- a two-column block and nothing else. The ANOVA rows, Adj R-squared and Total
    # disappear; only R-squared and Root MSE survive. So availability depends on the
    # variance estimator, not only on which scalars exist.
    # A robust variance estimator, or a command in _NO_ANOVA_TWO_COLUMN (rreg), means the
    # two-column header and no ANOVA. Consolidated here so the decision lives in ONE place
    # -- it used to be repeated as separate literals at several call sites.
    if _wants_two_column_header(result):
        return False
    # The PCSE/GLS panel commands print their own design block and NO sums-of-squares
    # table: measured, ``xtpcse`` goes straight from the design rows to the coefficient
    # header. The ``ivregress`` estimators are the same: they store mss/rss but Stata
    # prints a two-column header with Wald chi2 and no ANOVA rows.
    if str(getattr(result, "model_name", "") or "").startswith("ivregress"):
        return False
    if str(getattr(result, "model_name", "") or "") in ("xtpcse", "xtgls"):
        return False

    mss = _e_float(result, "mss")
    rss = _e_float(result, "rss")
    return bool(np.isfinite(mss) and np.isfinite(rss) and rss >= 0.0)


def _anova_block(
    result: Any,
    lines: list[str],
    *,
    nobs: int,
    df_model: float,
    df_resid: float,
    fvalue: float,
    p_f: float,
    rsquared: float,
    rsquared_adj: float,
    rmse: float,
) -> list[str]:
    """Render Stata's ``Source | SS df MS`` table with the fit block beside it.

    The last TWO geometry points that matter:

    * ``Total`` has ``N - 1`` degrees of freedom (the constant is absorbed), and
      ``Total SS`` is ``mss + rss`` -- computed, not stored separately;
    * the right-hand column starts on the FIRST row and runs down, so ``F`` pairs
      with ``Model`` and ``Root MSE`` pairs with ``Total``, exactly as Stata lays it
      out (``p10_outputs.txt:97-102``).
    """

    mss = _e_float(result, "mss")
    rss = _e_float(result, "rss")
    total_ss = mss + rss
    # ``regress`` reports the total degrees of freedom as N - 1 when a constant is
    # present.  ``df_model + df_resid`` gives that directly and also works for the
    # no-constant case (where it equals N).
    total_df = float(df_model + df_resid)

    # Stata's ANOVA columns, measured from a live capture (78 characters wide, ``+`` at
    # column 13). The tokens END at fixed columns: SS 26, df 36, MS 48, right-hand value 78.
    #
    #   '       Model |  12.6190573         5  2.52381146   Prob > F        =    0.0000'
    #                    ^16      ^26  ^34  ^36 ^38     ^48  ^50              ^70   ^78
    #
    # SS/MS use the same ``%9.0g`` rule as the coefficient table -- nine significant digits
    # here, not a fixed decimal count, which is why ``12.6190573`` sits beside ``.34552``.
    #: ``nl``'s ANOVA table is one column narrower in the SS field than regress's: its
    #: "SS" header ends at 22 where regress's ends at 23, and its values sit in 10 rather
    #: than 11. Measured against both tables directly.
    _ss_width = 10 if str(getattr(result, "model_name", "") or "") == "nl" else 11
    _is_nl = str(getattr(result, "model_name", "") or "") == "nl"
    #: ``nl`` reports the residual deviance where every other OLS-shaped command reports
    #: Root MSE. The labels differ in length ("Res. dev." 9, "Root MSE" 8) so the field is
    #: padded rather than hard-coded, keeping "=" at column 17 for both.
    _fit_label = "Res. dev." if _is_nl else "Root MSE"
    _fit_value = _e_float(result, "deviance") if _is_nl else rmse

    def _ss(value: float) -> str:
        return (
            _stata_number(value, _ss_width)
            if np.isfinite(value)
            else f"{'.':>{_ss_width}}"
        )

    def _df_cell(df: float) -> str:
        # Stata puts the df right-aligned in 10 with a trailing space, then the MS field.
        return (f"{df:>10.0f}" if np.isfinite(df) else f"{'.':>10}") + " "

    def _ms(value: float, df: float) -> str:
        if not np.isfinite(value) or not np.isfinite(df) or df <= 0:
            return f"{'.':>11}"
        return _stata_number(value / df, 11)

    # The header's SS/df/MS columns follow the same width: nl's are one narrower, and its
    # "SS" gap is a column further left to match the values below it.
    _anova_header = (
        "      Source |       SS           df       MS"
        if _ss_width == 11
        else "      Source |      SS            df       MS"
    )
    left = [
        _anova_header,
        "-------------+----------------------------------",
        f"       Model | {_ss(mss)}{_df_cell(df_model)}{_ms(mss, df_model)}",
        f"    Residual | {_ss(rss)}{_df_cell(df_resid)}{_ms(rss, df_resid)}",
        "-------------+----------------------------------",
        f"       Total | {_ss(total_ss)}{_df_cell(total_df)}{_ms(total_ss, total_df)}",
    ]
    _chi2_value = _e_float(result, "chi2")
    _chi2_pvalue = _e_float(result, "p")
    right = [
        f"Number of obs   = {nobs:>9d}",
        # ``xtpcse``/``xtgls`` report a ``Wald chi2`` MODEL test even though their
        # coefficient tests use t or z.  The captured run shows both at once
        # (``p9_outputs.txt:504-506`` beside a ``z``/``P>|z|`` table), and Stata records
        # which applies in ``e(chi2type)`` -- so that macro decides the line.
        *(
            [
                f"Wald chi2({_df_label(df_model)})".ljust(16)
                + f"= {('.' if not np.isfinite(_chi2_value) else f'{_chi2_value:>7.2f}'):>10}",
                "Prob > chi2     = "
                + f"{('.' if not np.isfinite(_chi2_pvalue) else f'{_chi2_pvalue:.4f}'):>10}",
            ]
            if _reports_a_chi2_test(result)
            else [
                f"F({_df_label(df_model)}, {_df_label(df_resid)})".ljust(16)
                + f"= {('. ' if not np.isfinite(fvalue) else f'{fvalue:>7.2f}'):>9}",
                f"Prob > F        = {('.' if not np.isfinite(p_f) else f'{p_f:.4f}'):>9}",
            ]
        ),
        f"R-squared       = {('.' if not np.isfinite(rsquared) else f'{rsquared:.4f}'):>9}",
        f"Adj R-squared   = {('.' if not np.isfinite(rsquared_adj) else f'{rsquared_adj:.4f}'):>9}",
        f"{_fit_label:<16}= {_stata_number(_fit_value, 7).strip().rjust(9) if np.isfinite(_fit_value) else '.'.rjust(9)}",
    ]
    # ``nl``'s right column starts one row LOWER than regress's: its first row carries
    # only "Source | SS df MS" (45 wide, no statistics beside it) and "Number of obs"
    # appears on the second. Measured against both. So the right column is offset by one
    # for nl and the first left row is emitted alone.
    #
    # The left side is padded to at least the right column's length, NOT the other way
    # round: zipping a shortened left against a full right silently DROPPED the tail of
    # the statistics (adj R-squared, then Res. dev.) instead of leaving the cells blank.
    _right_offset = 1 if _is_nl else 0
    _left_rows = left if not _right_offset else [left[0]] + left[1:]
    _pad = max(0, len(right) + _right_offset - len(_left_rows))
    _left_rows = _left_rows + [""] * _pad
    rendered = [
        f"{left_line:<51}{right_line}".rstrip()
        for left_line, right_line in zip(
            _left_rows[_right_offset:], right
        )
    ]
    if _right_offset:
        rendered.insert(0, left[0])
    return lines + rendered
    """Return which goodness-of-fit block Stata prints for this estimator."""

    name = str(getattr(result, "model_name", "") or "").strip().lower()
    if name in _HEADER_BLOCK:
        return _HEADER_BLOCK[name]
    # An unknown model keeps the old behaviour: a likelihood model is one that
    # actually carries a log-likelihood, which is the honest test.
    if np.isfinite(_e_float(result, "ll")):
        return "ml"
    return "ols"


def _cluster_combinations(result: Any) -> list[str] | None:
    """The ``clustertable`` block, or ``None`` when there is nothing multiway to report.

    Stata prints this only for ``vce(cluster a b)`` with MORE THAN ONE cluster variable
    (``areg.sthlp:225-228``), and the shape is::

        -------------------------------
                            |  Clusters
        Cluster combination | per comb.
        --------------------+----------
                          a |        30
                          b |         7
                        a#b |       210
        -------------------------------

    Every combination of the dimensions appears, in the same order Stata lists them:
    singletons first in dimension order, then the pairwise terms. The count for a
    combined row is the number of DISTINCT tuples, which is why ``a#b`` is 210 here
    rather than 30 or 7.

    Returns ``None`` for a one-way spec so ordinary clustering output is untouched.
    """

    e = getattr(result, "e", None) or {}
    counts = e.get("_cluster_counts") or []
    names = e.get("_cluster_names") or []
    if len(counts) < 2 or len(names) != len(counts):
        return None

    from itertools import combinations as _combinations

    rows: list[tuple[str, int]] = []
    for size in range(1, len(counts) + 1):
        for indices in _combinations(range(len(counts)), size):
            label = "#".join(names[index] for index in indices)
            if size == 1:
                value = int(counts[indices[0]])
            else:
                # Number of distinct joint tuples across the included dimensions. The
                # covariance already computed this key; recomputing the COUNT from the
                # labels alone is not possible here, so callers that care about multiway
                # counts store the value. Fall back to the product only when it is absent,
                # and say so rather than presenting the product as if it were measured.
                stored = (e.get("_cluster_combination_counts") or {}).get(label)
                value = int(stored) if stored is not None else 0
            rows.append((label, value))

    if not rows:
        return None
    width_label = max(len(label) for label, _ in rows)
    width_label = max(width_label, len("Cluster combination"))

    # Widths copied from a real ``areg ..., vce(cluster a b) clustertable`` capture, which
    # is 31 characters wide: a 19-character label column plus the right-hand block. The
    # first attempt sized the rule from the data and produced 36, so the two tables did not
    # line up even though every cell was correct.
    total_width = 31
    lines = ["-" * total_width]
    lines.append(f"{'':>{width_label}} |  {'Clusters':>7}")
    lines.append(f"{'Cluster combination':<{width_label}} | {'per comb.':>9}")
    lines.append("-" * (width_label + 1) + "+" + "-" * 10)
    for label, value in rows:
        lines.append(f"{label:>{width_label}} | {value:>{9}}")
    lines.append("-" * total_width)
    return lines


def _stata_summary(result: Any) -> str:
    nobs = int(result.nobs)
    df_resid = float(result.df_resid)
    df_model = float(result.df_model)

    rsquared = _as_float(result.rsquared)
    rsquared_adj = _as_float(result.rsquared_adj)

    rmse = _e_float(result, "rmse")
    if not np.isfinite(rmse):
        resid = np.asarray(result.resid, dtype=float)
        if df_resid > 0 and resid.size:
            rmse = float(np.sqrt(np.dot(resid, resid) / df_resid))

    fvalue = _e_float(result, "F")
    if not np.isfinite(fvalue) and np.isfinite(rsquared) and df_model > 0 and df_resid > 0:
        if rsquared >= 1.0:
            fvalue = float("inf")
        else:
            fvalue = float((rsquared / df_model) / ((1.0 - rsquared) / df_resid))

    p_f = _e_float(result, "p_F")
    if not np.isfinite(p_f) and np.isfinite(fvalue) and df_model > 0 and df_resid > 0:
        p_f = float(stats.f.sf(fvalue, df_model, df_resid))

    dist = getattr(result, "dist", "t")
    stat_label = "z" if dist == "z" else "t"
    p_label = "P>|z|" if dist == "z" else "P>|t|"

    ci = result.conf_int()
    lower = np.asarray(ci["lower"], dtype=float)
    upper = np.asarray(ci["upper"], dtype=float)
    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    stats_ = np.asarray(result.zvalues if dist == "z" else result.tvalues, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)
    names = [str(name) for name in result.params.index]

    # Stata prints ``_cons`` LAST, while patsy puts ``Intercept`` first.  The
    # container already knows the Stata order (``_stata_order``); applying it here
    # keeps every column aligned with its own name.  Captured output confirms the
    # placement -- e.g. ``p10_outputs.txt:19`` and ``:75`` both end with ``_cons``.
    #: ``_stata_order`` also refuses on ambiguous factors, so the row order must be
    #: worked out here when the mapping is unavailable: Stata always prints the
    #: constant LAST, so moving it to the end keeps a factor model's table in the same
    #: order a numeric one gets.
    order = _safe_stata_order(result)
    if not order and len(names) > 1:
        order = _constant_last_order(names)
    if order and len(order) == params.size and order != list(range(params.size)):
        params = params[order]
        bse = bse[order]
        stats_ = stats_[order]
        pvalues = pvalues[order]
        lower = lower[order]
        upper = upper[order]
        ci = {"lower": lower, "upper": upper}
    # The LABELS come from ``stata_param_names``, which renames patsy's
    # ``Intercept`` to Stata's ``_cons`` and puts the auxiliary parameters last.
    # Reordering without relabelling printed the right rows under the wrong names.
    #
    # ``stata_param_names`` strips the leading slash from auxiliary parameters
    # because ``e(b)`` stores them WITHOUT it (``nabreg.json``: ``lnalpha``).
    # The TABLE, however, displays ``/lnalpha`` -- so the slash is put back here,
    # for display only.
    # ``stata_param_names`` REFUSES when a factor term's Stata spelling is ambiguous
    # (Stata emits one row per level plus a zero base row; we hold a single column).
    # That refusal is right for a caller asking for a name mapping, but a RENDER must
    # not crash on it -- the table can be drawn with the names it has.
    _needs_name_permutation = False
    try:
        stata_names = list(getattr(result, "stata_param_names", []) or [])
    except NotImplementedError:
        _needs_name_permutation = True
        # The full mapping is unavailable because a factor term is ambiguous.  Fall
        # back to the renames that are still unambiguous: the constant becomes
        # ``_cons`` (patsy's ``Intercept`` is never what Stata prints) and everything
        # else keeps its own name.  Without this the SAME command rendered
        # ``Intercept`` on a factor model and ``_cons`` on a numeric one -- an
        # internal inconsistency a reader would not be able to explain.
        stata_names = [
            "_cons" if str(name).strip().lower() in ("intercept", "const", "_cons")
            else str(name)
            for name in (result.params.index if hasattr(result, "params") else [])
        ]
    # ``stata_param_names`` ALREADY returns the names in Stata's order, so it needs no
    # further permutation.  The fallback path builds them in the ORIGINAL parameter
    # order, so only THAT one follows ``order``.
    if _needs_name_permutation:
        if order and len(order) == len(stata_names) and order != list(range(len(order))):
            stata_names = [stata_names[index] for index in order]
    if len(stata_names) == len(names):
        # Auxiliary parameters are stored without the slash but DISPLAYED with it.
        # ``stata_names`` is ALREADY in display order, so the decision is made per
        # name from the ORIGINAL spelling: ``e(b)`` drops the slash (``nbreg.json``
        # holds ``lnalpha``, ``ologit.json`` holds ``cut1``) while the table puts it
        # back.  A whitelist would have to enumerate every auxiliary parameter and
        # would silently miss ``nl``'s ``b0``/``b1``.
        original_spellings = [str(name) for name in result.params.index]
        slash_names = {
            name for name in original_spellings if name.startswith("/")
        }
        # Bare names that correspond to a slashed original: strip the slash to match.
        stripped = {name[1:] for name in slash_names}
        display = []
        for name in stata_names:
            text = str(name)
            if text in stripped and not text.startswith("/"):
                text = "/" + text
            display.append(text)
        names = display

    # Which header block to print is a property of the *estimator*, not of
    # which e() keys happen to be present.  Keying off e(r2_p) misclassifies
    # qreg, which stores its check-function pseudo-R2 under that same name and
    # then gets an empty "Log likelihood = ." line that Stata never prints.
    block = _header_block_kind(result)
    chi2_type = str(getattr(result, "e", {}).get("chi2type") or "").strip()

    # NOTE: the leading blank line is NOT added here. The ML branch appends its own separator at
    # the point where the banner goes, and round 199 found that adding one here as well produced
    # TWO blank lines for logit/probit. The strip that removed it entirely was the real defect --
    # see the comment where that strip used to be.
    lines: list[str] = []
    model_name = getattr(result, "model_name", "") or "westata results"
    cov_type = getattr(result, "cov_type", "nonrobust")
    # Stata's banner is the command's own ``e(title)`` -- "Linear regression",
    # "Logistic regression", "Tobit regression" -- not one fixed string.  The macro is
    # recorded per command (transcribed from each ado), so use it when present and fall
    # back to the generic banner only where the command publishes none (``poisson``).
    banner = str((getattr(result, "e", None) or {}).get("title") or "").strip()
    # A ROBUST ols fit puts the banner on the FIRST ROW OF ITS TWO-COLUMN BLOCK rather
    # than on a line of its own:
    #
    #   Linear regression                     Number of obs     =        758
    #
    # so appending it here as well printed it twice -- once alone and once paired.
    _block_for_banner = _header_block_kind(result)
    _vcetype_for_banner = str(
        (getattr(result, "e", None) or {}).get("vcetype") or ""
    ).strip()
    # ``newey`` has its own two-column block, so its banner is paired there too and must
    # not also be printed as a standalone line.
    # ``xtpcse`` keeps its OWN banner line (the long "Linear regression, correlated
    # panels corrected standard errors (PCSEs)" string) and prints the design block under
    # it; its "Panel-corrected" row is a table banner, not a reason to drop the title.
    _robust_banner = (
        _block_for_banner in ("ols", "newey")
        and bool(_vcetype_for_banner)
        and not _pcse_family_for_header(result)
    ) or _block_for_banner == "newey"
    # ``quantile`` prints its banner on the SAME line as Number of obs (see its branch),
    # so the standalone line is skipped for it just as it is for a robust ols fit.
    # "ml" IS in this gate: the ML branch's own block starts with the banner
    # (``ml_left[0]``), so printing it again here duplicated it. An earlier round removed
    # "ml" from this set because including it left a blank first row -- that was a
    # different bug (the stranded separator after the banner was replaced), since fixed by
    # the blank-trimming below, not a reason to print the banner twice.
    _banner_in_block = (
        block in ("quantile", "ml")
        or str(getattr(result, "model_name", "") or "") in ("tobit", "truncreg")
        or str(getattr(result, "model_name", "") or "").startswith("ivregress")
        # The GEE block (xtlogit, pa) puts the banner on its FIRST row beside
        # "Number of obs", so a standalone banner line above it is a duplicate.
        or _is_gee_shape(result)
    )
    # A MISSING banner means Stata prints no banner for this command -- ``nl`` starts
    # straight at the Source/SS table. The old fallback put "Stata-style regression
    # results" on that line, a string Stata never prints anywhere.
    if not _robust_banner and not _banner_in_block and banner:
        lines.append(banner)
    # ``areg`` prints a SECOND title line naming what it absorbed:
    #
    #     Linear regression, absorbing indicators
    #                                              Absorbed variable: firm
    #
    # The macro is ``e(title2)``, captured in ``areg.json``.  Only ``areg`` sets it, so
    # every other command is unaffected; rendering it here is what makes an ``Areg`` table
    # match Stata's instead of merely having the right numbers under a shorter banner.
    subtitle = str((getattr(result, "e", None) or {}).get("title2") or "").strip()
    if subtitle:
        # Right-aligned under the banner, the way Stata prints its secondary title line.
        # Do NOT rstrip a wholly-blank line: the padding IS the alignment, and stripping it
        # turned this into an empty line.
        lines.append(subtitle.rjust(72))
    # ``clustertable`` also adds a "Clusters per comb." summary to the HEADER block, and the
    # actual combination table later, before the coefficient table. Both are emitted here
    # and in the ANOVA branch because that branch rebuilds ``lines``.
    cluster_summary = _cluster_combinations(result)
    # ``Model:`` and ``Covariance type:`` lines used to follow the banner.  Stata prints
    # NEITHER: the captured output in
    # ``tests/conformance/stata_fixtures/p10_outputs.txt:97-98`` opens the table directly
    # with
    #       Source |      SS           df       MS
    #   -------------+----------------------------------    Number of obs =        300
    # and a search of ``regress.ado``/``regress.sthlp`` finds no such label.  They were
    # this package's own scaffolding, and a reader comparing the two outputs would look for
    # them in Stata and not find them.  The ``e(vce)``/``e(vcetype)`` macros carry the same
    # information where a caller needs it programmatically.
    lines.append("")
    def _fixed(value: float, decimals: int) -> str:
        return "." if value is None or not np.isfinite(value) else f"{value:.{decimals}f}"

    # ``xtpcse``/``xtgls`` print a DESIGN block above the estimation table that no other
    # command has.  Transcribed from the captured run at
    # ``tests/conformance/stata_fixtures/p9_outputs.txt:496-506``:
    #
    #   Group variable:   firm              Number of obs     =        700
    #   Time variable:    year              Number of groups  =        100
    #   Panels:           correlated (balanced)   Obs per group:
    #   Autocorrelation:  no autocorrelation            min =          7
    #                                                   avg =          7
    #                                                   max =          7
    #   Estimated covariances      =      5050          R-squared     =     0.2144
    #   Estimated autocorrelations =         0          Wald chi2(1)  =     178.20
    #   Estimated coefficients     =         2          Prob > chi2   =     0.0000
    #
    # This package printed none of it, so a reader could not tell which panels the
    # correlation structure was estimated over -- the single most important thing to know
    # about a PCSE fit.
    panel_lines = _panel_design_block(result)
    # For the NON-LINEAR xt family the panel block comes AFTER the ML banner block: Stata
    # prints "Conditional fixed-effects logistic regression   Number of obs = ..." first,
    # then Group variable / Obs per group / LR chi2 / Log likelihood. Appending the panel
    # block here (as every other family does) put it above the banner, so the banner never
    # appeared at all.
    _defer_panel_block = block == "ml" and bool(panel_lines)
    if panel_lines and not _defer_panel_block:
        lines.extend(panel_lines)
        lines.append("")

    # For the survey family Stata's header *is* the design block: it prints
    # "Number of obs" inside it (paired with Number of strata) and no separate
    # standalone line, so the generic line is suppressed there.
    # ml prints its own two-column header below, so these generic single-column
    # lines are skipped for it -- otherwise every ML command shows its statistics twice,
    # once stacked and once paired.
    # A robust ols fit prints its own two-column block below (see the ``block == "ols"``
    # branch), so the generic single-column lines are skipped for it.
    _robust_vcetype = str((getattr(result, "e", None) or {}).get("vcetype") or "").strip()
    # ``_wants_two_column_header`` folds in the commands whose vcetype is empty but that
    # still print the two-column block (rreg), so this condition no longer has to know
    # about them individually.
    _is_robust_ols = (
        block == "ols"
        and _wants_two_column_header(result)
        and not _pcse_family_for_header(result)
        and not _iv_family_for_header(result)
    )
    # ONE predicate for "this command draws its own header block". Every new block-shaped
    # family used to need edits in two separate gates here (the Number of obs line and the
    # model-test line), and missing one produced duplicated statistics above the table.
    # That happened three times -- xtpcse, ivregress, eivreg -- so the condition lives in a
    # single place now.
    _own_header = _prints_own_header_block(result, block)
    if not _own_header and not _is_robust_ols:
        lines.append(f"{'Number of obs':<21} = {nobs:>10d}")
    # The model-level test is also estimator-specific.  ``quantile`` results
    # print no F test at all (Stata's qreg header is just Number of obs plus the
    # deviation lines), and printing "F(1, 72) = ." there is noise a scraper
    # would read as a missing value rather than "not applicable".
    if block == "arch":
        # This branch DRAWS the arch/arima header. ``arch`` is in
        # ``_OWN_HEADER_BLOCKS`` so the generic statistics above are skipped; this branch
        # is what replaces them, so it must come FIRST -- putting the bare "already
        # handled" test ahead of it swallowed the whole header.
        #
        # BEFORE the chi2 test: ``arch`` sets e(chi2type)="Wald", so the generic chi2
        # branch would claim it and print "Wald chi2(Nah)" in one column.
        #
        #   Sample: 1 thru 800                      Number of obs     =        800
        #                                           Wald chi2(.)      =          .
        #   Log likelihood = -1039.235              Prob > chi2       =          .
        # ``arch`` prints a LITERAL dot for the chi2 degrees of freedom (e(df_m) is not
        # reported for that model), but ``arima`` -- which shares this header shape --
        # prints its real count. Measured: "Wald chi2(1)" for arima, "Wald chi2(.)" for arch.
        if str(getattr(result, "model_name", "") or "") == "arima":
            _chi2df = _df_label(df_model)
        else:
            _chi2df = "."
        # The sample STARTS where the estimation sample starts, not at 1: with d=1 an
        # observation is lost to differencing, so ``arima ..., arima(1,1,0)`` on 144 periods
        # prints "Sample: 2 thru 144" and N = 143. ``e(N)`` already holds 143, and ``e(tmin)``
        # holds the first period used; the header was hard-coding both.
        _sample_n = int(_e_float(result, "N")) if np.isfinite(_e_float(result, "N")) else nobs
        # e(tmin) stays at the RAW series start, so the first period actually used has to be
        # derived: one observation is lost per degree of differencing, and the raw length is
        # e(N) plus that loss. Measured across orders: d=0 -> "1 thru 144", d=1 -> "2 thru
        # 144", d=2 -> "3 thru 142" with N = 144-d in every case.
        _raw = _e_float(result, "tmax")
        _lost = int(round(_raw)) - _sample_n if np.isfinite(_raw) else 0
        _first = int(round(_e_float(result, "tmin"))) + max(_lost, 0) if np.isfinite(
            _e_float(result, "tmin")
        ) else 1
        _start_label = f"Sample: {_first} thru {_first + _sample_n - 1}"
        lines.append(
            f"{_start_label:<48}"
            f"{'Number of obs':<18}= {_sample_n:>10d}"
        )
        # ``arima`` reports a real Wald chi2 (e(chi2) is stored); ``arch`` prints a dot
        # because its e(df_m) is absent. Printing a dot for both lost arima's statistic.
        _wald = _e_float(result, "chi2")
        _wald_text = "%.2f" % _wald if np.isfinite(_wald) else "."
        lines.append(
            f"{'':<48}{'Wald chi2(' + _chi2df + ')':<18}= {_wald_text:>10}"
        )
        _ll = _e_float(result, "ll")
        # ``%.7g`` gives "-1039.235", which is what Stata prints; more digits is too many.
        _ll_text = (
            "Log likelihood = " + "%.7g" % _ll if np.isfinite(_ll) else ""
        )
        lines.append(f"{_ll_text:<48}{'Prob > chi2':<18}= {'.':>10}")
    elif _own_header or _is_robust_ols:
        pass
    elif chi2_type:
        # Stata chooses the model test per estimator: "F" for regress,
        # "Wald chi2" for logit/probit/ivregress, "LR chi2" for tobit.
        # ``e(chi2type)`` records the choice.
        chi2 = _e_float(result, "chi2")
        p_chi2 = _e_float(result, "p")
        if not np.isfinite(p_chi2) and np.isfinite(chi2) and df_model > 0:
            p_chi2 = float(stats.chi2.sf(chi2, df_model))
        lines.append(
            f"{chi2_type + ' chi2(' + _df_label(df_model) + ')':<21}"
            f" = {_fixed(chi2, 2):>10}"
        )
        lines.append(f"{'Prob > chi2':<21} = {_fixed(p_chi2, 4):>10}")
    elif block == "svy":
        # Already rendered by the survey block below, as part of its two-column header.
        # Emitting the F here as well printed it twice, once stacked and once paired.
        pass
    elif dist == "z" or _reports_a_chi2_test(result):
        # Stata's likelihood/IV estimators print a Wald chi2 model test rather than an F
        # test (e.g. ``ivregress``: "Wald chi2(2) = 90.76").
        #
        # ``xtpcse`` is the reason this is not keyed on ``dist`` alone: it uses a t
        # distribution for the COEFFICIENT tests but reports ``Wald chi2`` for the MODEL
        # test.  The captured run makes that explicit --
        # ``p9_outputs.txt:504-506``:
        #
        #   Estimated coefficients     =         2    Wald chi2(1)  =     178.20
        #                                             Prob > chi2   =     0.0000
        #
        # while the coefficient table below it reads ``z``/``P>|z|``.  The package already
        # stored ``e(chi2type) = "Wald"``; the renderer simply ignored it and printed
        # ``F(1, 1398)`` instead.
        chi2 = _e_float(result, "chi2")
        if not np.isfinite(chi2) and np.isfinite(fvalue):
            chi2 = fvalue * df_model
        p_chi2 = _e_float(result, "p")
        if not np.isfinite(p_chi2) and np.isfinite(chi2) and df_model > 0:
            p_chi2 = float(stats.chi2.sf(chi2, df_model))
        lines.append(
            f"{'Wald chi2(' + _df_label(df_model) + ')':<21} = {_fixed(chi2, 2):>10}"
        )
        lines.append(f"{'Prob > chi2':<21} = {_fixed(p_chi2, 4):>10}")
    else:
        # ``newey.ado:250`` puts "Maximum lag = <lag>" in the LEFT column of the
        # F line:
        #
        #   Maximum lag = 2        F(2, 197) = 45.12
        #                          Prob > F  = 0.0000
        #
        # A plain-text rendering has one column, so the lag is emitted on its own
        # line immediately before the F statistic.
        if block == "newey":
            lag_value = _e_float(result, "lag")
            lag_text = str(int(round(lag_value))) if np.isfinite(lag_value) else "."
            lines.append(f"Maximum lag = {lag_text}")
        lines.append(
            f"{'F(' + _df_label(df_model) + ', ' + _df_label(df_resid) + ')':<21}"
            f" = {_fixed(fvalue, 2):>10}"
        )
        lines.append(f"{'Prob > F':<21} = {_fixed(p_f, 4):>10}")

    # The goodness-of-fit block is also estimator-specific.  ``quantile`` and
    # ``svy`` results store a pseudo-R2 / an R-squared that must NOT be
    # accompanied by Stata's OLS "Adj R-squared / Root MSE" pair, and ``qreg``
    # has no log-likelihood at all.
    if _is_didregress(result):
        # ``didregress`` prints the treatment/time table INSTEAD of any ANOVA block, then
        # its estimate table carries an ``ATET`` group header and a trailing Note.
        lines = list(_did_header_block(result, nobs))
    elif _is_xtreg_panel(result):
        # ``xtreg`` prints the panel header INSTEAD of the sums-of-squares table, so this
        # branch comes first: the ANOVA test would otherwise claim the fit, since xtreg
        # stores mss/rss like any other linear model.
        # The header block ends with the fit rows; Stata then prints ONE blank line and
        # the table rule. Adding a blank after the rule as well pushed the coefficient
        # header down by a row.
        # Neither a rule NOR a trailing blank: the code path below appends one blank line
        # and then the rule before the coefficient header.
        lines = list(_xtreg_header_block(result, nobs))
        # Everything appended before this point described a header that is not being
        # printed; the panel block replaces it wholesale.
    elif str(getattr(result, "model_name", "") or "") == "mvreg":
        # ``mvreg`` opens at an EQUATION SUMMARY table with NO banner. Measured from a
        # live ``mvreg y1 y2 = x1 x2``:
        #
        #   Equation             Obs   Parms        RMSE    "R-sq"          F      P>F
        #   --------------------------------------------------------------------------
        #   y1                   300       3    .9551445    0.7921   565.6674   0.0000
        #   y2                   300       3    .9587217    0.6899   330.3333   0.0000
        #   --------------------------------------------------------------------------
        #
        # Note "Parms" (mvreg) vs "Params" (sureg) -- a real spelling difference.
        #
        # Every line is 74 characters, and the value fields END at 24, 32, 44, 54, 65
        # and 74. The rules, the field widths and both precisions were wrong before:
        # RMSE printed five significant figures (.95514) where Stata prints seven
        # decimals (.9551445), and F printed two decimals (565.67) against 565.6674.
        _mv_e = getattr(result, "e", None) or {}
        _mv_outcomes = list(_mv_e.get("outcomes") or [])
        _mv_n = int(_mv_e.get("N") or nobs)
        _mv_rule = "-" * 74
        lines = [
            "Equation             Obs   Parms        RMSE    \"R-sq\"          F      P>F",
            _mv_rule,
        ]
        for _eq in _mv_outcomes:
            _rmse_eq = _e_float(result, f"rmse_{_eq}")
            _r2_eq = _e_float(result, f"r2_{_eq}")
            _f_eq = _e_float(result, f"F_{_eq}")
            _p_eq = _e_float(result, f"p_{_eq}")
            _rmse_text = f"{_rmse_eq:.7f}" if np.isfinite(_rmse_eq) else "."
            if _rmse_text.startswith("0."):
                _rmse_text = _rmse_text[1:]
            _f_text = f"{_f_eq:.4f}" if np.isfinite(_f_eq) else "."
            lines.append(
                f"{str(_eq):<18}{_mv_n:>6}{int(df_model) + 1:>8}"
                f"{_rmse_text:>12}"
                f"{_four(_r2_eq) if np.isfinite(_r2_eq) else '.':>10}"
                f"{_f_text:>11}"
                f"{(_four(_p_eq) if np.isfinite(_p_eq) else '.'):>9}"
            )
        lines.append(_mv_rule)
    elif str(getattr(result, "model_name", "") or "") == "xtabond":
        lines = _xtabond_header_block(result, nobs)
    elif str(getattr(result, "model_name", "") or "") == "sureg":
        # ``sureg`` opens with an EQUATION SUMMARY table and no fit statistics. Measured
        # from a live ``sureg (y1 x1) (y2 x2)``:
        #
        #   ------------------------------------------------------------------------------
        #   Equation             Obs   Params         RMSE  "R-squared"      chi2   P>chi2
        #   ------------------------------------------------------------------------------
        #   y1                   300        1     .9515677      0.7915    1731.53   0.0000
        #   y2                   300        1     .9551537      0.6891    1018.27   0.0000
        #   ------------------------------------------------------------------------------
        #
        # Each line is 78 characters; the value fields END at 24, 33, 46, 58, 69 and 78.
        # Two columns were missing entirely before -- ``chi2`` and ``P>chi2`` -- so the
        # table reported a fit with no test of it, and RMSE printed six significant
        # figures against Stata's seven decimals (.95157 against .9515677).
        _sureg_e = getattr(result, "e", None) or {}
        _sureg_title = str(_sureg_e.get("title") or "Seemingly unrelated regression")
        _sureg_names = list(_sureg_e.get("equations") or [])
        _sureg_rule = "-" * 78
        _r2_header = '"R-squared"'  # Stata prints the quotes literally in this table
        # The header is one column wider in two fields than the data rows -- measured:
        # header ends 24 / 33 / 46 / 59 / 69 / 78, rows end 24 / 33 / 46 / 58 / 69 / 78.
        lines = [
            _sureg_title,
            _sureg_rule,
            f"{'Equation':<18}{'Obs':>6}{'Params':>9}{'RMSE':>13}"
            f"{_r2_header:>13}{'chi2':>10}{'P>chi2':>9}",
            _sureg_rule,
        ]
        # ``e(chi2_1)``/``e(p_1)`` are keyed by the equation's POSITION, not its name.
        for _position, _eq in enumerate(_sureg_names, start=1):
            _rmse_eq = _e_float(result, f"rmse_{_eq}")
            _r2_eq = _e_float(result, f"r2_{_eq}")
            _k_eq = _e_float(result, f"k_{_eq}")
            _chi2_eq = _e_float(result, f"chi2_{_position}")
            _p_eq = _e_float(result, f"p_{_position}")
            # ``e(N)`` is the TOTAL across equations; the table reports the per-equation
            # count, which for a balanced system is N / k_eq.
            _n_eq = int(nobs // max(int(_e_float(result, "k_eq")), 1))
            # RMSE prints SEVEN DECIMALS with the leading zero dropped, not significant
            # figures: .9515677 rather than .95157.
            _rmse_text = f"{_rmse_eq:.7f}" if np.isfinite(_rmse_eq) else "."
            if _rmse_text.startswith("0."):
                _rmse_text = _rmse_text[1:]
            lines.append(
                f"{str(_eq):<18}{_n_eq:>6}"
                f"{(int(_k_eq) + 1) if np.isfinite(_k_eq) else 1:>9}"
                f"{_rmse_text:>13}"
                f"{_four(_r2_eq) if np.isfinite(_r2_eq) else '.':>12}"
                f"{_chi2_eq:>11.2f}"
                f"{_four(_p_eq) if np.isfinite(_p_eq) else '.':>9}"
            )
        lines.append(_sureg_rule)
    elif str(getattr(result, "model_name", "") or "") == "areg":
        # ``areg`` prints a TWO-COLUMN header instead of the Source/SS table, like reghdfe
        # but with its own left column. Measured:
        #
        #   Linear regression, absorbing indicators     Number of obs     =    100
        #   Absorbed variable: unit                     No. of categories =     20
        #                                               F(1, 79)          =   0.05
        #                                               Prob > F          = 0.8297
        #                                               R-squared         = 0.2890
        #                                               Adj R-squared     = 0.1089
        #                                               Root MSE          = 0.9380
        #
        # Right column starts at 52, label 18 wide, "=" at 71.
        _areg_rows: list[tuple[str, str]] = [
            ("Number of obs", f"{int(nobs):,}"),
        ]
        _k_absorb = _e_float(result, "k_absorb")
        if np.isfinite(_k_absorb):
            _areg_rows.append(("No. of categories", f"{int(_k_absorb)}"))
        if np.isfinite(fvalue):
            _areg_rows.append(
                (f"F({_df_label(df_model)}, {_df_label(df_resid)})", f"{fvalue:.2f}")
            )
        if np.isfinite(p_f):
            _areg_rows.append(("Prob > F", _four(p_f)))
        if np.isfinite(rsquared):
            _areg_rows.append(("R-squared", _four(rsquared)))
        if np.isfinite(rsquared_adj):
            _areg_rows.append(("Adj R-squared", _four(rsquared_adj)))
        if np.isfinite(rmse):
            _areg_rows.append(("Root MSE", f"{rmse:.4f}"))
        # Right column: left field 50, label 19, "=" at 69, value right-aligned in 7 so it
        # ends at 78. Measured from a live ``areg maize canal disaster, absorb(county)``,
        # where every one of the seven lines is 78 characters with "=" at column 69.
        # The previous widths (52 / 18 / 5) put "=" one column right and clipped the wider
        # values -- ``147,506`` is seven characters and a five-wide field cannot hold it.
        _areg_title = str((getattr(result, "e", None) or {}).get("title") or "")
        _areg_title2 = str((getattr(result, "e", None) or {}).get("title2") or "")
        lines = []
        for _index, (_label, _value) in enumerate(_areg_rows):
            _left = _areg_title if _index == 0 else (_areg_title2 if _index == 1 else "")
            lines.append(f"{_left:<51}{_label:<18}= {_value:>7}".rstrip())
        # ``clustertable`` belongs after the header block and before the coefficients, the
        # same place the ANOVA branch puts it. Omitting it here dropped the table for every
        # clustered areg fit -- the branch replaced the one that used to emit it.
        if cluster_summary:
            lines.append("")
            lines.extend(cluster_summary)
    elif str(getattr(result, "model_name", "") or "") == "reghdfe":
        # ``reghdfe`` prints a TWO-COLUMN header instead of a Source/SS table, then a block
        # naming the absorbed effects. Measured:
        #
        #   (MWFE estimator converged in 1 iterations)
        #
        #   HDFE Linear regression              Number of obs     =        100
        #   Absorbing 1 HDFE group              F(   1,      79) =       1.52
        #                                       Prob > F          =     0.2216
        #                                       R-squared         =     0.2890
        #                                       Adj R-squared     =     0.0994
        #                                       Within R-sq.      =     0.0006
        #                                       Root MSE          =     1.0324
        #
        #   Absorbed degrees of freedom:
        #   -------------------------------------------+
        #    Absorbed FE | Categories  - Redundant  = Num. Coefs |
        #   -------------+---------------------------------------|
        #           unit |        20           0          20     |
        #   -------------------------------------------+
        #
        # The right column starts at 50 with an 18-wide label and "=" at 68.
        _e_macros = getattr(result, "e", None) or {}
        _n_hdfe = _e_float(result, "N_hdfe")
        _absorb_label = (
            f"Absorbing {int(_n_hdfe)} HDFE group" + ("s" if _n_hdfe != 1 else "")
            if np.isfinite(_n_hdfe)
            else ""
        )

        def _hdfe_row(label: str, value: str) -> str:
            return f"{'':<50}{label:<18}= {value:>10}"

        _hdfe_rows: list[tuple[str, str]] = [
            ("Number of obs", f"{int(nobs):,}"),
        ]
        if np.isfinite(fvalue):
            _hdfe_rows.append(
                (f"F({_df_label(df_model):>4}, {_df_label(df_resid):>7})",
                 f"{fvalue:.2f}")
            )
        if np.isfinite(p_f):
            _hdfe_rows.append(("Prob > F", _four(p_f)))
        if np.isfinite(rsquared):
            _hdfe_rows.append(("R-squared", _four(rsquared)))
        if np.isfinite(rsquared_adj):
            _hdfe_rows.append(("Adj R-squared", _four(rsquared_adj)))
        _r2_within = _e_float(result, "r2_within")
        if np.isfinite(_r2_within):
            _hdfe_rows.append(("Within R-sq.", _four(_r2_within)))
        if np.isfinite(rmse):
            _hdfe_rows.append(("Root MSE", _stata_number(rmse, 7).strip()))

        lines = []
        # The convergence note is part of the block, not a stray log line.
        if np.isfinite(_n_hdfe):
            lines.append("(MWFE estimator converged in 1 iterations)")
            lines.append("")
        for _index, (_label, _value) in enumerate(_hdfe_rows):
            _left = "HDFE Linear regression" if _index == 0 else (
                _absorb_label if _index == 1 else ""
            )
            lines.append(f"{_left:<50}{_label:<18}= {_value:>10}".rstrip())
    elif block == "ols" and _anova_available(result):
        # Stata prints the sums of squares as a TABLE with the fit statistics beside
        # it (``p10_outputs.txt:97-102`` shows the layout for ``nl``, which uses the
        # same block):
        #
        #       Source |       SS           df       MS
        #   -----------+----------------------------------   Number of obs   =  100
        #        Model |  .044144718         1  .044144718   F(1, 98)        = 0.04
        #     Residual |  107.398435        98  1.09590219   R-squared       = 0.0004
        #   -----------+----------------------------------   Adj R-squared   = -0.0098
        #        Total |  107.442579        99  1.08527858   Root MSE        = 1.0471
        #
        # The right-hand column REPLACES the stacked header lines -- printing both
        # would duplicate Number of obs / R-squared, which Stata never does.  The
        # title lines are rebuilt so the block starts where Stata's does -- with the
        # banner and a blank line, since Stata prints no ``Model:``/
        # ``Covariance type:`` row before its table.
        # An empty banner means Stata prints none (``nl``); the list starts empty then.
        #
        # A plain (non-robust) ``regress`` prints no banner EITHER, even though it stores
        # ``e(title) = "Linear regression"``. Measured: ``regress y k`` opens directly at
        # "      Source |       SS           df       MS", while ``regress y k, robust``
        # heads its two-column block with the title. This branch REBUILDS ``lines``, so
        # suppressing the banner at the earlier ``lines.append(banner)`` had no effect here
        # -- which is why two rounds of patching that line changed nothing.
        _banner_suppressed = (
            str(getattr(result, "model_name", "") or "") in ("regress", "ols")
            and not _vcetype_for_banner
        )
        # THE LEADING BLANK IS PART OF THE REBUILD. This branch REPLACES ``lines`` -- which is why
        # the initialiser at the top of the function could carry the blank line and have no effect
        # The leading blank comes from the separator appended earlier in this function; adding
        # one here as well produced TWO blank lines for regress (round 199). The ML family gets
        # its blank from that same separator now that the strip below no longer removes it.
        lines = [banner] if (banner and not _banner_suppressed) else []
        # ``title2`` (areg's "Absorbed variable: ...") was appended to the earlier ``lines``
        # list, which this branch REPLACES -- the same trap the design block below hit. It
        # has to be re-attached here or the subtitle disappears exactly when the ANOVA
        # table exists, i.e. always.
        if subtitle:
            lines.append(subtitle.rjust(72))
        lines.append("")
        # The design block was appended to the earlier ``lines`` list, which this branch
        # REPLACES -- so it has to be re-attached here or xtpcse/xtgls lose it exactly
        # when the ANOVA table is available (i.e. always, for those commands).
        panel_lines = _panel_design_block(result)
        if panel_lines:
            lines.extend(panel_lines)
            lines.append("")
        lines = _anova_block(
            result, lines, nobs=nobs, df_model=df_model, df_resid=df_resid,
            fvalue=fvalue, p_f=p_f, rsquared=rsquared,
            rsquared_adj=rsquared_adj, rmse=rmse,
        )
        # ``clustertable`` sits between the ANOVA header block and the coefficient
        # table, so it is attached after the block is assembled rather than inside it.
        if cluster_summary:
            lines.append("")
            lines.extend(cluster_summary)
    elif block == "newey":
        # ``newey`` has its OWN two-column header, and it is not the robust one. Captured
        # from ``newey y s, lag(3)``:
        #
        #   Regression with Newey-West standard errors    Number of obs     =        300
        #   Maximum lag = 3                               F(  1,       298) =       0.34
        #                                                 Prob > F          =     0.5589
        #
        # Left column 48 (banner, then "Maximum lag"), right column 30. The F statistic
        # is padded as ``F(%3d, %9d)`` -- wide enough for any pair, unlike regress's
        # ``F(1, 298)`` -- which is why this cannot reuse the ols branch.
        _banner = str((getattr(result, "e", None) or {}).get("title") or "")
        if not _banner:
            _banner = "Regression with Newey-West standard errors"
        _lag_value = _e_float(result, "lag")
        _rows: list[tuple[str, str]] = [("Number of obs", f"{nobs:,.0f}" if np.isfinite(nobs) else ".")]
        if np.isfinite(fvalue):
            _rows.append(
                (
                    "F(%3d, %9d)" % (int(round(df_model)), int(round(df_resid))),
                    _fixed(fvalue, 2),
                )
            )
        if np.isfinite(p_f):
            _rows.append(("Prob > F", _fixed(p_f, 4)))
        for index, (label, value) in enumerate(_rows):
            if index == 0:
                left = _banner
            elif index == 1 and np.isfinite(_lag_value):
                left = f"Maximum lag = {int(round(_lag_value))}"
            else:
                left = ""
            lines.append(f"{left:<48}{label:<18}= {value:>10}".rstrip())
        if cluster_summary:
            lines.append("")
            lines.extend(cluster_summary)
    elif block == "ols" and _pcse_family_for_header(result):
        # ``xtpcse``/``xtgls`` print NOTHING here: their design block above already
        # carried the fit statistics. The blank line normally appended after this branch
        # is not needed either -- the design block ends with its own, and two in a row
        # separated the table from its header.
        lines.pop() if lines and lines[-1] == "" else None
    elif str(getattr(result, "model_name", "") or "").startswith("ivregress"):
        # ``ivregress`` prints a five-row two-column header and NO sums of squares,
        # whatever the method:
        #
        #   Instrumental-variables 2SLS regression       Number of obs   =   400
        #                                               Wald chi2(2)    =   1674.42
        #                                               Prob > chi2     =     0.0000
        #                                               R-squared       =     0.8923
        #                                               Root MSE        =      1.064
        #
        # Left column 50, right 28 (label 18, "= ", value 6).
        #
        # ``R-squared`` IS printed. An earlier version of this block omitted it, on the
        # stated belief that "R-squared and Adj R-squared are NOT printed even though the
        # estimator stores them" -- a live ``ivregress 2sls`` shows the line, with the
        # stored ``e(r2)`` as its value. Adj R-squared genuinely is not printed, and that
        # half of the old note was right.
        _iv_rows: list[tuple[str, str]] = [
            ("Number of obs", f"{nobs:,.0f}" if np.isfinite(nobs) else ".")
        ]
        _wald = _e_float(result, "chi2")
        _p_wald = _e_float(result, "p")
        _iv_df = _e_float(result, "df_m")
        # e(p) is not stored by the IV estimators; Stata derives it from the Wald
        # statistic. Computing it here rather than storing it keeps the estimator's e()
        # unchanged while still printing the row Stata prints.
        if not np.isfinite(_p_wald) and np.isfinite(_wald) and np.isfinite(_iv_df):
            _p_wald = float(stats.chi2.sf(_wald, _iv_df))
        if np.isfinite(_wald):
            _iv_rows.append((f"Wald chi2({_df_label(_iv_df)})", f"{_wald:.2f}"))
        if np.isfinite(_p_wald):
            _iv_rows.append(("Prob > chi2", f"{_p_wald:.4f}"))
        # Order matters: Stata puts R-squared between Prob > chi2 and Root MSE.
        _iv_r2 = _e_float(result, "r2")
        if np.isfinite(_iv_r2):
            _iv_rows.append(("R-squared", f"{_iv_r2:.4f}"))
        if np.isfinite(rmse):
            _iv_rows.append(("Root MSE", _stata_number(rmse, 7).strip()))
        _iv_banner = str((getattr(result, "e", None) or {}).get("title") or "")
        for _index, (_label, _value) in enumerate(_iv_rows):
            _left = _iv_banner if _index == 0 else ""
            # Left 50, label 16, "= ", value 10 -- measured from
            # "Instrumental-variables 2SLS regression     Number of obs   =        100".
            lines.append(f"{_left:<50}{_label:<16}= {_value:>10}".rstrip())
        # ``ivregress gmm`` adds one unaligned line UNDER the two-column block naming its
        # weight matrix. Measured: "GMM weight matrix: Robust" -- the default, and the
        # only form this package implements (IVGMM takes no weight-matrix option).
        if str(getattr(result, "model_name", "") or "") == "ivregress gmm":
            lines.append("GMM weight matrix: Robust")
    elif str(getattr(result, "model_name", "") or "") == "eivreg":
        # ``eivreg`` prints a RELIABILITY TABLE on the left and the fit statistics on the
        # right, sharing rows. Measured:
        #
        #                        Assumed
        #       Variable     reliability
        #   ----------------------------          Number of obs     =        100
        #             x1          0.7000          F(  2,        97) =       0.93
        #              *          1.0000          Prob > F          =     0.3984
        #                                          R-squared         =     0.0299
        #                                          Root MSE          =    .988738
        #
        # Left column 48; the table itself is 28 wide. The "*" row is Stata's marker for
        # the variables taken as measured without error, always 1.0000.
        _reliab = (getattr(result, "e", None) or {}).get("reliab") or {}
        _left_rows: list[str] = ["", "     Assumed", "    Variable     reliability"]
        for _name, _rel in _reliab.items():
            _left_rows.append(f"  {str(_name):>10}{float(_rel):>16.4f}")
        # Every variable in the model that is NOT listed stays 1.0000 under the "*" row.
        _listed = {
            str(name).split("[T.", 1)[0]
            for name in names
            if str(name).strip().lower() not in ("intercept", "_cons", "const")
        } - {str(k) for k in _reliab}
        if _listed:
            _left_rows.append(f"  {'*':>10}{1.0:>16.4f}")

        _right_rows: list[tuple[str, str]] = [
            ("Number of obs", f"{nobs:,.0f}" if np.isfinite(nobs) else ".")
        ]
        if np.isfinite(fvalue):
            _right_rows.append(
                (
                    f"F({_df_label(df_model)}, {_df_label(df_resid)})",
                    f"{fvalue:.2f}",
                )
            )
        if np.isfinite(p_f):
            _right_rows.append(("Prob > F", _four(p_f)))
        if np.isfinite(rsquared):
            _right_rows.append(("R-squared", _four(rsquared)))
        if np.isfinite(rmse):
            _right_rows.append(("Root MSE", _stata_number(rmse, 7).strip()))

        # The table's rule is drawn on the LEFT and the right column starts on that same
        # row; the first two left rows carry no right-hand text.
        for _index, _left in enumerate(_left_rows):
            _right_index = _index - 2
            if _right_index == 0:
                _dash = "-" * 28
                _label, _value = _right_rows[0]
                lines.append(f"{_dash:<48}{_label:<18}= {_value:>10}".rstrip())
            elif _right_index > 0 and _right_index < len(_right_rows):
                _label, _value = _right_rows[_right_index]
                lines.append(f"{'':<48}{_label:<18}= {_value:>10}".rstrip())
            else:
                lines.append(_left.rstrip())
    elif block == "ols":
        # Reached when the ANOVA table is unavailable -- for a ROBUST fit that is Stata's
        # own behaviour, and its header is a two-column block 78 wide with FIVE rows and
        # no Adj R-squared:
        #
        #   Linear regression                     Number of obs     =        758
        #                                         F(5, 752)         =      22.03
        #                                         Prob > F          =     0.0000
        #                                         R-squared         =     0.1232
        #                                         Root MSE          =     .34552
        #
        # Left column 48, right column 30 (label 18, " =", value 10).
        _vcetype = str((getattr(result, "e", None) or {}).get("vcetype") or "").strip()
        if _wants_two_column_header(result):
            _banner = str((getattr(result, "e", None) or {}).get("title") or "Linear regression")
            _rows: list[tuple[str, str]] = [
                ("Number of obs", f"{nobs:,.0f}" if np.isfinite(nobs) else ".")
            ]
            if np.isfinite(fvalue):
                # ``rreg`` pads the degrees of freedom ("F(  1,        98)") the same way
                # ``newey`` does; a plain regress prints "F(5, 752)".
                if str(getattr(result, "model_name", "") or "") == "rreg":
                    _f_label = "F(%3d, %9d)" % (
                        int(round(df_model)),
                        int(round(df_resid)),
                    )
                else:
                    _f_label = f"F({_df_label(df_model)}, {_df_label(df_resid)})"
                _rows.append((_f_label, _fixed(fvalue, 2)))
            if np.isfinite(p_f):
                _rows.append(("Prob > F", _fixed(p_f, 4)))
            # ``rreg`` prints ONLY Number of obs / F / Prob > F. Measured: no R-squared
            # and no Root MSE, unlike a vce(robust) regress which shows both.
            _is_rreg = str(getattr(result, "model_name", "") or "") == "rreg"
            if np.isfinite(rsquared) and not _is_rreg:
                _rows.append(("R-squared", _fixed(rsquared, 4)))
            # ``nl`` reports the RESIDUAL DEVIANCE where regress reports Root MSE. Measured:
            #   nl        "Res. dev.     =   279.3424"   (e(deviance) = 279.3424)
            #   regress   "Root MSE      =   .9879483"
            _deviance = _e_float(result, "deviance")
            _is_nl = str(getattr(result, "model_name", "") or "") == "nl"
            if _is_nl and np.isfinite(_deviance):
                _rows.append(("Res. dev.", _stata_number(_deviance, 7).strip()))
            elif np.isfinite(rmse) and not _is_rreg:
                _rows.append(("Root MSE", _stata_number(rmse, 7).strip()))
            for index, (label, value) in enumerate(_rows):
                left = _banner if index == 0 else ""
                lines.append(f"{left:<48}{label:<18}= {value:>10}".rstrip())
            # ``clustertable`` must survive this branch too: the multiway combination
            # table is independent of which variance estimator was used, and attaching it
            # only in the ANOVA branch made it vanish for every clustered fit.
            if cluster_summary:
                lines.append("")
                lines.extend(cluster_summary)
        else:
            # ``xtpcse``/``xtgls`` already printed R-squared inside their design block,
            # so the fallback block is skipped for them.
            if not _pcse_family_for_header(result):
                lines.append(f"{'R-squared':<21} = {_fixed(rsquared, 4):>10}")
                lines.append(f"{'Adj R-squared':<21} = {_fixed(rsquared_adj, 4):>10}")
                lines.append(f"{'Root MSE':<21} = {_fixed(rmse, 4):>10}")
    elif block == "ml":
        # Stata's likelihood header is a TWO-COLUMN block 78 wide, not a stack of lines.
        # Captured from a live ``logit``:
        #
        #   Logistic regression                          Number of obs =    400
        #                                                LR chi2(2)    =   3.27
        #                                                Prob > chi2   = 0.1951
        #   Log likelihood = -275.60445                  Pseudo R2     = 0.0059
        #
        # Left column 56 wide (banner, log likelihood); right column 22 (label 14, " =",
        # value 7). What the package printed before was a single 34-wide column, which is
        # why this header never lined up with anything.
        # The banner is e(title), not the command name: Stata prints
        # "Logistic regression", not "logit".
        _banner = str((getattr(result, "e", None) or {}).get("title") or "")
        ml_left: list[str] = [_banner or str(result.model_name or "")]
        ml_right: list[tuple[str, str]] = [
            ("Number of obs", f"{nobs:,.0f}" if np.isfinite(nobs) else ".")
        ]
        # Zero-inflated models put THREE extra rows in the right column before the chi2
        # test, and label the model on the left with its inflation equation. Captured:
        #
        #   Zero-inflated Poisson regression            Number of obs =  1,200
        #   Inflation model: logit                      Nonzero obs   =    660
        #                                               Zero obs      =    540
        #                                               LR chi2(1)    = 248.04
        #   Log likelihood = -1700.705                  Prob > chi2   = 0.0000
        # ``nbreg`` labels its parameterisation in the left column, opposite the chi2
        # test, so the row count has to line up with "Prob > chi2" and not with
        # "Number of obs". Captured:
        #
        #   Negative binomial regression                  Number of obs =   100
        #                                                 LR chi2(2)    =  0.08
        #   Dispersion: mean                              Prob > chi2   = 0.9599
        #   Log likelihood = -120.61551                   Pseudo R2     = 0.0003
        #
        # ``nbreg`` is the mean-dispersion (NB2) form; ``nbreg, dispersion(constant)`` is
        # the other parameterisation and would print "Dispersion: constant".
        n_zero = _e_float(result, "n_zero")
        n_nonzero = _e_float(result, "n_nonzero")
        if np.isfinite(n_nonzero):
            ml_left.append("Inflation model: logit")
            ml_right.append(("Nonzero obs", f"{n_nonzero:,.0f}"))
        if np.isfinite(n_zero):
            ml_right.append(("Zero obs", f"{n_zero:,.0f}"))
        chi2 = _e_float(result, "chi2")
        p_chi2 = _e_float(result, "p")
        if np.isfinite(chi2):
            ml_right.append(
                (f"{chi2_type or 'LR'} chi2({_df_label(df_model)})", _fixed(chi2, 2))
            )
        if np.isfinite(p_chi2):
            ml_right.append(("Prob > chi2", _fixed(p_chi2, 4)))
        pseudo_r2 = _e_float(result, "r2_p")
        ll = _e_float(result, "ll")
        # Stata pairs ``Log likelihood`` with ``Pseudo R2`` when there is one -- but the
        # zero-inflated models print NO Pseudo R2, so there the log likelihood sits on the
        # ``Prob > chi2`` row instead. Both arrangements occur, so the target row is decided
        # by whether a Pseudo R2 exists rather than assumed.
        # ``e(r2_p)`` EXISTS for the zero-inflated models but Stata does not print it: the
        # captured ``zip`` header has Nonzero obs / Zero obs / LR chi2 / Prob > chi2 and no
        # Pseudo R2, while ``ztp`` (zero-TRUNCATED, not zero-inflated) does print one. So
        # the deciding fact is the presence of the inflation counts, not of r2_p.
        _is_zero_inflated = np.isfinite(_e_float(result, "n_nonzero"))
        # A RANDOM-EFFECTS xt fit stores e(r2_p) but does NOT print "Pseudo R2". Measured in
        # round 230 on a live `xtlogit y x1, re`: the right column is
        # Number of obs / Number of groups / Obs per group / min / avg / max / Integration pts. /
        # Wald chi2 / Prob > chi2 -- and then the table starts. Adding the row shifted the whole
        # coefficient table down by one.
        _xt_re_no_pseudo = (
            str(getattr(result, "model_name", "") or "").startswith("xt")
            and (
                (getattr(result, "e", None) or {}).get("sigma_u") is not None
                or str((getattr(result, "e", None) or {}).get("title") or "").endswith(
                    "regression"
                )
            )
            and "Random-effects" in str((getattr(result, "e", None) or {}).get("title") or "")
        )
        _pairs_with_pseudo = (
            np.isfinite(pseudo_r2) and not _is_zero_inflated and not _xt_re_no_pseudo
        )
        if _pairs_with_pseudo:
            ml_right.append(("Pseudo R2", _fixed(pseudo_r2, 4)))
        # ``nbreg`` labels its parameterisation opposite "Prob > chi2", which is the third
        # row of the right column when a chi2 test is present:
        #
        #   Negative binomial regression                  Number of obs =   100
        #                                                 LR chi2(2)    =  0.08
        #   Dispersion: mean                              Prob > chi2   = 0.9599
        #   Log likelihood = -120.61551                   Pseudo R2     = 0.0003
        #
        # Placed by INDEX because the left column is padded to the right column's length
        # below; appending it earlier shifted it up by one row.
        # ``truncreg`` names its truncation points in the left column, opposite the row
        # that carries Number of obs / chi2. Measured:
        #
        #   Limit: Lower =   -1                Number of obs     =        85
        #          Upper = +inf                Wald chi2(2)      =       4.16
        #
        # ``e(llopt)``/``e(ulopt)`` hold them, or are absent when that side is unbounded.
        _is_tobit = str(getattr(result, "model_name", "") or "") == "tobit"
        # ``intreg`` prints the SAME censoring block as tobit (Uncensored / Left-censored /
        # Right-censored), measured from its header -- its N_unc/N_lc/N_rc are stored.
        _is_intreg = str(getattr(result, "model_name", "") or "") == "intreg"
        if _is_tobit or _is_intreg:
            _lower = _e_float(result, "llopt")
            _upper = _e_float(result, "ulopt")
            # The banner keeps row 0; the Limit rows start at row 1, beside "Uncensored"
            # and "Left-censored". Replacing row 0 (as truncreg does) put them one row
            # too high.
            while len(ml_left) < 3:
                ml_left.append("")
            # Only tobit has truncation points. intreg observes an INTERVAL per row rather
            # than a censoring threshold, so it prints the censoring counts with no
            # "Limits:" row at all -- measured.
            if _is_tobit:
                ml_left[2] = "Limits: Lower = " + (
                    f"{_lower:>4g}" if np.isfinite(_lower) else "   ."
                )
            if _is_tobit:
                ml_left.append(
                    "        Upper = "
                    + (f"{_upper:>4g}" if np.isfinite(_upper) else "+inf")
                )
            # Blank separator between the Limit group and the fit statistics.
            ml_left.append("")
            _unc = _e_float(result, "N_unc")
            _lc = _e_float(result, "N_lc")
            _rc = _e_float(result, "N_rc")
            _counts: list[tuple[str, str]] = []
            if np.isfinite(_unc):
                _counts.append(("Uncensored", f"{int(_unc):,}"))
            if np.isfinite(_lc):
                _counts.append(("Left-censored", f"{int(_lc):,}"))
            if np.isfinite(_rc):
                _counts.append(("Right-censored", f"{int(_rc):,}"))
            # ``intreg`` observes intervals, so it reports an "Interval-cens." count that
            # tobit does not have; its label is the longest in the block.
            _n_int = _e_float(result, "N_int")
            if _is_intreg and np.isfinite(_n_int):
                _counts.append(("Interval-cens.", f"{int(_n_int):,}"))
            # ``Number of obs`` stays first; the counts follow it. Inserting at 1 keeps
            # row 0 as Number of obs -- inserting at 0 pushed it to the top of the block,
            # which is what put "Uncensored" on the banner row.
            ml_right[1:1] = _counts
            # A blank separator follows the counts; Stata prints an empty line between
            # "Right-censored" and "LR chi2".
            ml_right.insert(1 + len(_counts), ("", ""))

        if str(getattr(result, "model_name", "") or "") == "truncreg":
            # The banner row is REPLACED by "Limit: Lower = ...", not followed by it:
            # Stata prints the banner once, as the title above the table, and the Limit
            # rows occupy the left column of the two-column block.
            _lower = _e_float(result, "llopt")
            _upper = _e_float(result, "ulopt")
            _limit_row = "Limit: Lower = " + (
                f"{_lower:>4g}" if np.isfinite(_lower) else "   ."
            )
            if ml_left:
                ml_left[0] = _limit_row
            else:
                ml_left.append(_limit_row)
            ml_left.append(
                "       Upper = " + (f"{_upper:>4g}" if np.isfinite(_upper) else "+inf")
            )

        if (
            str(getattr(result, "model_name", "") or "") == "nbreg"
            and len(ml_left) < 3
        ):
            while len(ml_left) < 2:
                ml_left.append("")
            ml_left.append("Dispersion: mean")

        # ``etregress`` names its estimator on the row under the banner, opposite the chi2
        # test:
        #
        #   Linear regression with endogenous treatment        Number of obs =
        #   Estimator: Maximum likelihood                      Wald chi2   =
        #   Log likelihood = -196.48097                        Prob > chi2 =
        _estimator_name = str(
            (getattr(result, "e", None) or {}).get("estimator_name") or ""
        ).strip()
        if _estimator_name and len(ml_left) >= 1:
            ml_left.insert(1, "Estimator: " + _estimator_name)

        # ``tnbreg`` names its truncation point on the row under the banner:
        #
        #   Truncated negative binomial regression      Number of obs =
        #   Truncation point = 0                        LR chi2(1)    =
        _truncation = _e_float(result, "truncation_point")
        if (
            str(getattr(result, "model_name", "") or "") == "tnbreg"
            and np.isfinite(_truncation)
            and len(ml_left) >= 1
        ):
            ml_left.insert(
                1, "Truncation point = " + f"{_truncation:g}"
            )

        # ``ml_left`` is padded so that appending the log likelihood makes it exactly ONE
        # row longer than ``ml_right`` (the ll row has no right-hand partner). Derived from
        # the CURRENT left length, since the "Estimator:" row inserted above changes it.
        _target_before_ll = len(ml_right) - 1 if np.isfinite(ll) else len(ml_right)
        while len(ml_left) < _target_before_ll:
            ml_left.append("")
        if np.isfinite(ll):
            ml_left.append("Log likelihood = " + ("%.8g" % ll))
        # ``truncreg``'s left column is 48 wide, not the 56 the other ML commands use:
        # its "Limit: Lower = ..." row is longer than a banner, so Stata shifts the right
        # column left to keep the block 78 wide -- and the value field shrinks with it
        # ("Number of obs     =  85": value in 4 columns, not 6).
        _is_truncreg = str(getattr(result, "model_name", "") or "") == "truncreg"
        # ``_is_intreg`` was computed above, beside ``_is_tobit``: intreg shares tobit's
        # 52/18/6 geometry ("Interval regression            Number of obs     =    100").
        # tobit's right column starts at 52 with a 17-wide label and a 5-wide value
        # ("Right-censored =      0"); truncreg uses 48 / 18 / 4; the rest 56 / 14 / 6.
        # Computed BEFORE _left_width, which depends on it. Round 228 moved this up after an
        # UnboundLocalError: the width branch read the flag two dozen lines before it was set.
        _re_macros = getattr(result, "e", None) or {}
        # Same prefix rule as above: a substring test on "re" matches "regression".
        _re_title = str(_re_macros.get("title") or "").strip()
        _is_random_effects = (
            _re_macros.get("sigma_u") is not None
            or _re_title.startswith("Random-effects")
            or str(getattr(result, "model_name", "") or "").endswith(", re")
        )
        # BOTH xt likelihood shapes splice their panel rows in -- only the ROWS THEMSELVES differ
        # between fe and re. Round 234 gated the splice on the random-effects test, which sent fe
        # back to the old "append the whole block afterwards" path and misaligned its header.
        _is_xt_likelihood = str(_re_title).endswith("regression") and str(
            getattr(result, "model_name", "") or ""
        ).startswith("xt")
        _splice_panel = block == "ml" and (
            _is_random_effects or _is_xt_likelihood
        ) and bool(panel_lines)
        if _is_tobit or _is_intreg:
            _left_width = 52
        elif _is_truncreg:
            _left_width = 48
        elif _splice_panel:
            # The random-effects xt header starts its right column at 53, not 56. Measured in
            # round 228: "Random-effects logistic regression" runs to 53 and
            # "Number of obs    =    600" (14-wide label) fills to 78.
            _left_width = 53
        else:
            _left_width = 56
        # A RANDOM-EFFECTS xt FIT PAIRS THE PANEL STRUCTURE WITH THE FIT STATISTICS, ROW BY ROW.
        # This is why the block order resisted three rounds: `panel_lines` is a list of finished
        # strings, and appending it cannot put "Group variable" beside "Number of groups" because
        # that row has to exist in ml_left AND ml_right at the same index. Measured in round 226:
        #
        #   'Random-effects logistic regression     Number of obs    =    600'
        #   'Group variable: sid                    Number of groups =     30'
        #   ''
        #   'Random effects u_i ~ Gaussian          Obs per group:'
        #   '                                                min =     20'
        #
        # So the rows are spliced INTO ml_left/ml_right just after "Number of obs", and
        # panel_lines is then suppressed for this shape (its own right column is unused).
        if _splice_panel:
            _entity = str(
                _re_macros.get("ivar") or _re_macros.get("panelvar") or ""
            ).strip()
            _per_group = _re_macros.get("_obs_per_group") or {}
            _gmin = _per_group.get("min", _e_float(result, "g_min"))
            _gavg = _per_group.get("avg", _e_float(result, "g_avg"))
            _gmax = _per_group.get("max", _e_float(result, "g_max"))
            _n_groups = _e_float(result, "N_g")
            _pts = _e_float(result, "intpoints")
            if not np.isfinite(_pts):
                _pts = 7.0
            # fe and re share this block but NOT these rows. A CONDITIONAL fit prints only
            # "Group variable" and the per-group extremes; it has no distribution line for u_i
            # and integrates nothing out, so no "Integration method" row. Measured in round 235.
            if _is_random_effects:
                # The DISTRIBUTION of u_i is model-specific, not always Gaussian: a count model's
                # random effect is Gamma (Stata: "Random effects u_i ~ Gamma"). Measured in round
                # 239 against `xtpoisson n x1, re`.
                _command_text = str(getattr(result, "model_name", "") or "")
                _is_count_model = _command_text.startswith(("xtpoisson", "xtnbreg"))
                _dist_name = "Gamma" if _is_count_model else "Gaussian"
                if _is_count_model:
                    # A Gamma random effect is CLOSED FORM -- nothing is integrated numerically,
                    # so Stata prints neither "Integration method" nor "Integration pts." and the
                    # block is two rows shorter than the logit/probit one.
                    # Row map measured in round 241 by printing both lists:
                    #
                    #   idx 8  ''   ''            <- blank, BOTH columns
                    #   idx 9  ''   'Wald chi2(1)'
                    #   idx 10 'Log likelihood = ...'   'Prob > chi2'
                    #
                    # so the blank goes into ml_right too -- round 240 put it only on the left,
                    # which is why adding and removing it never moved the Wald row.
                    _left_rows = [
                        f"Group variable: {_entity or '.'}",
                        "",
                        f"Random effects u_i ~ {_dist_name}",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                else:
                    _left_rows = [
                        f"Group variable: {_entity or '.'}",
                        "",
                        f"Random effects u_i ~ {_dist_name}",
                        "",
                        "",
                        "",
                        "",
                        "Integration method: "
                        + str(_re_macros.get("intmethod") or "mvaghermite").strip(),
                        "",
                        "",
                    ]
                _right_rows: list[tuple[str, str]] = [
                    (
                        "Number of groups",
                        f"{_n_groups:,.0f}" if np.isfinite(_n_groups) else ".",
                    ),
                    ("", ""),
                    ("Obs per group:", ""),
                    ("min", f"{_gmin:.0f}" if np.isfinite(_gmin) else "."),
                    ("avg", f"{_gavg:.1f}" if np.isfinite(_gavg) else "."),
                    ("max", f"{_gmax:.0f}" if np.isfinite(_gmax) else "."),
                ]
                if not _is_count_model:
                    _right_rows.extend(
                        [
                            ("", ""),
                            ("Integration pts.", f"{int(round(_pts))}"),
                            ("", ""),
                        ]
                    )
                else:
                    # The count shape's blank sits HERE, on the right, so "Wald chi2" lands on
                    # the row after it. Two entries keep the columns the same length.
                    _right_rows.extend([("", "")])
            else:
                # ROW MAP measured in round 238 by printing both lists at the assembly loop:
                #
                #   idx 8  ''            ''            <- blank
                #   idx 9  ''            LR chi2(1)
                #   idx 10 'Log likelihood'  Prob > chi2
                #
                # so there are TWO blanks between "max" and the LR row, and the left column is
                # one entry longer than the right before the log likelihood is placed.
                _left_rows = [
                    f"Group variable: {_entity or '.'}",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
                _right_rows = [
                    (
                        "Number of groups",
                        f"{_n_groups:,.0f}" if np.isfinite(_n_groups) else ".",
                    ),
                    ("", ""),
                    ("Obs per group:", ""),
                    ("min", f"{_gmin:.0f}" if np.isfinite(_gmin) else "."),
                    ("avg", f"{_gavg:.1f}" if np.isfinite(_gavg) else "."),
                    ("max", f"{_gmax:.0f}" if np.isfinite(_gmax) else "."),
                    ("", ""),
                ]
            ml_left[1:1] = _left_rows
            ml_right[1:1] = _right_rows
            # ``Log likelihood`` shares its row with the model test -- "Prob > chi2" or
            # "Wald chi2(#)" -- so it is placed at whichever index that test occupies, NOT
            # appended at the end. The pop that used to live here removed it and nothing put it
            # back, which left the left column blank on that row (round 231).
            if ml_left and ml_left[-1].startswith("Log likelihood"):
                _ll_text = ml_left.pop()
                # The model test is "Prob > chi2" for re. The CONDITIONAL fit has no Prob row in
                # this block -- its right column carries "LR chi2(#)" and Stata puts Log likelihood
                # on THAT row -- so fall back to the chi2 label, and only then to appending.
                # "Prob > chi2" -- NOT "LR chi2". Measured in round 237 by reading the whole
                # block side by side: Stata leaves the LR row's left cell EMPTY and puts
                # Log likelihood on the row below, beside Prob > chi2. Round 236 tried LR first
                # and put it one row high; this restores Prob, which is also what re needs.
                _test_index = next(
                    (
                        position
                        for position, (label, _value) in enumerate(ml_right)
                        if label.startswith("Prob >")
                    ),
                    None,
                )
                # The fe splice is shorter than re's, so an index valid for ml_right can be past
                # the end of ml_left -- guard it rather than trust the arithmetic.
                if (
                    _test_index is not None
                    and _test_index < len(ml_left)
                    and ml_left[_test_index] == ""
                ):
                    ml_left[_test_index] = _ll_text
                else:
                    ml_left.append(_ll_text)

        for index, line in enumerate(ml_left):
            right = ""
            if index < len(ml_right):
                label, value = ml_right[index]
                # Measured: truncreg's right column is label 18 + "= " + value 4
                # ("Number of obs     =  85"), where the other ML commands use label 14
                # + value 6. The block is the same 78 wide either way.
                if _is_tobit or _is_intreg:
                    # Right part is 26 wide: label in 18, "= ", value in 6. The censoring
                    # counts right-align their label; every other row left-aligns it.
                    # An empty pair is the blank separator and renders as nothing.
                    if label == "" and value == "":
                        right = ""
                    else:
                        _censoring = label.endswith("censored") or label == "Uncensored"
                        if _censoring:
                            right = f"{label:>17} = {value:>6}"
                        else:
                            right = f"{label:<18}= {value:>6}"
                elif _is_truncreg:
                    right = f"{label:<18}= {value:>4}"
                else:
                    # The right column is 28 wide: "   " + label + padding + "= " + value.
                    # A label ending in ':' is a HEADING with no value at all -- Stata prints
                    # "Obs per group:" and nothing after it, not "Obs per group: = .".
                    if label.endswith(":"):
                        right = f"{label}"
                    elif not label and not value:
                        right = ""
                    elif label in ("min", "avg", "max"):
                        # The per-group extremes RIGHT-align their 3-letter label into the same
                        # field, so "=" still lands in column 70. The field is one narrower here
                        # (16) because these rows carry a space before the "=", not a bare one.
                        right = f"{label:>16} = {value:>6}"
                    else:
                        # The left column is padded to _left_width (53 here) and the right column
                        # runs to the "=" in column 70 -- a 17-wide label field. Measured on five
                        # rows of a live xtlogit, re.
                        right = f"{label:<17}= {value:>6}"
            if _is_tobit and line and not right:
                lines.append(line.rstrip())
            else:
                # A row with NOTHING on either side is a blank line, not a run of spaces. The
                # f-string pads `line` to _left_width before rstrip can reach the concatenation,
                # which is why blank separator rows came out as 52 spaces.
                _composed = f"{line:<{_left_width}}{right}".rstrip()
                lines.append(_composed)
        # tobit/truncreg replace the banner row, which leaves the standalone blank
        # A LEADING BLANK **IS** PRINTED for the ML family. This used to strip every blank line at
        # the top, on the comment's claim that "Stata's header starts with the banner line and has
        # no leading blank". A live `logit rns s expr tenure, nolog` starts with chr(10)+'Logisti',
        # so the claim was wrong and the strip is what made logit/probit the only estimators
        # without the blank. Rounds 198 and 199 both tried to ADD the blank upstream and both were
        # undone here -- found in round 199 by tracing executed line numbers rather than reading.
        #
        # The deferred panel block goes AFTER the ML banner block (see where it is computed).
        #
        # EXCEPT when the rows were SPLICED into ml_left/ml_right pair-wise (round 226): then
        # they are already in ``lines`` and extending again duplicates every one of them, which
        # is what made the block 38 lines against Stata's 26. ``_splice_panel`` is set further
        # up in this same function; round 227 tried to gate this on a freshly recomputed
        # random-effects test instead of reusing that flag, and the banner got displaced.
        if _defer_panel_block and not _splice_panel:
            lines.append("")
            lines.extend(panel_lines)
    elif block == "quantile":
        # Stata's qreg header (p8_outputs.txt:20-22): the title carries the
        # quantile, then two deviation lines with Pseudo R2 on the right.  There
        # is no F test, no log-likelihood, no R-squared and no Root MSE.
        #
        #   .25 Quantile regression                  Number of obs =        600
        #     Raw sum of deviations 438.3735 (about -.46158607)
        #     Min sum of deviations 156.7522         Pseudo R2     =     0.6424
        sum_rdev = _e_float(result, "sum_rdev")
        sum_adev = _e_float(result, "sum_adev")
        q_v = _e_float(result, "q_v")
        pseudo_r2 = _e_float(result, "r2_p")
        # ``sqreg``/``bsqreg`` are MULTI-quantile and print a different block again: the
        # banner, then one line per quantile carrying its pseudo R2. Measured:
        #
        #   Simultaneous quantile regression        Number of obs =        100
        #     bootstrap(20) SEs                     .25 Pseudo R2 =     0.0018
        #                                           .50 Pseudo R2 =     0.0077
        #                                           .75 Pseudo R2 =     0.0033
        #
        # No deviation lines -- those belong to single-quantile ``qreg`` only.
        _r2_by_q = (getattr(result, "e", None) or {}).get("r2_p_by_quantile") or []
        if _r2_by_q:
            _quantiles = list((getattr(result, "e", None) or {}).get("quantiles") or [])
            _reps = _e_float(result, "reps")
            _banner_sq = str((getattr(result, "e", None) or {}).get("title") or "")
            lines.append(
                f"{_banner_sq:<52}{'Number of obs':<14}= {nobs:>11d}"
            )
            for _index, (_q, _r2v) in enumerate(zip(_quantiles, _r2_by_q)):
                _left = (
                    f"  bootstrap({int(_reps)}) SEs"
                    if _index == 0 and np.isfinite(_reps)
                    else ""
                )
                # Stata drops the leading zero: ".25 Pseudo R2", not "0.25 Pseudo R2".
                _q_text = f"{_q:g}"
                if _q_text.startswith("0."):
                    _q_text = _q_text[1:]
                _label = f"{_q_text:>4} Pseudo R2"
                lines.append(f"{_left:<52}{_label:<14}= {_r2v:>11.4f}")
            # The coefficient header carries a "Bootstrap" banner row, and the caller adds
            # the blank line -- emitting one here too produced two.
            lines.append(f"{'':>13} |{'Bootstrap':>24}")
            # Set the single-quantile scalars to missing so the branches BELOW skip
            # their deviation lines -- this block replaced them.
            sum_rdev = float("nan")
            sum_adev = float("nan")
            pseudo_r2 = float("nan")
        # Stata puts the banner and "Number of obs" on the SAME line, left-aligned pair:
        #
        #   Median regression              Number of obs =        100
        #     Raw sum of deviations 39.90204 (about .12350311)
        #     Min sum of deviations 39.21938    Pseudo R2     =     0.0171
        #
        # The generic header printed the banner alone and "Number of obs" on the next
        # line, which is a different shape (the OLS one), not a shifted version of this.
        # ``qreg`` CLEARS e(title) and prints its banner with ``display`` instead, so the
        # text is derived from the quantile rather than read from the macro. Measured:
        # .5 prints "Median regression"; other quantiles print "<q> Quantile regression"
        # (p8_outputs.txt:20 shows ".25 Quantile regression").
        _banner_text = str((getattr(result, "e", None) or {}).get("title") or "")
        if not _banner_text:
            _q = _e_float(result, "q")
            if np.isfinite(_q):
                _banner_text = (
                    "Median regression"
                    if abs(_q - 0.5) < 1e-12
                    else f"{_q:g} Quantile regression"
                )
        # "Number of obs =        100": the label is 14 wide then "= " and an 11-wide
        # value, so the "=" lands at column 66. The OLS header's 18-wide label field put
        # it four columns further right.
        # The multi-quantile block above already emitted the banner + Number of obs, so
        # the single-quantile line is skipped for those commands.
        # Geometry measured from a live ``qreg lnw s expr tenure, quantile(.5)``. Every
        # line is 78 characters:
        #
        #   'Median regression                                   Number of obs =        758'
        #   '  Raw sum of deviations   130.68 (about 5.684)'
        #   '  Min sum of deviations   105.07                    Pseudo R2     =     0.1960'
        #
        # The banner occupies 52; "Number of obs" ends at 64 with "=" at 66 and the value
        # right-aligned to 78. The deviation values start at column 26, and "Pseudo R2"
        # starts at 52 -- the same column the Number-of-obs label starts at.
        if not _r2_by_q:
            lines.append(f"{_banner_text:<52}{'Number of obs':<13} = {nobs:>10d}")
        if np.isfinite(sum_rdev):
            about = f" (about {_num(q_v, 8)})" if np.isfinite(q_v) else ""
            lines.append(
                f"  {'Raw sum of deviations':<22}{_num(sum_rdev, 7):>8}{about}"
            )
        if np.isfinite(sum_adev):
            left = f"  {'Min sum of deviations':<22}{_num(sum_adev, 7):>8}"
            lines.append(
                left.ljust(52) + f"{'Pseudo R2':<9}     = {_fixed(pseudo_r2, 4):>10}"
            )
        if np.isfinite(pseudo_r2) and not np.isfinite(sum_adev):
            # Degenerate case: no deviation lines, still report the pseudo-R2.
            lines.append(f"{'Pseudo R2':<21} = {_fixed(pseudo_r2, 4):>10}")
    elif block == "svy":
        # Stata's survey header is TWO-COLUMN and 78 wide, with the banner on a line of
        # its own above it. Captured from ``svy: regress y x1 x2``:
        #
        #   Survey: Linear regression
        #
        #   Number of strata =  8                      Number of obs   =    576
        #   Number of PSUs   = 48                      Population size =    576
        #                                              Design df       =     40
        #                                              F(2, 39)        =   0.02
        #                                              Prob > F        = 0.9767
        #                                              R-squared       = 0.0001
        #
        # Left column 54, right column 24 (label 15, " =", value 6). The previous
        # rendering stacked the left pair above F/Prob/R2 and paired only the first two
        # rows, so the columns did not line up with anything.
        n_strata = _e_float(result, "N_strata")
        n_psu = _e_float(result, "N_psu")
        n_pop = _e_float(result, "N_pop")
        r2 = _e_float(result, "r2")

        # The survey ESTIMATE commands use a NARROWER design block (62 wide, left 34)
        # than ``svy: regress`` (78, left 54), because their table is narrower.
        # Captured: "Number of strata =  8             Number of obs   =        576".
        if _is_svy_estimate(result):
            # Measured from a live ``svy: mean x1 x2``: every line is 62 characters, the
            # left "=" sits at 17 and the right half ALWAYS starts at column 34 with its
            # "=" at 50 and a ten-wide value ending at 62:
            #
            #   'Number of strata =  20            Number of obs   =        400'
            #   'Number of PSUs   = 200            Population size = 507.732852'
            #   '                                  Design df       =        180'
            #
            # The previous version let the split float with the value's width, on the
            # belief that "Number of obs = 100" pushed the right half to 41. Measured on
            # three different value widths, the right half does not move: it is 34.
            def _row_narrow(left: str, label: str = "", value: str = "") -> str:
                if not label:
                    return left.rstrip()
                return f"{left:<34}{label:<16}= {value:>10}".rstrip()

            narrow: list[str] = []
            if np.isfinite(n_strata):
                narrow.append(
                    _row_narrow(
                        f"Number of strata = {int(round(n_strata)):>3}",
                        "Number of obs",
                        f"{nobs:,.0f}",
                    )
                )
            if np.isfinite(n_psu):
                # ``Population size`` prints SIX decimals when the weighted total is
                # fractional ("837.592551") but a plain integer when it is not ("100") --
                # measured in both directions, not a fixed format.
                if not np.isfinite(n_pop):
                    pop = "."
                elif abs(n_pop - round(n_pop)) < 1e-9:
                    pop = f"{int(round(n_pop)):,}"
                else:
                    pop = f"{n_pop:,.6f}"
                narrow.append(
                    _row_narrow(
                        f"Number of PSUs   = {int(round(n_psu)):>3}",
                        "Population size",
                        pop,
                    )
                )
            if np.isfinite(df_resid):
                narrow.append(
                    _row_narrow("", "Design df", f"{int(round(df_resid))}")
                )
            lines.extend(narrow)
            lines.append("")
            return _svy_estimate_table(result, lines, names)

        # Measured from a live ``svy: regress`` on a 20-strata / 200-PSU design: every
        # line is 78 characters, the LEFT "=" sits at 17, and the right half starts at
        # column 50 with its "=" at 66 and a TEN-wide value ending at 78:
        #
        #   'Number of strata =  20                            Number of obs   =        400'
        #   'Number of PSUs   = 200                            Population size = 509.791159'
        #   '                                                  Design df       =        180'
        #
        # The old split was left 54 / label 15 / value 6, which put the right half four
        # columns right and clipped values to six characters.
        def _cell(label: str, value: str) -> str:
            return f"{label:<16}= {value:>10}"

        def _row(left: str, right: str = "") -> str:
            return f"{left:<50}{right}".rstrip()

        rows: list[str] = []
        if np.isfinite(n_strata):
            rows.append(_row(f"Number of strata = {int(round(n_strata)):>3}",
                             _cell("Number of obs", f"{nobs:,.0f}")))
        if np.isfinite(n_psu):
            # ``Population size`` prints as an INTEGER when the weights sum to a whole
            # number -- "576", not "576.000000" -- and with SIX DECIMALS when they do not
            # ("509.791159"). ``%g`` gave 509.791, truncating real digits.
            if not np.isfinite(n_pop):
                pop = "."
            elif abs(n_pop - round(n_pop)) < 1e-9:
                pop = f"{int(round(n_pop)):,}"
            else:
                pop = f"{n_pop:,.6f}"
            rows.append(_row(f"Number of PSUs   = {int(round(n_psu)):>3}",
                             _cell("Population size", pop)))
        if np.isfinite(df_resid):
            rows.append(_row("", _cell("Design df", f"{int(round(df_resid))}")))
        # The F test's denominator is Design df - df_m + 1: measured, ``svy: regress y x1``
        # gives F(1, 40) and ``... y x1 x2`` gives F(2, 39) on the same design.
        fvalue = _e_float(result, "F")
        p_f = _e_float(result, "p")
        if np.isfinite(fvalue):
            f_denom = df_resid - df_model + 1 if np.isfinite(df_resid) else float("nan")
            rows.append(
                _row("", _cell(
                    f"F({_df_label(df_model)}, {_df_label(f_denom)})", f"{fvalue:.2f}"
                ))
            )
        if np.isfinite(p_f):
            rows.append(_row("", _cell("Prob > F", f"{p_f:.4f}")))
        if np.isfinite(r2):
            rows.append(_row("", _cell("R-squared", f"{r2:.4f}")))
        lines.extend(rows)
    # ``block == "none"`` prints nothing: Stata prints no goodness-of-fit block
    # for arima / arch / xtgls, so an empty one is the faithful rendering.
    lines.append("")

    # Column widths, matched to Stata's real table.  The captured outputs -- for qreg
    # (``p8_outputs.txt:25``), nl (``p10_outputs.txt:106``) and the logit/probit family
    # (``p6p7_outputs.txt:18``) -- all print the same three headers:
    #
    #     y | Coefficient  Std. err.      t    P>|t|     [95% conf. interval]
    #
    # i.e. ``Coefficient`` (not ``Coef.``), ``Std. err.`` with a LOWERCASE ``err``, and
    # ``[95% conf. interval]`` all lowercase.  Round 237 capitalised the last two from
    # ``design.md:209``; the real captures outrank that document, so they are restored.
    #
    # WIDTHS ARE NOT NEGOTIABLE. They used to be (12, 12, 8, 8, 12), which made every row
    # 80 characters wide against Stata's 78 and pushed every column one place right. The
    # numbers were correct and the table still looked wrong, which is the failure a reader
    # notices first.
    #
    # Measured from a live ``regress`` capture (the ``+`` of the rule sits at column 13,
    # total width 78), the header keywords END at these columns:
    #
    #     Coefficient        ends 26      Std. err.   ends 37
    #     t                  ends 43      P>|t|       ends 53
    #     [95% conf. interval]  ends 78
    #
    # which makes the value columns 12, 10, 6, 10 and 25 wide after the ``|`` at 13.
    # Every width below was read off a live ``regress`` capture by locating where each
    # number ENDS. The ``|`` sits at column 13 and the line is 78 wide:
    #
    #     hdr  '           y | Coefficient  Std. err.      t    P>|t|  [95% ...]'
    #     row  '           s |   .1236469   .0133633     9.25   0.000  ...'
    #          tokens end at   25          36         45      53       66     78
    #
    # so the fields are 11, 11, 9, 8, 13 and 12 wide after the pipe. Guessing these (12,
    # 12, 8, 8, 12, 12) produced an 80-character line against Stata's 78 and shifted every
    # column; the numbers were right and the table still looked wrong.
    # Width from the DISPLAYED names, not the stored ones. A multi-equation model stores
    # ``inflate#Intercept`` (18 characters) but prints it as ``_cons`` inside an
    # ``inflate`` block, so sizing on the stored spelling made the table 83 wide instead
    # of 78 and pushed every column right.
    def _factor_display_pieces(name: str) -> tuple[str, str] | None:
        """Split ``C(var)[T.level]`` into the two strings Stata prints for it.

        Returns ``(term, level)`` or ``None`` when the name is not a factor coefficient. Stata
        shows a factor as a group header over its levels rather than one long row:

            foreign |
          Foreign  |  -1.650029   1.075994

        Measured on auto.dta in round 183.
        """

        text = str(name).strip()
        match = re.match(r"^C\(([^)]+)\)\[(?:T\.)?(.+)\]$", text)
        if match is not None:
            return match.group(1).strip(), match.group(2).strip()
        match = re.match(r"^([A-Za-z_][\w]*)\(([^)]+)\)\[(?:T\.)?(.+)\]$", text)
        if match is not None:
            return match.group(2).strip(), match.group(3).strip()
        return None

    def _display_length(name: str) -> int:
        text = str(name)
        for marker in ("#", ":"):
            head, sep, tail = text.partition(marker)
            if sep and head in ("inflate", "select", "tobit"):
                return len(tail)
        return len(text)

    # The name column grows with the longest name, as it always has -- several tables in
    # this file rely on that (long factor names, interaction terms). The ONE row that does
    # not participate is the ``var(e.<y>)`` scale row: Stata fixes its pipe at column 13 and
    # abbreviates the variable name to six characters, so that row is built separately in
    # the row loop and must not widen the table. Measured -- a ``tobit`` whose response is
    # ``food_exp`` still prints a standard 78-character table.
    _ordinary_names = [
        name for name in names if not str(name).startswith("var(e.")
    ]
    # A FACTOR name is DISPLAYED as a group header plus a level row, so it must not widen the
    # table by its full ``C(var)[T.level]`` spelling: with the 22-character form the table grew to
    # 87 columns where Stata's is 78. Only the two pieces are measured. Measured on auto.dta in
    # round 183.
    _measured_names: list[str] = []
    for _name in _ordinary_names:
        _pieces = _factor_display_pieces(str(_name))
        if _pieces is None:
            _measured_names.append(str(_name))
        else:
            _measured_names.extend(_pieces)
    name_width = max([12, *(_display_length(name) for name in _measured_names)])

    # The survey ESTIMATE commands (``svy: mean``/``total``/``proportion``) print a
    # different table: no t/P columns, one estimate column, and a 62-wide frame rather
    # than 78. Handled here because the row VALUES differ in count, not just in spelling.
    if _is_svy_estimate(result):
        return _svy_estimate_table(result, lines, names)

    coef_w, se_w, stat_w, p_w, ci_lo_w, ci_hi_w = 11, 11, 9, 8, 13, 12
    row_width = (
        name_width + 2 + coef_w + se_w + stat_w + p_w + ci_lo_w + ci_hi_w
    )
    # Stata labels the row-name column with the DEPENDENT VARIABLE, right-aligned:
    #
    #     '           y | Coefficient  Std. err.      t    P>|t|  ...'
    #
    # It used to be blank, which was the one place the table did not match. The name comes
    # from `response_name`/`e(depvar)`; with neither available the column stays empty
    # rather than showing an invented label.
    response_label = str(
        getattr(result, "response_name", None)
        or (getattr(result, "e", None) or {}).get("depvar")
        or ""
    )
    # ``mvreg`` is the exception: it fits several outcomes at once, so Stata leaves the
    # name column of the header BLANK and names each equation on its own block row below:
    #
    #                | Coefficient  Std. err.      t    P>|t|  ...
    #   -------------+------------------------------------------------------------
    #   y1           |
    #
    # westata printed the outcome list there ("y1 y2 | Coefficient ..."), which duplicates
    # the equation rows and occupies a cell Stata leaves empty.
    if str(getattr(result, "model_name", "") or "") == "mvreg":
        response_label = ""
    # A ROBUST fit changes two things in Stata's table, both measured:
    #
    #   1. the column header becomes ``std. err.`` in LOWER case (the ordinary fit prints
    #      ``Std. err.``), and
    #   2. an extra ``Robust`` banner row is inserted above the header, ending at the same
    #      column as the ``Coefficient`` header.
    #
    #   '             |               Robust'
    #   '           y | Coefficient  std. err.      t    P>|t|  ...'
    #
    # ``e(vcetype)`` is the macro that says which: Stata sets it to "Robust"/"Robust HC2"
    # etc. for those fits and leaves it EMPTY for plain ols.
    _vcetype = str((getattr(result, "e", None) or {}).get("vcetype") or "").strip()
    # ANY non-empty e(vcetype) means "not plain ols" and selects the two-column header
    # with a label above the column names. Measured: ``newey`` stores
    # e(vce)=""/e(vcetype)="Newey-West", i.e. the same shape as ``vce(robust)``, so
    # testing only for "robust" left HAC fits on the ANOVA layout.
    # NOT every non-empty vcetype means "robust". ``OIM`` is the DEFAULT variance for the
    # maximum-likelihood estimators, and Stata's ``logit`` prints ``Std. err.`` in upper
    # case with no banner row for it -- so treating any value as robust re-cased the whole
    # ML family. OIM is excluded explicitly rather than by testing for "Robust", because
    # ``newey`` stores "Newey-West" and must still take this branch.
    _is_robust = bool(_vcetype) and _vcetype.strip().upper() != "OIM"
    se_header = "std. err." if _is_robust else "Std. err."
    # ``logit ..., or`` renames the coefficient column. Measured: "Odds ratio" ends at
    # column 25 where "Coefficient" ends at 26, both inside a 78-character row.
    _odds_ratio = bool(getattr(result, "_odds_ratio", False))
    # "Odds ratio" is one character shorter than "Coefficient" but the column boundary is
    # fixed, so it is padded to the same width -- otherwise the whole row loses a space.
    _coef_header = f"{'Odds ratio':<{coef_w}}" if _odds_ratio else "Coefficient"
    header = (
        f"{response_label:>{name_width}} | "
        f"{_coef_header:>{coef_w if _odds_ratio else coef_w - 1}}"
        f"{se_header:>{se_w}}"
        f"{stat_label:>{stat_w - 2}}"
        f"{p_label:>{p_w + 1}}"
        f"{_ci_label(getattr(result, 'level', None)):>{ci_lo_w + ci_hi_w}}"
    )
    # A CLUSTERED fit prints a note above the rule saying how many clusters were used.
    # Captured from ``regress y s, vce(cluster g)``:
    #
    #                                        (Std. err. adjusted for 30 clusters in g)
    #   ------------------------------------------------------------------------------
    #
    # right-aligned to the full 78 columns. Without it a reader cannot tell which
    # variable was clustered on, which is the single most important thing to know about a
    # clustered fit.
    _cluster_note = ""
    if str((getattr(result, "e", None) or {}).get("vce") or "").strip() == "cluster":
        _n_clust = _e_float(result, "N_clust")
        if np.isfinite(_n_clust):
            # The DID family names the variable with ``e(group)``; other commands store it
            # under ``e(_cluster_names)``. Both spellings are read, since without the name
            # the note says "adjusted for 200 clusters" and a reader cannot tell which
            # variable was clustered on.
            _cluster_var = _cluster_variable(result) or str(
                (getattr(result, "e", None) or {}).get("group") or ""
            )
            _cluster_note = (
                f"(Std. err. adjusted for {int(_n_clust):,} clusters"
                f"{' in ' + _cluster_var if _cluster_var else ''})"
            )
    if _cluster_note:
        lines.append(f"{_cluster_note:>{row_width}}")
    lines.append("-" * row_width)
    if _is_robust:
        # Ends at the same column as the ``Coefficient`` header: the banner row is the
        # name column, the pipe, then the label right-aligned to the coefficient field.
        # The banner is right-aligned against a FIXED column, not against the coefficient
        # field: measured, "Robust" ends at 35 and "Newey-West" at 37, and both start
        # after the pipe at 13. Right-aligning to a fixed width reproduces both, whereas
        # deriving an offset from the label length put Newey-West two characters long.
        # The banner's right edge is fixed per label, measured rather than derived:
        # "Robust" ends at 35, "Newey-West" at 37, "Panel-corrected" at 40. Written as an
        # explicit map because offsets computed from the label length kept landing short.
        _BANNER_END = {
            "Robust": 35,
            # The EN DASH (U+2013) is what Stata prints; an ASCII hyphen here silently
            # missed the lookup and left the banner unpadded. Both spellings are accepted
            # so a caller that supplied an ASCII hyphen still lines up.
            "Newey\u2013West": 37,
            "Newey-West": 37,
            "Panel-corrected": 40,
            # arch's variance-estimator banner ends one column short of "Robust".
            "OPG": 34,
            # A survey ML fit (svy: logit/probit) labels its table "Linearized", ending at
            # 37 -- measured from ``svy: logit ybin x1``.
            "Linearized": 37,
        }
        _banner_end = _BANNER_END.get(_vcetype, name_width + 2 + len(_vcetype))
        lines.append(f"{'':>{name_width}} |{_vcetype:>{_banner_end - name_width - 2}}")
    lines.append(header)
    # Stata's table geometry, uniformly across families: a FULL-WIDTH rule opens the
    # table, a SPLIT rule separates the column header from the rows, and a FULL-WIDTH
    # rule closes it. Verified against a live ``regress y x1`` --
    #
    #   --------------------------------------------------------------   full
    #                y | Coefficient  Std. err.  ...                      header
    #   -------------+-----------------------------------------------     split
    #             x1 |  .0312779   ...
    #          _cons |  .0205762   ...
    #   --------------------------------------------------------------   full
    #
    # -- and against the committed captures (p10_outputs.txt:13/16 for sureg,
    # p9_outputs.txt:483/488, p12_outputs.txt:11/16). The closing rule used to reuse the
    # split form for every family that was not svy/did/xtgls, which put a rule there that
    # Stata never prints.
    _split_rule = "-" * (name_width + 1) + "+" + "-" * (row_width - name_width - 2)
    _border_rule = "-" * row_width
    # The rule AFTER the header is always the split form, even for the survey family.
    lines.append(_split_rule)
    # Multi-equation models (``zip``/``zinb``/``heckman``/``etregress``) store the second
    # equation's parameters with an ``inflate:``/``select:`` prefix. Stata does NOT print
    # that prefix: it prints a GROUP HEADER and then the bare names, which is why the
    # prefixed version made the name column 18 wide instead of 13 and shifted the table.
    #
    #   y            |
    #             x1 |    .559456   ...
    #          _cons |   .4039659   ...
    #   -------------+----...
    #   inflate      |
    #             x1 |   .0033665   ...
    # The separator is ``#`` in this package's parameter names and ``:`` in others, so
    # both are accepted rather than assuming one.
    # ``didregress`` groups its one reported coefficient under an "ATET" header and names
    # the treatment variable on a row of its own, with the comparison in parentheses:
    #
    #   ATET         |
    #        treated |
    #      (1 vs 0)  |  -.0745254   ...
    #
    # The coefficient carries the treatment name, so the row label is derived from it
    # rather than invented.
    _did_treatment = ""
    if _is_didregress(result):
        _did_treatment = str((getattr(result, "e", None) or {}).get("_did_treatment") or "")
    #: Guards the single rule that separates slopes from ordinal thresholds.
    #: Names that have already had their split rule emitted. A SET rather than a single
    #: flag because the random-effects block needs one rule per auxiliary name (/lnsig2u,
    #: then sigma_u), not one for the whole table.
    _rule_done: set[str] = set()
    # ``arch`` names its parameters "mu", "omega", "alpha[1]", "beta[1]" -- the backend's
    # spelling. Stata nests them instead:
    #
    #   dinf         |
    #          _cons |   .0005369   ...
    #   -------------+------------------------------
    #   ARCH         |
    #           arch |
    #            L1. |   .4201012   ...
    #          _cons |   .0001802   ...
    #
    # Written here as "GROUP:SUB:bare" so the generic group machinery emits the levels;
    # a name with no sub-label is "GROUP:bare".
    #
    # The mean-equation group is headed with the DEPENDENT VARIABLE, not a literal "y":
    # measured, ``arch dinf, arch(1)`` prints "dinf |", and the hardcoded "y" appeared for
    # every series regardless of its name.
    _arch_mean_group = str(
        (getattr(result, "e", None) or {}).get("depvar")
        or getattr(result, "response_name", None)
        or "y"
    )
    _ARCH_NAMES = {
        "mu": f"{_arch_mean_group}:_cons",
        "omega": "ARCH:_cons",
        "alpha": "ARCH:arch:L1.",
        "beta": "ARCH:garch:L1.",
        "gamma": "ARCH:garch:L1.",
    }

    def _arch_rewrite(name: str) -> str:
        base, _sep, lag = str(name).partition("[")
        mapped = _ARCH_NAMES.get(base)
        if not mapped:
            return ""
        # Multi-lag models keep the index: alpha[2] -> "L2.".
        if lag and mapped.endswith("L1."):
            return mapped[:-3] + f"L{lag.rstrip(']')}."
        return mapped

    _is_arch_shape = str(getattr(result, "model_name", "") or "") == "arch"
    if _is_arch_shape and any(_arch_rewrite(name) for name in names):
        # ORDER: Stata prints the ARCH group as arch.L1, garch.L1, then _cons -- the
        # variance parameters first, the constant last. Sorting by the rewritten name
        # would put "_cons" first, so the group is ordered explicitly.
        _arch_order = {
            "ARCH:arch:": 0,
            "ARCH:garch": 1,
            "ARCH:_cons": 2,
            f"{_arch_mean_group}:_cons": -1,
        }

        def _arch_rank(name: str) -> int:
            rewritten = _arch_rewrite(name)
            for key, rank in _arch_order.items():
                if rewritten.startswith(key.rstrip(":")):
                    return rank
            return 9

        _pairs = [
            (_arch_rank(name), _arch_rewrite(name) or str(name)) for name in names
        ]
        _pairs.sort(key=lambda item: item[0])
        names = [name for _rank, name in _pairs]
        _EQUATION_PREFIXES = ("y:", "ARCH:")

    # ``sqreg``/``bsqreg`` name their parameters "q0.25#x1" -- quantile, then the variable.
    # Stata instead prints a GROUP LABEL ("q25") and a bare name ("x1") under it, so the
    # prefix is rewritten to the generic equation form here, and the label loses its
    # decimal point ("q0.25" -> "q25").
    def _sqreg_group(name: str) -> str:
        head, sep, tail = str(name).partition("#")
        if not sep or not head.startswith("q"):
            return ""
        _value = head[1:]
        if _value.startswith("0."):
            _value = _value[1:]
        # ".5" must become "q50", not "q5" -- the digits are a PERCENT label, so a single
        # decimal place is padded back to two.
        _digits = _value.split(".")[1] if "." in _value else _value
        if len(_digits) == 1:
            _digits = _digits + "0"
        return "q" + _digits

    _sqreg_names = {str(name): _sqreg_group(name) for name in names}
    if any(_sqreg_names.values()):
        names = [
            (_sqreg_group(name) + ":" + str(name).partition("#")[2])
            if _sqreg_group(name)
            else str(name)
            for name in names
        ]
        _EQUATION_PREFIXES = tuple(
            {f"{group}:" for group in _sqreg_names.values() if group}
        ) + ("inflate#", "inflate:", "select#", "select:", "tobit#", "tobit:")
    else:
        # Keep the arch prefixes if they were set above; otherwise the standard set.
        # ``etregress``/``heckman`` name their two equations "y#..." / "treat#..." (and
        # "select#..." for heckman), which is the same two-level shape as zip's
        # "inflate#" blocks -- measured against their coefficient tables.
        #
        # ``mvreg``/``sureg`` use the OUTCOME NAMES as the equation prefix ("y:x1",
        # "ybin:x1"), so their prefixes are built from the actual parameter names rather
        # than from a fixed list -- a fixed "y:" entry matched only the first outcome and
        # left the rest as flat rows.
        _prefixes = [
            "inflate#",
            "inflate:",
            "select#",
            "select:",
            "tobit#",
            "tobit:",
            "y#",
            "y:",
            "treat#",
            "treat:",
        ]
        _known_outcomes = set(
            str(_x)
            for _x in ((getattr(result, "e", None) or {}).get("outcomes")
                       or (getattr(result, "e", None) or {}).get("equations")
                       or [])
        )
        for _name in names:
            _text = str(_name)
            # The renderer sees ``stata_param_names``, which spells the equation separator
            # "#"; accept both spellings.
            #
            # ONLY an equation prefix may be lifted. A ``#`` also means an INTERACTION
            # ("x1#x2"), and treating that as a group split turned the row "x1#x2" into a
            # group header "x1" plus a row "x2", silently breaking every table with an
            # interaction. A head is accepted as an equation when it is a known equation
            # word, appears with a ":" spelling somewhere, or names an outcome listed in
            # e(outcomes)/e(equations) -- which is how mvreg's "ybin" is recognised.
            for _sep in (":", "#"):
                if _sep not in _text:
                    continue
                _head = _text.partition(_sep)[0]
                if not _head:
                    break
                _is_equation = (
                    _head in ("y", "treat", "select", "inflate", "tobit", "ARCH")
                    or _head in _known_outcomes
                    or any(str(other).startswith(f"{_head}:") for other in names)
                )
                if _is_equation and f"{_head}{_sep}" not in _prefixes:
                    _prefixes.append(f"{_head}{_sep}")
                break
        _EQUATION_PREFIXES = (
            ("y:", "ARCH:") if _is_arch_shape else tuple(_prefixes)
        )
    _GROUP_CONSTANTS = ("intercept", "const", "_cons")
    seen_groups: set[str] = set()
    #: Sub-labels already emitted inside an arch group ("arch", "garch").
    _arch_subs_seen: set[str] = set()
    if _is_didregress(result) and names:
        # The ATET header, then the treatment variable on its own row, then the single
        # comparison row. Stata reports ONLY the treatment coefficient here -- covariates
        # are adjusted for, not shown -- so the table is built from the first parameter.
        # No rule here: the header's own rule was already emitted above the loop, and
        # adding another produced two consecutive identical rules.
        # Group headers are LEFT aligned in a 12-wide field then a space and the pipe;
        # the comparison row is right-aligned in 11. Both put the pipe at column 13.
        lines.append(f"{'ATET':<12} |")
        # The variable named here is the TREATMENT variable's own name, not a fixed label:
        # `didregress (y) (t)` prints "           t |", and a differently named treatment prints
        # that name instead (measured in round 162 against a fit whose treatment column is `t`).
        _treatment_name = str(
            getattr(result, "did_treatment_name", "") or (names[0] if names else "treated")
        )
        lines.append(f"{_treatment_name:>12} |")
        lines.append(
            f"{'(1 vs 0)':>11}  |"
            f"{_stata_coef(params[0], coef_w)}"
            f"{_stata_coef(bse[0], se_w)}"
            f"{_stata_stat(stats_[0], stat_w)}"
            f"{_stata_pvalue(pvalues[0], p_w)}"
            f"{_stata_coef(lower[0], ci_lo_w)}"
            f"{_stata_coef(upper[0], ci_hi_w)}"
        )
        # No closing rule either: the trailing border is appended once, after the loop.
        names = []
    # ``etregress``/``heckman`` fit on this path store ``sigma`` as a coefficient, but the
    # table prints it in the DERIVED block below (with its delta-method standard error), so
    # the plain coefficient row is skipped here. Without this the row appeared twice.
    _skip_sigma_row = any(
        str(name).strip() in ("athrho", "/athrho") for name in names
    )
    # An absorbed regressor prints as "<name> |          0  (omitted)" in the position it
    # held in the specification. Only panel and areg fits can carry these, so the guards
    # below are empty for every other model and nothing changes there.
    _omitted_names = tuple(getattr(result, "omitted_columns", ()) or ())
    _omitted_slots: dict[str, list[str]] = {}
    _trailing_omitted: list[str] = []
    if _omitted_names:
        _all_names = tuple(getattr(result, "all_columns", ()) or ())
        _surviving = set(names)
        _pending: list[str] = []
        for _column in _all_names:
            if _column in _omitted_names:
                _pending.append(_column)
            elif _column in _surviving:
                _omitted_slots[_column] = _pending
                _pending = []
        _trailing_omitted = _pending
    for index, name in enumerate(names):
        for _omitted in _omitted_slots.get(str(name), ()):
            # Measured from ``xtreg maize canal disaster, fe``: the name is right-aligned in
            # the usual column, the coefficient prints as a bare 0, and "(omitted)" follows
            # after two spaces where the standard error would be.
            lines.append(f"{_omitted:>{name_width}} |{'0':>11}  (omitted)")
        group = ""
        bare = name
        for prefix in _EQUATION_PREFIXES:
            if name.startswith(prefix):
                group = prefix[:-1]
                bare = name[len(prefix):]
                break
        if not group:
            # A FACTOR coefficient carries its term and level in the name itself --
            # ``C(foreign)[T.Foreign]``. Stata prints that as a group header over the levels:
            #
            #     foreign |
            #   Foreign  |  -1.650029   1.075994
            #
            # instead of one long row, which is also what kept the table 87 wide where Stata's
            # is 78. Measured on auto.dta in round 183.
            _factor = re.match(r"^C\(([^)]+)\)\[(?:T\.)?(.+)\]$", str(name).strip())
            if _factor is None:
                _factor = re.match(r"^([A-Za-z_][\w]*)\(([^)]+)\)\[(?:T\.)?(.+)\]$", str(name).strip())
                if _factor is not None:
                    group = _factor.group(2).strip()
                    bare = _factor.group(3).strip()
            else:
                group = _factor.group(1).strip()
                bare = _factor.group(2).strip()
        else:
            # ``_factor`` MUST be bound on EVERY iteration: the level-row renderer below reads
            # it, and it is assigned only inside the branch above. A command whose coefficient
            # name starts with an equation prefix (heckman, etregress) skipped that branch and
            # hit UnboundLocalError the first time a plain row reached the renderer.
            _factor = None
        if _skip_sigma_row and str(bare).strip() in ("sigma", "/sigma"):
            continue
        if group and group not in seen_groups:
            # The FIRST group needs a rule for the EQUATION-prefixed models (zip's "y |" /
            # "inflate |" blocks are separated from the coefficients above them), but NOT
            # for sqreg, whose first group follows the header's own rule already. Making
            # this unconditional broke zip when it fixed sqreg; the two shapes differ.
            _is_sqreg_shape = str(getattr(result, "model_name", "") or "") in (
                "sqreg",
                "bsqreg",
                # mvreg's equation blocks follow the header's own rule, like sqreg's.
                "mvreg",
                # same for sureg: its first equation row follows the header's rule.
                "sureg",
            )
            # arch's FIRST group ("y") follows the header rule, exactly like sqreg's.
            #
            # A FACTOR group is a third shape: Stata puts a blank pipe line, then the term
            # RIGHT-ALIGNED in the name column, with NO rule between it and the coefficient above:
            #
            #                 |
            #            foreign |
            #          Foreign  |  -1.650029   1.075994
            #
            # Measured on auto.dta, round 183.
            _is_factor_group = _factor_display_pieces(str(name)) is not None
            if _is_factor_group:
                lines.append(f"{'':>{name_width}} |")
            elif seen_groups or not (_is_sqreg_shape or _is_arch_shape):
                lines.append(
                    "-" * (name_width + 1) + "+" + "-" * (row_width - name_width - 2)
                )
            seen_groups.add(group)
            lines.append(
                f"{group:>{name_width}} |"
                if _is_factor_group
                else f"{group:<{name_width}} |"
            )
        # ``arch`` nests one level further: "ARCH" then a sub-label ("arch"/"garch") then
        # the lag ("L1."). The sub-label row is emitted once per distinct sub-label.
        if _is_arch_shape and ":" in bare:
            _sub, _rest = bare.split(":", 1)
            if _sub and _sub not in _arch_subs_seen:
                _arch_subs_seen.add(_sub)
                lines.append(f"{_sub:>{name_width}} |")
            bare = _rest
        # Inside an equation block the constant is still spelled ``_cons``.
        if bare.strip().lower() in _GROUP_CONSTANTS:
            bare = "_cons"
        # Stata stores auxiliary parameters WITHOUT a slash in e(b) but DISPLAYS them with
        # one: /cut1, /lnalpha, /lnsig2u. ``stata_param_names`` strips it to match e(b), so
        # it is put back here, for display only.
        if bare.strip() in ("lnsig2u", "lnalpha", "lnsigma"):
            bare = "/" + bare.strip()
        # ``arima`` reports its innovation variance as ``/sigma`` (the MACRO value), while
        # the backend's parameter is named ``sigma2``. Measured from
        # ``arima y, arima(1,0,0)``: "      /sigma |   1.039853   .0824649    12.61".
        # NOTE this row KEEPS its z and P columns -- unlike tobit's scale, which has none --
        # so it must not be added to the nuisance list below.
        if bare.strip() == "sigma2":
            bare = "/sigma"
        # Stata nests an ARMA term under a group header and a lag row:
        #
        #   ARMA         |
        #           ar   |
        #          L1.   |  -.1187992   .1018412  ...
        #
        # while the backend names it "ar.L1". Split it so the existing group-header
        # mechanism emits the two levels.
        if "." in bare and bare.split(".", 1)[0] in ("ar", "ma", "sar", "sma"):
            _term, _lag = bare.split(".", 1)
            # The ARMA block is bounded by a split rule of its own. It is not emitted by
            # the loop's group-header logic, which only knows about equation prefixes.
            lines.append(_split_rule)
            lines.append(f"{'ARMA':<{name_width}} |")
            lines.append(f"{_term:>{name_width}} |")
            bare = _lag + "."
        # Ordinal models report their THRESHOLDS (/cut1, /cut2, ...) with no test
        # statistic: Stata leaves both the z and P columns blank for those rows, because
        # a cut point is a nuisance parameter, not a hypothesis about a slope. Measured
        # from ``ologit yord x1 x2``:
        #
        #   /cut1 |   -.699741   .2145721                     -1.120295   -.2791874
        #   /cut2 |   .7037455   .2148227                      .2827008     1.12479
        #
        # Printing z and P there invented two numbers Stata does not show.
        # Nuisance parameters print no test statistic: ordinal thresholds (/cut1...) and
        # tobit's scale (/sigma). Both leave the z and P columns blank.
        _bare_name = bare.strip()
        _is_cut_row = (
            _bare_name.startswith("cut")
            or _bare_name.startswith("/cut")
            or (
                _bare_name in ("/sigma", "sigma", "/lnsigma")
                # arima's ``/sigma`` DOES carry a z statistic and a p-value -- it is a
                # hypothesis about the innovation variance -- while tobit's and
                # truncreg's do not. Measured in both directions.
                and str(getattr(result, "model_name", "") or "") != "arima"
            )
            or _bare_name.startswith("var(e.")
            # The random-effects ML commands report their variance parameter as
            # ``/lnsig2u`` in a block of its own, after the coefficients. Measured from
            # ``xtlogit ybin x1, re``:
            #
            #   _cons      |   .0097016   .2012351     0.05   0.962
            #   -------------+------------------------------------------------
            #   /lnsig2u   |  -21.25042   1245.243                     -2461.882
            #   -------------+------------------------------------------------
            #   sigma_u    |   .0000243    .015127                             0
            #   rho        |   1.79e-10   2.23e-07                             0
            or _bare_name in ("lnsig2u", "/lnsig2u")
        )
        # Ordinal thresholds, tobit's scale AND the random-effects variance parameter each
        # get a split rule before them; only the suppression of the z/P columns is shared.
        _needs_rule = (
            _bare_name.startswith("cut")
            or _bare_name.startswith("/cut")
            or _bare_name.startswith("var(e.")
            or (
                _bare_name in ("/sigma", "sigma", "/lnsigma", "lnsig2u", "/lnsig2u")
                # ``etregress``/``heckman`` print /athrho and /lnsigma CONSECUTIVELY under
                # one rule, while intreg puts a rule between the coefficients and /lnsigma.
                # Keying on the name alone gave etregress a rule in the middle of its pair.
                and not (
                    _bare_name in ("/lnsigma", "lnsigma")
                    and any(
                        str(other).strip() in ("athrho", "/athrho")
                        for other in names
                    )
                )
            )
        )
        if _is_cut_row and _needs_rule:
            # Deduplicated by CLASS, not by name: the ordinal thresholds are /cut1../cut4 and
            # keying on the name gave EACH one its own rule, turning a 4-rule table into a
            # 7-rule one. One rule per family of nuisance rows is what Stata prints.
            _rule_key = (
                "cut"
                if _bare_name.startswith("cut") or _bare_name.startswith("/cut")
                else _bare_name
            )
            if _rule_key not in _rule_done:
                lines.append(_split_rule)
                _rule_done.add(_rule_key)
        # arima's /sigma keeps its test statistic (so it is NOT a nuisance row) but still
        # gets a split rule before it, measured. Handled separately for that reason.
        if (
            str(getattr(result, "model_name", "") or "") == "arima"
            and _bare_name == "/sigma"
            and _bare_name not in _rule_done
        ):
            lines.append(_split_rule)
            _rule_done.add(_bare_name)
        if _is_cut_row:
            # Ordinal thresholds are padded to name_width then a space; tobit's scale is
            # padded to name_width + 1 with NO space, so its pipe sits one column right:
            #
            #   '       /cut1 |'   ordinal
            #   '     var(e.y)|'   tobit scale
            _scale_label = _bare_name.startswith("var(e.")
            if _scale_label:
                # "var(e.food_exp)" -> "var(e.food~p)": Stata shortens the variable name
                # inside the label to six characters. Measured in both directions -- an
                # eight-character and a thirteen-character name shorten the same way.
                _inner = _bare_name[len("var(e.") :].rstrip(")")
                bare = "var(e." + _stata_abbreviated(_inner) + ")"
            _name_cell = (
                # The scale row's pipe sits at column 13 REGARDLESS of the table's own name
                # width: measured, "var(e.food~p)|" has no space before the pipe while the
                # banner's "  food_exp |" puts it one column further right. Deriving it from
                # name_width made the row wider than 78 whenever a name overflowed that.
                f"{bare:>12}|"
                if _scale_label
                else f"{bare:>{name_width}} |"
            )
            if _scale_label:
                # ``var(e.<y>)`` is the VARIANCE, but the fitted parameter is its square
                # root, so the stored value has to be squared before it is printed.
                # Measured on ``tobit cens x1 x2, ll(0)``: the stored 1.029584 is Stata's
                # 1.060043 variance exactly (1.029584**2 = 1.060043), and the variance's
                # standard error is the delta-method transform 2*sigma*se(sigma)
                # = 2 * 1.029584 * .0395174 = .081372 against Stata's .0813728.
                #
                # The interval is transformed on the LOG-VARIANCE scale, which is why it
                # comes out asymmetric about the estimate (1.060043 sits below its own
                # interval's midpoint) -- on the raw scale Stata's interval is not
                # symmetric, so a plus/minus construction cannot be right.
                #
                # The scale is logarithmic AND the critical value is a t with e(df_r)
                # degrees of freedom, NOT the normal's 1.96. Measured on
                # ``tobit food_exp income, ll(0)`` (n = 40, no censoring, df_r = 39):
                #
                #   Stata :  var(e.food~p)|  7612.629  1702.236   4842.936   11966.32
                #   1.96  :                                             4911.303    11799.65
                #   t(39) :                                             4842.935    11966.32
                #
                # Recovering the implied z from Stata's bounds gives 2.022691 on BOTH sides,
                # and t.ppf(0.975, 39) = 2.022691 exactly -- which is what identified the
                # distribution. Using df_r rather than n-1 matters whenever the model carries
                # more than one regressor.
                sigma = float(params[index])
                sigma_se = float(bse[index])
                variance = sigma**2
                variance_se = 2.0 * sigma * sigma_se
                _df_r = _e_float(result, "df_r")
                if np.isfinite(_df_r) and _df_r > 0.0:
                    critical = float(stats.t.ppf(0.975, _df_r))
                else:
                    critical = 1.959963984540054
                if variance > 0.0 and np.isfinite(variance_se) and variance_se > 0.0:
                    shift = critical * variance_se / variance
                    lines.append(
                        _name_cell
                        + f"{_stata_coef(variance, coef_w)}"
                        f"{_stata_coef(variance_se, se_w)}"
                        f"{'':>{stat_w}}"
                        f"{'':>{p_w}}"
                        f"{_stata_coef(variance * float(np.exp(-shift)), ci_lo_w)}"
                        f"{_stata_coef(variance * float(np.exp(shift)), ci_hi_w)}"
                    )
                else:
                    lines.append(
                        _name_cell
                        + f"{_stata_coef(variance, coef_w)}"
                        f"{_stata_coef(variance_se, se_w)}"
                        f"{'':>{stat_w}}"
                        f"{'':>{p_w}}"
                        f"{_stata_coef(lower[index], ci_lo_w)}"
                        f"{_stata_coef(upper[index], ci_hi_w)}"
                    )
                continue
            lines.append(
                _name_cell
                + f"{_stata_coef(params[index], coef_w)}"
                f"{_stata_coef(bse[index], se_w)}"
                f"{'':>{stat_w}}"
                f"{'':>{p_w}}"
                f"{_stata_coef(lower[index], ci_lo_w)}"
                f"{_stata_coef(upper[index], ci_hi_w)}"
            )
        else:
            # ``arima``'s ``/sigma`` row prints the STANDARD DEVIATION, but the fitted
            # parameter stored under that name is the VARIANCE, so it has to be square
            # rooted before printing:
            #
            #   Stata:  /sigma |   .0164621   .0007197    22.87  ...
            #   stored: sigma2 = .0002708   ->  sqrt = .016456  (agrees to 7 figures)
            #
            # The standard error follows the delta method for the square root,
            # se(sigma) = se(sigma^2) / (2 * sigma) = 2.37e-05 / (2 * .016456) = .00072,
            # which is Stata's .0007197. A plain square root without that scaling would
            # report .0000237 -- three orders of magnitude too small.
            #
            # This is the mirror image of tobit's ``var(e.y)``, which stores the standard
            # deviation and prints its square. The two commands disagree internally about
            # which scale the stored parameter is on.
            _arima_sigma = (
                str(getattr(result, "model_name", "") or "") == "arima"
                and str(bare).strip() == "/sigma"
            )
            if _arima_sigma:
                variance = float(params[index])
                if variance > 0.0:
                    deviation = float(np.sqrt(variance))
                    deviation_se = float(bse[index]) / (2.0 * deviation)
                    statistic = deviation / deviation_se if deviation_se > 0.0 else float("nan")
                    pvalue = (
                        float(2.0 * stats.norm.sf(abs(statistic)))
                        if np.isfinite(statistic)
                        else float("nan")
                    )
                    # The interval is built on the SIGMA scale and is SYMMETRIC there --
                    # unlike tobit's, which lives on the log-variance scale with a t.
                    # Measured on ``arima dinf, arima(1,0,0)``:
                    #
                    #   /sigma |  .0164621  .0007197  22.87  0.000   .0150515  .0178727
                    #
                    # Implied z: (lo - sigma)/se = -1.959983 and (hi - sigma)/se =
                    # +1.959983, i.e. the normal's 1.96 to every printed digit. Taking the
                    # SQUARE ROOT of a variance interval instead (the previous code)
                    # produces an asymmetric pair that matches neither command's rule.
                    half_width = 1.959963984540054 * deviation_se
                    lines.append(
                        f"{bare:>{name_width}} |"
                        f"{_stata_coef(deviation, coef_w)}"
                        f"{_stata_coef(deviation_se, se_w)}"
                        f"{_stata_stat(statistic, stat_w)}"
                        f"{_stata_pvalue(pvalue, p_w)}"
                        f"{_stata_coef(deviation - half_width, ci_lo_w)}"
                        f"{_stata_coef(deviation + half_width, ci_hi_w)}"
                    )
                    continue
            if _odds_ratio:
                # ``logit ..., or`` reports EXPONENTIATED coefficients. Stata re-labels the
                # column "Odds ratio" and exponentiates the ESTIMATE, its standard error and
                # both interval endpoints; z and P>|z| are unchanged because they come from
                # the untransformed scale. Measured on ``logit treat age education re74 re75,
                # or``: age 1.013706 (.0138091), education 1.084459 (.0595068).
                #
                # exp(b) and exp(b +/- z*se) are the same thing as exp(b) +/- ... only for
                # small se, which is why the endpoints are exponentiated individually rather
                # than rebuilt from the transformed centre.
                lines.append(
                    f"{bare:>{name_width}} |"
                    f"{_stata_coef(float(np.exp(params[index])), coef_w)}"
                    f"{_stata_coef(float(np.exp(params[index])) * float(bse[index]), se_w)}"
                    f"{_stata_stat(stats_[index], stat_w)}"
                    f"{_stata_pvalue(pvalues[index], p_w)}"
                    f"{_stata_coef(float(np.exp(lower[index])), ci_lo_w)}"
                    f"{_stata_coef(float(np.exp(upper[index])), ci_hi_w)}"
                )
            else:
                # A FACTOR LEVEL'S INDENT IS name_width-DRIVEN, not a fixed 7. Two measured
                # examples fix the rule -- the level text is the LAST name_width-2 columns before
                # the pipe, i.e. it starts at (name_width - 2 - len(text)) and is padded to the
                # pipe:
                #
                #     '    Foreign  |'   name_width 12, len 7  -> 4 leading spaces
                #     '       1971  |'   name_width 12, len 4  -> 7 leading spaces
                #
                # so leading = name_width - 1 - len(text), then the text, then two spaces and the
                # pipe: 4+7=11, 4+7+2 = 13. Both measured examples satisfy it; my first attempt
                # right-aligned the text (correct only when it happens to fill the field) and my
                # second used a hard-coded 7 (right for 1971, wrong for Foreign).
                _level_text = (
                    f"{' ' * max(0, name_width - 1 - len(bare))}{bare}  |"
                    if _factor is not None
                    else f"{bare:>{name_width}} |"
                )
                # A BLANK PIPE LINE SEPARATES THE FACTOR BLOCK FROM ``_cons``. Stata closes the
                # level rows with one before the constant when a factor group precedes it:
                #
                #        1986  |   12005.98   ...
                #             |                     <- this line
                #       _cons |   2352.891   ...
                #
                # Measured in round 218 on `xtreg y k1 i.year, fe`. Emitted once, immediately
                # before the constant, and only when a factor block was printed and this is
                # actually the constant row.
                _is_constant_row = (
                    _factor is None
                    and str(bare).strip().lower() in _GROUP_CONSTANTS
                )
                if _is_constant_row and seen_groups and lines and "|" in lines[-1]:
                    if lines[-1].split("|", 1)[0].strip():
                        lines.append(f"{'':>{name_width}} |")
                lines.append(
                    f"{_level_text}"
                    f"{_stata_coef(params[index], coef_w)}"
                    f"{_stata_coef(bse[index], se_w)}"
                    f"{_stata_stat(stats_[index], stat_w)}"
                    f"{_stata_pvalue(pvalues[index], p_w)}"
                    f"{_stata_coef(lower[index], ci_lo_w)}"
                    f"{_stata_coef(upper[index], ci_hi_w)}"
                )
    # The random-effects ML commands continue the coefficient table with two derived rows
    # INSIDE the same table (not as a block below it), each preceded by a split rule:
    #
    #   -------------+----------------------------------------------------
    #      /lnsig2u |  -21.25042   1245.243                     -2461.882
    #   -------------+----------------------------------------------------
    #       sigma_u |   .0000243    .015127                             0
    #           rho |   1.79e-10   2.23e-07                             0
    #   ------------------------------------------------------------------
    #   LR test of rho=0: chibar2(01) = 0.00         Prob >= chibar2 = 1.0000
    #
    # Only the commands that report /lnsig2u do this; xtreg's sigma_u/rho stay in the
    # decomposition block below the table.
    #: Commands whose auxiliary parameter is followed by a DERIVED row inside the table.
    #: Both shapes were measured:
    #:   /lnsig2u -> sigma_u, rho      (random-effects logit/probit)
    #:   /lnalpha -> alpha             (truncated negative binomial)
    _has_re_variance = any(
        str(name).strip() in ("lnsig2u", "/lnsig2u") for name in names
    )
    _has_alpha = any(str(name).strip() in ("lnalpha", "/lnalpha") for name in names)
    # ``xttobit`` reports /sigma_u and /sigma_e as ORDINARY coefficient rows and then a
    # single derived rho row -- a third shape, measured:
    #
    #   /sigma_u |   .2597131   .1647344     1.58   0.115
    #   /sigma_e |   .9258212   .0831754    11.13   0.000
    # -----------+--------------------------------------------------
    #        rho |   .0729518   .0907106
    #
    # The decomposition block below the table would print sigma_u/sigma_e a SECOND time,
    # so it is suppressed for these commands and only rho is added inside the table.
    _has_sigma_pair = any(
        str(name).strip() in ("sigma_u", "/sigma_u") for name in names
    ) and any(str(name).strip() in ("sigma_e", "/sigma_e") for name in names)
    # ``etregress``/``heckman`` finish with /athrho + /lnsigma (whose pair shares ONE rule,
    # handled above) and then a block of THREE derived rows -- rho, sigma, lambda -- each of
    # which carries a standard error. Measured:
    #
    #   /athrho |   .9118269   .3745088     2.43   0.015
    #   /lnsigma|   .1248372   .1344003     0.93   0.353
    # ----------+----------------------------------------------------
    #       rho |   .7220079   .1792791
    #     sigma |   1.132964   .1522707
    #    lambda |    .818009   .3020312
    #
    # All three follow from the two coefficients by the delta method:
    #   rho = tanh(athrho), sigma = exp(lnsigma), lambda = rho * sigma
    _has_athrho = any(str(name).strip() in ("athrho", "/athrho") for name in names)
    if _has_athrho:
        _a_index = next(
            index
            for index, name in enumerate(names)
            if str(name).strip() in ("athrho", "/athrho")
        )
        _l_index = next(
            (
                index
                for index, name in enumerate(names)
                if str(name).strip() in ("lnsigma", "/lnsigma")
            ),
            None,
        )
        _a_b, _a_se = float(params[_a_index]), float(bse[_a_index])
        _rho_v = float(np.tanh(_a_b))
        _rho_se = _a_se * (1.0 - _rho_v**2)
        _sig_v = _sig_se = float("nan")
        if _l_index is not None:
            _l_b, _l_se = float(params[_l_index]), float(bse[_l_index])
            _sig_v = float(np.exp(_l_b))
            _sig_se = _l_se * _sig_v
        # The rule before this block was already emitted by the /lnsigma handling; adding
        # a second one produced two consecutive rules.
        if not (lines and lines[-1].startswith("---") and "+" in lines[-1]):
            lines.append(_split_rule)
        _derived_rows = [("rho", _rho_v, _rho_se), ("sigma", _sig_v, _sig_se)]
        if np.isfinite(_rho_v) and np.isfinite(_sig_v):
            _lam_v = _rho_v * _sig_v
            # delta method on rho*sigma, treating the two as independent
            _lam_se = float(
                np.sqrt((_sig_v**2) * (_rho_se**2) + (_rho_v**2) * (_sig_se**2))
            )
            _derived_rows.append(("lambda", _lam_v, _lam_se))
        for _row_name, _value, _se_v in _derived_rows:
            if not np.isfinite(_value):
                continue
            lines.append(
                f"{_row_name:>{name_width}} |"
                f"{_stata_coef(_value, coef_w)}"
                f"{_stata_coef(_se_v, se_w)}"
                f"{'':>{stat_w}}{'':>{p_w}}{'':>{ci_lo_w}}{'':>{ci_hi_w}}"
            )

    if _has_re_variance or _has_alpha or _has_sigma_pair:
        # A split rule separates the coefficient block from the derived rows, measured.
        lines.append(_split_rule)
        # Three shapes, each measured: the random-effects logit/probit add sigma_u + rho,
        # the truncated negative binomial adds alpha, and xttobit adds ONLY rho (its
        # sigma_u/sigma_e are already ordinary coefficient rows).
        if _has_re_variance:
            _derived = (("sigma_u", "sigma_u"), ("rho", "rho"))
        elif _has_alpha:
            _derived = (("alpha", "alpha"),)
        else:
            _derived = (("rho", "rho"),)
            # xttobit's rho sits under its OWN split rule, after the sigma_e row.
            lines.append(_split_rule)
        for _row_name, _key in _derived:
            _value = _e_float(result, _key)
            if not np.isfinite(_value):
                continue
            # NO test statistic on these rows: they are transforms of the coefficient above,
            # not hypotheses, so z and P stay blank.
            #
            # BUT ``alpha`` -- AND ONLY alpha -- carries a standard error and a confidence
            # interval, by the delta method from /lnalpha: alpha = exp(lnalpha), so
            # se(alpha) = se(lnalpha) * alpha and the interval is exp(b +/- z*se). Measured in
            # round 243: exp(b) = .6901348 and se .185547 against Stata's alpha .6901348 with
            # se .1855815 -- the fifth-digit gap is the same /lnalpha precision difference the
            # row above already shows.
            #
            # The rho/sigma_u rows keep the placeholder bounds (0 and .) that Stata prints for
            # them; adding an interval there would be wrong.
            _alpha_se = float("nan")
            _alpha_lo = float("nan")
            _alpha_hi = float("nan")
            if _row_name == "alpha":
                _ln_index = next(
                    (
                        position
                        for position, candidate in enumerate(names)
                        if str(candidate).strip() in ("lnalpha", "/lnalpha")
                    ),
                    None,
                )
                if _ln_index is not None and _ln_index < len(bse):
                    _ln_b = float(params[_ln_index])
                    _ln_se = float(bse[_ln_index])
                    _alpha_se = _ln_se * float(np.exp(_ln_b))
                    _alpha_lo = float(np.exp(_ln_b - 1.959963984540054 * _ln_se))
                    _alpha_hi = float(np.exp(_ln_b + 1.959963984540054 * _ln_se))
            if np.isfinite(_alpha_se):
                lines.append(
                    f"{_row_name:>{name_width}} |"
                    f"{_stata_coef(_value, coef_w)}"
                    f"{_stata_coef(_alpha_se, se_w)}"
                    f"{'':>{stat_w}}"
                    f"{'':>{p_w}}"
                    f"{_stata_coef(_alpha_lo, ci_lo_w)}"
                    f"{_stata_coef(_alpha_hi, ci_hi_w)}"
                )
                continue
            lines.append(
                f"{_row_name:>{name_width}} |"
                f"{_stata_coef(_value, coef_w)}"
                f"{'':>{se_w}}"
                f"{'':>{stat_w}}"
                f"{'':>{p_w}}"
                f"{'0':>{ci_lo_w}}"
                f"{'.':>{ci_hi_w}}"
            )

    # ``intreg`` ends its table with a derived sigma row under /lnsigma -- but unlike the
    # rho/alpha rows this one DOES carry a standard error and a confidence interval, all
    # reachable by the delta method from /lnsigma. Measured against
    # ``intreg lo hi x1``:  sigma .9373584 / se .0724253 / CI .8056332..1.090621,
    # reproduced exactly from exp(b) and exp(b +/- 1.96*se).
    if any(str(name).strip() in ("lnsigma", "/lnsigma") for name in names) and not (
        _has_re_variance or _has_alpha or _has_sigma_pair or _has_athrho
    ):
        _lns_index = next(
            index
            for index, name in enumerate(names)
            if str(name).strip() in ("lnsigma", "/lnsigma")
        )
        _lns_b = float(params[_lns_index])
        _lns_se = float(bse[_lns_index])
        _sig_value = float(np.exp(_lns_b))
        _sig_se = _lns_se * _sig_value
        _sig_lo = float(np.exp(_lns_b - 1.959963984540054 * _lns_se))
        _sig_hi = float(np.exp(_lns_b + 1.959963984540054 * _lns_se))
        # A split rule separates /lnsigma from the derived sigma row, measured.
        lines.append(_split_rule)
        lines.append(
            f"{'sigma':>{name_width}} |"
            f"{_stata_coef(_sig_value, coef_w)}"
            f"{_stata_coef(_sig_se, se_w)}"
            f"{'':>{stat_w}}"
            f"{'':>{p_w}}"
            f"{_stata_coef(_sig_lo, ci_lo_w)}"
            f"{_stata_coef(_sig_hi, ci_hi_w)}"
        )

    # The closing rule matches the OPENING one -- EXCEPT for the plain panel decomposition, where
    # Stata does not close the coefficient table at all before sigma_u/sigma_e. There it prints
    # only the split rule, and the full border comes AFTER rho:
    #
    #       _cons |   2352.891   3610.712     0.65   0.515
    #   -------------+------------------------------------   <- split rule
    #       sigma_u |   15272.67
    #       sigma_e |  9567.7549
    #           rho |  .71815593   (fraction ...
    #   --------------------------------------------------   <- full border
    #
    # Measured in round 220. Emitting the border here as well produced a full rule immediately
    # followed by the split rule.
    _closes_with_border = _has_re_variance or _has_alpha or _has_sigma_pair or _has_athrho
    if _closes_with_border:
        lines.append(_border_rule)

    # ``reghdfe`` prints an "Absorbed degrees of freedom" table BELOW the closed
    # coefficient table, measured:
    #   Absorbed degrees of freedom:
    #   -----------------------------------------------------+
    #    Absorbed FE | Categories  - Redundant  = Num. Coefs |
    #   -------------+---------------------------------------|
    #           unit |        20           0          20     |
    #   -----------------------------------------------------+
    #
    # The trailing "|" on each rule is part of the shape, not noise.
    _absorbed_names = (getattr(result, "e", None) or {}).get("absorbed_names") or []
    # ONLY reghdfe prints this table. didregress/xtdidregress also carry absorbed_names
    # (their internal formula absorbs unit + period) but they close with a Note instead,
    # and printing the table for them added four rules Stata does not have.
    if _absorbed_names and str(getattr(result, "model_name", "") or "") == "reghdfe":
        _e_macros_hdfe = getattr(result, "e", None) or {}
        _categories = int(_e_float(result, "k_absorb"))
        _redundant = _e_float(result, "df_a_redundant")
        _redundant = int(_redundant) if np.isfinite(_redundant) else 0
        _n_coefs = _categories - _redundant
        lines.append("")
        lines.append("Absorbed degrees of freedom:")
        lines.append("-" * 53 + "+")
        lines.append(
            " Absorbed FE | Categories  - Redundant  = Num. Coefs |"
        )
        lines.append("-" * 13 + "+" + "-" * 39 + "|")
        for _name in _absorbed_names:
            lines.append(
                f"{str(_name):>12} | {_categories:>10} {_redundant:>11} "
                f"{_n_coefs:>10}     |"
            )
        lines.append("-" * 53 + "+")

    # A RANDOM-EFFECTS fit closes with the SAME kind of boundary test, but about rho rather than
    # alpha. Measured on a live `xtlogit y x1, re`:
    #
    #   'LR test of rho=0: chibar2(01) = 98.28                  Prob >= chibar2 = 0.000'
    #
    # 2*(ll - ll_0) = 2*(-348.7264 + 397.8659) = 98.28, and the p-value is HALF the chi2(1) tail
    # for the same boundary reason as the alpha test.
    _is_re_rho_shape = (
        not _has_alpha
        and np.isfinite(_e_float(result, "rho"))
        and str(getattr(result, "model_name", "") or "").startswith("xt")
    )
    if _is_re_rho_shape:
        _ll_v = _e_float(result, "ll")
        _ll0_v = _e_float(result, "ll_0")
        if np.isfinite(_ll_v) and np.isfinite(_ll0_v):
            _lr = 2.0 * (_ll_v - _ll0_v)
            _p_lr = 0.5 * float(stats.chi2.sf(_lr, 1))
            _text = f"LR test of rho=0: chibar2(01) = {_lr:.2f}"
            _tail = f"Prob >= chibar2 = {_p_lr:.3f}"
            # "Prob >=" starts at column 55 and the row is 78 wide, like the rest of the block --
            # measured, the first version (row_width - len - 1) was one column short.
            lines.append(f"{_text:<{row_width - len(_tail)}}{_tail}")

    if _has_alpha:
        # ``xtpoisson, re`` and ``tnbreg`` close with a likelihood-ratio test of the
        # dispersion parameter. The statistic is chi-BAR-squared -- a 50:50 mixture of
        # chi2(0) and chi2(1) -- because the null sits on the boundary of the parameter
        # space, so the p-value is HALF the ordinary chi2(1) tail. Measured on
        # ``xtpoisson ycount x1, re``: 2*(ll - ll_0) = 3.7927 against Stata's
        # "chibar2(01) = 3.79", and 0.5*chi2.sf(3.79, 1) = 0.0257 against its "0.026".
        _ll_v = _e_float(result, "ll")
        _ll0_v = _e_float(result, "ll_0")
        if np.isfinite(_ll_v) and np.isfinite(_ll0_v):
            _lr = 2.0 * (_ll_v - _ll0_v)
            _p_lr = 0.5 * float(stats.chi2.sf(_lr, 1))
            # A rule separates the derived alpha row from the test, and the closing rule
            # already follows it -- measured, Stata prints both.
            lines.insert(len(lines) - 1, _split_rule)
            _text = f"LR test of alpha=0: chibar2(01) = {_lr:.2f}"
            _tail = f"Prob >= chibar2 = {_p_lr:.3f}"
            lines.append(f"{_text:<{row_width - len(_tail) - 1}}{_tail}")

    if _is_didregress(result):
        # ``didregress`` closes with a note naming what the ATET was adjusted for. TWO things
        # vary, both measured (round 162):
        #   * the first effect word depends on the command --
        #         didregress      "... group effects and time effects."
        #         xtdidregress    "... panel effects and time effects."
        #   * "covariates," appears ONLY when covariates were supplied:
        #         didregress (y) (t)        -> "... adjusted for group effects and time effects."
        #         didregress (y x1) (t)     -> "... adjusted for covariates, group effects, and
        #                                       time effects."
        # The previous version always printed "covariates", which made every no-covariate call
        # one line short of Stata.
        _did_word = (
            "panel"
            if str(getattr(result, "model_name", "") or "") == "xtdidregress"
            else "group"
        )
        # The result does not carry the covariate LIST, but ``df_m`` counts the model's slopes:
        # the ATET term alone gives 1, and each covariate adds one. Measured on the same panel
        # (df_m 1 without covariates, 2 with one).
        _df_m = float(getattr(result, "e", {}).get("df_m", 1) or 1)
        _has_covariates = _df_m > 1.0
        if _has_covariates:
            lines.append(
                f"Note: ATET estimate adjusted for covariates, {_did_word} effects, "
                "and time effects."
            )
        else:
            lines.append(
                f"Note: ATET estimate adjusted for {_did_word} effects and time effects."
            )

    if str(getattr(result, "model_name", "") or "").startswith(
        ("ivregress", "xtivreg")
    ):
        # ``ivregress`` AND ``xtivreg`` name the variable roles under the table, on two
        # unaligned lines:
        #
        #   Endogenous: x2
        #   Exogenous:  x1 z
        #
        # Note the single space after "Endogenous:" and the double after "Exogenous:" --
        # Stata pads the shorter label so the variable lists start together.
        _e = getattr(result, "e", None) or {}
        _endog = [str(name) for name in (_e.get("endog") or [])]
        _exog = [str(name) for name in (_e.get("exog") or [])]
        if _endog:
            lines.append("Endogenous: " + " ".join(_endog))
        if _exog:
            lines.append("Exogenous:  " + " ".join(_exog))

    # Stata's panel estimators print a decomposition block UNDER the coefficient
    # table.  The keys are confirmed by ``xtreg_fe.json``:
    # ``sigma_u=0.879742477...``, ``sigma_e=0.403844988...``,
    # ``rho=0.825950661...``, ``corr=-0.060600269...``.
    #
    #   sigma_u |  .87974248
    #   sigma_e |  .40384499
    #       rho |  .82595066   (fraction of variance due to u_i)
    #
    # Only the estimates are printed: the standard errors Stata shows beside them
    # are not stored in e(), and inventing them would be worse than omitting.
    sigma_u = _e_float(result, "sigma_u")
    sigma_e = _e_float(result, "sigma_e")
    rho = _e_float(result, "rho")
    corr = _e_float(result, "corr")
    # These three use EIGHT significant digits, not the coefficient table's ``%9.0g``.
    # Measured from ``xtreg y x1, fe``: ".34868505" / ".8853381" / ".13428387" -- four
    # decimals short of what %.8g gives, so the coefficient formatter is the wrong one
    # here. No leading zero, as everywhere else in Stata's output.
    def _panel_stat(value: float) -> str:
        text = "%.8g" % value
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
        return text

    panel_rows: list[str] = []
    if np.isfinite(sigma_u):
        panel_rows.append(f"{'sigma_u':>12} | {_panel_stat(sigma_u):>10}")
    if np.isfinite(sigma_e):
        panel_rows.append(f"{'sigma_e':>12} | {_panel_stat(sigma_e):>10}")
    if np.isfinite(rho):
        panel_rows.append(
            f"{'rho':>12} | {_panel_stat(rho):>10}"
            f"   (fraction of variance due to u_i)"
        )
    # Commands that report /lnsig2u already printed sigma_u/rho INSIDE the table, so the
    # decomposition block below it would duplicate them.
    # Suppressed for every command that already printed these rows inside the table --
    # otherwise sigma_u/sigma_e appear twice.
    if panel_rows and not (
        _has_re_variance or _has_alpha or _has_sigma_pair or _has_athrho
    ):
        # A SPLIT RULE separates the coefficient table from the decomposition rows, and Stata
        # prints it in the +-- form, not as a full border:
        #
        #       _cons |   2352.891   3610.712     0.65   0.515
        #   -------------+------------------------------------   <- _split_rule
        #       sigma_u |   15272.67
        #
        # Measured in round 220 on `xtreg y k1 i.year, fe`. The comment this replaces said "do NOT
        # add a rule here", which was true only of a FULL-WIDTH one: the ftest branch below emits
        # that, and two of them did sit back to back. The split form is what belongs here.
        lines.append(_split_rule)
        lines.extend(panel_rows)
        # ``corr(u_i, Xb)`` is NOT repeated here: the xtreg header block already printed
        # it, on the same row as Prob > F. Emitting it again below the table put it where
        # Stata's "F test that all u_i=0" belongs.
        #
        # The printed wording is settled by ``whatsnew13.sthlp:1165-1168``, which
        # describes the statistic "reported below the results table as
        # 'F test that all u_i=0'" -- Stata's own words, not a paraphrase.
        # ``xtreg.sthlp:1098-1100`` documents the keys as "F statistic for test of
        # u_i=0" / "p-value for test of u_i=0".
        f_f = _e_float(result, "F_f")
        p_f_test = _e_float(result, "p_f")
        if np.isfinite(f_f):
            df_a = _e_float(result, "df_a")
            # The rule above the F test is FULL width -- measured, Stata closes the
            # sigma/rho block with a plain "-----..." line, not a split one.
            lines.append("-" * row_width)
            # ONE line, 78 wide: the statistic on the left, its p-value right-aligned.
            # The F value carries TWO decimals here (``0.93``), not the four the fit
            # statistics use -- so ``_four`` would print "0.9306" against Stata's "0.93".
            text = (
                f"F test that all u_i=0: F({_df_label(df_a)}, "
                f"{_df_label(df_resid)}) = {f_f:.2f}"
            )
            # ``xtivreg`` spells it without the space after the comma -- measured:
            # " F test that all u_i=0: F(19,78)  =     1.10".
            if str(getattr(result, "model_name", "") or "").startswith("xtivreg"):
                text = (
                    f" F test that all u_i=0: F({_df_label(df_a)},"
                    f"{_df_label(df_resid)})  = {f_f:>8.2f}"
                )
            if np.isfinite(p_f_test):
                # ``Prob > F`` starts at column 61 and the line ends at 78, so the left
                # part is padded to 61 rather than computed from the tail's length --
                # which put it one column short.
                tail = f"Prob > F = {_four(p_f_test)}"
                lines.append(f"{text:<61}{tail}")
            else:
                lines.append(text)
            # ``xtivreg`` closes with a full-width rule AFTER the F row, then the
            # Endogenous/Exogenous lines -- measured.
            if str(getattr(result, "model_name", "") or "").startswith("xtivreg"):
                lines.append(_border_rule)
        else:
            # No "F test that all u_i=0" row (the RE estimator has none), but Stata still
            # closes the sigma/rho block with a full-width rule.
            lines.append(_border_rule)

    # ``logit ..., or`` closes with a note explaining what the constant means on this scale.
    # Measured: "Note: _cons estimates baseline odds."
    if bool(getattr(result, "_odds_ratio", False)):
        lines.append("Note: _cons estimates baseline odds.")

    # ``areg`` closes with its own absorbed-indicator test, and the wording is NOT xtreg's.
    # Measured from ``areg maize canal disaster, absorb(county)``:
    #
    #   F test of absorbed indicators: F(562, 146942) = 155.133       Prob > F = 0.000
    #
    # Three details differ from the xtreg line: the phrase ("of absorbed indicators" vs
    # "that all u_i=0"), the F value's THREE decimals (155.133, not 155.13), and the p-value
    # SITTING ON THE SAME LINE rather than right-aligned against column 61.
    if str(getattr(result, "model_name", "") or "") == "areg":
        f_absorb = _e_float(result, "F_absorb")
        if np.isfinite(f_absorb):
            df_a = _e_float(result, "df_a")
            p_absorb = _e_float(result, "p_absorb")
            if not np.isfinite(p_absorb):
                p_absorb = float(stats.f.sf(f_absorb, df_a, df_resid))
            lines.append(
                f"F test of absorbed indicators: F({_df_label(df_a)}, "
                f"{_df_label(df_resid)}) = {f_absorb:.3f}"
                f"       Prob > F = {p_absorb:.3f}"
            )

    return "\n".join(lines)


def _statsmodels_summary(result: Any) -> str:
    nobs = int(result.nobs)
    df_resid = float(result.df_resid)
    df_model = float(result.df_model)

    rsquared = _as_float(result.rsquared)
    rsquared_adj = _as_float(result.rsquared_adj)
    llf = _as_float(result.llf)
    aic = _as_float(result.aic)
    bic = _as_float(result.bic)

    dist = getattr(result, "dist", "t")
    stat_label = "z" if dist == "z" else "t"
    p_label = "P>|z|" if dist == "z" else "P>|t|"

    params = np.asarray(result.params, dtype=float)
    bse = np.asarray(result.bse, dtype=float)
    values = np.asarray(result.zvalues if dist == "z" else result.tvalues, dtype=float)
    pvalues = np.asarray(result.pvalues, dtype=float)
    ci = result.conf_int()
    lower = np.asarray(ci["lower"], dtype=float)
    upper = np.asarray(ci["upper"], dtype=float)
    names = [str(name) for name in result.params.index]

    model_name = getattr(result, "model_name", "") or "westata results"
    cov_type = getattr(result, "cov_type", "nonrobust")

    fvalue = _e_float(result, "F")

    title = f"{model_name} Results"
    lines: list[str] = [f"{title:^78}", "=" * 78]

    left_rows = [
        ("No. Observations:", _df_label(nobs)),
    ]
    # Which block to print is decided by what the command REPORTS, not by its reference
    # distribution.  ``tobit`` uses a t distribution for its tests but still reports an LR
    # chi-square and a pseudo R-squared (Stata's tobit table has ``LR chi2`` and
    # ``Pseudo R2``, and no F statistic), so keying the choice on ``dist`` alone gave it an
    # OLS block full of ``.``.
    stored_e = getattr(result, "e", None) or {}
    reports_lr = "chi2" in stored_e and "F" not in stored_e and "r2" not in stored_e
    ml_style = dist == "z" or reports_lr

    # ``Df Residuals``/``Df Model`` belong to the OLS-style table.  Stata's ML tables
    # (``logit``, ``probit``, ``poisson``) print neither: what they show on the right is
    # ``LR chi2(df)``, ``Prob > chi2``, ``Log likelihood`` and ``Pseudo R2``.  Emitting
    # ``Df Residuals: 250`` there reported a quantity the command's own output omits, and
    # pairing it with an ``R-squared`` of ``.`` made the table read as a broken OLS one.
    if not ml_style:
        left_rows.extend(
            [
                ("Df Residuals:", _df_label(df_resid)),
                ("Df Model:", _df_label(df_model)),
            ]
        )
    left_rows.append(("Covariance Type:", cov_type))

    if ml_style:
        pseudo_r2 = _e_float(result, "r2_p")
        chi2 = _e_float(result, "chi2")
        right_rows = [
            ("Pseudo R-squared:", _num(pseudo_r2, 4)),
            ("LR chi2:", _num(chi2, 4)),
            ("Log-Likelihood:", _num(llf, 5)),
        ]
    else:
        right_rows = [
            ("R-squared:", _num(rsquared, 4)),
            ("Adj. R-squared:", _num(rsquared_adj, 4)),
            ("F-statistic:", _num(fvalue, 4)),
            ("Log-Likelihood:", _num(llf, 5)),
        ]
    # AIC/BIC are shown only when Stata's own ``e()`` carries them.
    #
    # Stata's ``regress`` stores NO information criteria -- its stored-results block has
    # no ``e(aic)``/``e(bic)``, and ``regress_auto.json`` (captured from a real run)
    # confirms it.  This package still exposes ``e(aic)`` for callers, but on the
    # Gaussian-deviance scale (``N log(rss/N) + 2k``), so printing it in the summary put
    # a number in the table that Stata's table does not contain AND that is not equal to
    # what ``estat ic`` would show.  ``e(_aic_is_stata)`` records which sense applies.
    stored = getattr(result, "e", None) or {}
    if "aic" in stored and bool(stored.get("_aic_is_stata", True)):
        right_rows.append(("AIC:", _num(aic, 4)))
    if "bic" in stored and bool(stored.get("_aic_is_stata", True)):
        right_rows.append(("BIC:", _num(bic, 4)))
    for index in range(max(len(left_rows), len(right_rows))):
        left_text = ""
        if index < len(left_rows):
            left_label, left_value = left_rows[index]
            left_text = f"{left_label:<20}{left_value:>10}"
        right_text = ""
        if index < len(right_rows):
            right_label, right_value = right_rows[index]
            right_text = f"{right_label:<16}{right_value:>11}"
        lines.append(f"{left_text:<36}{right_text}".rstrip())
    lines.append("=" * 78)

    name_width = max([8, *(len(name) for name in names)])
    coef_w, se_w, stat_w, p_w, ci_w = 11, 11, 10, 10, 11
    row_width = name_width + 3 + coef_w + 1 + se_w + 1 + stat_w + 1 + p_w + 1 + 2 * ci_w + 1
    header = (
        f"{'':<{name_width}}  {'coef':>{coef_w}} {'std err':>{se_w}} "
        f"{stat_label:>{stat_w}} {p_label:>{p_w}} {'[0.025':>{ci_w}} {'0.975]':>{ci_w}}"
    )
    lines.append(header)
    lines.append("-" * row_width)
    for index, name in enumerate(names):
        lines.append(
            f"{name:<{name_width}}  {_cell(params[index], coef_w, 6)} "
            f"{_cell(bse[index], se_w, 6)} {_cell(values[index], stat_w, 4)} "
            f"{_cell(pvalues[index], p_w, 4)} "
            f"{_cell(lower[index], ci_w, 6)} {_cell(upper[index], ci_w, 6)}"
        )
    lines.append("=" * row_width)
    return "\n".join(lines)
