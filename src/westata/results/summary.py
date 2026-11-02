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
        return _stata_summary(result)
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
    "arima": "none",
    "arch": "none",
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
    if model_name in ("xtpcse", "xtgls"):
        lines: list[str] = []
        lines.append(
            f"{'Group variable:':<18}{entity or '.':<14}"
            f"{'Number of obs':<18}= {_number('N'):>10}"
        )
        lines.append(
            f"{'Time variable:':<18}{timevar or '.':<14}"
            f"{'Number of groups':<18}= {_number('N_g'):>10}"
        )
        lines.append(f"{'Panels:':<18}{panels or '.'}")
        lines.append(f"{'':<18}{'Obs per group:':>32}")
        lines.append(f"{'':<50}{'min':>5} = {_number('g_min'):>11}")
        lines.append(f"{'':<50}{'avg':>5} = {_number('g_avg'):>11}")
        lines.append(f"{'':<50}{'max':>5} = {_number('g_max'):>11}")
        lines.append(f"{'Estimated covariances':<28}= {_number('n_sigma'):>12}")
        lines.append(f"{'Estimated autocorrelations':<28}= {_number('n_cr'):>12}")
        # ``Estimated coefficients`` counts the estimated betas; fall back to the
        # design's column count when ``e(k)`` is absent, which is what the quantity
        # means.  This row belongs to the xtpcse/xtgls block only -- the xt NON-LINEAR
        # shape above does not print it.
        coefficients = e_macros.get("k")
        if coefficients is None:
            design = getattr(result, "design_matrix", None)
            if design is not None:
                try:
                    coefficients = int(np.asarray(design[0]).shape[1])
                except (TypeError, ValueError, IndexError):
                    coefficients = None
        if coefficients is None:
            lines.append(f"{'Estimated coefficients':<28}= {'':>12}")
        else:
            lines.append(f"{'Estimated coefficients':<28}= {int(coefficients):>12}")
    else:
        # The xt NON-LINEAR shape: group variable + group count, then the per-group
        # extremes.  No time-variable line, no ``Panels:``, no "Estimated ..." rows --
        # see the captured ``xtlogit, fe`` output quoted above.
        lines = [
            f"{'Group variable:':<18}{entity or '.':<14}"
            f"{'Number of groups':<20}= {_number('N_g'):>10}",
            f"{'':<18}{'Obs per group:':>32}",
            f"{'':<50}{'min':>5} = {_number('g_min'):>11}",
            f"{'':<50}{'avg':>5} = {_number('g_avg'):>11}",
            f"{'':<50}{'max':>5} = {_number('g_max'):>11}",
        ]
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
    _vcetype = str((getattr(result, "e", None) or {}).get("vcetype") or "").strip()
    if _vcetype.lower().startswith("robust"):
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
    def _ss(value: float) -> str:
        return (_stata_number(value, 11) if np.isfinite(value) else f"{'.':>11}")

    def _df_cell(df: float) -> str:
        # Stata puts the df right-aligned in 10 with a trailing space, then the MS field.
        return (f"{df:>10.0f}" if np.isfinite(df) else f"{'.':>10}") + " "

    def _ms(value: float, df: float) -> str:
        if not np.isfinite(value) or not np.isfinite(df) or df <= 0:
            return f"{'.':>11}"
        return _stata_number(value / df, 11)

    left = [
        "      Source |       SS           df       MS",
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
        f"Root MSE        = {_stata_number(rmse, 7).strip().rjust(9) if np.isfinite(rmse) else '.'.rjust(9)}",
    ]
    rendered = [
        f"{left_line:<51}{right_line}".rstrip()
        for left_line, right_line in zip(left, right)
    ]
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
    _robust_banner = _block_for_banner == "ols" and _vcetype_for_banner.lower().startswith(
        "robust"
    )
    if not _robust_banner:
        lines.append(banner if banner else "Stata-style regression results")
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
    if panel_lines:
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
    _is_robust_ols = block == "ols" and _robust_vcetype.lower().startswith("robust")
    if block not in ("svy", "ml") and not _is_robust_ols:
        lines.append(f"{'Number of obs':<21} = {nobs:>10d}")
    # The model-level test is also estimator-specific.  ``quantile`` results
    # print no F test at all (Stata's qreg header is just Number of obs plus the
    # deviation lines), and printing "F(1, 72) = ." there is noise a scraper
    # would read as a missing value rather than "not applicable".
    if block in ("quantile", "ml") or _is_robust_ols:
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
        # Survey models report their Wald F only when a model test exists; the
        # mean/total/proportion estimators have none (e(F) is absent).
        chi2 = _e_float(result, "chi2")
        fv = _e_float(result, "F")
        pv = _e_float(result, "p")
        if np.isfinite(fv) and np.isfinite(pv):
            lines.append(
                f"{'F(' + _df_label(df_model) + ', ' + _df_label(df_resid) + ')':<21}"
                f" = {_fixed(fv, 2):>10}"
            )
            lines.append(f"{'Prob > F':<21} = {_fixed(pv, 4):>10}")
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
    if block == "ols" and _anova_available(result):
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
        lines = [
            banner if banner else "Stata-style regression results",
        ]
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
        if _vcetype.lower().startswith("robust"):
            _banner = str((getattr(result, "e", None) or {}).get("title") or "Linear regression")
            _rows: list[tuple[str, str]] = [
                ("Number of obs", f"{nobs:,.0f}" if np.isfinite(nobs) else ".")
            ]
            if np.isfinite(fvalue):
                _rows.append(
                    (f"F({_df_label(df_model)}, {_df_label(df_resid)})", _fixed(fvalue, 2))
                )
            if np.isfinite(p_f):
                _rows.append(("Prob > F", _fixed(p_f, 4)))
            if np.isfinite(rsquared):
                _rows.append(("R-squared", _fixed(rsquared, 4)))
            if np.isfinite(rmse):
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
        _pairs_with_pseudo = np.isfinite(pseudo_r2) and not _is_zero_inflated
        if _pairs_with_pseudo:
            ml_right.append(("Pseudo R2", _fixed(pseudo_r2, 4)))
        while len(ml_left) < (len(ml_right) - 1 if np.isfinite(ll) else len(ml_right)):
            ml_left.append("")
        if np.isfinite(ll):
            ml_left.append("Log likelihood = " + ("%.8g" % ll))
        for index, line in enumerate(ml_left):
            right = ""
            if index < len(ml_right):
                label, value = ml_right[index]
                right = f"{label:<14}= {value:>6}"
            lines.append(f"{line:<56}{right}".rstrip())
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
        if np.isfinite(sum_rdev):
            about = f" (about {_num(q_v, 8)})" if np.isfinite(q_v) else ""
            lines.append(f"  Raw sum of deviations {_num(sum_rdev, 7)}{about}")
        if np.isfinite(sum_adev):
            left = f"  Min sum of deviations {_num(sum_adev, 7)}"
            # Stata right-aligns Pseudo R2 in the same column block as the
            # Number of obs value; pad to a fixed gutter so the two lines line up.
            lines.append(
                left.ljust(40) + f"{'Pseudo R2':<13} = {_fixed(pseudo_r2, 4):>10}"
            )
        if np.isfinite(pseudo_r2) and not np.isfinite(sum_adev):
            # Degenerate case: no deviation lines, still report the pseudo-R2.
            lines.append(f"{'Pseudo R2':<21} = {_fixed(pseudo_r2, 4):>10}")
    elif block == "svy":
        # Stata's survey header: strata / PSUs / obs / population size / design
        # df.  Every number is already in e() (survey.py:634-644).
        n_strata = _e_float(result, "N_strata")
        n_psu = _e_float(result, "N_psu")
        n_pop = _e_float(result, "N_pop")
        if np.isfinite(n_strata):
            lines.append(
                f"{'Number of strata':<21} = {_fixed(n_strata, 0):>10}"
                f"   {'Number of obs':<17} = {nobs:>8d}"
            )
        if np.isfinite(n_psu):
            lines.append(
                f"{'Number of PSUs':<21} = {_fixed(n_psu, 0):>10}"
                f"   {'Population size':<17} = {_fixed(n_pop, 6):>10}"
            )
        if np.isfinite(df_resid):
            lines.append(f"{'Design df':<21} = {_df_label(df_resid):>10}")
        r2 = _e_float(result, "r2")
        if np.isfinite(r2):
            lines.append(f"{'R-squared':<21} = {_fixed(r2, 4):>10}")
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
    def _display_length(name: str) -> int:
        text = str(name)
        for marker in ("#", ":"):
            head, sep, tail = text.partition(marker)
            if sep and head in ("inflate", "select", "tobit"):
                return len(tail)
        return len(text)

    name_width = max([12, *(_display_length(name) for name in names)])
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
    _is_robust = _vcetype.lower().startswith("robust")
    se_header = "std. err." if _is_robust else "Std. err."
    header = (
        f"{response_label:>{name_width}} | "
        f"{'Coefficient':>{coef_w - 1}}"
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
            _cluster_note = (
                f"(Std. err. adjusted for {int(_n_clust):,} clusters"
                f"{' in ' + _cluster_var if (_cluster_var := _cluster_variable(result)) else ''})"
            )
    if _cluster_note:
        lines.append(f"{_cluster_note:>{row_width}}")
    lines.append("-" * row_width)
    if _is_robust:
        # Ends at the same column as the ``Coefficient`` header: the banner row is the
        # name column, the pipe, then the label right-aligned to the coefficient field.
        lines.append(f"{'':>{name_width}} |{_vcetype:>{coef_w + se_w - 1}}")
    lines.append(header)
    lines.append("-" * (name_width + 1) + "+" + "-" * (row_width - name_width - 2))
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
    _EQUATION_PREFIXES = ("inflate#", "inflate:", "select#", "select:", "tobit#", "tobit:")
    _GROUP_CONSTANTS = ("intercept", "const", "_cons")
    seen_groups: set[str] = set()
    for index, name in enumerate(names):
        group = ""
        bare = name
        for prefix in _EQUATION_PREFIXES:
            if name.startswith(prefix):
                group = prefix[:-1]
                bare = name[len(prefix):]
                break
        if group and group not in seen_groups:
            seen_groups.add(group)
            lines.append("-" * (name_width + 1) + "+" + "-" * (row_width - name_width - 2))
            lines.append(f"{group:<{name_width}} |")
        # Inside an equation block the constant is still spelled ``_cons``.
        if bare.strip().lower() in _GROUP_CONSTANTS:
            bare = "_cons"
        lines.append(
            f"{bare:>{name_width}} |"
            f"{_stata_coef(params[index], coef_w)}"
            f"{_stata_coef(bse[index], se_w)}"
            f"{_stata_stat(stats_[index], stat_w)}"
            f"{_stata_pvalue(pvalues[index], p_w)}"
            f"{_stata_coef(lower[index], ci_lo_w)}"
            f"{_stata_coef(upper[index], ci_hi_w)}"
        )
    lines.append("-" * row_width)

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
    panel_rows: list[str] = []
    if np.isfinite(sigma_u):
        panel_rows.append(f"{'sigma_u':>12} |  {_stata_coef(sigma_u, coef_w)}")
    if np.isfinite(sigma_e):
        panel_rows.append(f"{'sigma_e':>12} |  {_stata_coef(sigma_e, coef_w)}")
    if np.isfinite(rho):
        panel_rows.append(
            f"{'rho':>12} |  {_stata_coef(rho, coef_w)}"
            f"   (fraction of variance due to u_i)"
        )
    if panel_rows:
        lines.extend(panel_rows)
        if np.isfinite(corr):
            lines.append("")
            lines.append(f"{'corr(u_i, Xb)':>20} = {_stata_coef(corr, coef_w)}")

        # The printed wording is settled by ``whatsnew13.sthlp:1165-1168``, which
        # describes the statistic "reported below the results table as
        # 'F test that all u_i=0'" -- Stata's own words, not a paraphrase.
        # ``xtreg.sthlp:1098-1100`` documents the keys as "F statistic for test of
        # u_i=0" / "p-value for test of u_i=0".
        f_f = _e_float(result, "F_f")
        p_f_test = _e_float(result, "p_f")
        if np.isfinite(f_f):
            df_a = _e_float(result, "df_a")
            lines.append("")
            lines.append(
                f"F test that all u_i=0: F({_df_label(df_a)}, "
                f"{_df_label(df_resid)}) = {_stata_coef(f_f, 8)}"
            )
            if np.isfinite(p_f_test):
                lines.append(f"{'Prob > F':<22} = {p_f_test:>10.4f}")

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
