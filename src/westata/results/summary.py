"""Human-readable summaries for :class:`westata.results.StataResults`.

Two renderers are provided:

``statsmodels``
    Compact statsmodels-looking table (coefficients, standard errors, test
    statistics, p-values and confidence intervals).
``stata``
    Stata ``regress``-looking layout with ``Number of obs``, model F test,
    R-squared block and the ``Coef. / Std. err. / P>|t| / [95% conf. interval]``
    coefficient table.  For ``dist="z"`` models the test labels switch to
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
_CI_LABEL = "[95% conf. interval]"


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


def _stata_coef(value: float, width: int) -> str:
    """Stata coefficient format: 7 significant digits, no leading zero."""

    text = _num(value, 7)
    if text.startswith("0."):
        text = text[1:]
    elif text.startswith("-0."):
        text = "-" + text[2:]
    return f"{text:>{width}}"


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

    lines: list[str] = []
    model_name = getattr(result, "model_name", "") or "westata results"
    cov_type = getattr(result, "cov_type", "nonrobust")
    lines.append("Stata-style regression results")
    lines.append(f"Model:            {model_name}")
    lines.append(f"Covariance type:  {cov_type}")
    lines.append("")
    def _fixed(value: float, decimals: int) -> str:
        return "." if value is None or not np.isfinite(value) else f"{value:.{decimals}f}"

    lines.append(f"{'Number of obs':<21} = {nobs:>10d}")
    lines.append(
        f"{'F(' + _df_label(df_model) + ', ' + _df_label(df_resid) + ')':<21}"
        f" = {_fixed(fvalue, 2):>10}"
    )
    lines.append(f"{'Prob > F':<21} = {_fixed(p_f, 4):>10}")
    lines.append(f"{'R-squared':<21} = {_fixed(rsquared, 4):>10}")
    lines.append(f"{'Adj R-squared':<21} = {_fixed(rsquared_adj, 4):>10}")
    lines.append(f"{'Root MSE':<21} = {_fixed(rmse, 4):>10}")
    lines.append("")

    name_width = max([8, *(len(name) for name in names)])
    coef_w, se_w, stat_w, p_w, ci_w = 12, 12, 8, 8, 12
    row_width = name_width + 3 + coef_w + 1 + se_w + 1 + stat_w + 1 + p_w + 1 + 2 * ci_w + 1
    header = (
        f"{'':<{name_width}} | {'Coef.':>{coef_w}} {'Std. err.':>{se_w}} "
        f"{stat_label:>{stat_w}} {p_label:>{p_w}} {_CI_LABEL:>{2 * ci_w + 1}}"
    )
    lines.append("-" * row_width)
    lines.append(header)
    lines.append("-" * (name_width + 1) + "+" + "-" * (row_width - name_width - 2))
    for index, name in enumerate(names):
        lines.append(
            f"{name:<{name_width}} | {_stata_coef(params[index], coef_w)} "
            f"{_stata_coef(bse[index], se_w)} {_stata_stat(stats_[index], stat_w)} "
            f"{_stata_pvalue(pvalues[index], p_w)} "
            f"{_stata_coef(lower[index], ci_w)} {_stata_coef(upper[index], ci_w)}"
        )
    lines.append("-" * row_width)
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

    lines: list[str] = []
    lines.append(model_name)
    lines.append(f"{'Covariance type:':<21}{cov_type}")
    lines.append("=" * 78)
    lines.append(f"{'Number of obs:':<21}{nobs}")
    lines.append(f"{'Df residuals:':<21}{_df_label(df_resid)}")
    lines.append(f"{'Df model:':<21}{_df_label(df_model)}")
    lines.append(f"{'R-squared:':<21}{_num(rsquared, 7)}")
    lines.append(f"{'Adj. R-squared:':<21}{_num(rsquared_adj, 7)}")
    lines.append(f"{'Log-likelihood:':<21}{_num(llf, 7)}")
    lines.append(f"{'AIC:':<21}{_num(aic, 7)}")
    lines.append(f"{'BIC:':<21}{_num(bic, 7)}")
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
