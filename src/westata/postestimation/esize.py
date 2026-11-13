"""``estat esize`` -- effect sizes for a linear model.

``regress_postestimation.sthlp:599-614``:

    estat esize calculates effect sizes for linear models after regress or anova.  By
    default, estat esize reports eta-squared estimates (Kerlinger 1964), which are
    equivalent to R-squared estimates.  If the option epsilon is specified, estat esize
    reports epsilon-squared estimates (Grisson and Kim 2012).  If the option omega is
    specified, estat esize reports omega-squared estimates (Grisson and Kim 2012).  Both
    epsilon-squared and omega-squared are adjusted R-squared estimates.  Confidence
    intervals for eta-squared estimates are estimated by using the noncentral F
    distribution (Smithson 2001).

``esize.ado`` dispatches to compiled builtins for the arithmetic, so the formulas below
come from the sources the help file itself cites rather than from Stata source:

    eta2     = SS_effect / SS_total                        ( == R^2 )
    epsilon2 = (SS_effect - df_effect * MS_error) / SS_total
    omega2   = (SS_effect - df_effect * MS_error) / (SS_total + MS_error)

``ms_error = SS_residual / df_residual``.  Epsilon and omega therefore differ only in the
denominator -- omega adds one ``MS_error``, which makes it the smaller of the two and keeps
it from going as far negative on a null effect.

``r(eta2)``/``r(epsilon2)``/``r(omega2)`` are listed in ``esize.sthlp:284-285``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats as scipy_stats
from scipy.optimize import brentq


@dataclass
class EffectSizeResult:
    """One effect-size estimate and its confidence interval.

    Attributes
    ----------
    kind:
        ``"eta2"``, ``"epsilon2"`` or ``"omega2"``.
    estimate:
        The point estimate.
    ci_lower, ci_upper:
        Confidence bounds for the eta-squared form, from the noncentral F
        (``sthlp:610-612``).  ``None`` for epsilon/omega, where the help file documents no
        interval.
    level:
        Confidence level as a FRACTION (``0.95``), never a percentage (R3.3).
    """

    kind: str
    estimate: float
    ci_lower: float | None = None
    ci_upper: float | None = None
    level: float = 0.95

    def __str__(self) -> str:
        labels = {
            "eta2": "Eta-squared",
            "epsilon2": "Epsilon-squared",
            "omega2": "Omega-squared",
        }
        label = labels.get(self.kind, self.kind)
        lines = ["", "Effect sizes for linear models", ""]
        header = f"  {label:>18}"
        if self.ci_lower is not None:
            # `level` is a fraction (0.95), so the printed tail is (1 - level)/2 = 2.5%.
            header += f"   [{(1.0 - self.level) * 50.0:>4.1f}% Conf. Interval]"
        lines.append(header)
        row = f"  {self.estimate:>18.6f}"
        if self.ci_lower is not None:
            row += f"   {self.ci_lower:>12.6f} {self.ci_upper:>12.6f}"
        lines.append(row)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, float]:
        """Flatten to Stata's ``r()`` names.

        ``esize.sthlp:284-285`` names ``r(eta2)``, ``r(epsilon2)`` and ``r(omega2)``; only
        the one that was computed is returned, so a caller checking for a key learns which
        estimate the command produced.
        """

        return {self.kind: float(self.estimate)}


def _noncentral_f_bounds(
    statistic: float, df_num: float, df_denom: float, level: float
) -> tuple[float, float]:
    """CI for the noncentrality parameter, then map it to an eta-squared interval.

    Smithson (2001), cited at ``sthlp:610-612``: the interval is found by solving for the
    noncentrality values whose noncentral-F quantiles pass through the observed ``F``.

    ``eta2 = lambda / (lambda + N)``, so the bounds on ``lambda`` convert directly.
    """

    alpha = 1.0 - level
    total = df_num + df_denom + 1.0

    def lower_gap(noncentrality: float) -> float:
        # The LOWER confidence limit is the noncentrality whose UPPER alpha/2 quantile
        # equals the observed statistic.
        return float(
            scipy_stats.ncf.ppf(1.0 - alpha / 2.0, df_num, df_denom, noncentrality)
        ) - statistic

    def upper_gap(noncentrality: float) -> float:
        return float(
            scipy_stats.ncf.ppf(alpha / 2.0, df_num, df_denom, noncentrality)
        ) - statistic

    def solve(function, low: float, high: float) -> float:
        try:
            return float(brentq(function, low, high, maxiter=200))
        except ValueError:
            # No sign change on [low, high].  That is not a numerical accident: it is
            # what happens when the observed F is SMALLER than every noncentral-F
            # upper quantile over the whole domain (``lower_gap`` stays positive) or
            # larger than every lower quantile (``upper_gap`` stays negative).  The
            # correct limit is the edge of the parameter space, i.e. noncentrality 0
            # -- an effect size of 0 -- not NaN.  Returning NaN here made
            # ``estat_esize().ci_lower`` NaN for every ordinary fit (measured: F=1.24
            # on df=(2,197) gave ci=(nan, 0.0522)), which `test_estat_esize` then
            # asserted as `nan < nan`.
            if function(low) > 0 and function(high) > 0:
                return 0.0
            if function(low) < 0 and function(high) < 0:
                return float(high)
            return float("nan")  # pragma: no cover - unreachable bracket failure

    lower_noncentrality = solve(lower_gap, 0.0, 1e4)
    upper_noncentrality = solve(upper_gap, 0.0, 1e4)

    def to_eta(noncentrality: float) -> float:
        if not np.isfinite(noncentrality):
            return float("nan")
        return noncentrality / (noncentrality + total)

    return to_eta(lower_noncentrality), to_eta(upper_noncentrality)


def estat_esize(
    result: Any,
    *,
    epsilon: bool = False,
    omega: bool = False,
    level: float = 0.95,
) -> EffectSizeResult:
    """Stata's ``estat esize``.

    Parameters
    ----------
    epsilon:
        Report epsilon-squared (``sthlp:620-622``).
    omega:
        Report omega-squared (``sthlp:624-626``).  Mutually exclusive with ``epsilon``
        (``esize.ado:14``: ``opts_exclusive "`epsilon' `omega'"``).
    level:
        Confidence level as a FRACTION (``0.95``), never a percentage (R3.3).

        This used to default to ``95.0``.  Every other ``level`` in the package is a
        fraction (``margins``, ``nlcom``, ``lincom_as_stata``, ``contrast``), so a
        caller who passed ``level=0.95`` here got ``(100 - 0.95) / 2 = 49.5%``
        intervals -- a plausible-looking number, silently wrong.  Verified: the old
        expression printed ``[49.5% Conf. Interval]``.

    Raises
    ------
    NotImplementedError
        When the fit is not ``regress`` or ``anova`` -- ``sthlp:599-601`` restricts it.
    ValueError
        When both ``epsilon`` and ``omega`` are requested.

    Notes
    -----
    ``regress`` stores ``e(mss)`` and ``e(rss)`` but NOT ``e(tss)``, so the total is
    reconstructed as their sum.  That is an identity for a model with a constant; a
    ``noconstant`` fit has no such decomposition, which is why Stata's own total is not
    consulted here.
    """

    if epsilon and omega:
        raise ValueError(
            "estat esize: epsilon and omega are mutually exclusive "
            "(esize.ado:14: opts_exclusive)"
        )

    model_name = str(getattr(result, "model_name", "") or "").strip().lower()
    if model_name not in ("regress", "areg", "anova"):
        raise NotImplementedError(
            "estat esize is only defined for linear models after regress or anova "
            f"(regress_postestimation.sthlp:599-601); got {model_name!r}"
        )

    e = getattr(result, "e", {}) or {}
    mss = e.get("mss")
    rss = e.get("rss")
    if mss is None or rss is None:
        raise ValueError(
            "estat esize needs e(mss) and e(rss); this result stores "
            f"mss={mss!r} and rss={rss!r}"
        )
    mss = float(mss)
    rss = float(rss)
    total = mss + rss
    if total <= 0.0:  # pragma: no cover - a perfectly constant response
        raise ValueError(
            "estat esize: the total sum of squares is zero, so no effect size exists"
        )

    df_effect = float(e.get("df_m", getattr(result, "df_model", 1.0)))
    df_residual = float(e.get("df_r", getattr(result, "df_resid", 1.0)))
    if df_residual <= 0.0:  # pragma: no cover
        raise ValueError(
            "estat esize: no residual degrees of freedom, so MS_error is undefined"
        )
    mean_square_error = rss / df_residual

    eta_squared = mss / total
    if omega:
        estimate = (mss - df_effect * mean_square_error) / (
            total + mean_square_error
        )
        return EffectSizeResult(kind="omega2", estimate=float(estimate), level=level)
    if epsilon:
        estimate = (mss - df_effect * mean_square_error) / total
        return EffectSizeResult(kind="epsilon2", estimate=float(estimate), level=level)

    statistic = (mss / df_effect) / mean_square_error if mean_square_error > 0 else 0.0
    lower, upper = _noncentral_f_bounds(
        float(statistic), df_effect, df_residual, level
    )
    return EffectSizeResult(
        kind="eta2",
        estimate=float(eta_squared),
        ci_lower=lower,
        ci_upper=upper,
        level=level,
    )
