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

    def __str__(self) -> str:
        stat_label = "z" if self.distribution == "z" else "t"
        lines = [self.expression]
        lines.append(
            f"{'':>12}{'Coef.':>12}{'Std. err.':>12}{stat_label:>8}"
            f"{'P>|' + stat_label + '|':>8}{'[95% conf. interval]':>28}"
        )
        lines.append(
            f"{self.label or '':<12}{self.estimate:>12.7g}{self.std_err:>12.7g}"
            f"{self.statistic:>8.2f}{self.pvalue:>8.3f}"
            f"{self.ci_lower:>14.7g}{self.ci_upper:>14.7g}"
        )
        return "\n".join(lines)


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
    )
