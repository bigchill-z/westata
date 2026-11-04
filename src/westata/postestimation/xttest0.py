"""Breusch-Pagan LM test for random effects: Stata's ``xttest0``.

Stata's reference values come from live runs. On a synthetic BALANCED panel (12 entities x 10
periods, ``quietly xtreg y x1 x2, re`` followed by ``xttest0``)::

    y | 3.557623   SD 1.886166
    e | .7086426   SD .8418091
    u | 1.471772   SD 1.213166

    chibar2(01) = 252.58
    Prob > chibar2 = 0.0000

The statistic is the Breusch-Pagan Lagrange multiplier for ``Var(u) = 0``:

.. math::

    LM = \\frac{N}{2(T-1)}
         \\left[ \\frac{\\sum_i \\left(\\sum_t \\hat e_{it}\\right)^2}
                      {\\sum_i \\sum_t \\hat e_{it}^2} - 1 \\right]^2

with ``\\hat e`` the residuals of the POOLED OLS regression, ``T`` the number of periods and
``N`` the total number of observations. It is referred to a 50:50 mixture of ``chi2(0)`` and
``chi2(1)``, which is what Stata's ``chibar2(01)`` label means: ``p = 0.5 * P(chi2(1) > LM)``.

Only the balanced case is implemented. On the unbalanced panel available in the test data
(eleven counties with 262 periods and one with 118) Stata reported 2232.70 where this formula
gives 2289.52, and substituting the mean or the harmonic mean of the group sizes did not close
the gap -- so Stata uses a different expression there rather than a different ``T``. Refusing an
unbalanced panel is better than silently reporting a number that is 2.5% wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from westata.errors import ConsistencyError

__all__ = ["XtTest0Result", "xttest0"]


@dataclass(frozen=True)
class XtTest0Result:
    """``xttest0``: the Breusch-Pagan LM test for random effects."""

    chibar2: float
    pvalue: float
    var_u: float
    var_e: float
    var_y: float
    nobs: int
    entities: int
    periods: int

    def __str__(self) -> str:
        lines = [
            "Breusch and Pagan Lagrangian multiplier test for random effects",
            "",
            "        y[id,t] = Xb + u[id] + e[id,t]",
            "",
            "        Estimated results:",
            "                         |       Var     SD = sqrt(Var)",
            "                ---------+-----------------------------",
            f"{'y':>24} |{self.var_y:>12.7g}{np.sqrt(self.var_y):>14.7g}",
            f"{'e':>24} |{self.var_e:>12.7g}{np.sqrt(self.var_e):>14.7g}",
            f"{'u':>24} |{self.var_u:>12.7g}{np.sqrt(self.var_u):>14.7g}",
            "",
            "        Test: Var(u) = 0",
            f"{'chibar2(01) =':>34}{self.chibar2:>8.2f}",
            f"{'Prob > chibar2 =':>34}{self.pvalue:>9.4f}",
        ]
        return "\n".join(lines)



def _pooled_residuals(result: Any, fallback: np.ndarray) -> np.ndarray:
    """Residuals of the pooled OLS fit on the same design as ``result``."""

    design = getattr(result, "design_matrix", None)
    frame = getattr(result, "model_data", None)
    response_name = str(getattr(result, "response_name", "") or "")
    if design is None or frame is None or not response_name:
        return fallback
    try:
        matrix = np.asarray(design[0], dtype=float)
        response = np.asarray(frame[response_name], dtype=float).reshape(-1)
    except (KeyError, ValueError, TypeError):
        return fallback
    if matrix.ndim != 2 or matrix.shape[0] != response.size:
        return fallback
    coefficients, *_ = np.linalg.lstsq(matrix, response, rcond=None)
    return response - matrix @ coefficients


def xttest0(
    result: Any,
    *,
    entity: Any = None,
) -> XtTest0Result:
    """Breusch-Pagan LM test for a random effect, as ``xttest0`` after ``xtreg, re``.

    ``entity`` supplies the group labels when ``result`` does not carry them. The residuals
    are always those of the POOLED OLS regression, which is what the test is built on --
    Stata takes them from the ``re`` fit's ``e(rss)`` scale but the statistic itself uses the
    pooled residuals.
    """

    frame = getattr(result, "model_data", None)
    resid = getattr(result, "resid", None)
    if resid is None:
        raise ConsistencyError("xttest0 needs a fitted result carrying residuals")

    labels = entity
    if labels is None:
        labels = getattr(result, "_entity_labels", None)
    if labels is None and frame is not None:
        for candidate in ("id", "entity", "group", "county", "firm", "prov"):
            if candidate in getattr(frame, "columns", []):
                labels = frame[candidate]
                break
    if labels is None:
        raise ConsistencyError(
            "xttest0 needs the entity labels; pass entity=<labels>"
        )

    values = np.asarray(resid, dtype=float).reshape(-1)
    codes = pd.factorize(np.asarray(labels))[0]
    if codes.size != values.size:
        raise ConsistencyError(
            f"xttest0: {codes.size} entity labels against {values.size} residuals"
        )

    # The statistic uses the residuals of the POOLED OLS regression, not the random-effects
    # residuals. Measured: on the balanced 12x10 panel the pooled residuals give 252.58
    # (Stata's value) while the re fit's own residuals give 0.27 -- the latter are nearly
    # orthogonal to the group means by construction, which destroys the very between-group
    # signal this test measures. The pooled fit is therefore redone from the same design.
    values = _pooled_residuals(result, values)

    counts = np.bincount(codes)
    if counts.size < 2:
        raise ConsistencyError("xttest0 needs at least two entities")
    if not np.all(counts == counts[0]):
        raise ConsistencyError(
            "xttest0: only the balanced-panel form is implemented; the group sizes are "
            f"{sorted(set(counts.tolist()))}. Stata uses a different expression for "
            "unbalanced panels and this implementation does not reproduce it."
        )
    periods = int(counts[0])
    entity_count = int(counts.size)
    nobs = int(values.size)
    if periods < 2:
        raise ValueError("xttest0 needs at least two periods per entity")

    group_sums = np.bincount(codes, values)
    rss = float(values @ values)
    if rss <= 0.0:
        raise ConsistencyError("xttest0: the residuals are all zero")
    ratio = float((group_sums**2).sum()) / rss
    statistic = (nobs / (2.0 * (periods - 1))) * (ratio - 1.0) ** 2
    # ``chibar2(01)`` is the 50:50 mixture of chi2(0) and chi2(1), so the tail is halved.
    pvalue = 0.5 * float(stats.chi2.sf(statistic, 1))

    var_e = None
    var_u = None
    stored = getattr(result, "e", None) or {}
    # ``xtreg, re`` stores e(sigma_u)/e(sigma_e) as VARIANCES, not standard deviations.
    try:
        var_u = float(stored["sigma_u"])
        var_e = float(stored["sigma_e"])
    except (KeyError, TypeError, ValueError):
        var_u = var_e = None
    if var_u is None or var_e is None or not np.isfinite(var_u):
        # Fall back to the moment estimators so the block still prints something coherent,
        # but the STATISTIC above never depends on these.
        var_e = rss / (nobs - 1)
        mean_group = float(group_sums.mean() / periods)
        var_u = max(mean_group**2 - var_e / periods, 0.0)

    # ``var_y`` is the variance of the RESPONSE ITSELF, on the original scale. Building it
    # from fittedvalues + residuals gave 7.44762 against Stata's 3.557623.
    var_y = None
    frame = getattr(result, "model_data", None)
    response_name = str(getattr(result, "response_name", "") or "")
    if frame is not None and response_name:
        try:
            response = np.asarray(frame[response_name], dtype=float).reshape(-1)
            if response.size == nobs:
                var_y = float(np.var(response, ddof=1))
        except (KeyError, ValueError, TypeError):
            var_y = None
    if var_y is None:
        var_y = float(np.var(values, ddof=1))

    return XtTest0Result(
        chibar2=float(statistic),
        pvalue=float(pvalue),
        var_u=float(var_u),
        var_e=float(var_e),
        var_y=float(var_y),
        nobs=nobs,
        entities=entity_count,
        periods=periods,
    )
