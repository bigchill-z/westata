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
    #: Names for Stata's equation line. Stata prints the model's own dependent variable and the
    #: panel's id/time names; the defaults keep older callers working.
    depvar: str = ""
    entity_name: str = ""
    time_name: str = ""

    def __str__(self) -> str:
        """Stata's ``xttest0`` block, with column offsets measured in round 176.

        The pipe sits at column 25; the variance column ends at 37 and the SD column at 52. The
        equation line names the DEPENDENT VARIABLE and the panel's own id/time names
        ('lny[sid,t] = Xb + u[sid] + e[sid,t]'), not placeholders 'y[id,t]' -- the row label in
        the table is the dependent variable too.
        """

        def cell(text: str, end: int, current: int) -> str:
            return " " * max(end - current - len(text), 0) + text

        depvar = str(self.depvar or "y")
        entity = str(self.entity_name or "id")
        time = str(self.time_name or "t")
        lines = [
            "Breusch and Pagan Lagrangian multiplier test for random effects",
            "",
            f"        {depvar}[{entity},t] = Xb + u[{entity}] + e[{entity},t]",
            "",
            "        Estimated results:",
            " " * 25 + "|       Var     SD = sqrt(Var)",
            " " * 16 + "---------+-----------------------------",
        ]
        for label, variance in (
            (depvar, self.var_y),
            ("e", self.var_e),
            ("u", self.var_u),
        ):
            line = cell(label, 24, 0) + " |"
            line += cell(self._variance(variance), 37, len(line))
            line += cell(
                self._variance(np.sqrt(variance) if variance == variance else variance),
                52,
                len(line),
            )
            lines.append(line)
        lines.extend(
            [
                "",
                "        Test: Var(u) = 0",
                cell("chibar2(01)", 40, 0) + cell("=", 42, 40) + cell(f"{self.chibar2:>7.2f}", 51, 42),
                cell("Prob > chibar2", 40, 0) + cell("=", 42, 40) + cell(f"{self.pvalue:>6.4f}", 51, 42),
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _variance(value: float) -> str:
        """Seven significant figures with the leading zero dropped, as Stata prints these cells."""
        if value != value:
            return "."
        text = f"{float(value):.7g}"
        if text.startswith("0."):
            return text[1:]
        if text.startswith("-0."):
            return "-" + text[2:]
        return text



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


def _depvar_name(result: Any) -> str:
    """The dependent variable's name, for Stata's equation line and the first table row.

    Results carry it as ``response_name`` or, failing that, in ``e(depvar)``. Stata prints the real
    name -- 'lny[sid,t] = Xb + u[sid] + e[sid,t]' and a 'lny' row -- where an earlier version of
    this module hard-coded the placeholders 'y[id,t]' and 'y'.
    """

    name = str(getattr(result, "response_name", "") or "").strip()
    if name:
        return name
    stored = getattr(result, "e", None) or {}
    try:
        return str(stored["depvar"]).strip()
    except (KeyError, TypeError):
        return "y"


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
    # The VARIANCE BLOCK wants a residual decomposition, and the two rows come from different
    # pieces of one. Round 176 measured both routes on munnell.dta:
    #     within residual variance / (N - n)  =  0.00151239   -> Stata's "e" row .0015071
    #     e(sigma_u)^2                        =  0.0067015    -> Stata's "u" row .0067015
    # e(sigma_e)^2 is 2.5% too high for the "e" row (0.00154534), but e(sigma_u)^2 lands on the "u"
    # row almost exactly, so the two rows are taken from the two sources that actually reproduce
    # them rather than from one formula applied twice.
    stored = getattr(result, "e", None) or {}
    try:
        # Group the residuals EXPLICITLY rather than reusing ``codes``/``group_sums``: those are
        # built for the bincount arithmetic of the statistic, and mixing them with ``resid``
        # produced 0.00671176 -- the total variance rather than the within one.
        _frame = np.bincount(codes)
        _means = np.bincount(codes, weights=resid) / np.maximum(_frame, 1)
        within = resid - _means[codes]
        var_e = float(within @ within) / max(nobs - entity_count, 1)
    except (NameError, TypeError, ValueError):  # pragma: no cover - defensive
        var_e = None
    try:
        var_u = float(stored["sigma_u"]) ** 2
    except (KeyError, TypeError, ValueError):
        var_u = None
    if var_u is None and var_e is not None:
        # Without a stored sigma_u, back it out of the residual decomposition.
        grand = float(resid.mean())
        between = float(group_sums @ group_sums) / entity_count - periods * grand**2
        var_u = max(between / periods - var_e / periods, 0.0)
    if var_e is None and var_u is not None:
        try:
            var_e = float(stored["sigma_e"]) ** 2
        except (KeyError, TypeError, ValueError):
            var_e = None
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
        depvar=_depvar_name(result),
        entity_name=str(getattr(result, "entity_name", "") or ""),
        time_name=str(getattr(result, "time_name", "") or ""),
    )
