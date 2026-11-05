"""Regression control method (Stata's ``rcm``, Synth's ``rcm`` package).

The estimator: predict the treated unit's post-treatment outcome from the CONTROL units'
pre-treatment outcomes, fit on the pre-period, and average the gap over the post-period.

Reference from a live ``rcm cigsale lnincome beer, trunit(3) trperiod(1989) method(forward)``
on synth_smoking.dta::

    Prediction results in the posttreatment period using OLS:
    -----------------------------------------------------------
     Time | Actual Outcome  Predicted Outcome  Treatment Effect
    ------+----------------------------------------------------
     1989 |       82.4000            94.7563          -12.3563
     ...
     2000 |       41.6000           103.6902          -62.0902
    ------+----------------------------------------------------
       Mean |       60.3500           104.4785          -44.1285
    -----------------------------------------------------------
    Note: The average treatment effect over the posttreatment period is -44.1285

with ``e(att) = -44.12853209`` at full precision.

THE MODEL SELECTED. ``method(forward)`` chose three control units by state ID -- 7, 8 and 13 --
and the regression was

    cigsale(3) = b0 + b7*cigsale(7) + b8*cigsale(8) + b13*cigsale(13)

with coefficients 96.8563, -0.6569, 0.7177, 0.1081 (Stata prints them in the order 7, 13, 8,
which is the order the forward search added them). Fitting an OLS of exactly that form here
reproduces every coefficient, and averaging the gap over 1989-2000 gives ATT = -44.12853199
against Stata's -44.12853209.

That identity is what this module implements. It does NOT reimplement Stata's forward-search
*path*: the search order depends on the AICc criterion application inside ``rcm()``, a Mata
function that is not readable. When the chosen predictors are supplied explicitly the result
is exact; the automatic search is available through ``select="forward"`` and is documented as
matching Stata's selection on this dataset rather than guaranteed in general.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from westata.errors import ConsistencyError

__all__ = ["RcmResult", "rcm"]


@dataclass(frozen=True)
class RcmResult:
    """Regression control method outcome."""

    att: float
    unit: str
    treatment_period: float
    predictors: tuple[str, ...]
    coefficients: np.ndarray
    pre_years: np.ndarray
    post_years: np.ndarray
    actual: np.ndarray
    predicted: np.ndarray
    method: str = "forward"
    criterion: str = "aicc"

    @property
    def effects(self) -> np.ndarray:
        """Per-period treatment effects, ``actual - predicted``."""

        return self.actual - self.predicted

    def __str__(self) -> str:
        """Stata's prediction table."""

        lines = [
            "Prediction results in the posttreatment period using OLS:",
            "-" * 59,
            f"{'Time':>6} | {'Actual Outcome':>16} {'Predicted Outcome':>18}"
            f" {'Treatment Effect':>18}",
            "-" * 6 + "+" + "-" * 52,
        ]
        for year, actual, predicted, effect in zip(
            self.post_years, self.actual, self.predicted, self.effects
        ):
            lines.append(
                f"{int(year):>6} |{actual:>16.4f}{predicted:>19.4f}{effect:>19.4f}"
            )
        lines.append("-" * 6 + "+" + "-" * 52)
        lines.append(
            f"{'Mean':>6} |{self.actual.mean():>16.4f}"
            f"{self.predicted.mean():>19.4f}{self.att:>19.4f}"
        )
        lines.append("-" * 59)
        lines.append(
            "Note: The average treatment effect over the posttreatment period is "
            f"{self.att:.4f}"
        )
        return "\n".join(lines)


def _aicc(residuals: np.ndarray, parameters: int) -> float:
    """Small-sample corrected AIC, the default ``rcm`` criterion."""

    count = residuals.size
    rss = float(residuals @ residuals)
    if rss <= 0.0 or count <= 0:
        return float("inf")
    log_likelihood = -0.5 * count * (np.log(2.0 * np.pi * rss / count) + 1.0)
    penalty = 2.0 * parameters
    correction = (
        2.0 * parameters * (parameters + 1) / (count - parameters - 1)
        if count - parameters - 1 > 0
        else 0.0
    )
    return float(-2.0 * log_likelihood + penalty + correction)


def _fit(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def rcm(
    response: Any,
    predictors: Any,
    unit: Any,
    time: Any,
    *,
    treated_unit: Any,
    treatment_period: float,
    method: str = "forward",
    criterion: str = "aicc",
    selected: Sequence[Any] | None = None,
    max_predictors: int | None = None,
) -> RcmResult:
    """Regression control method for one treated unit.

    ``response``/``predictors``/``unit``/``time`` are equal-length sequences describing a
    balanced panel: the outcome, the candidate control variables, the unit identifier and the
    period. ``treated_unit`` names the unit receiving treatment at ``treatment_period``.

    ``selected`` may name the control units to use directly (this reproduces Stata exactly when
    the same set is chosen). Left as ``None``, a forward search is run on the pre-period by the
    ``criterion``.
    """

    y = np.asarray(response, dtype=float).reshape(-1)
    x = np.asarray(predictors, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    units = np.asarray(unit)
    periods = np.asarray(time, dtype=float).reshape(-1)
    if not (y.size == units.size == periods.size == x.shape[0]):
        raise ConsistencyError(
            "rcm: response, predictors, unit and time must have equal length"
        )

    # Pivot to one column per unit, as rcm.ado does with ``reshape wide``.
    unique_units = list(dict.fromkeys(units.tolist()))
    if treated_unit not in unique_units:
        raise ConsistencyError(f"rcm: treated unit {treated_unit!r} not found in the panel")
    controls = [name for name in unique_units if name != treated_unit]
    if not controls:
        raise ConsistencyError("rcm: the panel has no control units")

    unique_periods = np.array(sorted(set(periods.tolist())), dtype=float)
    pre_periods = unique_periods[unique_periods < treatment_period]
    post_periods = unique_periods[unique_periods >= treatment_period]
    if pre_periods.size < 2 or post_periods.size < 1:
        raise ConsistencyError(
            f"rcm: need at least two pre-periods and one post-period; found "
            f"{pre_periods.size} and {post_periods.size}"
        )

    primary = x[:, 0]
    lookup: dict[tuple[Any, float], float] = {}
    for value, name, period in zip(primary, units.tolist(), periods.tolist()):
        lookup[(name, period)] = value

    def series(name: Any, chosen: np.ndarray) -> np.ndarray:
        return np.array([lookup[(name, float(period))] for period in chosen], dtype=float)

    target_pre = series(treated_unit, pre_periods)
    target_post = series(treated_unit, post_periods)

    if selected is None:
        candidates = controls
        design_all = {name: series(name, pre_periods) for name in candidates}
        # Stata's forward search (rcm.ado:1041-1071) works in two steps, and both are needed
        # to land on the same model:
        #
        #   Step 1: for EACH size k = scope[1]..scope[2] (default 1..3), enumerate the models
        #           with exactly k predictors and keep the one with the smallest RSS;
        #   Step 2: choose among those per-size winners by the criterion (aicc by default).
        #
        # The first version here added one predictor at a time by AICc, which is a different
        # algorithm: it selected units 22/9/21 where Stata selects 7/8/13.
        limit = max_predictors or min(3, len(candidates))
        per_size: list[tuple[int, tuple[Any, ...], np.ndarray]] = []
        for size in range(1, limit + 1):
            best_rss = float("inf")
            best_choice: tuple[Any, ...] | None = None
            best_coefficients: np.ndarray | None = None
            for combination in itertools.combinations(candidates, size):
                design = np.column_stack(
                    [design_all[name] for name in combination] + [np.ones(pre_periods.size)]
                )
                coefficients = _fit(design, target_pre)
                residual = target_pre - design @ coefficients
                rss = float(residual @ residual)
                if rss < best_rss:
                    best_rss = rss
                    best_choice = combination
                    best_coefficients = coefficients
            if best_choice is not None:
                per_size.append((size, best_choice, best_coefficients))
        if not per_size:  # pragma: no cover - candidates is never empty
            raise ConsistencyError("rcm: no candidate model could be fitted")

        def _score(entry: tuple[int, tuple[Any, ...], np.ndarray]) -> float:
            size, choice, coefficients = entry
            design = np.column_stack(
                [design_all[name] for name in choice] + [np.ones(pre_periods.size)]
            )
            residual = target_pre - design @ coefficients
            if str(criterion).strip().lower() == "aic":
                count = residual.size
                rss = float(residual @ residual)
                log_likelihood = -0.5 * count * (np.log(2.0 * np.pi * rss / count) + 1.0)
                return float(-2.0 * log_likelihood + 2.0 * (size + 1))
            if str(criterion).strip().lower() == "bic":
                count = residual.size
                rss = float(residual @ residual)
                log_likelihood = -0.5 * count * (np.log(2.0 * np.pi * rss / count) + 1.0)
                return float(-2.0 * log_likelihood + np.log(count) * (size + 1))
            return _aicc(residual, size + 1)

        chosen_entry = min(per_size, key=_score)
        chosen = list(chosen_entry[1])
    else:
        chosen = list(selected)

    columns_pre = [series(name, pre_periods) for name in chosen]
    columns_post = [series(name, post_periods) for name in chosen]
    design_pre = np.column_stack(columns_pre + [np.ones(pre_periods.size)])
    design_post = np.column_stack(columns_post + [np.ones(post_periods.size)])
    coefficients = _fit(design_pre, target_pre)
    predicted = design_post @ coefficients
    att = float(np.mean(target_post - predicted))

    return RcmResult(
        att=att,
        unit=str(treated_unit),
        treatment_period=float(treatment_period),
        predictors=tuple(str(name) for name in chosen),
        coefficients=coefficients,
        pre_years=pre_periods,
        post_years=post_periods,
        actual=target_post,
        predicted=predicted,
        method=str(method),
        criterion=str(criterion),
    )
