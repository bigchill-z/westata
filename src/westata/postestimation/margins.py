"""Marginal effects and adjusted predictions, targeting Stata's ``margins``.

Supported subset (deliberately small and explicit):

``margins(dydx=...)``
    Average marginal effects (the default, equivalent to ``margins, dydx(x)``).
``margins(dydx=..., at={...})``
    Conditional marginal effects at fixed covariate values
    (``margins, dydx(x) at(z=1)``).
``margins(dydx=..., over=...)``
    Marginal effects separately per group (``margins, dydx(x) over(g)``).
``margins(at={...})`` / ``margins()``
    Adjusted predictions / predictive margins (``margins, at(...)``).

Standard errors use the delta method against the model's VCE, which is
Stata's ``vce(delta)`` default: ``Var = G V G'`` where ``G`` is the gradient of
the averaged statistic with respect to the parameters.

What is *not* implemented, and raises :class:`UnsupportedError` rather than
approximating: ``contrast()``, ``pwcompare()``, ``expression()``,
``vce(unconditional)``, ``atmeans`` with factor variables, and multiple
comparison adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from westata.errors import UnsupportedError
from westata.postestimation.predict import predict

__all__ = ["MarginsResult", "margins"]


@dataclass
class MarginsResult:
    """One row of a ``margins`` table plus its covariance."""

    label: str
    estimate: float
    std_err: float
    statistic: float
    pvalue: float
    ci_lower: float
    ci_upper: float
    prediction: str = ""

    def __str__(self) -> str:
        return (
            f"{self.label:<24}{self.estimate:>12.7g}{self.std_err:>12.7g}"
            f"{self.statistic:>8.2f}{self.pvalue:>8.3f}"
            f"{self.ci_lower:>14.7g}{self.ci_upper:>14.7g}"
        )


def _statistic_and_gradient(
    result: Any,
    exog: np.ndarray,
    kind: str,
    *,
    derivative_with_respect_to: int | None = None,
    h: float = 1e-6,
) -> tuple[float, np.ndarray]:
    """Return the averaged statistic and its gradient w.r.t. the parameters.

    The gradient is computed by central differences on the *parameter vector*
    while holding the covariates fixed.  ``derivative_with_respect_to`` selects
    the covariate whose marginal effect is wanted; ``None`` means the
    prediction itself (a predictive margin).
    """

    params = np.asarray(result.params, dtype=float).copy()
    k = params.size

    def evaluate(parameter_vector: np.ndarray) -> np.ndarray:
        # Temporarily swap the parameters so ``predict`` sees the perturbed
        # values without mutating the caller's result object.
        original = result._params
        try:
            result._params = pd.Series(parameter_vector, index=original.index)
            if derivative_with_respect_to is None:
                return predict(result, exog, kind)
            shifted = exog.copy()
            shifted[:, derivative_with_respect_to] += h
            forward = predict(result, shifted, kind)
            shifted = exog.copy()
            shifted[:, derivative_with_respect_to] -= h
            backward = predict(result, shifted, kind)
            return (forward - backward) / (2.0 * h)
        finally:
            result._params = original

    value = float(np.mean(evaluate(params)))
    gradient = np.zeros(k, dtype=float)
    for index in range(k):
        step = 1e-6 * max(1.0, abs(params[index]))
        forward = params.copy()
        forward[index] += step
        backward = params.copy()
        backward[index] -= step
        gradient[index] = (
            float(np.mean(evaluate(forward))) - float(np.mean(evaluate(backward)))
        ) / (2.0 * step)
    return value, gradient


def _row_from_statistic(
    label: str,
    estimate: float,
    gradient: np.ndarray,
    cov: np.ndarray,
    *,
    dist: str,
    df_resid: float,
    level: float,
    prediction: str,
) -> MarginsResult:
    from scipy import stats

    variance = float(gradient @ cov @ gradient)
    std_err = float(np.sqrt(max(variance, 0.0)))

    alpha = 1.0 - level
    if dist == "z":
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
        statistic = estimate / std_err if std_err > 0 else float("nan")
        pvalue = float(2.0 * stats.norm.sf(abs(statistic)))
    else:
        critical = float(stats.t.ppf(1.0 - alpha / 2.0, df_resid))
        statistic = estimate / std_err if std_err > 0 else float("nan")
        pvalue = float(2.0 * stats.t.sf(abs(statistic), df_resid))

    return MarginsResult(
        label=label,
        estimate=estimate,
        std_err=std_err,
        statistic=statistic,
        pvalue=pvalue,
        ci_lower=estimate - critical * std_err,
        ci_upper=estimate + critical * std_err,
        prediction=prediction,
    )


#: Stata's default prediction type per model for ``margins, dydx()``.  For a
#: linear model the response *is* the linear prediction, but for logit/probit
#: Stata differentiates the predicted probability (``pr``), not ``xb``; getting
#: this wrong turns an average marginal effect into a plain coefficient.
DEFAULT_PREDICT: dict[str, str] = {
    "regress": "xb",
    "ivregress 2sls": "xb",
    "ivregress liml": "xb",
    "ivregress gmm": "xb",
    "reghdfe": "xb",
    "xtreg, fe": "xb",
    "xtreg, re": "xb",
    "xtreg, be": "xb",
    "xtreg, fd": "xb",
    "logit": "pr",
    "probit": "pr",
    "poisson": "n",
    "nbreg": "n",
    "ologit": "pr",
    "oprobit": "pr",
    "tobit": "e",
}


def default_predict_kind(model_name: str) -> str:
    """Return the prediction type ``margins`` differentiates by default."""

    return DEFAULT_PREDICT.get(model_name, "xb")


def margins(
    result: Any,
    *,
    dydx: Sequence[str] | str | None = None,
    at: Mapping[str, Any] | None = None,
    over: str | None = None,
    exog: np.ndarray | None = None,
    param_names: Sequence[str] | None = None,
    level: float = 0.95,
    predict_kind: str | None = None,
    factor_vars: Sequence[str] | None = None,
) -> list[MarginsResult] | MarginsResult:
    """Compute marginal effects or predictive margins.

    Parameters
    ----------
    dydx:
        Name(s) of the covariates to differentiate with respect to.  Omit to
        get adjusted predictions (``margins`` without ``dydx()``).
    at:
        ``{"z": 1}`` fixes covariates at the given values before averaging.
    over:
        Column name (or index array) defining groups to report separately.
    exog, param_names:
        The design matrix and its column names.  If omitted, the estimator must
        have stored them; otherwise a clear error is raised.
    factor_vars:
        Names that are factor (dummy) variables.  Stata reports the *discrete
        change* for these (prediction at 1 minus prediction at 0) rather than a
        derivative; passing them switches to that definition.
    """

    if exog is None or param_names is None:
        stored = getattr(result, "design_matrix", None)
        if stored is None:
            raise UnsupportedError(
                "margins: the design matrix was not retained on this result. "
                "Pass exog=... and param_names=... explicitly, or refit with an "
                "estimator that stores them."
            )
        exog = stored[0]
        param_names = list(stored[1])

    exog = np.asarray(exog, dtype=float)
    names = list(param_names)
    factor_set = set(factor_vars or ())

    # Stata differentiates the response, not the linear index: logit/probit
    # marginals are on the probability scale and count models on the mean scale.
    kind = predict_kind or default_predict_kind(str(getattr(result, "model_name", "")))

    if at:
        exog = _apply_at(exog, names, at)

    if dydx is None:
        rows = _predictive_margins(
            result, exog, names, over=over, level=level, predict_kind=kind
        )
    else:
        requested = [dydx] if isinstance(dydx, str) else list(dydx)
        rows = _marginal_effects(
            result,
            exog,
            names,
            requested,
            factor_set=factor_set,
            over=over,
            level=level,
            predict_kind=kind,
        )

    return rows[0] if len(rows) == 1 else rows


def _apply_at(
    exog: np.ndarray, names: Sequence[str], at: Mapping[str, Any]
) -> np.ndarray:
    adjusted = exog.copy()
    for name, value in at.items():
        if name not in names:
            raise KeyError(f"at(): column {name!r} not in {list(names)}")
        position = list(names).index(name)
        adjusted[:, position] = float(value)
    return adjusted


def _groups(over: Any, n: int, model_data: Any = None) -> dict[Any, np.ndarray] | None:
    if over is None:
        return None
    # `over` is usually a column name (Stata's `over(group)`); resolve it
    # against the estimation sample before falling back to raw labels.
    if isinstance(over, str):
        columns = getattr(model_data, "columns", [])
        if over not in columns:
            raise KeyError(
                f"over(): column {over!r} is not available on this result; "
                "pass over= an array of labels instead"
            )
        labels = np.asarray(model_data[over]).reshape(-1)
    else:
        labels = np.asarray(over).reshape(-1)
    if labels.size != n:
        raise ValueError(
            f"over has {labels.size} entries but the data have {n} rows"
        )
    return {
        group: np.flatnonzero(labels == group)
        for group in pd.unique(labels)
    }


def _marginal_effects(
    result: Any,
    exog: np.ndarray,
    names: list[str],
    requested: list[str],
    *,
    factor_set: set[str],
    over: Any,
    level: float,
    predict_kind: str,
) -> list[MarginsResult]:
    dist = "z" if getattr(result, "dist", "t") == "z" else "t"
    df_resid = float(result.df_resid)
    cov = result.cov_params()

    groups = _groups(over, exog.shape[0], getattr(result, "model_data", None))
    rows: list[MarginsResult] = []

    for name in requested:
        if name not in names:
            raise KeyError(f"dydx: column {name!r} not in {names}")
        position = names.index(name)
        is_factor = name in factor_set

        if groups is None:
            if is_factor:
                estimate, gradient = _discrete_change(
                    result, exog, predict_kind, position
                )
            else:
                estimate, gradient = _statistic_and_gradient(
                    result,
                    exog,
                    predict_kind,
                    derivative_with_respect_to=position,
                )
            rows.append(
                _row_from_statistic(
                    name,
                    estimate,
                    gradient,
                    cov,
                    dist=dist,
                    df_resid=df_resid,
                    level=level,
                    prediction=predict_kind,
                )
            )
        else:
            for group, mask in groups.items():
                subset = exog[mask]
                if is_factor:
                    estimate, gradient = _discrete_change(
                        result, subset, predict_kind, position
                    )
                else:
                    estimate, gradient = _statistic_and_gradient(
                        result,
                        subset,
                        predict_kind,
                        derivative_with_respect_to=position,
                    )
                rows.append(
                    _row_from_statistic(
                        f"{group}",
                        estimate,
                        gradient,
                        cov,
                        dist=dist,
                        df_resid=df_resid,
                        level=level,
                        prediction=predict_kind,
                    )
                )
    return rows


def _discrete_change(
    result: Any, exog: np.ndarray, kind: str, position: int
) -> tuple[float, np.ndarray]:
    """Stata's factor-variable marginal effect: P(x=1) - P(x=0)."""

    params = np.asarray(result.params, dtype=float)
    k = params.size

    def evaluate(parameter_vector: np.ndarray) -> float:
        original = result._params
        try:
            result._params = pd.Series(parameter_vector, index=original.index)
            high = exog.copy()
            high[:, position] = 1.0
            low = exog.copy()
            low[:, position] = 0.0
            return float(np.mean(predict(result, high, kind) - predict(result, low, kind)))
        finally:
            result._params = original

    estimate = evaluate(params)
    gradient = np.zeros(k, dtype=float)
    for index in range(k):
        step = 1e-6 * max(1.0, abs(params[index]))
        forward = params.copy()
        forward[index] += step
        backward = params.copy()
        backward[index] -= step
        gradient[index] = (evaluate(forward) - evaluate(backward)) / (2.0 * step)
    return estimate, gradient


def _predictive_margins(
    result: Any,
    exog: np.ndarray,
    names: list[str],
    *,
    over: Any,
    level: float,
    predict_kind: str,
) -> list[MarginsResult]:
    dist = "z" if getattr(result, "dist", "t") == "z" else "t"
    df_resid = float(result.df_resid)
    cov = result.cov_params()

    groups = _groups(over, exog.shape[0], getattr(result, "model_data", None))
    rows: list[MarginsResult] = []

    if groups is None:
        estimate, gradient = _statistic_and_gradient(result, exog, predict_kind)
        rows.append(
            _row_from_statistic(
                "margin",
                estimate,
                gradient,
                cov,
                dist=dist,
                df_resid=df_resid,
                level=level,
                prediction=predict_kind,
            )
        )
    else:
        for group, mask in groups.items():
            estimate, gradient = _statistic_and_gradient(
                result, exog[mask], predict_kind
            )
            rows.append(
                _row_from_statistic(
                    str(group),
                    estimate,
                    gradient,
                    cov,
                    dist=dist,
                    df_resid=df_resid,
                    level=level,
                    prediction=predict_kind,
                )
            )
    return rows
