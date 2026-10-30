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
from westata.postestimation.predict import (
    family_slopes,
    ordered_structure,
    predict,
    prediction_family,
)

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

    def to_dict(self) -> dict[str, float]:
        """Flatten to the rows of Stata's ``r(table)``.

        ``margins.sthlp:1101`` documents ``r(table)``: a matrix whose rows are ``b``,
        ``se``, the test statistic, ``p``, and the two confidence bounds.  The keys here
        are those row labels, which is how a caller reading the help file asks for them.

        Stata labels the statistic row ``t`` or ``z`` according to the model's
        distribution.  This object does not carry that flag, so the key is ``t`` --
        naming it after a distribution we cannot verify would be a guess.  Use
        ``str(...)`` when the label itself matters.
        """

        return {
            "b": float(self.estimate),
            "se": float(self.std_err),
            "t": float(self.statistic),
            "p": float(self.pvalue),
            "ll": float(self.ci_lower),
            "ul": float(self.ci_upper),
        }


def _statistic_and_gradient(
    result: Any,
    exog: np.ndarray,
    kind: str,
    *,
    derivative_with_respect_to: int | None = None,
    h: float | None = None,
    outcome: int | None = None,
) -> tuple[float, np.ndarray]:
    """Return the averaged statistic and its gradient w.r.t. the parameters.

    The delta-method standard error needs ``G = d(statistic)/d(beta)``.  Two
    routes are used, and the analytic one always wins when it exists:

    * **Analytic.**  For predictors with a known response-scale family
      (``logit``/``probit``/``poisson``/``nbreg`` and the linear ``xb``), both
      the average marginal effect and its gradient are closed-form functions of
      the linear index:

          AME_j = mean_i g'(eta_i) b_j
          d AME_j / d b_l = mean_i g''(eta_i) X_il b_j + [l == j] mean_i g'(eta_i)

      This is what Stata differentiates (its ``r(Jacobian)`` after
      ``margins, dydx(weight)`` is ``[-1.10858047e-01, -6.64926510e-04,
      -4.79699615e-05]``, matching the formula to every digit).  The ordered
      models are also analytic: their ``pr`` is a cut-point difference, so the
      gradient runs over the slopes *and* the cut points (see
      :func:`_analytic_ordered_margin`).

    * **Adaptive numeric.**  For predictors with no closed form in the linear
      index (tobit's ``e()``/``ystar()`` expectations), central differences use
      a step scaled to the magnitude of the value being perturbed,
      ``h = eps^(1/4) * max(1, |value|)``, refined once by Richardson
      extrapolation.  A *fixed* ``h = 1e-6`` is what made the old version lose
      3-5 significant digits: ``(p(x+h) - p(x-h)) / 2h`` for a probability
      around 0.1 subtracts two numbers that agree to 10 decimal places, so the
      relative error of the difference grows like ``eps * p / (h * p')`` and the
      absolute error floor of the derivative is ``eps * p / h ≈ 4e-11`` -- the
      same order as the average marginal effect being reported (3.5e-4 here,
      i.e. 1e-7 relative).  Measured step sweep on ``margins, dydx(weight)``
      after ``logit``: h=1e-3 -> 4.9e-6, 1e-6 -> 7.1e-3, 1e-8 -> 7.0e-1.
    """

    if derivative_with_respect_to is not None:
        analytic = _analytic_marginal_effect(
            result, exog, kind, derivative_with_respect_to, outcome=outcome
        )
        if analytic is not None:
            return analytic

    if derivative_with_respect_to is None:
        analytic = _analytic_predictive_margin(result, exog, kind, outcome=outcome)
        if analytic is not None:
            return analytic

    return _numeric_statistic_and_gradient(
        result,
        exog,
        kind,
        derivative_with_respect_to=derivative_with_respect_to,
        h=h,
    )


#: Steps for the numeric fallback, by *what is being perturbed*.
#:
#: A central first difference of a smooth function has truncation error
#: ``O(h^2 f''')`` and round-off error ``O(eps * f / h)``; one Richardson
#: extrapolation removes the ``h^2`` term, leaving ``O(h^4)`` against ``eps/h``.
#: The optimum is then ``h ~ eps^(1/5)`` in theory, but the prediction itself is
#: only known to ~1e-16 relative, so the practical optimum is flatter and is
#: measured, not derived.  Sweep on ``tobit mpg weight foreign, ll(17)`` against
#: Stata's ``r(Jacobian)`` and ``r(table)`` SEs (StataNow 19.5, 2026-10-24):
#:
#: =================  ==============  =============
#: covariate step      worst b rel     worst SE rel
#: =================  ==============  =============
#: 1.0 * eps^(1/3)       5.6e-09         2.5e-05
#: 3.0 * eps^(1/3)       5.6e-09         2.4e-06
#: 3.0 * eps^(1/4)       1.5e-08         1.1e-07
#: 30  * eps^(1/3)       5.6e-09         2.8e-06
#: =================  ==============  =============
#:
#: The covariate step therefore uses ``3 eps^(1/4)`` (the derivative of the
#: prediction with respect to a covariate is computed at every observation, so a
#: wider step is safe) and the *parameter* step, which feeds a second difference
#: of an already-differenced quantity, stays at ``eps^(1/3)``.
_COVARIATE_STEP = 3.0 * float(np.finfo(float).eps ** 0.25)
_PARAMETER_STEP = float(np.finfo(float).eps ** (1.0 / 3.0))


def _covariate_steps(values: np.ndarray) -> np.ndarray:
    return _COVARIATE_STEP * np.maximum(1.0, np.abs(np.asarray(values, dtype=float)))


def _parameter_steps(values: np.ndarray) -> np.ndarray:
    return _PARAMETER_STEP * np.maximum(1.0, np.abs(np.asarray(values, dtype=float)))


def _richardson(first: np.ndarray, first_half: np.ndarray) -> np.ndarray:
    """One Richardson extrapolation step: error ``O(h^2) -> O(h^4)``."""

    return (4.0 * first_half - first) / 3.0


def _parameter_setter(result: Any) -> Any:
    """Context manager swapping ``result._params`` without mutating the caller."""

    import contextlib

    @contextlib.contextmanager
    def swapped(parameter_vector: np.ndarray) -> Any:
        original = result._params
        result._params = pd.Series(
            np.asarray(parameter_vector, dtype=float), index=original.index
        )
        try:
            yield
        finally:
            result._params = original

    return swapped


def _analytic_predictive_margin(
    result: Any, exog: np.ndarray, kind: str, *, outcome: int | None = None
) -> tuple[float, np.ndarray] | None:
    """``d mean(g(x b)) / d b`` in closed form, or ``None`` if unavailable."""

    ordered = _analytic_ordered_margin(result, exog, kind, outcome=outcome)
    if ordered is not None:
        return ordered

    family = prediction_family(result, kind)
    if family is None:
        return None
    params = np.asarray(result.params, dtype=float).reshape(-1)
    n_columns = exog.shape[1]
    eta = exog @ params[:n_columns]
    first, _second = family_slopes(family, eta)
    value = float(np.mean(_prediction_at_eta(result, kind, family, eta)))
    gradient = np.zeros(params.size, dtype=float)
    gradient[:n_columns] = first @ exog / float(exog.shape[0])
    return value, gradient


def _analytic_ordered_margin(
    result: Any,
    exog: np.ndarray,
    kind: str,
    *,
    position: int | None = None,
    outcome: int | None = None,
) -> tuple[float, np.ndarray] | None:
    """Analytic AME (or predictive margin) for an ordered model's ``pr``.

    The ordered response is ``P(y = k) = F(c_k - eta) - F(c_{k-1} - eta)``, so

        dP/deta  = -f(c_k - eta) + f(c_{k-1} - eta)
        dP/dc_m  = f(c_m - eta) ([m = k] - [m = k - 1])

    and the delta-method gradient with respect to the whole parameter vector
    (slopes *and* cut points) follows by the chain rule.  Stata's
    ``margins, dydx(weight)`` after ``ologit`` is exactly this quantity with
    ``k = 1`` (the default ``predict(pr)``); differencing it numerically loses
    the small AME (9.4e-6 here) to cancellation, which is the M1 defect.
    """

    if str(kind).strip() != "pr":
        return None
    structure = ordered_structure(result)
    if structure is None:
        return None

    from westata.postestimation.predict import _ordered_pieces

    pieces = _ordered_pieces(result, exog, outcome=outcome)
    if pieces is None:  # pragma: no cover - structure was just verified
        return None
    probability, first, second, cut_first, cut_second = pieces

    params = np.asarray(result.params, dtype=float).reshape(-1)
    slope_positions, cut_positions, _link = structure
    beta = params[slope_positions]
    n_obs = int(exog.shape[0])
    gradient = np.zeros(params.size, dtype=float)

    if position is None:
        # Predictive margin: d mean(P) / d beta and / d cuts.
        gradient[slope_positions] = first @ exog / n_obs
        gradient[cut_positions] = cut_first.sum(axis=1) / n_obs
        return float(np.mean(probability)), gradient

    # Marginal effect with respect to one covariate.
    if position >= exog.shape[1]:
        return None
    value = float(np.mean(first * beta[position]))
    gradient[slope_positions] = (second * beta[position]) @ exog / n_obs
    gradient[position] += float(np.mean(first))
    gradient[cut_positions] = cut_second.sum(axis=1) * beta[position] / n_obs
    return value, gradient


def _analytic_marginal_effect(
    result: Any, exog: np.ndarray, kind: str, position: int, *, outcome: int | None = None
) -> tuple[float, np.ndarray] | None:
    """``d mean(g(x b)) / d x_position`` and its gradient, in closed form."""

    ordered = _analytic_ordered_margin(
        result, exog, kind, position=position, outcome=outcome
    )
    if ordered is not None:
        return ordered

    family = prediction_family(result, kind)
    if family is None:
        return None
    params = np.asarray(result.params, dtype=float).reshape(-1)
    n_columns = exog.shape[1]
    if not 0 <= position < n_columns:
        return None
    eta = exog @ params[:n_columns]
    first, second = family_slopes(family, eta)
    slope = float(params[position])

    value = float(np.mean(first * slope))

    gradient = np.zeros(params.size, dtype=float)
    gradient[:n_columns] = (second * slope) @ exog / float(exog.shape[0])
    gradient[position] += float(np.mean(first))
    return value, gradient


def _numeric_statistic_and_gradient(
    result: Any,
    exog: np.ndarray,
    kind: str,
    *,
    derivative_with_respect_to: int | None,
    h: float | None,
) -> tuple[float, np.ndarray]:
    """Adaptive-step central differences with one Richardson refinement."""

    params = np.asarray(result.params, dtype=float).copy()
    k = params.size
    exog = np.asarray(exog, dtype=float)
    swapped = _parameter_setter(result)

    def prediction(values: np.ndarray, design: np.ndarray) -> np.ndarray:
        with swapped(values):
            return np.asarray(predict(result, design, kind), dtype=float).reshape(-1)

    def evaluate(parameter_vector: np.ndarray) -> np.ndarray:
        """The statistic's integrand, as a per-observation array."""

        if derivative_with_respect_to is None:
            return prediction(parameter_vector, exog)
        position = derivative_with_respect_to
        column = exog[:, position]
        step = (
            np.full(column.shape, float(h))
            if h is not None
            else _covariate_steps(column)
        )
        forward = exog.copy()
        forward[:, position] = column + step
        backward = exog.copy()
        backward[:, position] = column - step
        first = (prediction(parameter_vector, forward) - prediction(
            parameter_vector, backward
        )) / (2.0 * step)
        half = step / 2.0
        forward[:, position] = column + half
        backward[:, position] = column - half
        second = (prediction(parameter_vector, forward) - prediction(
            parameter_vector, backward
        )) / (2.0 * half)
        return _richardson(first, second)

    value = float(np.mean(evaluate(params)))

    gradient = np.zeros(k, dtype=float)
    for index in range(k):
        if h is not None:
            step = float(h)
        else:
            step = _parameter_steps(np.array([params[index]]))[0]
        forward = params.copy()
        forward[index] += step
        backward = params.copy()
        backward[index] -= step
        difference = (
            float(np.mean(evaluate(forward))) - float(np.mean(evaluate(backward)))
        ) / (2.0 * step)

        half_forward = params.copy()
        half_forward[index] += step / 2.0
        half_backward = params.copy()
        half_backward[index] -= step / 2.0
        half_difference = (
            float(np.mean(evaluate(half_forward)))
            - float(np.mean(evaluate(half_backward)))
        ) / (2.0 * (step / 2.0))

        gradient[index] = float(_richardson(np.array(difference), np.array(half_difference)))
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


#: Stata's default prediction type per model for ``margins``.  For a linear
#: model the response *is* the linear prediction, but for logit/probit Stata
#: differentiates the predicted probability (``pr``), not ``xb``; getting this
#: wrong turns an average marginal effect into a plain coefficient.
#:
#: ``tobit``'s default is ``xb``, *not* ``e(a,b)``.  Measured on StataNow 19.5
#: (2026-10-24, ``sysuse auto``, ``tobit mpg weight foreign, ll(17)``):
#:
#: =====================================  =================  ===============
#: command                                estimate           equals
#: =====================================  =================  ===============
#: ``margins, dydx(weight)``              -0.00779786618190   coefficient
#: ``margins, dydx(weight) predict(xb)``  -0.00779786618190   coefficient
#: ``margins, dydx(weight) predict(e(17,.))``  -0.00477824532202  censored mean
#: ``margins``                            20.6918844403444   mean(xb)
#: =====================================  =================  ===============
#:
#: Stata's own ``e(marginsok)`` for tobit lists ``default XB Pr(passthru)
#: E(passthru) YStar(passthru)``, i.e. ``xb`` is the flagged default.  Reporting
#: ``e(a,b)`` here would silently answer a different question from Stata's.
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
    "tobit": "xb",
}


def default_predict_kind(model_name: str) -> str:
    """Return the prediction type ``margins`` differentiates by default."""

    return DEFAULT_PREDICT.get(model_name, "xb")


def margins(
    result: Any,
    *,
    dydx: Sequence[str] | str | None = None,
    eyex: Sequence[str] | str | None = None,
    dyex: Sequence[str] | str | None = None,
    eydx: Sequence[str] | str | None = None,
    at: Mapping[str, Any] | None = None,
    over: str | None = None,
    exog: np.ndarray | None = None,
    param_names: Sequence[str] | None = None,
    level: float = 0.95,
    predict_kind: str | None = None,
    factor_vars: Sequence[str] | None = None,
    outcome: int | None = None,
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
    outcome:
        Ordered models only: the category whose probability ``predict, pr``
        refers to (Stata's ``predict, pr outcome(#k)``).  ``None`` is Stata's
        default, category 1.
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

    # ``at()`` replaces covariate values, so it must be applied BEFORE any form is
    # computed -- including the ``dyex`` factor, which is mean(x) over the ADJUSTED
    # design.  Applying it later made ``margins(dyex="x1", at={"x1": 2})`` return
    # exactly the unadjusted number.
    if at:
        exog = _apply_at(exog, names, at)

    # ``margins.sthlp:265-266`` documents four derivative forms:
    #
    #   dydx(varlist)   d response / d x
    #   eyex(varlist)   d ln(response) / d ln(x)   -- elasticity
    #   dyex(varlist)   d response / d ln(x)       -- semi-elasticity
    #   eydx(varlist)   d ln(response) / d x       -- semi-elasticity
    #
    # Only ``dydx`` is implemented.  The three elasticity forms need a PER-OBSERVATION
    # scaling factor (x/response, x, 1/response) applied to both the estimate and its
    # gradient BEFORE averaging -- scaling the averaged number instead would give a
    # different, wrong result, and the standard error has to carry the same factor.
    # Refusing here beats the ``TypeError: unexpected keyword argument`` a caller got
    # before, which does not say the form exists at all.
    requested_forms = {
        "eyex": eyex,
        "dyex": dyex,
        "eydx": eydx,
    }
    named = [form for form, value in requested_forms.items() if value is not None]
    if named:
        if dydx is not None:
            raise ValueError(
                "margins accepts one derivative form at a time; got dydx= together "
                f"with {named[0]}="
            )
        if len(named) > 1:
            raise ValueError(
                "margins accepts one derivative form at a time; got "
                + " and ".join(named)
            )
        form = named[0]
        if dydx is not None:  # pragma: no cover - guarded above
            raise ValueError("margins accepts one derivative form at a time")
        # ``dyex`` = d E[y] / d ln(x) = x * d E[y] / d x, factor ``x_i``.
        # ``eyex`` = d ln E[y] / d ln(x) = (x_i / E[y_i]) * d E[y] / d x.
        # ``eydx`` = d ln E[y] / d x    = (1 / E[y_i]) * d E[y] / d x.
        #
        # The last two factors depend on ``E[y_i]``, which changes with the covariates,
        # so they cannot be pulled out of the average as a constant coefficient can.
        # They are computed per observation and then averaged -- which is what Stata
        # does and what makes ``eyex``/``eydx`` different from a rescaled ``dydx``.
        requested = None
        for candidate in (dyex, eyex, eydx):
            if candidate is not None:
                requested = (
                    [candidate] if isinstance(candidate, str) else list(candidate)
                )
                break
        if requested is None:  # pragma: no cover - the caller always passes one
            raise ValueError("margins needs a derivative form")
        # ``kind`` is defined further down for the ``dydx`` path; the elastic branch
        # returns early, so it has to compute the same value here.
        elastic_kind = predict_kind or default_predict_kind(
            str(getattr(result, "model_name", ""))
        )
        rows = _marginal_effects(
            result,
            exog,
            names,
            requested,
            factor_set=factor_set,
            over=over,
            level=level,
            predict_kind=elastic_kind,
            outcome=outcome,
            elasticity_form=form,
        )
        return rows[0] if len(rows) == 1 else rows

    # Stata differentiates the response, not the linear index: logit/probit
    # marginals are on the probability scale and count models on the mean scale.
    #
    # ``default`` is a KEYWORD in ``e(marginsok)``, not a prediction type -- passing it
    # through as a literal made ``margins(dydx=..., predict_kind="default")`` fail even
    # though Stata accepts it.  It resolves to the command's own default.
    requested_kind = predict_kind
    if requested_kind is not None and str(requested_kind).strip().lower() == "default":
        requested_kind = None
    kind = requested_kind or default_predict_kind(
        str(getattr(result, "model_name", ""))
    )

    # ``e(marginsnotok)`` lists the prediction types ``margins`` must REFUSE after this
    # command.  It is a real macro that ``logit.ado:254-263`` and ``probit.ado`` set, and
    # Stata's own estimators rely on it -- ``areg.ado:255``, ``arch.ado:1531``,
    # ``arima.ado:727`` and dozens more each forbid a different subset.  This module read
    # it only in a comment, so ``margins`` accepted ``stdp``/``residuals`` after logit
    # even though the command declares them disallowed.
    forbidden = str(result.e.get("marginsnotok", "") or "").split()
    if forbidden:
        normalised_forbidden = {
            entry.strip().lower() for entry in forbidden if entry.strip()
        }
        if str(kind).strip().lower() in normalised_forbidden:
            raise ValueError(
                f"margins: predict type {kind!r} is not allowed after "
                f"{getattr(result, 'model_name', 'this command')!r} "
                f"(e(marginsnotok) = {result.e.get('marginsnotok')!r}); "
                "use one of e(marginsok) "
                f"= {result.e.get('marginsok')!r}"
            )

    # ``at()`` was already applied above, before the derivative form was chosen.
    if dydx is None:
        rows = _predictive_margins(
            result, exog, names, over=over, level=level, predict_kind=kind,
            outcome=outcome,
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
            outcome=outcome,
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
    outcome: int | None = None,
    elasticity_form: str | None = None,
) -> list[MarginsResult]:
    dist = "z" if getattr(result, "dist", "t") == "z" else "t"
    df_resid = float(result.df_resid)
    cov = result.cov_params()

    groups = _groups(over, exog.shape[0], getattr(result, "model_data", None))
    rows: list[MarginsResult] = []

    def _scale(
        name: str,
        estimate: float,
        gradient: np.ndarray,
        subset: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        """Apply the elasticity factor to the estimate AND its gradient.

        ``margins.sthlp:265-266`` defines three elastic forms, and the factor is the
        whole difference between them:

        * ``dyex``  = ``d E[y] / d ln(x)``  factor ``x_i``;
        * ``eyex``  = ``d ln E[y] / d ln(x)``  factor ``x_i / E[y_i]``;
        * ``eydx``  = ``d ln E[y] / d x``  factor ``1 / E[y_i]``.

        ``x_i`` alone is a constant for the linear predictor, but ``E[y_i]`` is not, so
        the last two are averaged PER OBSERVATION.  The gradient has to carry the same
        factor at every observation, because the delta-method variance is ``G' V G``
        with ``G`` averaged over the same terms.
        """

        if elasticity_form is None:
            return estimate, gradient

        position = names.index(name)
        covariate = np.asarray(subset[:, position], dtype=float)
        if elasticity_form == "dyex":
            factor = float(np.mean(covariate))
            return estimate * factor, gradient * factor

        # ``eyex``/``eydx`` divide by the predicted response at each observation.
        prediction = np.asarray(
            predict(result, subset, predict_kind), dtype=float
        ).reshape(-1)
        if prediction.size != covariate.size:  # pragma: no cover - defensive
            raise ValueError(
                "margins: the predicted response has "
                f"{prediction.size} entries for {covariate.size} rows"
            )
        if np.any(prediction == 0.0):
            raise ValueError(
                f"margins, {elasticity_form}() divides by the predicted response, and "
                "at least one fitted value is exactly zero; the elasticity is undefined "
                "there."
            )
        # ``eyex``/``eydx`` are derivatives of ln E[y].  A NEGATIVE fitted value makes
        # the logarithm undefined, and ``1/E[y]`` still evaluates there -- returning a
        # finite number for an expression that does not exist.  Stata's own behaviour
        # for this case cannot be read from the ado (the factors are computed inside a
        # Mata routine with no source in ado/base), so the honest answer is a refusal
        # naming the offending count rather than a plausible number.
        if np.any(prediction < 0.0):
            raise ValueError(
                f"margins, {elasticity_form}() is a derivative of ln(E[y]), and "
                f"{int(np.sum(prediction < 0.0))} fitted value(s) are negative, where "
                "the log is undefined.  Use dydx() for the derivative of the level, or "
                "model a strictly positive response."
            )
        per_observation = (
            covariate / prediction if elasticity_form == "eyex" else 1.0 / prediction
        )
        factor = float(np.mean(per_observation))
        return estimate * factor, gradient * factor

    for name in requested:
        if name not in names:
            raise KeyError(f"dydx: column {name!r} not in {names}")
        position = names.index(name)
        is_factor = name in factor_set

        if groups is None:
            if is_factor:
                estimate, gradient = _discrete_change(
                    result, exog, predict_kind, position, outcome=outcome
                )
            else:
                estimate, gradient = _statistic_and_gradient(
                    result,
                    exog,
                    predict_kind,
                    derivative_with_respect_to=position,
                    outcome=outcome,
                )
            estimate, gradient = _scale(name, estimate, gradient, exog)
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
                        result, subset, predict_kind, position, outcome=outcome
                    )
                else:
                    estimate, gradient = _statistic_and_gradient(
                        result,
                        subset,
                        predict_kind,
                        derivative_with_respect_to=position,
                        outcome=outcome,
                    )
                # The semi-elasticity factor is the mean of the covariate WITHIN the
                # group, because each ``over`` row reports that group's own margin.
                estimate, gradient = _scale(name, estimate, gradient, subset)
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
    result: Any,
    exog: np.ndarray,
    kind: str,
    position: int,
    *,
    outcome: int | None = None,
) -> tuple[float, np.ndarray]:
    """Stata's factor-variable marginal effect: ``P(x=1) - P(x=0)``.

    Stata differentiates this exactly as it differentiates a continuous
    marginal effect, so the same split applies: when the prediction has a known
    response-scale family the gradient is

        d/db_l mean_i [ g(eta_i^1) - g(eta_i^0) ]
            = mean_i [ g'(eta_i^1) X_i^1_l - g'(eta_i^0) X_i^0_l ]

    which is what Stata's ``r(Jacobian)`` holds (for a linear model it collapses
    to the coefficient itself, so the SE equals ``bse`` -- verified:
    ``margins, dydx(foreign)`` after ``regress mpg weight i.foreign`` reports
    ``se = 1.07599411819307``, exactly the coefficient's own SE).  Otherwise the
    adaptive numeric route is used.
    """

    params = np.asarray(result.params, dtype=float).copy()
    k = params.size
    exog = np.asarray(exog, dtype=float)
    high = exog.copy()
    high[:, position] = 1.0
    low = exog.copy()
    low[:, position] = 0.0
    swapped = _parameter_setter(result)

    def predictions(parameter_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with swapped(parameter_vector):
            return (
                np.asarray(
                    predict(result, high, kind, outcome=outcome), dtype=float
                ).reshape(-1),
                np.asarray(
                    predict(result, low, kind, outcome=outcome), dtype=float
                ).reshape(-1),
            )

    def evaluate(parameter_vector: np.ndarray) -> float:
        upper, lower = predictions(parameter_vector)
        return float(np.mean(upper - lower))

    family = prediction_family(result, kind)
    if family is not None:
        eta_high = high @ params[: high.shape[1]]
        eta_low = low @ params[: low.shape[1]]
        first_high, _ = family_slopes(family, eta_high)
        first_low, _ = family_slopes(family, eta_low)
        estimate = float(np.mean(_prediction_at_eta(result, kind, family, eta_high)
                                 - _prediction_at_eta(result, kind, family, eta_low)))
        gradient = np.zeros(k, dtype=float)
        gradient[: high.shape[1]] = (
            first_high @ high - first_low @ low
        ) / float(high.shape[0])
        return estimate, gradient

    estimate = evaluate(params)
    gradient = np.zeros(k, dtype=float)
    swapped_steps = _adaptive_steps(params)
    for index in range(k):
        step = float(swapped_steps[index])
        forward = params.copy()
        forward[index] += step
        backward = params.copy()
        backward[index] -= step
        full = (evaluate(forward) - evaluate(backward)) / (2.0 * step)

        half_forward = params.copy()
        half_forward[index] += step / 2.0
        half_backward = params.copy()
        half_backward[index] -= step / 2.0
        half = (evaluate(half_forward) - evaluate(half_backward)) / step
        gradient[index] = float(_richardson(np.array(full), np.array(half)))
    return estimate, gradient


def _prediction_at_eta(
    result: Any, kind: str, family: str, eta: np.ndarray
) -> np.ndarray:
    """``g(eta)`` for a known family (used to keep the estimate exact)."""

    from scipy import stats as _stats

    if family == "identity":
        return np.asarray(eta, dtype=float)
    if family == "logit":
        return 1.0 / (1.0 + np.exp(-np.asarray(eta, dtype=float)))
    if family == "probit":
        return _stats.norm.cdf(np.asarray(eta, dtype=float))
    if family == "exp":
        return np.exp(np.asarray(eta, dtype=float))
    raise ValueError(f"predict: unknown prediction family {family!r}")


def _predictive_margins(
    result: Any,
    exog: np.ndarray,
    names: list[str],
    *,
    over: Any,
    level: float,
    predict_kind: str,
    outcome: int | None = None,
) -> list[MarginsResult]:
    dist = "z" if getattr(result, "dist", "t") == "z" else "t"
    df_resid = float(result.df_resid)
    cov = result.cov_params()

    groups = _groups(over, exog.shape[0], getattr(result, "model_data", None))
    rows: list[MarginsResult] = []

    if groups is None:
        estimate, gradient = _statistic_and_gradient(
            result, exog, predict_kind, outcome=outcome
        )
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
                result, exog[mask], predict_kind, outcome=outcome
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
