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

import itertools

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from westata.errors import UnsupportedError
from westata.postestimation.predict import (
    DEFAULT_PREDICT as _DEFAULT_PREDICT,
)
from westata.postestimation.predict import (
    default_predict_kind as _default_predict_kind,
)
from westata.postestimation.predict import (
    family_slopes,
    ordered_structure,
    predict,
    prediction_family,
)

from westata.results.summary import _stata_number

__all__ = [
    "DEFAULT_PREDICT",
    "MarginsResult",
    "MarginsTable",
    "default_predict_kind",
    "margins",
]


#: The 78-character fields of Stata's margins table, measured off a live run. Each value
#: is right-aligned so that it ENDS at the given exclusive column, which is what makes a
#: 10-character estimate and an 8-character one line up under the same heading.
_MARGIN_END_ESTIMATE = 25
_MARGIN_END_STDERR = 36
_MARGIN_END_STATISTIC = 45
_MARGIN_END_PVALUE = 53
_MARGIN_END_LOWER = 66
_MARGIN_END_UPPER = 78
_MARGIN_WIDTH = 78


def _margin_number(value: float, end_column: int, used: int) -> str:
    """``value`` right-aligned so that it ends at ``end_column``.

    Margins cells use SEVEN DECIMAL PLACES with trailing zeros dropped -- not a
    significant-digit rule. Verified against every number in the reference table:

        .0135531672 -> .0135532      .0282773592 -> .0282774
        .0075124809 -> .0075125      .0011710248 -> .001171

    A six-significant-figure rule reproduces the first and third but gives ``.00751248``
    and ``.00117102`` for the second and fourth, which is what Stata does NOT print.
    Values too wide for seven decimals fall back to significant figures.
    """

    if value is None or not np.isfinite(value):
        text = "."
    else:
        number = float(value)
        text = f"{number:.7f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        if text.startswith("0."):
            text = text[1:]
        elif text.startswith("-0."):
            text = "-" + text[2:]
        if len(text) > 9:
            # Too wide for the cell: fall back to significant figures that fit.
            for digits in range(7, 0, -1):
                candidate = f"{number:.{digits}g}"
                if candidate.startswith("0."):
                    candidate = candidate[1:]
                elif candidate.startswith("-0."):
                    candidate = "-" + candidate[2:]
                if len(candidate) <= 9:
                    text = candidate
                    break
    return " " * max(end_column - used - len(text), 0) + text


def _margin_rate(value: float, decimals: int, end_column: int, used: int) -> str:
    """A fixed-decimal field (statistic, p-value) right-aligned to ``end_column``."""

    text = "." if value is None or not np.isfinite(value) else f"{value:.{decimals}f}"
    return " " * max(end_column - used - len(text), 0) + text


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
    #: Table context, stamped onto every row by ``margins()``: the heading, the sample
    #: size, the VCE label, the prediction expression and the ``wrt`` list. They describe
    #: the whole table rather than this row, but keeping them here lets a single-row
    #: result render itself without a separate container.
    title: str = ""
    nobs: float = float("nan")
    vce_label: str = ""
    expression: str = ""
    variables: tuple[str, ...] = ()
    column_label: str = ""
    stat_label: str = "z"
    #: ``atmeans`` fixes the covariates at their sample means; Stata prints the values it
    #: used under an "At:" block. Carried here so the renderer can show them.
    at_means: tuple[tuple[str, float], ...] = ()

    def __str__(self) -> str:
        """Reproduce Stata's margins table for a single row.

        Reference (``margins, dydx(s)`` after a logit, n = 758)::

            Average marginal effects                                   Number of obs = 758
            Model VCE: OIM
            Expression: Pr(rns), predict()
            dy/dx wrt:  s
            ------------------------------------------------------------------------------
                         |            Delta-method
                         |      dy/dx   std. err.      z    P>|z|     [95% conf. interval]
            -------------+----------------------------------------------------------------
                       s |  -.0135532   .0075125    -1.80   0.071    -.0282774     .001171
            ------------------------------------------------------------------------------

        Every line is 78 characters. The rule of the table is that each value is
        right-aligned to a fixed END column (25, 36, 45, 53, 66, 78), which is what makes
        a ten-character estimate and an eight-character one line up under one heading.
        """

        return MarginsTable([self]).render()

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


class MarginsTable(list):
    """A whole ``margins`` table: N rows rendered as ONE Stata table.

    Subclasses ``list`` on purpose. ``margins()`` has always returned a plain list when it
    produced several rows and a bare result when it produced one, and callers rely on that
    -- ``isinstance(x, list)`` is asserted in the test suite. Subclassing keeps every one of
    those behaviours (iteration, indexing, ``len``) while giving the table a ``__str__``, so
    a multi-term call finally renders instead of printing a list of dataclass reprs.

    A single-row call still returns a bare :class:`MarginsResult`, which is not a list; its
    ``__str__`` builds a one-row ``MarginsTable`` and renders that, so both shapes print the
    same table.
    """

    def __str__(self) -> str:
        """Print the table rather than the list-of-dataclasses repr.

        A ``list`` subclass inherits ``list.__repr__``, so without this override a
        multi-term call would still print ``[MarginsResult(...), ...]``.
        """

        return self.render()

    def render(self) -> str:
        """Stata's full margins table."""

        if not self:
            return ""
        first = self[0]
        rule = "-" * _MARGIN_WIDTH

        lines: list[str] = []
        if first.title:
            # "…effects" padded to column 59, then the sample size in three columns.
            lines.append(
                f"{first.title:<59}Number of obs = {first.nobs:>3.0f}"
            )
        if first.vce_label:
            lines.append(f"Model VCE: {first.vce_label}")
        if first.expression:
            lines.append(f"Expression: {first.expression}")
        if first.variables:
            # Stata writes "dy/dx wrt:" followed by TWO spaces before the list.
            lines.append(f"{first.column_label} wrt:  " + " ".join(first.variables))
        # ``atmeans`` prints the values it fixed, whatever the form. This used to live INSIDE
        # the ``if first.variables`` block, so `margins(atmeans=True)` -- which has no dydx and
        # therefore no wrt line -- never printed them. The means were computed correctly and the
        # table simply dropped them, which is why the statistic also stayed wrong: with no "At:"
        # values the adjusted design was never built, and the margin was the unweighted one.
        #
        #   At: age       = 25.37079 (mean)
        #       education = 10.19551 (mean)
        #
        # Measured: the name is LEFT-aligned in a 10-wide field (not right-aligned like a
        # coefficient row), the continuation lines are indented four spaces to sit under
        # "At: ", and the suffix is "(mean)".
        for position, (name, value) in enumerate(first.at_means):
            prefix = "At: " if position == 0 else "    "
            rendered_value = _stata_number(value, 9).strip()
            lines.append(f"{prefix}{str(name):<10}= {rendered_value} (mean)")
        lines.extend(
            [
                rule,
                f"{'':<13}|{'Delta-method':>24}",
                self._header(first),
                "-" * 13 + "+" + "-" * (_MARGIN_WIDTH - 14),
            ]
        )
        for row in self:
            lines.append(self._row(row))
        lines.append(rule)
        return "\n".join(lines)

    @staticmethod
    def _header(first: MarginsResult) -> str:
        """The two-line header, whose only variable parts are the column and stat labels."""

        column = first.column_label or "dy/dx"
        stat = first.stat_label or "z"
        return (
            f"{'':<13}|{column:>11}   std. err.      {stat}"
            f"    P>|{stat}|     [95% conf. interval]"
        )

    @staticmethod
    def _row(row: MarginsResult) -> str:
        """One data line, every value ending at its measured column."""

        # Stata labels a single predictive margin "_cons"; the upstream row calls it
        # "margin", which is not a name any Stata table shows.
        label = row.label
        if label == "margin":
            label = "_cons"
        text = f"{label:>12} |"
        text += _margin_number(row.estimate, _MARGIN_END_ESTIMATE, len(text))
        text += _margin_number(row.std_err, _MARGIN_END_STDERR, len(text))
        text += _margin_rate(row.statistic, 2, _MARGIN_END_STATISTIC, len(text))
        text += _margin_rate(row.pvalue, 3, _MARGIN_END_PVALUE, len(text))
        text += _margin_number(row.ci_lower, _MARGIN_END_LOWER, len(text))
        text += _margin_number(row.ci_upper, _MARGIN_END_UPPER, len(text))
        return text


def _finish_margins(
    rows: list[MarginsResult],
    result: Any,
    *,
    variables: Sequence[str] = (),
    column_label: str = "",
    atmeans: bool = False,
    at_values: Mapping[str, Any] | None = None,
) -> Any:
    """Stamp the table context onto every row and wrap a multi-row result.

    Stata prints ONE table per ``margins`` call, so the heading, sample size, VCE label and
    prediction expression belong to the table rather than to a row. They are stored on each
    row anyway (a row can then render itself) and the rows are returned as a
    :class:`MarginsTable` when there is more than one.

    Anything this function cannot determine is left EMPTY, which omits the line entirely.
    That is deliberate: printing "Model VCE: OIM" for a model whose VCE was never checked
    would assert something the code does not know.
    """

    if not rows:
        return rows
    model = str(getattr(result, "model_name", "") or "").lower()
    depvar = str(
        getattr(result, "response_name", None)
        or (getattr(result, "e", None) or {}).get("depvar")
        or ""
    )
    cov_type = str(getattr(result, "cov_type", "") or "").lower()
    total = getattr(result, "nobs", None)
    if total is None:
        total = len(getattr(result, "resid", []) or [])

    # Only the cases that have been checked against Stata are labelled.
    ml_models = {"logit", "probit", "logistic", "cloglog", "ologit", "oprobit"}
    if model in ml_models and cov_type in {"", "nonrobust"}:
        vce_label = "OIM"
    else:
        vce_label = ""
    if model in ml_models and depvar:
        expression = f"Pr({depvar}), predict()"
    elif model in {"ols", "regress"}:
        expression = "Linear prediction, predict()"
    else:
        expression = ""
    # ``atmeans`` changes the TITLE, not just the numbers: Stata calls the result
    # "Conditional marginal effects" because it is the effect AT a point, versus the
    # average over the sample. Measured on ``logit treat age education re74 re75`` then
    # ``margins, dydx(*) atmeans``.
    if column_label:
        title = (
            "Conditional marginal effects" if atmeans else "Average marginal effects"
        )
    else:
        title = "Predictive margins"
    at_means = (
        tuple((str(name), float(value)) for name, value in (at_values or {}).items())
        if atmeans
        else ()
    )
    for row in rows:
        row.title = title
        row.nobs = float(total)
        row.vce_label = vce_label
        row.expression = expression
        row.variables = tuple(variables)
        row.column_label = column_label or "Margin"
        row.stat_label = "z" if model in ml_models else "t"
        row.at_means = at_means
    return rows[0] if len(rows) == 1 else MarginsTable(rows)


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


def _per_observation_derivative(
    result: Any, exog: np.ndarray, kind: str, position: int, *, outcome: int | None = None
) -> np.ndarray:
    """``d g(x_i b) / d x_position`` at EACH observation.

    Needed for elasticities, which average the per-observation elasticity rather than
    multiplying averaged pieces together (see the ``_scale`` comment). The analytic slope
    ``g'(eta) * b_j`` is the same expression ``_analytic_marginal_effect`` averages, so it is
    reused here rather than differentiated a second way.
    """

    family = prediction_family(result, kind)
    params = np.asarray(result.params, dtype=float).reshape(-1)
    n_columns = exog.shape[1]
    if family is None or not 0 <= position < n_columns:
        # Fall back to the numeric route's average, broadcast back over the sample.
        value, _ = _statistic_and_gradient(
            result, exog, kind, derivative_with_respect_to=position, outcome=outcome
        )
        return np.full(exog.shape[0], float(value))
    eta = exog @ params[:n_columns]
    first, _ = family_slopes(family, eta)
    return np.asarray(first, dtype=float) * float(params[position])


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
#: ``model_name -> the predict type Stata assumes`` -- re-exported from
#: :mod:`westata.postestimation.predict`, which now owns the single copy. ``margins`` and
#: ``StataResults.predict`` disagreed for several releases because each had its own default;
#: keeping one table is what makes the two entry points agree by construction.
DEFAULT_PREDICT = _DEFAULT_PREDICT
default_predict_kind = _default_predict_kind


def margins(
    result: Any,
    *,
    dydx: Sequence[str] | str | None = None,
    eyex: Sequence[str] | str | None = None,
    dyex: Sequence[str] | str | None = None,
    eydx: Sequence[str] | str | None = None,
    at: Mapping[str, Any] | None = None,
    atmeans: bool = False,
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

    # ``margins`` averages over the ESTIMATION SAMPLE, and under ``fweight`` Stata's sample
    # is the EXPANDED one. The design matrix holds the collapsed rows (24 for the titanic
    # logit), so averaging over it gave marginal effects 4-5% away from Stata's -- measured:
    # ``dydx(child)`` .1809351 here against .1732315 in Stata, and every other covariate off
    # by a similar amount (C11-03 in the 0.0.6 gap report).
    #
    # This expansion MUST run BEFORE ``atmeans`` computes the means, not after. When it ran
    # afterwards the "At:" block reported the UNWEIGHTED means of the 24 collapsed rows --
    # child .3333333 where Stata prints .0495229, female .5 against .2135393, class1 .25
    # against .1476602 -- and the conditional MEM inherited that error (child .2628119
    # against .2223422, 18% off). The AME was unaffected, which is why the bug survived: it
    # only shows in the atmeans form.
    _freq = getattr(result, "freq_weights", None)
    if _freq is not None:
        _freq = np.asarray(_freq, dtype=float).reshape(-1)
        if _freq.size == exog.shape[0] and np.all(np.isfinite(_freq)):
            _repeats = np.rint(_freq).astype(int)
            if np.all(_repeats >= 0) and int(_repeats.sum()) > 0:
                exog = np.repeat(exog, _repeats, axis=0)

    # ``atmeans`` fixes every covariate at its SAMPLE MEAN and reports a CONDITIONAL marginal
    # effect there, which is a different number from the average marginal effect -- measured
    # on grilic's logit: s is -.0135868 at the means against -.0135532 averaged. Stata prints
    # the means it used under "At:", so the values are not a hidden detail.
    if atmeans:
        if at:
            raise ValueError(
                "margins accepts at= or atmeans=True, not both: atmeans fixes EVERY covariate "
                "at its mean, so an explicit at= would contradict it"
            )
        means: dict[str, float] = {}
        position_by_name = {str(name): index for index, name in enumerate(names)}
        # Stata's atmeans uses WEIGHTED means when the fit carried weights. Measured on a
        # titanic logit with [fweight=freq]:
        #
        #   covariate   unweighted   weighted     Stata's "At:" line
        #   child       .3333333     .0495229     .0495229
        #   female      .5000000     .2135393     .2135393
        #   class1      .2500000     .1476602     .1476602
        #
        # and the difference is not cosmetic: the reported Margin came out .5532798 unweighted
        # against Stata's .3027798. The data has only 24 rows carrying a frequency column, so an
        # unweighted mean of it describes the cell layout rather than the population.
        # The fit EXPANDS frequency weights into repeated rows, so ``exog`` has sum(weights) rows
        # while ``result.freq_weights`` still has one entry per source row. Comparing the two
        # lengths directly made this guard reject the weights and silently fall back to the
        # unweighted mean -- the original defect, wearing a new coat. Expand first, then check.
        weight_vector = getattr(result, "freq_weights", None)
        if weight_vector is not None:
            weight_vector = np.asarray(weight_vector, dtype=float).ravel()
            if weight_vector.size != exog.shape[0]:
                counts = np.rint(weight_vector).astype(int)
                if counts.sum() == exog.shape[0] and np.all(counts >= 0):
                    weight_vector = np.repeat(np.ones(counts.size), counts)
                else:
                    weight_vector = None
        for name, position in position_by_name.items():
            if str(name) == "Intercept":
                continue
            base = str(name).split("[")[0]
            if base in means:
                continue
            column = exog[:, position]
            if weight_vector is None:
                means[base] = float(np.mean(column))
            else:
                means[base] = float(np.average(column, weights=weight_vector))
        at = means

    # ``at()`` with a SEQUENCE of values produces one result per value (and one per
    # combination when several covariates carry sequences -- ``margins.sthlp:1000-1005``).
    # Handling that by recursing keeps every downstream form correct: the derivative, the
    # elasticity scaling and the standard error are all computed on the ADJUSTED design, so
    # they cannot be reused across scenarios.
    scenarios = _at_scenarios(at) if at else [{}]
    if len(scenarios) > 1:
        collected: list[MarginsResult] = []
        for scenario in scenarios:
            produced = margins(
                result,
                dydx=dydx,
                eyex=eyex,
                dyex=dyex,
                eydx=eydx,
                at=scenario,
                over=over,
                exog=exog,
                param_names=param_names,
                level=level,
                predict_kind=predict_kind,
                factor_vars=factor_vars,
                outcome=outcome,
            )
            rows_for_scenario = produced if isinstance(produced, list) else [produced]
            label = ", ".join(f"{name}={value}" for name, value in scenario.items())
            for row in rows_for_scenario:
                # Stata labels each block with the at() setting; carrying the label is what
                # makes a multi-scenario table readable rather than a list of identical
                # rows a reader cannot tell apart.
                row.label = f"{row.label} ({label})" if row.label else label
                collected.append(row)
        return collected

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
        #: What Stata prints as the column heading for each elasticity form. ``dydx`` is
        #: handled further down; these three are the semi- and full-elasticity variants.
        elastic_label = ""
        for candidate, form_name in (
            (dyex, "dy/ex"),
            (eyex, "ey/ex"),
            (eydx, "ey/dx"),
        ):
            if candidate is not None:
                requested = (
                    [candidate] if isinstance(candidate, str) else list(candidate)
                )
                elastic_label = form_name
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
        return _finish_margins(
            rows, result, variables=requested, column_label=elastic_label
        )

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
        # ``atmeans`` was resolved into ``at`` further up, but this branch used to finish
        # without passing it back, so ``at_means`` stayed empty and Stata's "At:" lines never
        # printed. The means themselves were computed correctly all along.
        return _finish_margins(
            rows, result, atmeans=atmeans, at_values=at if atmeans else None
        )
    requested = [dydx] if isinstance(dydx, str) else list(dydx)
    # ``dydx(s expr)`` -- Stata accepts a SPACE-SEPARATED varlist inside one pair of
    # parentheses, and a caller writing that is asking for two effects, not for a column named
    # "s expr". Without this split the resolver raised
    # ``KeyError: column 's expr' not in [...]``, which reads like a typo in the data rather
    # than an unsupported spelling.
    split_request: list[str] = []
    for entry in requested:
        split_request.extend(str(entry).split())
    requested = split_request
    # ``dydx(*)`` means EVERY covariate, which in Stata is every parameter except the
    # constant: a live ``margins, dydx(*)`` after ``regress y k l`` reports
    # "dy/dx wrt:  k l". Passing '*' straight to the resolver raised
    # ``KeyError: column '*' not in [...]``, which does not tell the caller that the
    # wildcard exists at all.
    if "*" in requested:
        # The constant is excluded by name: patsy spells it "Intercept", and an explicit
        # "_cons" column appears in designs built without patsy.
        everything = [
            str(name)
            for name in names
            if str(name) not in ("Intercept", "_cons", "const")
        ]
        if len(requested) == 1:
            requested = everything
        else:
            # A mixed list such as ['*', 'x1'] names x1 twice; Stata prints each
            # covariate once, so the wildcard is expanded and duplicates collapse.
            expanded: list[str] = []
            for entry in requested:
                for name in everything if entry == "*" else [entry]:
                    if name not in expanded:
                        expanded.append(name)
            requested = expanded
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
    return _finish_margins(
        rows,
        result,
        variables=requested,
        column_label="dy/dx",
        atmeans=atmeans,
        at_values=at,
    )


def _at_scenarios(at: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand ``at=`` into the scenarios Stata evaluates.

    ``margins.sthlp:995-1005`` states the rule outright:

        at(x1=5)                     one scenario
        at(x1=(5 10))                two: x1=5, then x1=10
        at(x1=(5 10) x2=(0 1))       FOUR: every combination, in turn

    So a sequence means "evaluate at each value", and several sequences mean their
    CARTESIAN PRODUCT -- not a zipped pass. Zipping would silently drop scenarios whenever
    the lists differ in length, and dropping a scenario is invisible in the output: the
    table just has fewer rows than the caller asked for.

    A scalar (or a one-element sequence) stays one scenario, so existing calls are
    unaffected.
    """

    if not at:
        return [{}]

    columns: list[str] = []
    value_lists: list[list[Any]] = []
    for name, value in at.items():
        columns.append(name)
        value_lists.append(list(value) if _is_sequence(value) else [value])

    scenarios: list[dict[str, Any]] = []
    for combination in itertools.product(*value_lists):
        scenarios.append(dict(zip(columns, combination)))
    return scenarios


def _is_sequence(value: Any) -> bool:
    """True for a list/tuple/array/Series, False for a string or a number.

    ``str`` is deliberately excluded: ``at={"g": "treated"}`` is a scalar label, and
    iterating it into characters would produce one scenario per letter.
    """

    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim > 0
    return isinstance(value, (list, tuple, pd.Series))


def _apply_at(
    exog: np.ndarray, names: Sequence[str], at: Mapping[str, Any]
) -> np.ndarray:
    adjusted = exog.copy()
    for name, value in at.items():
        if name not in names:
            raise KeyError(f"at(): column {name!r} not in {list(names)}")
        position = list(names).index(name)
        if _is_sequence(value):
            raise TypeError(
                f"at(): {name!r} received a sequence inside a single scenario; "
                "sequences are expanded by margins() before this point"
            )
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
        # The elasticity is the average of the PER-OBSERVATION elasticities, not the
        # average derivative times the average factor. Those differ whenever the derivative
        # and the factor covary -- which they do, since both carry the fitted probability.
        #
        # Measured on ``logit treat age education re74 re75`` then ``margins, eyex(age)``:
        #
        #   Stata                        .1992726  = mean(dy_i * x_i / p_i)
        #   mean(dy_i)*mean(x_i/p_i)     .1996857  <- 0.2% high: the E[ab] vs E[a]E[b] gap
        #
        # The per-observation derivative is recomputed here rather than reusing the averaged
        # ``estimate``; the gradient keeps the averaged factor, which is what the delta
        # method needs.
        beta_derivative, _ = _statistic_and_gradient(
            result,
            subset,
            predict_kind,
            derivative_with_respect_to=position,
            outcome=outcome,
        )
        derivative_per_observation = _per_observation_derivative(
            result, subset, predict_kind, position, outcome=outcome
        )
        factor = float(np.mean(per_observation))
        combined = float(np.mean(derivative_per_observation * per_observation))
        _ = beta_derivative
        return combined, gradient * factor

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
