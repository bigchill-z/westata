"""Nonlinear least squares (``nl``) and the stochastic frontier (``frontier``).

======================  ==================================================
westata                 Stata
======================  ==================================================
``Nl``                  ``nl``
``Frontier``            ``frontier`` (half-normal only)
======================  ==================================================

``nl``
------
Stata writes the model with the parameters in braces::

    nl (y = {b0} * exp(-{b1} * x))

westata spells the same formula ``"y = {b0} * exp(-{b1} * x)"``.  The braces are
replaced by plain identifiers so the right-hand side parses as a Python
expression, and the expression is then validated against an explicit ``ast``
node whitelist and *interpreted* -- ``eval``, ``exec`` and ``compile`` are never
called, so the formula cannot reach the interpreter.  ``__import__('os')``,
attribute access, comprehensions, lambdas, conditionals, subscripts and any
function outside the whitelist are rejected with :class:`ValueError` before a
single observation is touched.  This mirrors
:mod:`westata.postestimation.nonlinear`, which draws the same boundary for
``nlcom``/``testnl``.

The parameters are reported with the leading slash Stata prints (``/b0``,
``/b1``).  Point estimates minimise the residual sum of squares; inference uses
the usual Gauss-Newton covariance ``s^2 (J'J)^-1`` with ``s^2 = RSS/(n-k)`` and
``J`` the central-difference Jacobian of the prediction.

**Centred versus uncentred sums of squares.**  Stata decides from the model
specification whether the fit "contains a constant term": a parameter that
appears as a bare additive term (``{b0}`` in ``{b0} + {b1}*x``) is a constant,
while ``{b0}*exp(...)`` is not.  When there is no constant, Stata reports the
*uncentred* total sum of squares, ``sum(y^2)``.  That is why the captured
``nl_0`` output has ``Total 1017.4143 = Model 1010.4869 + Residual 6.9274269``
and ``R-squared .9932``: the centred total for that sample is only ~810 and
would print ``.9914``.  ``e(df_m)`` is ``k - 1`` with a constant and ``k``
without, and the adjusted R-squared uses ``(n - c)/(n - k)`` with ``c`` the
constant indicator.  A ``{b0} + {b1}*x`` model therefore reproduces ``regress``
exactly, including ``e(r2)`` and the SS decomposition.

``e(deviance)`` is Stata's "Res. dev." -- the residual deviance ``-2 lnL``
evaluated at the NLS estimate with the maximum-likelihood scale
``sigma^2 = RSS/n`` (not ``RSS/(n-k)``, which belongs to ``e(rmse)``).  Against
``nl_0`` this gives ``-279.1251`` exactly.

``frontier``
------------
The normal/half-normal stochastic production frontier

    y = x'b + v - u,     v ~ N(0, sigma_v^2),   u ~ |N(0, sigma_u^2)|

with the composed error ``e = y - x'b`` and ``sigma^2 = sigma_v^2 + sigma_u^2``
has the observation-level log-likelihood

    lnf = ln 2 - ln sigma + ln phi(e/sigma) + ln Phi(-e * lambda / sigma),
    lambda = sigma_u / sigma_v.

The tail term is evaluated with :func:`scipy.stats.norm.logcdf`; forming
``ln(Phi(z))`` directly underflows to ``-inf`` for ``z < -37`` and would silently
truncate the sample.

Parameters are estimated in Stata's own ``/lnsig2v``, ``/lnsig2u``
parameterisation -- a bijective reparameterisation of ``(sigma2, lambda)``, since
``sigma2 = exp(lnsig2v) + exp(lnsig2u)`` and ``lambda = sqrt(exp(lnsig2u -
lnsig2v))``.  It is used because it is the parameterisation Stata's output
reports, so the coefficient table can be compared line by line with the fixture.
``e()`` carries the derived ``sigma_u``, ``sigma_v``, ``sigma2`` and ``lambda``.

Standard errors are the observed information (the numerical Hessian of the
log-likelihood, inverted), which is Stata's default.  Against ``frontier_0``
this reproduces the coefficients, ``sigma_v``, ``sigma_u``, ``sigma2``, the
``/lnsig2v`` and ``/lnsig2u`` standard errors, the ``Wald chi2(2)`` of the slopes
and the ``LR test of sigma_u = 0`` ``chibar2(01)`` to the digits Stata prints.

``lambda`` (``sigma_u/sigma_v``) is a ratio, so the delta-method standard error
of a *ratio* is not what Stata prints for it; westata reports the ratio and its
point estimate only, and does not invent a delta-method standard error for it
(the same reason Stata's ``sigma_u``/``sigma_v`` do get one but ``lambda`` is
reported against ``/lnsig2v``/``/lnsig2u``).

Estimator options
-----------------
``distribution="halfnormal"`` (or ``"half-normal"``) is the only implemented
frontier; anything else raises :class:`~westata.errors.UnsupportedError`, because
the truncated and exponential variants need different likelihoods rather than a
flag.  ``cov_type`` accepts only ``"nonrobust"`` (Stata's default and only option
for these commands).

Convergence
-----------
``strict=True`` raises :class:`~westata.errors.ConvergenceError` when the
optimiser does not converge; otherwise a ``RuntimeWarning`` is emitted.  scipy's
BFGS frequently reports "precision loss" at a point whose gradient is already
~1e-6, so a non-successful BFGS run is polished with Nelder-Mead and the fit is
recorded as converged when the gradient of the objective is below ``1e-4``.
That is deliberate: forwarding scipy's status verbatim would report failure on
the exact optimum of the fixtures above.
"""

from __future__ import annotations

import ast
import math
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats

from westata.errors import ConsistencyError, ConvergenceError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Frontier", "Nl", "compile_expression", "frontier_loglike"]

#: Gradient infinity-norm below which a polished fit counts as converged.
_GRADIENT_TOLERANCE = 1e-4

#: Relative step of the central differences used for the Jacobian/Hessian.
_DIFFERENCE_STEP = 1e-6
#: Larger relative step for the second derivatives (cancellation is worse there).
_HESSIAN_STEP = 1e-5

# ---------------------------------------------------------------------------
# The expression language
# ---------------------------------------------------------------------------

#: Stata's ``nl`` expression functions.  ``ln`` is Stata's natural logarithm and
#: ``log`` is accepted as its documented alias; ``normal``/``normalden`` are
#: ``Phi``/``phi``.  Every entry is vectorised over a NumPy array.
_FUNCTIONS: dict[str, Callable[[Any], Any]] = {
    "exp": np.exp,
    "ln": np.log,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "atan": np.arctan,
    "logit": lambda value: np.log(value / (1.0 - value)),
    "invlogit": lambda value: 1.0 / (1.0 + np.exp(-value)),
    "normal": lambda value: stats.norm.cdf(value),
    "normalden": lambda value: stats.norm.pdf(value),
}

_ALLOWED_FUNCTION_TEXT = ", ".join(sorted(_FUNCTIONS))


def _reject(node: ast.AST) -> ValueError:
    """Build the :class:`ValueError` for one node outside the whitelist."""

    kind = type(node).__name__
    if isinstance(node, ast.Attribute):
        detail = "attribute access is not allowed"
    elif isinstance(node, ast.Subscript):
        detail = "subscripts are not allowed"
    elif isinstance(node, (ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                           ast.GeneratorExp)):
        detail = "lambdas and comprehensions are not allowed"
    elif isinstance(node, ast.Call):
        function = getattr(node.func, "id", None)
        if function is not None:
            return ValueError(
                f"nl: unknown function {function!r}; the only functions allowed are "
                f"{_ALLOWED_FUNCTION_TEXT}"
            )
        detail = f"only these functions are allowed: {_ALLOWED_FUNCTION_TEXT}"
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        detail = "imports are not allowed"
    else:
        detail = (
            "only parameters ({name}), data columns, numeric constants, "
            "parentheses, + - * / ** and the functions "
            f"{_ALLOWED_FUNCTION_TEXT} are permitted"
        )
    return ValueError(f"nl: {kind} is not allowed in the expression; {detail}")


def _parse(
    expression: str,
    parameter_names: Sequence[str],
    variable_names: Sequence[str],
) -> tuple[Callable[[Mapping[str, float], pd.DataFrame], np.ndarray], bool]:
    """Validate ``expression`` and return a callable plus its constant flag.

    The returned callable takes the parameter mapping and the data frame and
    returns a length-``n`` prediction.  The flag is ``True`` when the expression
    contains a bare additive parameter, which is how Stata decides that the model
    has a constant term (and therefore whether the total sum of squares is
    centred).

    Raises
    ------
    ValueError
        For any construct outside the whitelist.  Nothing is compiled or
        evaluated, so a hostile formula fails validation instead of running.
    """

    if not isinstance(expression, str):
        raise TypeError(
            f"nl: the expression must be a string, got {type(expression).__name__}"
        )
    text = expression.strip()
    if not text:
        raise ValueError("nl: the right-hand side of the formula is empty")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"nl: cannot parse {text!r}: {exc.msg}") from None

    parameters = list(parameter_names)
    variables = list(variable_names)
    allowed_names = set(parameters) | set(variables)

    # ---- structural validation (recursive, so the message names the culprit)
    def validate(node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(
                    "nl: only numeric constants are allowed, got "
                    f"{node.value!r}"
                )
            return
        if isinstance(node, ast.Name):
            if node.id not in allowed_names:
                raise ValueError(
                    f"nl: {node.id!r} is neither a parameter nor a column; "
                    f"parameters are {parameters} and columns are {variables}"
                )
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, (ast.USub, ast.UAdd)):
                raise _reject(node)
            validate(node.operand)
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
                raise _reject(node)
            validate(node.left)
            validate(node.right)
            return
        if isinstance(node, ast.Call):
            # A call is legal only on a whitelisted bare function name; that also
            # rules out __import__(...).system(...) and obj.method(...).
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise _reject(node)
            if node.keywords:
                raise ValueError("nl: keyword arguments are not allowed")
            if len(node.args) != 1:
                raise ValueError(
                    f"nl: {node.func.id}() takes exactly one argument"
                )
            validate(node.args[0])
            return
        raise _reject(node)

    validate(tree.body)

    # ---- Stata's "does the model contain a constant term?" test
    def additive_terms(node: ast.AST) -> list[ast.AST]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return additive_terms(node.left) + additive_terms(node.right)
        return [node]

    has_constant = False
    for term in additive_terms(tree.body):
        inner = term
        while isinstance(inner, ast.UnaryOp) and isinstance(
            inner.op, (ast.USub, ast.UAdd)
        ):
            inner = inner.operand
        if isinstance(inner, ast.Name) and inner.id in set(parameters):
            has_constant = True
            break

    # ---- interpretation (never compilation)
    def evaluate(node: ast.AST, values: Mapping[str, float], frame: pd.DataFrame) -> Any:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in values:
                return values[node.id]
            return frame[node.id].to_numpy(dtype=float)
        if isinstance(node, ast.UnaryOp):
            operand = evaluate(node.operand, values, frame)
            return -operand if isinstance(node.op, ast.USub) else operand
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left, values, frame)
            right = evaluate(node.right, values, frame)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left**right  # Pow, validated above
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            return _FUNCTIONS[node.func.id](evaluate(node.args[0], values, frame))
        raise _reject(node)  # unreachable after validation

    def predict(
        values: Mapping[str, float], frame: pd.DataFrame
    ) -> np.ndarray:
        value = np.asarray(evaluate(tree.body, values, frame), dtype=float)
        if value.ndim == 0:
            return np.full(len(frame), float(value))
        if value.size == 1:
            return np.full(len(frame), float(value.reshape(-1)[0]))
        return np.broadcast_to(value, (len(frame),)).astype(float)

    return predict, has_constant


def compile_expression(
    expression: str,
    parameter_names: Sequence[str],
    variable_names: Sequence[str],
) -> Callable[[Mapping[str, float], pd.DataFrame], np.ndarray]:
    """Compile a whitelisted ``nl`` expression into a callable.

    Parameters
    ----------
    expression:
        The right-hand side with the ``{name}`` placeholders already replaced by
        their parameter names.
    parameter_names:
        Symbols the expression may use as parameters.
    variable_names:
        Data columns it may read.

    Returns
    -------
    callable
        ``f(values, frame) -> ndarray`` of length ``len(frame)``.

    Raises
    ------
    ValueError
        For any construct outside the whitelist.  See the module docstring: the
        expression is interpreted, never handed to ``eval``.
    """

    predict, _ = _parse(expression, parameter_names, variable_names)
    return predict


def _warn_or_raise(converged: bool, message: str, strict: bool) -> None:
    """Shared ``strict=`` handling for both estimators."""

    if converged:
        return
    if strict:
        raise ConvergenceError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _gradient_infinity_norm(objective: Callable[[np.ndarray], float],
                            point: np.ndarray) -> float:
    """Central-difference gradient's infinity norm, for convergence reporting."""

    gradient = np.zeros(point.size, dtype=float)
    for position in range(point.size):
        step = _DIFFERENCE_STEP * max(1.0, abs(point[position]))
        forward = point.copy()
        backward = point.copy()
        forward[position] += step
        backward[position] -= step
        gradient[position] = (
            objective(forward) - objective(backward)
        ) / (2.0 * step)
    return float(np.max(np.abs(gradient))) if gradient.size else 0.0


def _jacobian(
    predict: Callable[[np.ndarray], np.ndarray], point: np.ndarray
) -> np.ndarray:
    """Central-difference Jacobian of ``predict`` at ``point``."""

    base = np.asarray(predict(point), dtype=float)
    jacobian = np.empty((base.size, point.size), dtype=float)
    for position in range(point.size):
        step = _DIFFERENCE_STEP * max(1.0, abs(point[position]))
        forward = point.copy()
        backward = point.copy()
        forward[position] += step
        backward[position] -= step
        jacobian[:, position] = (
            np.asarray(predict(forward), dtype=float)
            - np.asarray(predict(backward), dtype=float)
        ) / (2.0 * step)
    return jacobian


def _numerical_hessian(
    objective: Callable[[np.ndarray], float], point: np.ndarray
) -> np.ndarray:
    """Central-difference Hessian with a scale-aware step."""

    size = point.size
    hessian = np.zeros((size, size), dtype=float)
    steps = _HESSIAN_STEP * np.maximum(np.abs(point), 1.0)
    centre = objective(point)
    for i in range(size):
        for j in range(i, size):
            ei = np.zeros(size)
            ej = np.zeros(size)
            ei[i] = steps[i]
            ej[j] = steps[j]
            if i == j:
                value = (objective(point + ei) - 2.0 * centre + objective(point - ei)) / (
                    steps[i] ** 2
                )
            else:
                value = (
                    objective(point + ei + ej)
                    - objective(point + ei - ej)
                    - objective(point - ei + ej)
                    + objective(point - ei - ej)
                ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def frontier_loglike(
    theta: np.ndarray, y: np.ndarray, exog: np.ndarray
) -> float:
    """Normal/half-normal frontier log-likelihood (the function ``fit`` maximises).

    Parameters
    ----------
    theta:
        ``[beta, lnsig2v, lnsig2u]`` -- the parameter vector in the order the
        fitted result reports it.
    y:
        Dependent variable, length ``n``.
    exog:
        Design matrix ``n x k``.

    Returns
    -------
    float
        ``sum_i [ln 2 - ln sigma + ln phi(e_i/sigma) + ln Phi(-e_i lambda/sigma)]``
        with ``e = y - x'b``, ``sigma^2 = exp(lnsig2v) + exp(lnsig2u)`` and
        ``lambda = sqrt(exp(lnsig2u - lnsig2v))``.

    Notes
    -----
    This is public (not an underscore helper) so the likelihood can be checked
    against an independent implementation, and so that the ``logcdf`` guarantee
    is testable: at ``z = -40`` the tail term is about ``-804``, while
    ``ln(norm.cdf(z))`` would return ``-inf`` and silently drop the observation.
    """

    beta = np.asarray(theta, dtype=float)[: exog.shape[1]]
    lnsig2v = float(theta[exog.shape[1]])
    lnsig2u = float(theta[exog.shape[1] + 1])
    sigma2 = math.exp(lnsig2v) + math.exp(lnsig2u)
    sigma = math.sqrt(sigma2)
    lam = math.exp(0.5 * (lnsig2u - lnsig2v))
    z = (np.asarray(y, dtype=float) - exog @ beta) / sigma
    log_density = -0.5 * z**2 - 0.5 * math.log(2.0 * math.pi)
    tail = stats.norm.logcdf(-z * lam)
    total = float(
        np.sum(math.log(2.0) - math.log(sigma) + log_density + tail)
    )
    return total


# ---------------------------------------------------------------------------
# nl
# ---------------------------------------------------------------------------


class Nl:
    """Nonlinear least squares (Stata ``nl``).

    Parameters
    ----------
    endog:
        Dependent variable, length ``n``.
    frame:
        Data frame the expression reads its columns from.
    parameter_names:
        Parameter names in reporting order (without the ``/`` prefix).
    predictor:
        ``f(values, frame) -> ndarray``, as returned by :func:`compile_expression`.
    has_constant:
        Whether a parameter enters additively, i.e. whether Stata centres the
        total sum of squares and drops one model degree of freedom.
    """

    _stata_command = "nl"

    def __init__(
        self,
        endog: Any,
        frame: pd.DataFrame,
        parameter_names: Sequence[str],
        predictor: Callable[[Mapping[str, float], pd.DataFrame], np.ndarray],
        *,
        has_constant: bool = False,
        start: Mapping[str, float] | None = None,
        formula: str = "",
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        self._frame = frame.reset_index(drop=True)
        self._parameter_names = list(parameter_names)
        self._predictor = predictor
        self._has_constant = bool(has_constant)
        self._start = dict(start) if start else {}
        self._formula = formula
        self._model_data = model_data
        unknown = set(self._start) - set(self._parameter_names)
        if unknown:
            raise ValueError(
                f"nl: start values given for unknown parameters {sorted(unknown)}; "
                f"the parameters are {self._parameter_names}"
            )

    # ------------------------------------------------------------------ build
    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        start: Mapping[str, float] | None = None,
        subset: Any = None,
    ) -> "Nl":
        """Build the model from ``"y = {b0} * exp(-{b1} * x)"``.

        Starting values may be given with ``start={"b0": 4, "b1": 0.8}`` or
        inline as ``{b0=4}``; inline values are overridden by ``start``.

        Raises
        ------
        ValueError
            For a formula that is not of the form ``y = <expression>``, or whose
            expression uses anything outside the whitelist.
        ConsistencyError
            When the data cannot support the model (unknown column, no
            parameters, more parameters than observations).
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        if "=" not in formula:
            raise ValueError(
                "nl: the formula must be 'y = <expression>' with '=' between the "
                "dependent variable and the model, as in "
                "'y = {b0} * exp(-{b1} * x)'"
            )
        lhs, rhs = formula.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if not lhs:
            raise ValueError("nl: the formula has no dependent variable before '='")
        if lhs not in data.columns:
            raise ConsistencyError(
                f"nl: dependent variable {lhs!r} not found; columns are "
                f"{list(data.columns)}"
            )

        # Stata marks parameters with braces.  Swap each {name} -- or {name=#}
        # when an initial value is written inline -- for a plain identifier.
        parameter_names: list[str] = []
        inline_start: dict[str, float] = {}
        pieces: list[str] = []
        index = 0
        while index < len(rhs):
            character = rhs[index]
            if character != "{":
                pieces.append(character)
                index += 1
                continue
            closing = rhs.find("}", index)
            if closing == -1:
                raise ValueError(f"nl: unclosed '{{' in {rhs!r}")
            declaration = rhs[index + 1 : closing].strip()
            if not declaration:
                raise ValueError("nl: an empty parameter declaration '{}' is not allowed")
            name, _, initial = declaration.partition("=")
            name = name.strip()
            if not name.isidentifier():
                raise ValueError(
                    f"nl: {name!r} is not a valid parameter name; parameter names "
                    "must be identifiers, as in {b0}"
                )
            if initial.strip():
                try:
                    inline_start[name] = float(initial)
                except ValueError:
                    raise ValueError(
                        f"nl: the initial value {initial.strip()!r} for parameter "
                        f"{name!r} is not a number"
                    ) from None
            if name not in parameter_names:
                parameter_names.append(name)
            pieces.append(name)
            index = closing + 1

        expression = "".join(pieces)
        if not parameter_names:
            raise ConsistencyError(
                "nl: the expression contains no parameters; declare them in "
                "braces, as in {b0} * exp(-{b1} * x)"
            )

        variable_names = [column for column in data.columns if column != lhs]
        predictor, has_constant = _parse(expression, parameter_names, variable_names)

        merged_start: dict[str, float] = dict(inline_start)
        if start:
            merged_start.update({str(k): float(v) for k, v in dict(start).items()})

        endog = data[lhs].to_numpy(dtype=float)
        if endog.size <= len(parameter_names):
            raise ConsistencyError(
                f"nl: {endog.size} observations but {len(parameter_names)} "
                "parameters; the model is not identified"
            )
        return cls(
            endog,
            data,
            parameter_names,
            predictor,
            has_constant=has_constant,
            start=merged_start,
            formula=formula.strip(),
            model_data=data,
        )

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 200,
        strict: bool = False,
    ) -> StataResults:
        """Minimise the residual sum of squares.

        Parameters
        ----------
        maxiter:
            Stata's iteration limit, applied here as the residual-evaluation
            budget of Levenberg-Marquardt.
        strict:
            Raise :class:`~westata.errors.ConvergenceError` instead of warning
            when the optimiser does not converge.
        """

        requested = "nonrobust" if cov_type is None else str(cov_type).strip()
        if requested.lower() != "nonrobust":
            raise ValueError(
                f"nl: only cov_type='nonrobust' is implemented (Stata's nl has no "
                f"vce() option); got {cov_type!r}"
            )

        names = self._parameter_names
        k = len(names)
        y = self._y
        n = y.size
        if n <= k:
            raise ConsistencyError(
                f"nl: {n} observations but {k} parameters; the model is not "
                "identified"
            )

        # Stata grid-searches its starting values; the default here is 1 for every
        # parameter, which is what a user gets without start=/inline values.
        initial = np.array(
            [float(self._start.get(name, 1.0)) for name in names], dtype=float
        )

        def predict(values: np.ndarray) -> np.ndarray:
            return self._predictor(dict(zip(names, values)), self._frame)

        def residuals(values: np.ndarray) -> np.ndarray:
            return y - np.asarray(predict(values), dtype=float)

        def objective(values: np.ndarray) -> float:
            with np.errstate(all="ignore"):
                try:
                    residual = residuals(values)
                    total = float(residual @ residual)
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    return 1e12
            return total if np.isfinite(total) else 1e12

        with np.errstate(all="ignore"):
            result = optimize.least_squares(
                residuals,
                initial,
                method="lm",
                xtol=1e-15,
                ftol=1e-15,
                gtol=1e-15,
                max_nfev=max(int(maxiter), 1),
            )
        theta = np.asarray(result.x, dtype=float)
        best = objective(theta)
        converged = bool(result.success)

        # Polish only when the optimiser stopped for a reason other than the
        # evaluation budget.  ``least_squares`` reports status 0 when
        # ``max_nfev`` is exhausted, so a caller who asked for ``maxiter=1`` gets
        # exactly one iteration rather than a derivative-free polish silently
        # overriding the request.
        if not converged and int(result.status) != 0:
            refined = optimize.minimize(
                objective,
                theta,
                method="Nelder-Mead",
                options={
                    "maxiter": max(2000, int(maxiter) * 20),
                    "xatol": 1e-12,
                    "fatol": 1e-12,
                },
            )
            if float(refined.fun) <= best:
                theta = np.asarray(refined.x, dtype=float)
                best = float(refined.fun)
                converged = bool(refined.success)
        if not converged and best < 1e12:
            converged = _gradient_infinity_norm(objective, theta) < _GRADIENT_TOLERANCE

        _warn_or_raise(
            converged,
            f"nl: the optimiser did not converge ({getattr(result, 'message', '')})",
            strict,
        )

        fitted = np.asarray(predict(theta), dtype=float)
        residual = y - fitted
        rss = float(residual @ residual)
        df_r = n - k
        sigma2 = rss / df_r

        # Stata's constant-term test decides the SS basis (see module docstring).
        if self._has_constant:
            centre = float(np.mean(y))
            total = float(np.sum((y - centre) ** 2))
            constant_terms = 1
        else:
            total = float(np.sum(y**2))
            constant_terms = 0
        mss = total - rss
        r2 = 1.0 - rss / total if total > 0 else float("nan")
        r2_adj = (
            1.0 - (1.0 - r2) * (n - constant_terms) / df_r
            if np.isfinite(r2)
            else float("nan")
        )
        df_m = k - constant_terms
        # "Res. dev." is -2 lnL at the ML scale RSS/n, not at RSS/(n-k).
        if rss > 0:
            log_likelihood = -0.5 * n * (math.log(2.0 * math.pi) + math.log(rss / n) + 1.0)
        else:
            log_likelihood = float("inf")

        jacobian = _jacobian(predict, theta)
        covariance = sigma2 * np.linalg.pinv(jacobian.T @ jacobian)
        if not np.all(np.isfinite(covariance)):
            raise ConvergenceError(
                "nl: the Gauss-Newton covariance is not finite; the parameters are "
                "likely not identified at this point"
            )

        e: dict[str, Any] = {
            "N": n,
            "df_m": df_m,
            "df_r": float(df_r),
            "rank": k,
            "rss": rss,
            "mss": mss,
            "tss": total,
            "r2": r2,
            "r2_a": r2_adj,
            "rmse": float(np.sqrt(sigma2)),
            "sigma": float(np.sqrt(sigma2)),
            "ll": log_likelihood,
            "deviance": -2.0 * log_likelihood,
            "converged": int(converged),
            "n_parameters": k,
            "has_constant": int(constant_terms),
            "cmd": self._stata_command,
            "cmdline": f"nl ({self._formula})" if self._formula else "nl",
        }

        return StataResults(
            params=theta,
            param_names=[f"/{name}" for name in names],
            cov=covariance,
            resid=residual,
            fittedvalues=fitted,
            nobs=n,
            df_resid=float(df_r),
            df_model=float(df_m),
            dist="t",
            model_name=self._stata_command,
            cov_type=requested,
            e=e,
            model_data=self._model_data,
        )


# ---------------------------------------------------------------------------
# frontier
# ---------------------------------------------------------------------------


class Frontier:
    """Stochastic production frontier (Stata ``frontier``), half-normal.

    The model is ``y = x'b + v - u`` with ``v ~ N(0, sigma_v^2)`` and
    ``u ~ |N(0, sigma_u^2)|``.  See the module docstring for the likelihood and
    the parameterisation.
    """

    _stata_command = "frontier"

    def __init__(
        self,
        endog: Any,
        exog: Any,
        *,
        param_names: Sequence[str],
        distribution: str = "halfnormal",
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        X = np.asarray(exog, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self._X = X
        self._param_names = list(param_names)
        if len(self._param_names) != X.shape[1]:
            raise ValueError(
                f"param_names has {len(self._param_names)} entries but exog has "
                f"{X.shape[1]} columns"
            )
        self._distribution = _normalize_distribution(distribution)
        self._model_data = model_data

    # ------------------------------------------------------------------ build
    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        distribution: str = "halfnormal",
        subset: Any = None,
    ) -> "Frontier":
        """Build the model from ``"y ~ x1 + x2"``.

        Raises
        ------
        UnsupportedError
            For any ``distribution`` other than half-normal.  It is raised here,
            at model construction, rather than deferred to ``fit``.
        """

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        # Validate the option before touching the data so the failure is about
        # the option and not about the sample.
        _normalize_distribution(distribution)
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        if "~" not in formula:
            raise ValueError(
                "frontier: the formula must be 'y ~ x1 + x2', as in "
                "'y ~ x1 + x2'"
            )
        lhs = formula.split("~", 1)[0].strip()
        if lhs not in data.columns:
            raise ConsistencyError(
                f"frontier: dependent variable {lhs!r} not found; columns are "
                f"{list(data.columns)}"
            )

        X, names, kept = build_design(formula, data)
        return cls(
            data.loc[kept, lhs].to_numpy(dtype=float),
            X,
            param_names=names,
            distribution=distribution,
            model_data=data.loc[kept].reset_index(drop=True),
        )

    # -------------------------------------------------------------------- fit
    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        """Maximise the half-normal frontier likelihood."""

        requested = "nonrobust" if cov_type is None else str(cov_type).strip()
        if requested.lower() != "nonrobust":
            raise ValueError(
                f"frontier: only cov_type='nonrobust' is implemented (the observed "
                f"information, Stata's default); got {cov_type!r}"
            )
        if self._distribution != "halfnormal":
            raise UnsupportedError(
                f"frontier: distribution={self._distribution!r} is not implemented; "
                "only 'halfnormal' is"
            )

        y, X = self._y, self._X
        n, k = X.shape
        if n <= k + 2:
            raise ConsistencyError(
                f"frontier: {n} observations but {k + 2} parameters; the model is "
                "not identified"
            )
        has_constant = any(name == "Intercept" for name in self._param_names)

        def negative(theta: np.ndarray) -> float:
            total = frontier_loglike(theta, y, X)
            return -total if np.isfinite(total) else 1e12

        beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual0 = y - X @ beta0
        variance0 = max(float(np.var(residual0)), 1e-6)
        # Split the OLS variance evenly between the two components.
        start = np.concatenate([beta0, [math.log(variance0) - math.log(2.0)] * 2])

        result = optimize.minimize(
            negative, start, method="BFGS", options={"maxiter": int(maxiter)}
        )
        theta = np.asarray(result.x, dtype=float)
        best = negative(theta)
        converged = bool(result.success)

        # Skip the polish when the user's own iteration budget is what stopped the
        # optimiser: ``maxiter=1`` must mean one iteration, not "one BFGS
        # iteration plus 4000 Nelder-Mead ones".
        if not converged and int(result.status) != 1:
            refined = optimize.minimize(
                negative,
                theta,
                method="Nelder-Mead",
                options={
                    "maxiter": max(4000, int(maxiter) * 8),
                    "xatol": 1e-11,
                    "fatol": 1e-13,
                },
            )
            if float(refined.fun) <= best:
                theta = np.asarray(refined.x, dtype=float)
                best = float(refined.fun)
                converged = bool(refined.success)
        if not converged and best < 1e12:
            converged = _gradient_infinity_norm(negative, theta) < _GRADIENT_TOLERANCE

        _warn_or_raise(
            converged,
            f"frontier: the optimiser did not converge "
            f"({getattr(result, 'message', '')})",
            strict,
        )

        sigma2_v = math.exp(theta[k])
        sigma2_u = math.exp(theta[k + 1])
        sigma2 = sigma2_v + sigma2_u
        sigma_u = math.sqrt(sigma2_u)
        sigma_v = math.sqrt(sigma2_v)
        lam = sigma_u / sigma_v

        beta = theta[:k]
        residual = y - X @ beta
        hessian = _numerical_hessian(negative, theta)
        try:
            covariance = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            # A singular Hessian whose diagonal is still non-negative is a
            # flat direction, not a saddle: fall back to the pseudo-inverse,
            # which reports zero variance along it instead of inventing a value.
            if np.any(np.diag(hessian) < 0):
                raise ConvergenceError(
                    "frontier: the observed information is not positive definite; "
                    "the estimates are not a maximum (try different starting values)"
                ) from None
            covariance = np.linalg.pinv(hessian)
        if not np.all(np.isfinite(covariance)):
            raise ConvergenceError(
                "frontier: the observed information is not finite; the model is "
                "likely not identified (check for collinear regressors)"
            )
        if np.any(np.diag(covariance) < 0):
            raise ConvergenceError(
                "frontier: the observed information yields a negative variance; the "
                "estimates are not a maximum (try different starting values)"
            )

        llf = -negative(theta)
        # OLS (sigma_u = 0) log-likelihood, needed for Stata's boundary LR test.
        rss_ols = float(residual0 @ residual0)
        ll_ols = (
            -0.5 * n * (math.log(2.0 * math.pi) + math.log(rss_ols / n) + 1.0)
        )
        lr_chi2 = 2.0 * (llf - ll_ols)

        slopes = slice(1 if has_constant else 0, k)
        slope_params = beta[slopes]
        slope_cov = covariance[slopes, slopes]
        try:
            wald = float(slope_params @ np.linalg.solve(slope_cov, slope_params))
        except np.linalg.LinAlgError:
            wald = float("nan")
        wald_df = k - (1 if has_constant else 0)

        e: dict[str, Any] = {
            "N": n,
            "df_m": wald_df,
            "df_r": float(n - k),
            "rank": k + 2,
            "ll": llf,
            "ll_0": ll_ols,
            "chi2": wald,
            "chi2type": "Wald",
            "p": float(stats.chi2.sf(wald, wald_df)) if wald_df > 0 else float("nan"),
            "lr_chi2": lr_chi2,
            # Stata's test is one-sided at the sigma_u = 0 boundary: chibar2(01).
            "p_lr": float(0.5 * stats.chi2.sf(lr_chi2, 1.0)),
            "sigma_u": sigma_u,
            "sigma_v": sigma_v,
            "sigma2_u": sigma2_u,
            "sigma2_v": sigma2_v,
            "sigma2": sigma2,
            "lambda": lam,
            "gamma": sigma2_u / sigma2,
            "distribution": self._distribution,
            "converged": int(converged),
            "cmd": self._stata_command,
            "cmdline": "frontier " + " ".join(self._param_names),
        }

        return StataResults(
            params=theta,
            param_names=list(self._param_names) + ["/lnsig2v", "/lnsig2u"],
            cov=covariance,
            resid=residual,
            fittedvalues=X @ beta,
            nobs=n,
            df_resid=float(n - k),
            df_model=float(wald_df),
            dist="z",
            model_name=self._stata_command,
            cov_type=requested,
            e=e,
            design_matrix=(X, list(self._param_names)),
            model_data=self._model_data,
        )


def _normalize_distribution(distribution: Any) -> str:
    """Validate the frontier's inefficiency distribution."""

    normalized = str(distribution).strip().lower().replace("-", "").replace("_", "")
    if normalized != "halfnormal":
        raise UnsupportedError(
            f"frontier: distribution={distribution!r} is not implemented; only "
            "'halfnormal' is. The truncated and exponential variants need different "
            "likelihoods, not a flag."
        )
    return "halfnormal"
