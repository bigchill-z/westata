"""Non-linear least squares and stochastic frontier.

===============  ==========
westata           Stata
===============  ==========
``Nl``             ``nl``
``Frontier``       ``frontier``
===============  ==========

``nl`` takes Stata's parameter syntax, ``y = {b0} * exp(-{b1} * x)``. The braces are
turned into named parameters and the right-hand side is compiled once into a
callable. Compilation goes through an ``ast`` whitelist rather than ``eval``: the
expression comes from the user, and ``nl`` must not become an execution primitive.

``frontier`` fits the standard normal/half-normal stochastic production frontier

    y = x'b + v - u,     v ~ N(0, sigma_v^2),   u ~ |N(0, sigma_u^2)|

whose log-likelihood for one observation, with ``e = y - x'b`` and
``sigma^2 = sigma_v^2 + sigma_u^2``, is

    lnf = ln 2 - ln sigma + ln phi(e/sigma) + ln Phi(-e * lambda / sigma)

where ``lambda = sigma_u / sigma_v``. The tail term uses ``norm.logcdf``; computing
``ln(Phi(z))`` directly underflows for ``z < -7`` and would silently return -inf.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import log_ndtr

from westata.errors import ConsistencyError, ConvergenceError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Frontier", "Nl", "compile_expression"]

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Name,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    # Context nodes: every Name and Constant carries one, and they grant no
    # capability on their own, so excluding them makes the whitelist unusable.
    ast.Load,
    ast.Store,
    ast.Del,
)

_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "exp": np.exp,
    "log": np.log,
    "ln": np.log,
    "sqrt": np.sqrt,
    "sin": np.sin,
    "cos": np.cos,
    "abs": np.abs,
    "invlogit": lambda value: 1.0 / (1.0 + np.exp(-value)),
}


def compile_expression(
    expression: str, parameter_names: Sequence[str], variable_names: Sequence[str]
) -> Callable[[Mapping[str, float], pd.DataFrame], np.ndarray]:
    """Compile a whitelisted expression into a callable over parameters and data.

    Parameters
    ----------
    expression:
        The right-hand side with ``{name}`` placeholders already extracted.
    parameter_names:
        Symbols the expression may reference as parameters.
    variable_names:
        Data columns it may reference.

    Raises
    ------
    ValueError
        For any node outside the whitelist. There is no ``eval`` anywhere, so a
        hostile formula fails validation instead of executing.
    """

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"nl: cannot parse {expression!r}: {exc.msg}") from None

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"nl: {type(node).__name__} is not allowed in an expression; only "
                f"arithmetic, parentheses and the functions {sorted(_FUNCTIONS)} "
                "are permitted"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise ValueError(
                    f"nl: unknown function; allowed functions are "
                    f"{sorted(_FUNCTIONS)}"
                )
            if node.keywords:
                raise ValueError("nl: keyword arguments are not allowed")
        if isinstance(node, ast.Attribute):
            raise ValueError("nl: attribute access is not allowed")

    parameters = set(parameter_names)
    variables = set(variable_names)

    def evaluate(node: ast.AST, values: Mapping[str, float], frame: pd.DataFrame) -> Any:
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise ValueError("nl: only numeric constants are allowed")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in values:
                return values[node.id]
            if node.id in frame.columns:
                return frame[node.id].to_numpy(dtype=float)
            raise ValueError(
                f"nl: {node.id!r} is neither a parameter nor a column; parameters "
                f"are {sorted(parameters)} and columns are {sorted(variables)}"
            )
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
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError("nl: unsupported operator")
        if isinstance(node, ast.Call):
            function = _FUNCTIONS[node.func.id]  # validated above
            if len(node.args) != 1:
                raise ValueError(
                    f"nl: {node.func.id}() takes exactly one argument"
                )
            return function(evaluate(node.args[0], values, frame))
        raise ValueError(f"nl: unsupported element {type(node).__name__}")

    def callable_(
        values: Mapping[str, float], frame: pd.DataFrame
    ) -> np.ndarray:
        result = evaluate(tree.body, values, frame)
        return np.asarray(result, dtype=float) * np.ones(len(frame))

    return callable_


class Nl:
    """Non-linear least squares (Stata ``nl``)."""

    _stata_command = "nl"

    def __init__(
        self,
        endog: Any,
        exog_frame: pd.DataFrame,
        parameter_names: Sequence[str],
        predictor: Callable[[Mapping[str, float], pd.DataFrame], np.ndarray],
        *,
        start: Mapping[str, float] | None = None,
        model_data: pd.DataFrame | None = None,
    ) -> None:
        self._y = np.asarray(endog, dtype=float).reshape(-1)
        self._frame = exog_frame.reset_index(drop=True)
        self._parameter_names = list(parameter_names)
        self._predictor = predictor
        self._start = dict(start) if start else {}
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        start: Mapping[str, float] | None = None,
        subset: Any = None,
    ) -> "Nl":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)

        if "=" not in formula:
            raise ConsistencyError(
                "nl: the formula must be 'y = <expression>', as in "
                "'y = {b0} * exp(-{b1} * x)'"
            )
        lhs, rhs = formula.split("=", 1)
        lhs = lhs.strip()
        rhs = rhs.strip()
        if lhs not in data.columns:
            raise ConsistencyError(f"nl: column {lhs!r} not found in the data")

        # Stata marks parameters with braces; swap them for plain identifiers so
        # the expression parses, and remember the names.
        parameter_names: list[str] = []
        replaced = []
        index = 0
        while index < len(rhs):
            character = rhs[index]
            if character == "{":
                closing = rhs.find("}", index)
                if closing == -1:
                    raise ValueError(f"nl: unclosed '{{' in {rhs!r}")
                name = rhs[index + 1 : closing].strip()
                if not name:
                    raise ValueError("nl: an empty parameter name is not allowed")
                if name not in parameter_names:
                    parameter_names.append(name)
                replaced.append(name)
                index = closing + 1
            else:
                replaced.append(character)
                index += 1
        expression = "".join(replaced)
        if not parameter_names:
            raise ConsistencyError(
                "nl: the expression contains no parameters; use {name} to declare one"
            )

        variable_names = [
            column for column in data.columns if column != lhs
        ]
        predictor = compile_expression(expression, parameter_names, variable_names)
        return cls(
            data[lhs].to_numpy(dtype=float),
            data,
            parameter_names,
            predictor,
            start=start,
            model_data=data,
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 200,
        strict: bool = False,
    ) -> StataResults:
        names = self._parameter_names
        k = len(names)

        initial = np.zeros(k, dtype=float)
        for position, name in enumerate(names):
            if name in self._start:
                initial[position] = float(self._start[name])
            else:
                initial[position] = 1.0

        def predict(values: np.ndarray) -> np.ndarray:
            mapping = dict(zip(names, values))
            return self._predictor(mapping, self._frame)

        def objective(values: np.ndarray) -> float:
            try:
                residual = self._y - predict(values)
            except (ValueError, FloatingPointError):
                return 1e12
            total = float(residual @ residual)
            return total if np.isfinite(total) else 1e12

        result = optimize.minimize(
            objective, initial, method="BFGS", options={"maxiter": maxiter}
        )
        if not result.success:
            refined = optimize.minimize(
                objective,
                np.asarray(result.x, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-10, "fatol": 1e-10},
            )
            if refined.fun <= result.fun:
                result = refined
        if not result.success:
            message = f"nl: optimiser did not converge ({result.message})"
            if strict:
                raise ConvergenceError(message)

        theta = np.asarray(result.x, dtype=float)
        residual = self._y - predict(theta)
        n = self._y.size
        rss = float(residual @ residual)
        dof = max(n - k, 1)
        sigma2 = rss / dof

        # Gauss-Newton style covariance: s^2 (J'J)^-1 with J the numerical Jacobian
        # of the prediction with respect to the parameters.
        jacobian = np.empty((n, k), dtype=float)
        for position in range(k):
            step = 1e-6 * max(1.0, abs(theta[position]))
            forward = theta.copy()
            backward = theta.copy()
            forward[position] += step
            backward[position] -= step
            jacobian[:, position] = (
                predict(forward) - predict(backward)
            ) / (2.0 * step)
        covariance = sigma2 * np.linalg.pinv(jacobian.T @ jacobian)

        total = float(((self._y - self._y.mean()) ** 2).sum())
        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "df_r": float(dof),
            "rank": k,
            "rss": rss,
            "mss": total - rss,
            "r2": 1.0 - rss / total if total > 0 else float("nan"),
            "rmse": float(np.sqrt(sigma2)),
            "deviance": float(n * np.log(rss / n)) if rss > 0 else float("nan"),
        }

        # Stata prints nl parameters with a leading slash.
        display_names = [f"/{name}" for name in names]
        return StataResults(
            params=theta,
            param_names=display_names,
            cov=covariance,
            resid=residual,
            fittedvalues=predict(theta),
            nobs=n,
            df_resid=float(dof),
            df_model=float(k - 1),
            dist="t",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            model_data=self._model_data,
        )


class Frontier:
    """Stochastic production frontier (Stata ``frontier``), half-normal."""

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
        self._X = np.atleast_2d(np.asarray(exog, dtype=float))
        self._param_names = list(param_names)
        normalized = str(distribution).strip().lower()
        if normalized not in ("halfnormal", "half-normal"):
            raise UnsupportedError(
                f"frontier: distribution={distribution!r} is not implemented; only "
                "'halfnormal' is. The truncated and exponential variants need "
                "different likelihoods, not a flag."
            )
        self._distribution = "halfnormal"
        self._model_data = model_data

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        *,
        distribution: str = "halfnormal",
        subset: Any = None,
    ) -> "Frontier":
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if subset is not None:
            data = data.loc[subset]
        data = data.reset_index(drop=True)
        X, names, kept = build_design(formula, data)
        lhs = formula.split("~", 1)[0].strip()
        return cls(
            data.loc[kept, lhs].to_numpy(dtype=float),
            X,
            param_names=names,
            distribution=distribution,
            model_data=data,
        )

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 500,
        strict: bool = False,
    ) -> StataResults:
        n, k = self._X.shape
        # Parameterise on log scales so both variances stay positive.
        # theta = [beta, log(sigma2), logit(gamma)] with gamma = sigma_u^2/sigma2.

        def negative(theta: np.ndarray) -> float:
            beta = theta[:k]
            sigma2 = float(np.exp(theta[k]))
            gamma = float(1.0 / (1.0 + np.exp(-theta[k + 1])))
            sigma_u2 = max(gamma * sigma2, 1e-12)
            sigma_v2 = max((1.0 - gamma) * sigma2, 1e-12)
            sigma = math.sqrt(sigma2)
            lam = math.sqrt(sigma_u2 / sigma_v2)

            residual = self._y - self._X @ beta
            z = residual / sigma
            # lnf = ln 2 - ln sigma + ln phi(z) + ln Phi(-z * lambda)
            log_density = -0.5 * np.log(2 * np.pi) - 0.5 * z**2
            tail = log_ndtr(-z * lam)
            total = float(
                np.sum(np.log(2.0) - np.log(sigma) + log_density + tail)
            )
            return -total if np.isfinite(total) else 1e12

        start = np.zeros(k + 2)
        beta0, *_ = np.linalg.lstsq(self._X, self._y, rcond=None)
        start[:k] = beta0
        residual0 = self._y - self._X @ beta0
        start[k] = np.log(max(float(np.var(residual0)), 1e-6))
        start[k + 1] = 0.0

        result = optimize.minimize(
            negative, start, method="BFGS", options={"maxiter": maxiter}
        )
        if not result.success:
            refined = optimize.minimize(
                negative,
                np.asarray(result.x, dtype=float),
                method="Nelder-Mead",
                options={"maxiter": maxiter, "xatol": 1e-9, "fatol": 1e-9},
            )
            if refined.fun <= result.fun:
                result = refined
        if not result.success:
            message = f"frontier: optimiser did not converge ({result.message})"
            if strict:
                raise ConvergenceError(message)

        theta = np.asarray(result.x, dtype=float)
        beta = theta[:k]
        sigma2 = float(np.exp(theta[k]))
        gamma = float(1.0 / (1.0 + np.exp(-theta[k + 1])))
        sigma_u2 = max(gamma * sigma2, 1e-12)
        sigma_v2 = max((1.0 - gamma) * sigma2, 1e-12)
        sigma_u = math.sqrt(sigma_u2)
        sigma_v = math.sqrt(sigma_v2)
        lam = sigma_u / sigma_v

        residual = self._y - self._X @ beta
        dof = max(n - k - 2, 1)
        covariance = self._covariance(negative, theta)

        llf = -float(negative(theta))
        e: dict[str, Any] = {
            "N": n,
            "df_m": float(k - 1),
            "df_r": float(dof),
            "rank": k + 2,
            "ll": llf,
            "sigma2": sigma2,
            "sigma_u": sigma_u,
            "sigma_v": sigma_v,
            "lambda": lam,
            "gamma": gamma,
            "distribution": self._distribution,
        }

        return StataResults(
            params=beta,
            param_names=list(self._param_names),
            cov=covariance[:k, :k],
            resid=residual,
            fittedvalues=self._X @ beta,
            nobs=n,
            df_resid=float(dof),
            df_model=float(k - 1),
            dist="z",
            model_name=self._stata_command,
            cov_type="nonrobust",
            e=e,
            design_matrix=(self._X, list(self._param_names)),
            model_data=self._model_data,
        )

    @staticmethod
    def _covariance(objective: Any, point: np.ndarray) -> np.ndarray:
        size = point.size
        step = 1e-5
        hessian = np.zeros((size, size))
        increments = step * np.maximum(np.abs(point), 1.0)
        centre = objective(point)
        for i in range(size):
            for j in range(i, size):
                ei = np.zeros(size)
                ej = np.zeros(size)
                ei[i] = increments[i]
                ej[j] = increments[j]
                if i == j:
                    value = (
                        objective(point + ei) - 2 * centre + objective(point - ei)
                    ) / increments[i] ** 2
                else:
                    value = (
                        objective(point + ei + ej)
                        - objective(point + ei - ej)
                        - objective(point - ei + ej)
                        + objective(point - ei - ej)
                    ) / (4 * increments[i] * increments[j])
                hessian[i, j] = value
                hessian[j, i] = value
        try:
            covariance = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(hessian)
        if not np.all(np.isfinite(covariance)):
            raise ConvergenceError(
                "frontier: the observed information is not invertible; the model "
                "is likely not identified"
            )
        return covariance
