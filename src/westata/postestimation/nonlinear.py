"""Nonlinear combinations (``nlcom``) and nonlinear Wald tests (``testnl``).

Reference: ``[R] nlcom`` and ``[R] testnl``.  For a differentiable function
``g(b)`` the delta method gives

    Var(g(b_hat)) = G V G',        G = dg/db evaluated at b_hat,

so the standard error is ``sqrt(G V G')``.  ``testnl`` applies the same
linearisation to the constraint ``g(b) = c``:

    W = (g(b) - c)' (G V G')^-1 (g(b) - c)  ~  chi2(q).

``G`` has no closed form for an arbitrary expression, so it is computed by
central differences with the scale-aware step ``h_j = 1e-6 * max(1, |b_j|)``.
For a linear ``g`` a central difference returns the exact slope whatever ``h``
is, so ``nlcom`` reproduces ``lincom`` exactly on linear combinations; the
approximation only enters for genuinely nonlinear ``g``.

``testnl`` is reported as ``chi2(q)`` with ``q`` the number of constraints,
which is what ``[R] testnl`` prints.

Inference distribution for ``nlcom``
------------------------------------
``nlcom`` returns a :class:`~westata.postestimation.lincom.LincomResult` and
follows the *model's* reference distribution, exactly as ``lincom`` does: ``t``
with ``e(df_r)`` after ``regress``, the normal after a ``z`` model such as
``logit``.  That keeps the documented equivalence
``nlcom("_b[x1]") == lincom("x1")`` exact in every reported field, including
the p-value and the interval.

Known deviation from Stata: Stata prints ``nlcom`` with ``z``/``P>|z|`` and a
normal interval even after ``regress``.  The captured StataNow 19.5 fixture in
``tests/conformance/stata_fixtures/p5_postestimation_outputs.txt`` shows
``7.298097  .3291511  22.17  [6.652973, 7.943221]``, and the upper bound is
``estimate + 1.959964 * se`` (the ``t(196)`` bound would be ``7.947229``).
westata deliberately keeps the model's distribution here, because the same
capability feeding ``lincom`` must not silently disagree with it; a user who
needs Stata's exact interval can compute it from ``std_err``.  This is recorded
as an open question rather than hidden.

Expression language
-------------------
Expressions are parsed with :func:`ast.parse` and validated against an explicit
node whitelist.  ``eval``, ``exec`` and ``compile`` are never called, so user
text cannot reach the interpreter: ``__import__('os').system(...)`` fails
validation instead of running.

Accepted syntax::

    _b[x1]                       coefficient x1
    _b["C(g)[T.2]"]              coefficient names that are not Python names
    _b[x1] + _b[x2]              + - * / **, unary +/-, parentheses
    exp(_b[x1]) / sqrt(2)        numeric constants

Functions: ``exp``, ``log`` (natural logarithm, Stata's name), ``sqrt``,
``sin``, ``cos``, ``logit``, ``invlogit``.  Stata's ``ln()`` alias is
deliberately not accepted: the whitelist is the safety boundary and the contract
names exactly these seven functions.

``testnl`` additionally accepts ``=`` to form one constraint, and several
constraints separated by commas (``"a = 0, b = 0"``) or by whitespace
(``"a = 0 b = 0"``, including Stata's grouped form ``"(a = 0) (b = 0)"``), which
are then tested jointly.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Sequence

import numpy as np
from scipy import stats

from westata.postestimation.lincom import LincomResult
from westata.postestimation.tests import WaldTest, run_wald_test

__all__ = ["nlcom", "testnl"]

#: Relative step of the central-difference gradient: ``h = step * max(1, |b_j|)``.
_GRADIENT_STEP = 1e-6

#: Stata expression functions.  ``log`` is the natural logarithm (Stata's name).
_FUNCTIONS: dict[str, Callable[[float], float]] = {
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "logit": lambda p: math.log(p / (1.0 - p)),
    "invlogit": lambda x: (
        1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
    ),
}

_ALLOWED_FUNCTIONS = ", ".join(sorted(_FUNCTIONS))

_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_UNARY_OPERATORS = (ast.USub, ast.UAdd)

_BINARY_CHARS = "+-*/="


class DivisionByZero(ZeroDivisionError, ValueError):
    """Division by zero inside an expression.

    It is both a :class:`ZeroDivisionError` (Python's natural error for the
    operation) and a :class:`ValueError` (this module's documented error for an
    expression it cannot evaluate), so a caller may catch either.
    """


def _unsupported(node: ast.AST) -> ValueError:
    """Build the rejection message for one disallowed node."""

    if isinstance(node, ast.Attribute):
        detail = "attribute access is not allowed"
    elif isinstance(node, ast.Call):
        detail = f"only these functions are allowed: {_ALLOWED_FUNCTIONS}"
    elif isinstance(node, ast.Subscript):
        detail = "subscripts are only allowed on _b, as in _b[x1]"
    else:
        detail = (
            "only coefficient references (_b[name]), numeric constants, "
            f"parentheses and the functions {_ALLOWED_FUNCTIONS} are permitted"
        )
    return ValueError(
        f"nlcom/testnl: {type(node).__name__} is not allowed in an expression; "
        f"{detail}"
    )


# ---------------------------------------------------------------------------
# The expression language
# ---------------------------------------------------------------------------


class _Expression:
    """A validated expression ``g(b)``, evaluable at a coefficient vector.

    Validation is a recursive structural check against the whitelist; evaluation
    is a recursive interpretation of the same tree.  No source text is ever
    compiled or evaluated by Python.
    """

    def __init__(self, text: str, names: Sequence[str]) -> None:
        if not isinstance(text, str):
            raise TypeError(f"expression must be a string, got {type(text).__name__}")
        stripped = text.strip()
        if not stripped:
            raise ValueError("nlcom/testnl: the expression is empty")
        try:
            tree = ast.parse(stripped, mode="eval")
        except SyntaxError as exc:
            raise ValueError(
                f"nlcom/testnl: cannot parse {text!r}: {exc.msg}"
            ) from None
        self._validate(tree.body)
        self._body = tree.body
        self._names = list(names)
        self._positions = {name: position for position, name in enumerate(names)}
        self.text = stripped

    # ------------------------------------------------------------ validate
    def _validate(self, node: ast.AST) -> None:
        if isinstance(node, ast.Constant):
            # bool is an int subclass; a boolean is not a number here.  A string
            # constant is legal only as the ``_b["name"]`` index, which the
            # Subscript branch consumes without recursing into it.
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(
                    "nlcom/testnl: only numeric constants are allowed, got "
                    f"{node.value!r}"
                )
            return

        if isinstance(node, ast.Name):
            if node.id == "_b":
                raise ValueError(
                    "nlcom/testnl: _b must be subscripted with a coefficient "
                    "name, as in _b[x1]"
                )
            raise ValueError(
                f"nlcom/testnl: bare name {node.id!r} is not allowed; write "
                f"_b[{node.id}] for a coefficient"
            )

        if isinstance(node, ast.Subscript):
            target = node.value
            if not isinstance(target, ast.Name) or target.id != "_b":
                raise _unsupported(node)
            index = node.slice
            # ``_b[x1]`` parses as Subscript(Name('_b'), Name('x1')): a bare
            # coefficient name is a Name node, not a string.
            if isinstance(index, ast.Name):
                return
            if isinstance(index, ast.Constant) and isinstance(index.value, str):
                if not index.value:
                    raise ValueError(
                        "nlcom/testnl: the coefficient name inside _b[...] is empty"
                    )
                return
            raise ValueError(
                "nlcom/testnl: the subscript of _b must be a coefficient name, as "
                'in _b[x1] or _b["C(g)[T.2]"]'
            )

        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARY_OPERATORS):
                raise _unsupported(node)
            self._validate(node.operand)
            return

        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINARY_OPERATORS):
                raise _unsupported(node)
            self._validate(node.left)
            self._validate(node.right)
            return

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
                raise _unsupported(node)
            if node.keywords:
                raise ValueError("nlcom/testnl: keyword arguments are not allowed")
            if len(node.args) != 1:
                raise ValueError(
                    f"nlcom/testnl: {node.func.id}() takes exactly one argument"
                )
            self._validate(node.args[0])
            return

        raise _unsupported(node)

    # --------------------------------------------------------------- apply
    def __call__(self, params: np.ndarray) -> float:
        return self._evaluate(self._body, params)

    def _evaluate(self, node: ast.AST, params: np.ndarray) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)

        if isinstance(node, ast.Subscript):
            key = _subscript_key(node)
            try:
                position = self._positions[key]
            except KeyError:
                raise KeyError(
                    f"nlcom/testnl: coefficient {key!r} not found; available: "
                    f"{self._names}"
                ) from None
            return float(params[position])

        if isinstance(node, ast.UnaryOp):
            value = self._evaluate(node.operand, params)
            return -value if isinstance(node.op, ast.USub) else value

        if isinstance(node, ast.BinOp):
            left = self._evaluate(node.left, params)
            right = self._evaluate(node.right, params)
            return _apply_binary(node.op, left, right)

        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            return _apply_function(node.func.id, self._evaluate(node.args[0], params))

        # Unreachable after validation; kept so a future node type cannot pass
        # through silently.
        raise _unsupported(node)


def _subscript_key(node: ast.Subscript) -> str:
    index = node.slice
    if isinstance(index, ast.Name):
        return index.id
    assert isinstance(index, ast.Constant) and isinstance(index.value, str)
    return index.value


def _apply_binary(operator: ast.operator, left: float, right: float) -> float:
    if isinstance(operator, ast.Add):
        return left + right
    if isinstance(operator, ast.Sub):
        return left - right
    if isinstance(operator, ast.Mult):
        return left * right
    if isinstance(operator, ast.Div):
        if right == 0.0:
            raise DivisionByZero("nlcom/testnl: division by zero in the expression")
        return left / right
    assert isinstance(operator, ast.Pow)
    try:
        # ``math.pow`` keeps real-valued semantics: a negative base with a
        # fractional exponent is a domain error, not a complex number.
        return math.pow(left, right)
    except OverflowError as exc:
        raise ValueError(
            f"nlcom/testnl: {left!r}**{right!r} overflows ({exc})"
        ) from None
    except ValueError as exc:
        raise ValueError(
            f"nlcom/testnl: {left!r}**{right!r} is undefined ({exc})"
        ) from None


def _apply_function(name: str, argument: float) -> float:
    try:
        return float(_FUNCTIONS[name](argument))
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(
            f"nlcom/testnl: {name}({argument!r}) is undefined ({exc})"
        ) from None


def _gradient(
    function: Callable[[np.ndarray], float],
    params: np.ndarray,
    *,
    step: float = _GRADIENT_STEP,
) -> np.ndarray:
    """Central-difference gradient ``dg/db`` with step ``step * max(1, |b_j|)``."""

    params = np.asarray(params, dtype=float).reshape(-1)
    gradient = np.zeros(params.size, dtype=float)
    for index in range(params.size):
        delta = step * max(1.0, abs(params[index]))
        forward = params.copy()
        forward[index] += delta
        backward = params.copy()
        backward[index] -= delta
        try:
            gradient[index] = (function(forward) - function(backward)) / (2.0 * delta)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(
                "nlcom/testnl: the delta-method gradient cannot be evaluated near "
                f"the estimate (parameter {index} perturbed by {delta:g}): {exc}"
            ) from None
    if not np.all(np.isfinite(gradient)):
        raise ValueError(
            "nlcom/testnl: the delta-method gradient is not finite at the estimate; "
            "the expression is not differentiable here"
        )
    return gradient


# ---------------------------------------------------------------------------
# testnl argument splitting
# ---------------------------------------------------------------------------


def _top_level_positions(text: str, targets: str) -> list[int]:
    """Indices of ``targets`` characters outside brackets and string literals.

    A coefficient name may itself contain any of them: ``_b["C(g, base=0)[T.1]"]``
    is one token whose name holds a comma, an ``=`` and brackets, so a plain
    ``str.split`` would cut it apart.
    """

    positions: list[int] = []
    depth = 0
    quote: str | None = None
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        elif depth == 0 and char in targets:
            positions.append(index)
        index += 1
    return positions


def _closes_at_end(text: str) -> bool:
    """Whether the leading ``(`` closes exactly at the final character."""

    depth = 0
    quote: str | None = None
    for index, char in enumerate(text):
        if quote is not None:
            if char == quote:
                quote = None
            continue
        if char in "'\"":
            quote = char
        elif char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
            if depth == 0:
                return index == len(text) - 1
    return False


def _strip_outer_parens(text: str) -> str:
    """Remove balanced parentheses that wrap the whole text."""

    text = text.strip()
    while text.startswith("(") and text.endswith(")") and _closes_at_end(text):
        text = text[1:-1].strip()
    return text


def _parses_as_expression(text: str) -> bool:
    try:
        ast.parse(text, mode="eval")
    except SyntaxError:
        return False
    return True


def _looks_like_constraint(text: str) -> bool:
    """Whether ``text`` is a complete ``g(b) [= c]`` clause."""

    body = _strip_outer_parens(text)
    if not body:
        return False
    equals = _top_level_positions(body, "=")
    if not equals:
        return _parses_as_expression(body)
    if len(equals) > 1:
        return False
    position = equals[0]
    left = body[:position].strip()
    right = body[position + 1 :].strip()
    if not left or not right:
        return False
    return _parses_as_expression(left) and _parses_as_expression(right)


def _find_constraint_boundary(text: str) -> int | None:
    """Start index of the next constraint in ``"a = 0 b = 0"``, if any.

    A boundary is top-level whitespace where the text to its left is already a
    complete constraint and the text to its right does not start with a binary
    operator.  That keeps ``"2 * _b[x1] + _b[x2] = 0"`` in one piece while
    splitting ``"_b[x1] = 0 _b[x2] = 0"``.
    """

    for index in _top_level_positions(text, " \t\n\r"):
        left = text[:index].strip()
        right = text[index:].strip()
        if not left or not right:
            continue
        if left[-1] in _BINARY_CHARS or right[0] in _BINARY_CHARS:
            continue
        # A hypothesis is about the model's coefficients, so the left-hand side
        # must reference at least one of them; otherwise "2 _b[x1]" would be
        # silently treated as two constraints.
        if "_b[" not in left:
            continue
        if _looks_like_constraint(left):
            return index
    return None


def _split_constraints(expression: str) -> list[str]:
    """Split a ``testnl`` argument into individual constraint texts."""

    if not isinstance(expression, str):
        raise TypeError(
            f"testnl: the constraint must be a string, got "
            f"{type(expression).__name__}"
        )
    text = expression.strip()
    if not text:
        raise ValueError("testnl: no constraint was given")

    pieces: list[str] = []
    start = 0
    for position in _top_level_positions(text, ","):
        pieces.append(text[start:position])
        start = position + 1
    pieces.append(text[start:])

    constraints: list[str] = []
    for piece in pieces:
        remainder = piece.strip()
        if not remainder:
            raise ValueError(f"testnl: cannot parse the constraint {expression!r}")
        while True:
            boundary = _find_constraint_boundary(remainder)
            if boundary is None:
                head = _strip_outer_parens(remainder)
                if not head:
                    raise ValueError(
                        f"testnl: cannot parse the constraint {expression!r}"
                    )
                constraints.append(head)
                break
            head = _strip_outer_parens(remainder[:boundary])
            if not head:
                raise ValueError(
                    f"testnl: cannot parse the constraint {expression!r}"
                )
            constraints.append(head)
            remainder = remainder[boundary:].strip()
            if not remainder:
                raise ValueError(
                    f"testnl: cannot parse the constraint {expression!r}"
                )
    if not constraints:
        raise ValueError("testnl: no constraint was given")
    return constraints


def _split_equality(constraint: str) -> tuple[str, str]:
    """Split one clause into ``g(b)`` and the quantity it is tested against."""

    equals = _top_level_positions(constraint, "=")
    if not equals:
        # A bare expression is tested against zero, as ``test`` treats a bare
        # coefficient name.
        return constraint, "0"
    if len(equals) > 1:
        raise ValueError(
            "testnl: only one '=' is allowed in a constraint, as in "
            "'exp(_b[x1]) = 2'"
        )
    position = equals[0]
    left = constraint[:position].strip()
    right = constraint[position + 1 :].strip()
    if not left or not right:
        raise ValueError(f"testnl: cannot parse the constraint {constraint!r}")
    return left, right


class _Constraint:
    """``g(b) - c`` for one ``g(b) = c`` constraint, plus its gradient."""

    def __init__(self, text: str, names: Sequence[str]) -> None:
        left, right = _split_equality(text)
        self.text = text
        self._left = _Expression(left, names)
        self._right = _Expression(right, names)

    def __call__(self, params: np.ndarray) -> float:
        return self._left(params) - self._right(params)

    def gradient(self, params: np.ndarray) -> np.ndarray:
        return _gradient(self, params)


def _strip_quoted(text: str) -> str:
    """Remove string literals, so a quoted name cannot look like an ``=``."""

    out: list[str] = []
    quote: str | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if quote is not None:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
        else:
            out.append(char)
        index += 1
    return "".join(out)


def _params_and_cov(result: Any) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Read ``b``, ``V`` and the coefficient names from any westata result."""

    names = list(result.param_names)
    params = np.asarray(result.params, dtype=float).reshape(-1)
    cov = np.asarray(result.cov_params(), dtype=float)
    if params.size != len(names):
        raise ValueError(
            f"params has {params.size} entries but the result reports "
            f"{len(names)} parameter names"
        )
    if cov.shape != (params.size, params.size):
        raise ValueError(
            f"cov_params has shape {cov.shape}, expected "
            f"({params.size}, {params.size})"
        )
    return params, cov, names


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def nlcom(
    result: Any,
    expression: str,
    *,
    level: float = 0.95,
    eform: bool = False,
) -> LincomResult:
    """Point estimate, standard error and interval for ``g(b)``.

    Parameters
    ----------
    result:
        Any estimation result exposing ``params``, ``param_names`` and
        ``cov_params()`` (every westata estimator does).
    expression:
        A function of the coefficients written with ``_b[name]``, e.g.
        ``"exp(_b[x1])"``, ``"_b[x1] / _b[x2]"`` or ``"_b[x1] + _b[x2]"``.
    level:
        Confidence level as a fraction (Stata's ``level(95)`` is ``0.95``).
    eform:
        Stata's ``eform`` option: report ``exp(g(b))`` and exponentiate the
        interval, exactly as :func:`run_lincom` does for ``lincom, eform``.

    Returns
    -------
    LincomResult
        The same container ``lincom`` returns, so downstream code needs no
        special case.  Inference follows the model's distribution (``t`` with
        ``e(df_r)``, or the normal for a ``z`` model); see the module docstring
        for the known deviation from Stata's printed ``z``.
    """

    if not isinstance(expression, str):
        raise TypeError(
            f"nlcom: the expression must be a string, got {type(expression).__name__}"
        )
    # A quoted coefficient name may contain '=' (e.g. "C(g, base=0)[T.1]"), so
    # only unquoted text counts as a hypothesis.
    if "=" in _strip_quoted(expression):
        raise ValueError(
            "nlcom: the expression must not contain '='; use results.testnl() "
            "for hypotheses"
        )
    if not 0.0 < level < 1.0:
        raise ValueError(f"nlcom: level must lie in (0, 1), got {level!r}")

    params, cov, names = _params_and_cov(result)
    function = _Expression(expression, names)
    estimate = function(params)
    gradient = _gradient(function, params)

    variance = float(gradient @ cov @ gradient)
    if variance < 0:
        raise ValueError(
            "nlcom: the delta-method variance is negative; the covariance matrix "
            "is not positive semidefinite"
        )
    std_err = float(np.sqrt(variance))

    alpha = 1.0 - level
    # Stata's nlcom always reports z / P>|z| and a normal-theory interval, even
    # after regress. The committed fixture proves it:
    #     nlcom exp(_b[x1]) -> 7.298097 .3291511 22.17 [6.652973, 7.943221]
    # which is est -/+ 1.959964*se; the t(196) interval would be
    # [6.648965, 7.947229]. This is a deliberate difference from lincom, which
    # follows the model's own reference distribution.
    is_z = True
    df_resid = float("nan")
    critical = float(stats.norm.ppf(1.0 - alpha / 2.0))

    if std_err > 0:
        statistic = estimate / std_err
        pvalue = (
            float(2.0 * stats.norm.sf(abs(statistic)))
            if is_z
            else float(2.0 * stats.t.sf(abs(statistic), df_resid))
        )
    else:
        statistic = float("nan")
        pvalue = float("nan")

    lower = estimate - critical * std_err
    upper = estimate + critical * std_err
    if eform:
        estimate = math.exp(estimate)
        lower = math.exp(lower)
        upper = math.exp(upper)

    return LincomResult(
        estimate=estimate,
        std_err=std_err,
        statistic=statistic,
        pvalue=pvalue,
        ci_lower=lower,
        ci_upper=upper,
        distribution="z" if is_z else "t",
        df=None if is_z else df_resid,
        expression=function.text,
        eform=eform,
        label="_nl_1",
    )


def testnl(result: Any, expression: str) -> WaldTest:
    """Wald test of one or more nonlinear constraints (Stata ``testnl``).

    ``expression`` holds one constraint (``"exp(_b[x1]) = 2"``) or several
    separated by commas or whitespace, which are tested jointly.  A bare
    expression is tested against zero.  The reference distribution is
    ``chi2(q)`` with ``q`` the number of constraints, as ``[R] testnl`` reports.
    """

    constraints = _split_constraints(expression)
    params, cov, names = _params_and_cov(result)

    rows: list[np.ndarray] = []
    gaps: list[float] = []
    for text in constraints:
        constraint = _Constraint(text, names)
        gaps.append(constraint(params))
        rows.append(constraint.gradient(params))

    gradient = np.vstack(rows)
    gap = np.asarray(gaps, dtype=float)

    # Reuse the linear Wald machinery: with R = G and r = G b - (g(b) - c), the
    # residual R b - r is exactly the nonlinear gap and R V R' is G V G'.
    rhs = gradient @ params - gap
    rendered = "\n".join(
        f" ({index:2d}) {text}" for index, text in enumerate(constraints, start=1)
    )

    return run_wald_test(
        params,
        cov,
        gradient,
        rhs,
        dist="z",
        df_resid=float(getattr(result, "df_resid", float("nan"))),
        constraint=rendered,
    )


# pytest collects any module-level callable whose name starts with "test", so a
# consumer writing ``from westata.postestimation.nonlinear import testnl`` would
# otherwise have pytest try to run ``testnl`` as a test function.  Opt out.
testnl.__test__ = False  # type: ignore[attr-defined]
