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
``J`` the central-difference Jacobian of the prediction.  ``J'J`` is **never
formed**: it is computed from the QR factor of the **column-normalised**
Jacobian after a rank test in that normalised space, because a conditioned but
perfectly estimable design (``{b0} + {b1}*weight + {b2}*weight^2``:
``cond(J) = 1.6e8``) makes ``pinv(J'J)`` drop a direction that Stata keeps, which
reported ``se(/b0) = 3.95e-07`` for Stata's ``5.77``.  A genuinely
rank-deficient Jacobian raises :class:`~westata.errors.ConsistencyError` instead
of inventing a standard error.

**The rank test is scale-free, in two places.**  Two separate things had to become
dimensionless.

*The criterion.*  Rank is decided on ``J_n = J D^-1`` with ``D = diag(||J[:,k]||)``,
so the singular values are those of a unit-column matrix and the decision does not
move when a variable's units change.  A test on the raw singular values of ``J`` is
a statement about the caller's units: rescaling ``u`` in
``{b0} + {b1}*u + {b2}*u^2`` by ``1e3`` leaves ``cond(J_n) = 65.26`` bit-for-bit
unchanged while the raw ``s_min/s_max`` moves from ``6.25e-09`` to ``4.97e-03``, so
a raw test accepts the original and refuses the rescaled model -- the same
identifiable model, two answers.

*The tolerance, and why not simply rescale by ``s_max``.*  Column normalisation
alone is not enough, because it also changes how the differencing noise propagates.
Each column is a central difference with absolute error about
``eps * ||f|| / h`` -- a constant, independent of the parameter's scale -- and
dividing by ``||J[:,k]||`` turns that into a *relative* error of
``eps * ||f|| / (h ||J[:,k]||)``.  A regressor entered at ``1e-6`` scale therefore
arrives in ``J_n`` with a large relative error even though the model is fine, and an
exactly collinear pair entered at that scale produces a spurious singular value of
about ``8e-6`` instead of ``0``.  The tolerance is consequently built *per column*
from that noise floor, ``||E||_F`` over the normalised matrix, and never from
``s_max`` times a relative constant: measuring against ``s_max`` is what made the
earlier guard flip with units, and measuring against a fixed relative constant is
what let small-scale collinearity through.  Both are covered by
``tests/test_nl.py::test_nl_rank_test_is_scale_free`` and
``::test_nl_scale_free_rank_test_still_catches_exact_collinearity``.

Starting values are searched, not hard-coded: without ``start=`` the fit begins
from the caller's values and sweeps a multiplicative ladder coordinate-wise,
keeping any move that lowers the residual sum of squares.  This is the same job
Stata's own grid search does, and it is why ``nl (price = {b0}*exp({b1}*mpg))``
now reaches ``b0 = 19581.4`` where a flat start of 1.0 used to stall at
``2.3e-10`` with a residual sum of squares 7.5 times too large.

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

Standard errors are the observed information: a numerical Hessian,
inverted, which is Stata's default.  Against ``frontier_0`` this reproduces the
coefficients, ``sigma_v``, ``sigma_u``, ``sigma2``, the ``/lnsig2v`` and
``/lnsig2u`` standard errors, the ``Wald chi2(2)`` of the slopes and the
``LR test of sigma_u = 0`` ``chibar2(01)`` to the digits Stata prints.

**What the Hessian differentiates, and where the remaining gap comes from.**
The Hessian differentiates the closed-form :func:`frontier_gradient`, not the
log-likelihood itself: the half-normal likelihood is nearly flat along
``sigma_u -> 0``, so second-differencing the objective loses about half its
digits there and diverges outright at relative step ``1e-6``.  With the gradient
route in place:

* SEs evaluated at **Stata's ``e(b)``** agree with Stata's ``e(V)`` to
  ``4.6e-6`` on the n=400 real-inefficiency sample (and ``1.7e-07`` on the
  ``frontier_0`` fixture).  That is the Hessian's own accuracy.
* SEs evaluated at **westata's ``params``** -- what a fitted result reports --
  differ by ``2.5e-5`` on that n=400 sample.  The gap is the **stopping point**,
  not the Hessian: westata's log-likelihood is ``2.1e-10`` higher and Stata's
  gradient infinity-norm is ``1.4e-4`` against westata's ``3.2e-6``.

So ``se`` is the observed information at a slightly *better* optimum than
Stata's.  Closing the last ``2.5e-5`` would mean reproducing Stata's stopping
rule for every ``frontier`` fit -- a larger change than a defect repair, with
collateral risk to the other likelihood paths -- so it is documented rather than
chased.  To compare against Stata digit for digit, evaluate the Hessian at a
common parameter vector.

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
``strict=True`` -- the default -- raises
:class:`~westata.errors.ConvergenceError` when the optimiser does not converge;
``strict=False`` downgrades that to a ``RuntimeWarning`` and returns the last
iterate.  A non-converged nonlinear least-squares fit is not an estimate, which
is why returning one behind a warning was the wrong default.  scipy's BFGS
frequently reports "precision loss" at a point whose gradient is already ~1e-6,
so a non-successful BFGS run is polished with Nelder-Mead and the fit is recorded
as converged when the gradient of the objective is below ``1e-4``.  That is
deliberate: forwarding scipy's status verbatim would report failure on the exact
optimum of the fixtures above.

``frontier``'s observed information differentiates the *analytic* gradient
(:func:`frontier_gradient`) rather than the log-likelihood itself: the
half-normal likelihood is nearly flat along ``sigma_u -> 0``, so a second
difference of the objective loses about half its digits there.  Measured on a
500-observation sample with real inefficiency, the SEs agree with Stata to
``1.7e-07`` on the fixture DGP and the objective Hessian diverged to ``NaN`` at
step ``1e-6``.
"""

from __future__ import annotations

import ast
import math
import warnings
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.linalg import solve_triangular

from westata.errors import ConsistencyError, ConvergenceError, UnsupportedError
from westata.formula.factor import build_design
from westata.results.base import StataResults

__all__ = ["Frontier", "Nl", "compile_expression", "frontier_loglike"]

#: Gradient infinity-norm below which a polished fit counts as converged.
_GRADIENT_TOLERANCE = 1e-4

#: Relative step of the central differences used for the Jacobian/Hessian.
_DIFFERENCE_STEP = 1e-6

#: Relative step of the central differences used for the Gauss-Newton Jacobian.
#: The covariance is the *inverse* of ``J'J``, so a residual error in ``J`` is
#: amplified by the condition number of the cross-product; a coarser step keeps
#: the truncation error (which is ``O(step^2)``) below the rounding error.
#: Measured on ``nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)``: the standard
#: errors reproduce Stata to 1.4e-12 with 1e-5 against 7.1e-11 with 1e-6.
_JACOBIAN_STEP = 1e-5

#: Candidate multipliers used by the starting-value search (see
#: :meth:`Nl._starting_values`).  Multiplicative, because ``nl`` parameters are
#: scale parameters as often as they are slopes: ``{b0}*exp({b1}*x)`` needs
#: ``b0`` near ``2e4`` on ``auto.dta``, which no additive ladder reaches.
_START_LADDER = (1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4)

#: Hard cap on the number of grid points the starting-value search evaluates.
_START_GRID_LIMIT = 4096

#: Larger relative step for the second derivatives (cancellation is worse there).
_HESSIAN_STEP = 1e-5

#: Safety factor on the rank test's differencing-noise floor.
#: A singular value has to clear ``_RANK_NOISE_FACTOR`` times the estimated noise
#: to count as a real direction -- i.e. the decision sits a full decade away from
#: the boundary instead of on it.  See ``_gauss_newton_covariance`` for the
#: measurement: the thinnest true-collinearity margin was 0.46x the raw floor
#: (only a factor 2.2 from flipping), while legitimate models clear the floor by
#: 7-10 orders of magnitude.
_RANK_NOISE_FACTOR = 10.0

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


def _gauss_newton_covariance(
    jacobian: np.ndarray,
    sigma2: float,
    parameter_names: Sequence[str],
    *,
    prediction_scale: float = 1.0,
) -> np.ndarray:
    """``s^2 (J'J)^-1``, via a column-normalised QR with an explicit rank test.

    Three decisions, each forced by measurement on
    ``nl (mpg = {b0} + {b1}*weight + {b2}*weight^2)`` (``auto.dta``):

    1. **Never form ``J'J``.**  Squaring the design doubles the condition
       number: ``cond(J) = 1.6e8`` here, so ``cond(J'J) = 2.6e16`` and the
       smallest eigenvalue of the cross-product (``3.4e-1`` against
       ``8.7e15``) sits **below** numpy's default ``pinv`` cut-off
       (``max(shape) * eps = 6.7e-16`` relative).  ``np.linalg.pinv`` then
       silently zeroes that direction, which is how the standard error of
       ``/b0`` came out as ``3.95e-07`` where Stata reports ``5.77``, with every
       point estimate still exact.  Solving with ``R`` from ``qr(J)`` keeps the
       problem at ``cond(J)``: the SEs then reproduce Stata to ``1.4e-12``.
    2. **Test the rank of ``J``, not of ``J'J``.**  A rank test on the
       cross-product would reject this perfectly estimable model (it reports
       ``rank = 2`` of 3).  The rank of ``J`` is what decides identification.
    3. **Test it in the column-normalised space.**  With ``D = diag(||J[:,k]||)``
       and ``J_n = J D^-1``, rank is decided from ``svd(J_n)``.  ``J`` mixes
       parameters with wildly different units -- ``weight^2`` is ``1e8`` times the
       constant -- so a threshold on the raw singular values of ``J`` is a
       statement about the caller's units, not about identification.  Measured:
       rescaling ``u`` by ``1e3`` in ``{b0} + {b1}*u + {b2}*u^2`` leaves
       ``cond(J_n) = 65.26`` bit-for-bit unchanged while the raw
       ``s_min/s_max`` moves from ``6.25e-09`` to ``4.97e-03``, and the raw test
       REFUSED the rescaled model while accepting the original -- the same
       identifiable model, two different answers.  The normalised test accepts
       all five scalings and still refuses an exactly collinear pair.

    The tolerance on ``J_n`` is ``max(shape) * eps / h`` (``h`` the relative
    differencing step, default :data:`_JACOBIAN_STEP`): ``J`` is a central
    difference, so its entries carry a rounding error of about ``eps * |f| / h``,
    and a test below that noise floor would call an exactly collinear design full
    rank.  Because ``J_n``'s columns are unit-norm this is a *dimensionless*
    number -- it does not move when the caller changes units.

    The covariance follows from ``J'J = D J_n' J_n D``, so
    ``(J'J)^-1 = D^-1 (J_n'J_n)^-1 D^-1``; ``(J_n'J_n)^-1`` comes from ``J_n``'s QR
    factor and ``D^-1`` is applied on both sides.  Normalising here does **not**
    change the reported standard errors: it is an exact algebraic rearrangement of
    the same inverse, plus a rank test that is no longer unit-dependent.

    Raises
    ------
    ConsistencyError
        When ``J_n`` is rank deficient: the model does not identify every
        parameter, and reporting *some* standard error for an unidentified
        direction is worse than refusing.  The message names the collinear
        columns (from the null space of ``J_n``), so a caller can act on it.
    """

    jacobian = np.asarray(jacobian, dtype=float)
    n, k = jacobian.shape
    if k == 0:
        return np.zeros((0, 0), dtype=float)
    if not np.all(np.isfinite(jacobian)):
        raise ConsistencyError(
            "nl: the Jacobian of the prediction is not finite; the parameters are "
            "likely not identified at this point"
        )

    # D = diag(||J[:,k]||); a parameter whose column is identically zero carries no
    # information at all and is handled by the rank test below (norm -> 1 keeps
    # the normalisation defined so the message can still list it).
    norms = np.linalg.norm(jacobian, axis=0)
    zero_columns = [
        name for name, norm in zip(parameter_names, norms) if norm == 0.0
    ]
    safe_norms = np.where(norms > 0.0, norms, 1.0)
    normalised = jacobian / safe_norms

    left, singular, right = np.linalg.svd(normalised, full_matrices=False)

    # ---- noise floor of the *normalised* matrix
    # Every column of J is a central difference, so its absolute error is about
    # ``eps * ||f|| / h`` -- the same for all columns, independent of the
    # parameter's scale.  Normalising by ||J[:,k]|| divides that constant by the
    # column norm, so a column with a small norm (a regressor entered at 1e-6
    # scale) arrives with a *large relative* error.  Measured: with
    # ``z = x*1e-6`` the absolute error is 8.8e-10 but ||col_b|| = 2.7e-5, so the
    # normalised column is only good to 3.3e-5 -- which is why a single fixed
    # tolerance cannot work and the exactly collinear direction looked like a
    # singular value of 8e-6.
    magnitude = max(float(prediction_scale), 1.0)
    column_noise = np.finfo(float).eps * magnitude / (_JACOBIAN_STEP * safe_norms)
    # ``||E||_2 <= ||E||_F`` bounds how far the computed J_n can be from a
    # rank-deficient matrix: a singular value inside that band is
    # indistinguishable from zero, so it is treated as zero.
    #
    # The *_RANK_NOISE_FACTOR* margin exists because the decision must not sit on
    # the boundary.  Measured on the exactly collinear triple
    # ``{a}*x + {b}*z + {c}``, the smallest singular value lands at 0.049x the raw
    # noise floor at ``z = x``, 0.46x at ``z = x*1e-6`` and 0.012x at ``z = x*1e6``
    # -- a 0.46x margin is only a factor 2.2 from flipping, and that is thin
    # enough for a different BLAS path or a different summation order to decide
    # differently.  Legitimate models sit 1e7-1e10 times *above* the floor
    # (quadratic 3.3e8, exp-decay 3.0e10, linear 1.1e7), so widening the band by
    # this factor costs nothing in discrimination and removes the knife edge.
    tolerance = _RANK_NOISE_FACTOR * max(
        float(np.linalg.norm(column_noise)),
        np.finfo(float).eps * max(normalised.shape),
    )
    rank = int(np.sum(singular > tolerance))
    if rank < k:
        null_directions = right[rank:]
        culprits: list[str] = []
        for direction in null_directions:
            # The parameters that move together along this null direction: the
            # entries of the right singular vector at the normalised scale.
            worst = int(np.argmax(np.abs(direction)))
            label = parameter_names[worst]
            if label not in culprits:
                culprits.append(label)
        detail = ", ".join(zero_columns) if zero_columns else ", ".join(culprits)
        condition = (
            float(singular[0] / singular[-1]) if singular[-1] > 0.0 else float("inf")
        )
        raise ConsistencyError(
            f"nl: the column-normalised Jacobian has rank {rank} but the model has "
            f"{k} parameters, so at least one parameter is not identified and its "
            f"standard error does not exist. cond(J_n) = {condition:.3e} "
            f"(dimensionless, column-normalised); singular values "
            f"{', '.join(f'{value:.3e}' for value in singular)} against a noise "
            f"floor of {tolerance:.3e}. The offending parameters are {detail}: "
            "they enter the model only through a linear combination of each other "
            "(a duplicate regressor, or a parameter whose column is identically "
            "zero). Rescaling the variables does not fix this -- the test is "
            "scale-free by construction -- so the model itself must change."
        )

    try:
        r_factor = np.linalg.qr(normalised, mode="r")
        inverse = solve_triangular(r_factor, np.eye(k), lower=False)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded by the rank test
        raise ConsistencyError(
            f"nl: the Gauss-Newton cross-product is singular ({exc}); the model is "
            "not identified on these data"
        ) from None
    # (J'J)^-1 = D^-1 (J_n'J_n)^-1 D^-1 = D^-1 R^-1 R^-T D^-1
    inner = inverse @ inverse.T
    covariance = float(sigma2) * (inner / safe_norms[:, None] / safe_norms[None, :])
    return np.asarray(covariance, dtype=float)


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
    """Central-difference Jacobian of ``predict`` at ``point``.

    Uses :data:`_JACOBIAN_STEP`, not the optimiser's difference step: the
    covariance inverts ``J'J``, so the step has to be chosen for the *inverse*
    problem (see the constant's own comment).

    The caller passes the response's magnitude to
    :func:`_gauss_newton_covariance` separately because the rank test needs it: a
    central difference is accurate to about ``eps * ||f|| / h``, and that absolute
    error becomes a *relative* error of ``eps * ||f|| / (h * ||J[:,k]||)`` for a
    column of norm ``||J[:,k]||``.  Without the magnitude the caller cannot tell
    whether a small singular value is a real deficiency or differencing noise.
    """

    base = np.asarray(predict(point), dtype=float)
    jacobian = np.empty((base.size, point.size), dtype=float)
    for position in range(point.size):
        step = _JACOBIAN_STEP * max(1.0, abs(point[position]))
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
    objective: Callable[[np.ndarray], float],
    point: np.ndarray,
    *,
    gradient: Callable[[np.ndarray], np.ndarray] | None = None,
    step: float | None = None,
) -> np.ndarray:
    """Central-difference Hessian, of the objective or of its analytic gradient.

    ``gradient`` switches the routine to differentiating a *closed-form*
    gradient.  That matters for the frontier likelihood, which is almost flat
    along ``sigma_u -> 0``: second-differencing the objective itself loses
    roughly half its digits to cancellation there, while the gradient is
    available analytically and its central difference stays first order.  It is
    the standard way to get a usable observed information out of a flat
    likelihood without inventing a step size per direction.
    """

    size = point.size
    if gradient is not None:
        steps = (
            (_HESSIAN_STEP if step is None else float(step))
            * np.maximum(np.abs(point), 1.0)
        )
        hessian = np.empty((size, size), dtype=float)
        for i in range(size):
            ei = np.zeros(size)
            ei[i] = steps[i]
            hessian[:, i] = (gradient(point + ei) - gradient(point - ei)) / (
                2.0 * steps[i]
            )
        return 0.5 * (hessian + hessian.T)

    hessian = np.zeros((size, size), dtype=float)
    steps = (_HESSIAN_STEP if step is None else float(step)) * np.maximum(
        np.abs(point), 1.0
    )
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


def frontier_exponential_loglike(
    theta: np.ndarray,
    y: np.ndarray,
    exog: np.ndarray,
    *,
    cost: bool = False,
) -> float:
    """Normal/exponential frontier log-likelihood, from ``fron_ex.ado``.

    ``theta = [beta, lnsig2v, lnsig2u]`` with ``lnsig2v = ln(sigma_v^2)`` and ``lnsig2u = ln(sigma_u^2)``,
    matching the half-normal path's parameter order so the two are interchangeable from the caller's side.

    ``cost=False`` is a production frontier (``S_COST = 1``), ``cost=True`` a cost frontier (``S_COST = -1``),
    which is how ``frontier.ado`` selects the sign.

        e  = y - x'b
        z  = -S_COST e exp(-lnsig2v/2) - exp((lnsig2v - lnsig2u)/2)
        ll = sum[ -1/2 lnsig2u + 1/2 exp(lnsig2v - lnsig2u) + ln Phi(z) + S_COST e exp(-lnsig2u/2) ]

    The tail is evaluated through ``stats.norm.logcdf`` rather than ``log(norm.cdf)`` for the reason recorded on
    the half-normal version: at ``z = -40`` the term is about -804 while ``log(cdf)`` returns ``-inf`` and
    silently drops the observation.
    """

    beta = np.asarray(theta, dtype=float)[: exog.shape[1]]
    lnsig2v = float(theta[exog.shape[1]])
    lnsig2u = float(theta[exog.shape[1] + 1])
    sign = -1.0 if cost else 1.0

    residual = np.asarray(y, dtype=float) - exog @ beta
    z = -sign * residual * math.exp(-0.5 * lnsig2v) - math.exp(0.5 * (lnsig2v - lnsig2u))

    total = (
        -0.5 * lnsig2u
        + 0.5 * math.exp(lnsig2v - lnsig2u)
        + stats.norm.logcdf(z)
        + sign * residual * math.exp(-0.5 * lnsig2u)
    )
    return float(np.sum(total))


def frontier_exponential_gradient(
    theta: np.ndarray,
    y: np.ndarray,
    exog: np.ndarray,
    *,
    cost: bool = False,
) -> np.ndarray:
    """Analytic gradient of :func:`frontier_exponential_loglike`, from ``fron_ex.ado``'s g1/g2/g3.

    Transcribed rather than differentiated: the score in the ado file is the authority, and this project has
    twice been burned by a hand-derived second derivative that looked right and was not.
    """

    k = exog.shape[1]
    beta = np.asarray(theta, dtype=float)[:k]
    lnsig2v = float(theta[k])
    lnsig2u = float(theta[k + 1])
    sign = -1.0 if cost else 1.0

    residual = np.asarray(y, dtype=float) - exog @ beta
    z = -sign * residual * math.exp(-0.5 * lnsig2v) - math.exp(0.5 * (lnsig2v - lnsig2u))

    # zr = phi(z)/Phi(z), guarded so a far tail yields a finite gradient instead of NaN.
    density = stats.norm.pdf(z)
    cdf = np.maximum(stats.norm.cdf(z), 1e-300)
    zr = density / cdf

    half_v = math.exp(-0.5 * lnsig2v)
    half_u = math.exp(-0.5 * lnsig2u)
    ratio = math.exp(lnsig2v - lnsig2u)

    ztd1 = sign * half_v
    ztd2 = 0.5 * sign * residual * half_v - 0.5 * math.exp(0.5 * (lnsig2v - lnsig2u))
    ztd3 = 0.5 * math.exp(0.5 * (lnsig2v - lnsig2u))

    g1 = zr * ztd1 - sign * half_u
    g2 = 0.5 * ratio + zr * ztd2
    g3 = -0.5 - 0.5 * ratio + zr * ztd3 - 0.5 * sign * residual * half_u

    out = np.empty(k + 2)
    out[:k] = exog.T @ g1
    out[k] = float(np.sum(g2))
    out[k + 1] = float(np.sum(g3))
    return out


def frontier_exponential_information(
    theta: np.ndarray,
    y: np.ndarray,
    exog: np.ndarray,
    *,
    cost: bool = False,
) -> np.ndarray:
    """Observed information: minus the Jacobian of the verified analytic gradient.

    A central difference of the gradient converges as eps/h rather than eps/h^2 and involves no cancellation of
    the objective, which is why the half-normal path takes the same route.
    """

    size = theta.size
    step = 1e-6 * np.maximum(np.abs(theta), 1.0)
    jacobian = np.zeros((size, size))
    for column in range(size):
        plus = theta.copy()
        minus = theta.copy()
        plus[column] += step[column]
        minus[column] -= step[column]
        jacobian[:, column] = (
            frontier_exponential_gradient(plus, y, exog, cost=cost)
            - frontier_exponential_gradient(minus, y, exog, cost=cost)
        ) / (2.0 * step[column])
    return -(jacobian + jacobian.T) / 2.0


def _inverse_mills(z: np.ndarray) -> np.ndarray:
    """``phi(z)/Phi(z)``, clamped where the tail would underflow.

    ``Phi(z)`` is a normal CDF, so ``phi/Phi`` is at most ``~1e154`` for
    ``z = -37``; below that the ratio is not representable and the log-likelihood
    is already ``-inf``.  Clamping keeps the *gradient* finite so the Hessian is
    a matrix rather than a NaN cloud, without changing any evaluated likelihood.
    """

    denominator = stats.norm.cdf(z)
    numerator = stats.norm.pdf(z)
    safe = np.maximum(denominator, 1e-300)
    return numerator / safe


def frontier_gradient(
    theta: np.ndarray, y: np.ndarray, exog: np.ndarray
) -> np.ndarray:
    """Analytic gradient of :func:`frontier_loglike`.

    Written out so the observed information can be obtained by differentiating
    the *gradient*: the likelihood is nearly flat in the direction that raises
    ``sigma_u`` from zero, and a second difference of the objective itself loses
    too many digits there (measured on a 600-observation sample with a real
    ``u ~ |N(0, 0.7^2)|`` component: SE of ``/lnsig2u`` was 2.73 from the
    objective Hessian against Stata's 2.70).

    Derivation, with ``e = y - x'b``, ``z = e/sigma``, ``a = -z lambda`` and
    ``m = phi(a)/Phi(a)``::

        d lnL / d beta = sum_i x_i (z_i + lambda m_i) / sigma
        d lnL / d ln s2_v = sum_i [ (s2_v/sigma^2)(z_i^2-1)/2
                                    + lambda z_i m_i (1 + s2_v/sigma^2)/2 ]
        d lnL / d ln s2_u = sum_i [ (s2_u/sigma^2)(z_i^2-1)/2
                                    + lambda z_i m_i (s2_u/sigma^2 - 1)/2 ]

    The ``lambda`` terms come from ``d ln Phi(-z lambda)/d lnsig2``; the sign
    pattern was checked against a central difference of :func:`frontier_loglike`
    to 1e-8 relative on the fixture sample.
    """

    theta = np.asarray(theta, dtype=float)
    k = exog.shape[1]
    beta = theta[:k]
    lnsig2v = float(theta[k])
    lnsig2u = float(theta[k + 1])
    sigma2_v = math.exp(lnsig2v)
    sigma2_u = math.exp(lnsig2u)
    sigma2 = sigma2_v + sigma2_u
    sigma = math.sqrt(sigma2)
    lam = math.sqrt(sigma2_u / sigma2_v)
    z = (np.asarray(y, dtype=float) - exog @ beta) / sigma
    mills = _inverse_mills(-z * lam)

    gradient = np.empty(k + 2, dtype=float)
    gradient[:k] = (exog * ((z + lam * mills) / sigma)[:, None]).sum(axis=0)
    share_v = sigma2_v / sigma2
    share_u = sigma2_u / sigma2
    gradient[k] = float(
        np.sum(
            0.5 * share_v * (z**2 - 1.0)
            + 0.5 * lam * z * mills * (1.0 + share_v)
        )
    )
    gradient[k + 1] = float(
        np.sum(
            0.5 * share_u * (z**2 - 1.0)
            + 0.5 * lam * z * mills * (share_u - 1.0)
        )
    )
    return gradient


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
    def _starting_values(self, objective: Callable[[np.ndarray], float]) -> np.ndarray:
        """Starting values: the caller's, refined by a coarse grid search.

        Stata does not start ``nl`` at 1 either -- it runs a grid search over a
        ladder of values (``[R] nl`` documents the "starting values" search) and
        only then iterates.  Starting every parameter at 1 is what made
        ``nl (price = {b0}*exp({b1}*mpg))`` stall at ``b0 = 2.3e-10, b1 = 0.75``
        with ``rss = 3.42e9`` against Stata's ``4.54e8``: ``b0`` lives near
        ``2e4`` on ``auto.dta`` and no Levenberg-Marquardt step from 1 reaches it.

        The search is coordinate-wise over :data:`_START_LADDER` (multiplicative,
        because ``nl`` parameters are as often scale parameters as slopes), and
        only replaces a coordinate when it lowers the residual sum of squares.
        A caller-supplied ``start=`` seeds the search rather than being
        overridden, so an explicit value still anchors the fit and the sweep can
        only improve on it.
        """

        names = self._parameter_names
        k = len(names)
        start = np.array(
            [float(self._start.get(name, 1.0)) for name in names], dtype=float
        )
        if k == 0 or len(_START_LADDER) ** k > _START_GRID_LIMIT:
            # Too many parameters for a grid; keep the caller's values (a full
            # factorial would be exponential in k).
            return start

        best = start.copy()
        best_value = objective(best)
        for _ in range(2):
            improved = False
            for position in range(k):
                current = best[position]
                candidates = list(_START_LADDER)
                if current != 0.0:
                    candidates.append(abs(current))
                for magnitude in candidates:
                    for value in (magnitude, -magnitude):
                        if value == current:
                            continue
                        trial = best.copy()
                        trial[position] = value
                        score = objective(trial)
                        if score < best_value:
                            best, best_value, improved = trial, score, True
            if not improved:
                break
        return best

    def fit(
        self,
        cov_type: str | None = None,
        cov_kwds: Mapping[str, Any] | None = None,
        ssc: Any = None,
        *,
        maxiter: int = 200,
        strict: bool = True,
    ) -> StataResults:
        """Minimise the residual sum of squares.

        Parameters
        ----------
        maxiter:
            Stata's iteration limit, applied here as the residual-evaluation
            budget of Levenberg-Marquardt.
        strict:
            Raise :class:`~westata.errors.ConvergenceError` when the optimiser
            does not converge.  **The default is now** ``True``: a non-converged
            NLS fit is not an estimate, and returning one behind a
            ``RuntimeWarning`` is how ``nl (price = {b0}*exp({b1}*mpg))`` came to
            report ``rss 3.42e9`` where Stata reports ``4.54e8``.  Pass
            ``strict=False`` to inspect the optimiser's last iterate.
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

        initial = self._starting_values(objective)

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
        # Rank-checked QR solve, never ``pinv(J'J)``: see the helper's docstring
        # for the measurements that forced it.
        covariance = _gauss_newton_covariance(
            jacobian,
            sigma2,
            names,
            prediction_scale=float(np.max(np.abs(fitted))),
        )
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
        if self._distribution not in ("halfnormal", "exponential"):
            raise UnsupportedError(
                f"frontier: distribution={self._distribution!r} is not implemented; "
                "supported values are 'halfnormal' and 'exponential'"
            )
        exponential = self._distribution == "exponential"

        y, X = self._y, self._X
        n, k = X.shape
        if n <= k + 2:
            raise ConsistencyError(
                f"frontier: {n} observations but {k + 2} parameters; the model is "
                "not identified"
            )
        has_constant = any(name == "Intercept" for name in self._param_names)

        def loglike(theta: np.ndarray) -> float:
            if exponential:
                return frontier_exponential_loglike(theta, y, X)
            return frontier_loglike(theta, y, X)

        def negative(theta: np.ndarray) -> float:
            total = loglike(theta)
            return -total if np.isfinite(total) else 1e12

        beta0, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual0 = y - X @ beta0
        variance0 = max(float(np.var(residual0)), 1e-6)
        if exponential:
            # frontier.ado lines 347-350: sigma2_u = (-S_COST m3 / 2)^(2/3), sigma2_v = m2 - sigma2_u with a
            # 1e-4 floor. S_COST is 1 here (production frontier).
            third = float(np.mean((residual0 - residual0.mean()) ** 3))
            sigma2_u0 = (-third / 2.0) ** (2.0 / 3.0) if -third / 2.0 > 0 else variance0 / 2.0
            sigma2_v0 = variance0 - sigma2_u0
            if not np.isfinite(sigma2_v0) or sigma2_v0 <= 0:
                sigma2_v0 = 1e-4
            if not np.isfinite(sigma2_u0) or sigma2_u0 <= 0:
                sigma2_u0 = variance0 / 2.0
            start = np.concatenate(
                [beta0, [math.log(sigma2_v0), math.log(sigma2_u0)]]
            )
        else:
            # Split the OLS variance evenly between the two components.
            start = np.concatenate([beta0, [math.log(variance0) - math.log(2.0)] * 2])

        def gradient_of_negative(theta: np.ndarray) -> np.ndarray:
            # Supplying the analytic gradient removes finite-difference noise from the step direction. On the
            # flat region where sigma_u is barely identified (lnsig2u near -3.8 with sigma_v near 1.1) that is
            # the difference between stopping at |grad| ~ 2e-5 and at ~1e-8: the surface itself is shallow, so
            # the step direction is all an optimiser has to work with.
            if exponential:
                return -frontier_exponential_gradient(theta, y, X)
            return -frontier_gradient(theta, y, X)

        result = optimize.minimize(
            negative,
            start,
            method="BFGS",
            jac=gradient_of_negative,
            options={"maxiter": int(maxiter), "gtol": 1e-10},
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
        # Observed information from the *analytic* gradient: the half-normal
        # likelihood is nearly flat along sigma_u -> 0, and second-differencing
        # the objective there costs about half the digits to cancellation.
        # Measured on the 400-observation frontier DGP with real inefficiency
        # (u ~ |N(0, 0.7^2)|, seed 99): the objective Hessian gives 4.0e-4 SE
        # error at step 1e-5 and 1.3e-1 at 1e-6, while this route is flat at
        # 4.6e-6 from step 1e-4 through 1e-6.  What remains of the reported SE's
        # 2.5e-5 is `theta`'s stopping point, not the Hessian -- see the module
        # docstring and retest/fix/systems/repro_T8_stopping_point.py.
        gradient_for = (
            frontier_exponential_gradient if exponential else frontier_gradient
        )
        hessian = _numerical_hessian(
            negative, theta, gradient=lambda point: -gradient_for(point, y, X)
        )
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
    if normalized == "halfnormal":
        return "halfnormal"
    if normalized == "exponential":
        return "exponential"
    raise UnsupportedError(
        f"frontier: distribution={distribution!r} is not implemented; supported values "
        "are 'halfnormal' and 'exponential'. The truncated-normal variants need their own "
        "likelihood, not a flag."
    )
