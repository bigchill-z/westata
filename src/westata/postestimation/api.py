"""User-facing postestimation entry points.

Two audiences are served here:

* **Stata users** get ``results.test(...)`` / ``results.lincom(...)`` that accept
  coefficient names and print ``( 1) x1 = 0`` / ``F( 1, 71) = ...`` the way
  Stata does.
* **statsmodels users** get ``results.wald_test(R, q, scalar=True)`` with the
  same semantics as ``statsmodels``' method, so existing downstream code works.

The Stata parsing is deliberately small: coefficient names, ``*`` wildcards,
``a = b`` and ``a = b = c`` equality chains, and integer/float constants.  It
is not a general expression parser, and anything it cannot parse raises
``ValueError`` instead of guessing.
"""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from westata.postestimation.lincom import LincomResult, run_lincom
from westata.postestimation.tests import (
    WaldTest,
    build_contrast,
    describe_constraint,
    resolve_names,
    run_wald_test,
)

__all__ = [
    "lincom_as_stata",
    "statsmodels_wald_test",
    "test_as_stata",
]

_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _constraint_with_constant(expression: str, constant: float) -> str:
    """Render ``expr + c`` the way Stata labels it: ``expr = -c``.

    Stata moves the constant to the right-hand side and prints
    ``( 1)  2*weight - 3*foreign = -10``; the ``+ 10`` disappears from the left
    because it is folded into the right-hand side.  ``lincom 10`` (a constant
    only) prints ``10 = 0``.
    """

    symbolic: list[str] = []
    for sign, body in _split_signed_terms(expression):
        if _NUMBER.match(body):
            continue
        symbolic.append(("- " if sign < 0 else "+ ") + body)
    if not symbolic:
        return f"{expression.strip()} = 0"
    rendered = " ".join(symbolic).lstrip("+").strip()
    return f"{rendered} = {-constant:g}"


def _dist_and_df(result: Any) -> tuple[str, float]:
    dist = "z" if getattr(result, "dist", "t") == "z" else "t"
    df_resid = float(getattr(result, "df_resid", float("nan")))
    return dist, df_resid


def _split_expansion_terms(side: str, names: Sequence[str]) -> list[str]:
    """Split a term list into individual coefficient references.

    A wildcard such as ``x*`` expands to *several separate* constraints in
    Stata's ``test`` syntax (``test x*`` is a joint test that each is zero),
    whereas ``x1 + x2`` is a single linear combination.  This distinction is
    the reason the splitter returns a list of references rather than a single
    term dictionary.
    """

    references: list[str] = []
    # Split on whitespace first so ``"x1 x2"`` stays two references, then split
    # each chunk on signs so ``"x1+x2"`` also works.
    for chunk in side.split():
        for token in re.findall(r"[+-]?[^+-]+", chunk):
            body = token.lstrip("+-")
            if "*" in body and not body.split("*", 1)[1]:
                positions = resolve_names([body], names)
                references.extend(names[position] for position in positions)
            else:
                references.append(token)
    return references


def _split_signed_terms(side: str) -> list[tuple[float, str]]:
    """Split ``"2*x - 3*y + 10"`` into ``[(+2, "2*x"), (-3*? ...)]`` pairs.

    A ``+``/``-`` separates terms unless it belongs to a floating-point exponent
    (``1e-3``) or sits inside brackets (``C(g)[T.1]``).  This matters because
    Stata's expressions contain both: ``lincom 2*weight - 3*foreign + 1e3`` is a
    single expression, not six.
    """

    compact = side.replace(" ", "")
    if not compact:
        raise ValueError("empty side in hypothesis")

    tokens: list[tuple[float, str]] = []
    sign = 1.0
    body: list[str] = []
    depth = 0

    def flush() -> None:
        nonlocal sign
        text = "".join(body).strip()
        if text:
            tokens.append((sign, text))
        body.clear()
        sign = 1.0

    for index, char in enumerate(compact):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(depth - 1, 0)

        exponent_sign = (
            char in "+-"
            and depth == 0
            and body
            and body[-1] in "eE"
            and index + 1 < len(compact)
            and (compact[index + 1].isdigit() or compact[index + 1] == ".")
        )
        if char in "+-" and depth == 0 and not exponent_sign:
            # A leading sign belongs to the first term; otherwise it starts a
            # new one.
            flush()
            sign = -1.0 if char == "-" else 1.0
            continue
        body.append(char)
    flush()

    if not tokens:
        raise ValueError(f"cannot parse {side!r}")
    return tokens


def _split_linear_form(
    side: str, names: Sequence[str]
) -> tuple[dict[int, float], float]:
    """Parse one side of a hypothesis into ``(coefficient weights, constant)``.

    Stata's ``lincom``/``test`` accept a bare numeric constant anywhere in the
    expression and move it to the right-hand side::

        lincom 2*weight - 3*foreign + 10   ->   (1) 2*weight - 3*foreign = -10
        test   weight + 10 = 0             ->   (1) weight = -10

    The constant shifts the estimate but never touches the variance, which is
    why ``lincom 2*weight - 3*foreign + 10`` has exactly the same standard error
    as ``lincom 2*weight - 3*foreign`` (14.9369115 vs 4.9369115, both
    ``se = 3.2272271``; verified on StataNow 19.5).
    """

    terms: dict[int, float] = {}
    constant = 0.0

    for sign, body in _split_signed_terms(side):
        if "*" in body:
            coefficient, variable = body.split("*", 1)
            # ``2*x`` is a weighted term; ``x*`` is a varlist wildcard (Stata
            # shorthand), recognised by an empty right-hand side.
            if not variable:
                weight = sign
                positions = resolve_names([body], names)
            elif not _NUMBER.match(coefficient):
                raise ValueError(
                    f"only numeric weights are supported in {body!r}; "
                    "write 2*x rather than x*2"
                )
            else:
                weight = sign * float(coefficient)
                positions = resolve_names([variable], names)
        elif _NUMBER.match(body):
            constant += sign * float(body)
            continue
        else:
            weight = sign
            positions = resolve_names([body], names)

        for position in positions:
            terms[position] = terms.get(position, 0.0) + weight

    return terms, constant


def _split_terms(side: str, names: Sequence[str]) -> dict[int, float]:
    """Parse ``"a + b - 2*c"`` into ``{position: coefficient}``.

    Constants are rejected here: this is the path used by a bare ``test x1 x2``
    list, where Stata treats each token as a coefficient name and answers a
    constant with ``r(111) '<n>' not found``.  Expressions with an ``=`` go
    through :func:`_split_linear_form`, which does accept constants.
    """

    terms, constant = _split_linear_form(side, names)
    if constant != 0.0:
        raise ValueError(
            f"constant {constant:g} is not a coefficient reference; put it on the "
            "right-hand side of '=' (Stata: r(111) '<n>' not found)"
        )
    return terms


def _parse_hypothesis(
    spec: str, names: Sequence[str]
) -> tuple[list[dict[int, float]], list[float], str]:
    """Parse one ``test`` specification into rows of ``Rb = r``."""

    if "=" not in spec:
        # ``test x1 x2`` and ``test x*`` are joint tests that *each* is zero,
        # not a test of their sum.
        parts = _split_expansion_terms(spec, names)
        if len(parts) > 1:
            return [ _split_terms(part, names) for part in parts], [0.0] * len(parts), ""
        terms = _split_terms(spec, names)
        return [terms], [0.0], ""

    sides = [chunk.strip() for chunk in spec.split("=")]
    if any(not chunk for chunk in sides):
        raise ValueError(f"cannot parse hypothesis {spec!r}")

    # "a = b = c" is a chain of pairwise equalities against the first term.
    rows: list[dict[int, float]] = []
    rhs: list[float] = []

    left_terms, left_const = _split_linear_form(sides[0], names)
    for other in sides[1:]:
        other_terms, other_const = _split_linear_form(other, names)
        row: dict[int, float] = {}
        for position, weight in left_terms.items():
            row[position] = row.get(position, 0.0) + weight
        for position, weight in other_terms.items():
            row[position] = row.get(position, 0.0) - weight
        rows.append(row)
        # ``a + c1 = b + c2`` tests ``a - b = c2 - c1``.
        rhs.append(other_const - left_const)

    return rows, rhs, ""


def test_as_stata(
    result: Any,
    *specs: Any,
    coef: bool = False,
) -> WaldTest | dict[str, Any]:
    """Wald test of one or more linear hypotheses.

    Parameters
    ----------
    *specs:
        Each argument is a Stata-style hypothesis.  A bare coefficient list
        (``"x1 x2"``) tests that all are zero; an equality chain (``"x1 = x2"``)
        tests that difference.  Multiple specs are tested jointly.
    coef:
        Stata's ``coef`` option: also return the constrained estimates
        ``b_c = b - V R'(RVR')^-1(Rb - r)`` and ``V_c = V - VR'(RVR')^-1 RV``.
    """

    if not specs:
        raise ValueError("test requires at least one hypothesis")

    names = list(result.param_names)
    params = np.asarray(result.params, dtype=float)
    cov = result.cov_params()

    rows: list[dict[int, float]] = []
    rhs: list[float] = []
    for spec in specs:
        if not isinstance(spec, str):
            raise TypeError(f"hypothesis must be a string, got {type(spec).__name__}")
        spec_rows, spec_rhs, _ = _parse_hypothesis(spec, names)
        rows.extend(spec_rows)
        rhs.extend(spec_rhs)

    R, r = build_contrast(len(names), rows, rhs)
    dist, df_resid = _dist_and_df(result)
    constraint = describe_constraint(R, r, names, zero_rhs=bool(np.all(r == 0)))

    outcome = run_wald_test(
        params, cov, R, r, dist=dist, df_resid=df_resid, constraint=constraint
    )
    if not coef:
        return outcome

    # [R] test: two-step constrained estimates.
    middle = R @ cov @ R.T
    correction = cov @ R.T @ np.linalg.pinv(middle)
    constrained = params - correction @ (R @ params - r)
    constrained_cov = cov - correction @ R @ cov
    return {
        "test": outcome,
        "params": pd.Series(constrained, index=names),
        "cov": constrained_cov,
    }


def lincom_as_stata(
    result: Any,
    expression: str,
    *,
    level: float = 0.95,
    eform: bool = False,
) -> Any:
    """Evaluate one linear combination (Stata ``lincom``).

    Stata moves a bare constant to the right-hand side of the printed
    constraint and leaves the variance untouched, so
    ``lincom 2*weight - 3*foreign + 10`` reports ``14.9369115`` with the same
    ``se = 3.2272271`` as ``lincom 2*weight - 3*foreign`` reports for
    ``4.9369115``.  The constant is added to the estimate and never enters
    :math:`L V L'`.
    """

    names = list(result.param_names)
    if "=" in expression:
        raise ValueError(
            "lincom does not accept an equals sign; use results.test() for "
            "hypotheses"
        )
    terms, constant = _split_linear_form(expression, names)
    weights = np.zeros(len(names), dtype=float)
    for position, weight in terms.items():
        weights[position] = weight

    dist, df_resid = _dist_and_df(result)
    params = np.asarray(result.params, dtype=float)
    cov = result.cov_params()

    # The standard error depends on the weights only, so it comes from the
    # shared implementation; the constant is then applied to the point estimate
    # and the interval, exactly as Stata prints the constraint
    # ``2*weight - 3*foreign = -10`` with ``se`` unchanged.
    unshifted = run_lincom(
        params,
        cov,
        weights,
        dist=dist,
        df_resid=df_resid,
        expression=f"{expression} = 0",
        level=level,
        eform=False,
    )
    if constant == 0.0 and not eform:
        return unshifted

    linear = float(weights @ params) + constant
    std_err = unshifted.std_err
    alpha = 1.0 - level
    if dist == "z":
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
        statistic = linear / std_err if std_err > 0 else float("nan")
        pvalue = float(2.0 * stats.norm.sf(abs(statistic)))
        df_value: float | None = None
    else:
        critical = float(stats.t.ppf(1.0 - alpha / 2.0, df_resid))
        statistic = linear / std_err if std_err > 0 else float("nan")
        pvalue = float(2.0 * stats.t.sf(abs(statistic), df_resid))
        df_value = float(df_resid)

    estimate = linear
    lower = linear - critical * std_err
    upper = linear + critical * std_err
    if eform:
        estimate = float(np.exp(linear))
        lower = float(np.exp(lower))
        upper = float(np.exp(upper))

    constraint = (
        expression if constant == 0.0 else _constraint_with_constant(expression, constant)
    )
    return LincomResult(
        estimate=estimate,
        std_err=std_err,
        statistic=statistic,
        pvalue=pvalue,
        ci_lower=lower,
        ci_upper=upper,
        distribution=dist,
        df=df_value,
        expression=constraint,
        eform=eform,
        label="(1)",
    )


def statsmodels_wald_test(
    result: Any, R: Any, q: Any = None, scalar: bool = True
) -> Any:
    """statsmodels-compatible ``wald_test``.

    ``scalar=True`` returns an F test with ``df_num = q`` and
    ``df_denom = df_resid`` (statsmodels' default); ``scalar=False`` returns a
    chi2 test, which matches ``statsmodels``' ``scalar=False`` behaviour.
    """

    names = list(result.param_names)
    R_arr = np.atleast_2d(np.asarray(R, dtype=float))
    if R_arr.shape[1] != len(names):
        raise ValueError(
            f"R must have {len(names)} columns (one per parameter), got "
            f"{R_arr.shape[1]}"
        )
    q_arr = (
        np.zeros(R_arr.shape[0], dtype=float)
        if q is None
        else np.atleast_1d(np.asarray(q, dtype=float))
    )
    if q_arr.shape != (R_arr.shape[0],):
        raise ValueError(
            f"q must have {R_arr.shape[0]} entries, got shape {q_arr.shape}"
        )

    params = np.asarray(result.params, dtype=float)
    cov = result.cov_params()
    constraint = describe_constraint(R_arr, q_arr, names)
    df_resid = float(result.df_resid)

    if scalar:
        return run_wald_test(
            params,
            cov,
            R_arr,
            q_arr,
            dist="t",
            df_resid=df_resid,
            constraint=constraint,
        )
    return run_wald_test(
        params,
        cov,
        R_arr,
        q_arr,
        dist="z",
        df_resid=df_resid,
        constraint=constraint,
    )
