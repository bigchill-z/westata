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

from westata.postestimation.lincom import run_lincom
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


def _split_terms(side: str, names: Sequence[str]) -> dict[int, float]:
    """Parse ``"a + b - 2*c"`` into ``{position: coefficient}``."""

    side = side.strip()
    if not side:
        raise ValueError("empty side in hypothesis")

    terms: dict[int, float] = {}
    # Insert a space around signs that are not part of an exponent.
    tokens = re.findall(r"[+-]?[^+-]+", side.replace(" ", ""))
    if not tokens:
        raise ValueError(f"cannot parse {side!r}")

    for token in tokens:
        sign = 1.0
        body = token
        if body[0] in "+-":
            if body[0] == "-":
                sign = -1.0
            body = body[1:]
        body = body.strip()
        if not body:
            raise ValueError(f"cannot parse {side!r}")

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
            # A bare number is a constant term, handled by the caller.
            raise ValueError(
                f"constant {body!r} is not a coefficient reference; "
                "put constants on the right-hand side of '='"
            )
        else:
            weight = sign
            positions = resolve_names([body], names)

        for position in positions:
            terms[position] = terms.get(position, 0.0) + weight

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

    def _terms_or_constant(side: str) -> tuple[dict[int, float], float]:
        try:
            return _split_terms(side, names), 0.0
        except ValueError:
            if _NUMBER.match(side):
                return {}, float(side)
            raise

    left_terms, left_const = _terms_or_constant(sides[0])
    for other in sides[1:]:
        other_terms, other_const = _terms_or_constant(other)
        row: dict[int, float] = {}
        for position, weight in left_terms.items():
            row[position] = row.get(position, 0.0) + weight
        for position, weight in other_terms.items():
            row[position] = row.get(position, 0.0) - weight
        rows.append(row)
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
    """Evaluate one linear combination (Stata ``lincom``)."""

    names = list(result.param_names)
    if "=" in expression:
        raise ValueError(
            "lincom does not accept an equals sign; use results.test() for "
            "hypotheses"
        )
    terms = _split_terms(expression, names)
    weights = np.zeros(len(names), dtype=float)
    for position, weight in terms.items():
        weights[position] = weight

    dist, df_resid = _dist_and_df(result)
    return run_lincom(
        np.asarray(result.params, dtype=float),
        result.cov_params(),
        weights,
        dist=dist,
        df_resid=df_resid,
        expression=f"{expression} = 0",
        level=level,
        eform=eform,
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
