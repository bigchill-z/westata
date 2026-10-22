"""Wald tests of linear hypotheses, targeting Stata's ``test``.

Reference: ``[R] test`` -> "Methods and formulas".

    W = (Rb - r)' (R V R')^-1 (Rb - r)

If the estimation command reports z statistics, ``W ~ chi2(q)``.  If it reports
t statistics with ``d`` denominator degrees of freedom, ``F = W/q ~ F(q, d)``.

The ``coef`` option's constrained estimates are

    b_c = b - V R' (R V R')^-1 (R b - r)
    V_c = V - V R' (R V R')^-1 R V

which is also what :func:`westata.postestimation.lincom` uses.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["WaldTest", "build_contrast", "run_wald_test"]


@dataclass(frozen=True)
class WaldTest:
    """One Wald test, rendered the way Stata prints it.

    Attributes
    ----------
    statistic:
        ``W`` for chi2 inference, ``W/q`` for F inference.
    df:
        Numerator degrees of freedom ``q``.
    df_denom:
        Denominator degrees of freedom for the F form; ``None`` for chi2.
    pvalue:
        ``P > F`` or ``P > chi2``.
    distribution:
        ``"F"`` or ``"chi2"``.
    constraint:
        Human-readable description of ``Rb = r``.
    """

    statistic: float
    df: float
    df_denom: float | None
    pvalue: float
    distribution: str
    constraint: str

    def __str__(self) -> str:
        if self.distribution == "F":
            head = f"F({self.df:g}, {self.df_denom:g}) = {self.statistic:10.2f}"
            tail = f"Prob > F        = {self.pvalue:10.4f}"
        else:
            head = f"chi2({self.df:g}) = {self.statistic:10.2f}"
            tail = f"Prob > chi2     = {self.pvalue:10.4f}"
        return f"{head}\n{tail}"


def resolve_names(
    requested: Sequence[str], available: Sequence[str], *, wildcards: bool = True
) -> list[int]:
    """Map coefficient references to column positions.

    Stata accepts exact coefficient names plus ``*`` wildcards (``x*``).  A
    reference that matches nothing is an error rather than a silent no-op,
    because a test of an empty hypothesis would otherwise "pass".
    """

    available = list(available)
    positions: list[int] = []
    for ref in requested:
        if wildcards and any(ch in ref for ch in "*?["):
            matches = [
                i for i, name in enumerate(available) if fnmatch.fnmatchcase(name, ref)
            ]
            if not matches:
                raise KeyError(
                    f"no coefficient matches {ref!r}; available: {available}"
                )
            positions.extend(matches)
        else:
            if ref not in available:
                raise KeyError(f"coefficient {ref!r} not found; available: {available}")
            positions.append(available.index(ref))
    # Deduplicate but keep order (Stata tests each coefficient once).
    seen: set[int] = set()
    ordered: list[int] = []
    for position in positions:
        if position not in seen:
            seen.add(position)
            ordered.append(position)
    return ordered


def build_contrast(
    n_params: int,
    rows: Sequence[Mapping[int, float]],
    rhs: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble ``R`` (q x k) and ``r`` (q,) from sparse row specs."""

    q = len(rows)
    R = np.zeros((q, n_params), dtype=float)
    for i, row in enumerate(rows):
        for position, weight in row.items():
            if not 0 <= position < n_params:
                raise IndexError(f"parameter index {position} out of range")
            R[i, position] = float(weight)
    r = np.zeros(q, dtype=float) if rhs is None else np.asarray(rhs, dtype=float)
    if r.shape != (q,):
        raise ValueError(f"rhs must have {q} entries, got shape {r.shape}")
    return R, r


def run_wald_test(
    params: np.ndarray,
    cov: np.ndarray,
    R: np.ndarray,
    r: np.ndarray,
    *,
    dist: str,
    df_resid: float,
    constraint: str = "",
) -> WaldTest:
    """Compute the Wald statistic and its p-value.

    ``dist="z"`` selects chi2 inference, ``dist="t"`` selects the F form, as
    described in ``[R] test``.
    """

    params = np.asarray(params, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    R = np.asarray(R, dtype=float)
    r = np.asarray(r, dtype=float).reshape(-1)

    diff = R @ params - r
    middle = R @ cov @ R.T
    try:
        solved = np.linalg.solve(middle, diff)
    except np.linalg.LinAlgError:
        # Singular (e.g. testing a collinear combination): use the pseudo-inverse
        # and report it, rather than returning an arbitrary large number.
        solved = np.linalg.pinv(middle) @ diff
    W = float(diff @ solved)
    q = R.shape[0]

    if dist == "z":
        return WaldTest(
            statistic=W,
            df=float(q),
            df_denom=None,
            pvalue=float(stats.chi2.sf(W, q)),
            distribution="chi2",
            constraint=constraint,
        )

    F = W / q
    return WaldTest(
        statistic=F,
        df=float(q),
        df_denom=float(df_resid),
        pvalue=float(stats.f.sf(F, q, df_resid)),
        distribution="F",
        constraint=constraint,
    )


def describe_constraint(
    R: np.ndarray,
    r: np.ndarray,
    names: Sequence[str],
    *,
    zero_rhs: bool = False,
) -> str:
    """Render ``Rb = r`` the way Stata prints ``( 1) x1 = 0``."""

    lines: list[str] = []
    for i in range(R.shape[0]):
        terms: list[str] = []
        for j, weight in enumerate(R[i]):
            if weight == 0:
                continue
            if not terms:
                prefix = "-" if weight < 0 else ""
                magnitude = "" if abs(weight) == 1 else f"{abs(weight):g}*"
                terms.append(f"{prefix}{magnitude}{names[j]}")
            else:
                sign = "-" if weight < 0 else "+"
                magnitude = "" if abs(weight) == 1 else f"{abs(weight):g}*"
                terms.append(f" {sign} {magnitude}{names[j]}")
        expression = "".join(terms) if terms else "0"
        if zero_rhs and r[i] == 0:
            lines.append(f"({i + 1:2d}) {expression} = 0")
        else:
            lines.append(f"({i + 1:2d}) {expression} = {r[i]:g}")
    return "\n".join(lines)
