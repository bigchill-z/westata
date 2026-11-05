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


def _space_after_marker(line: str) -> str:
    """Stata writes ``( 1)  temp = 0`` -- two spaces after the closing parenthesis."""

    close = line.find(")")
    if close < 0:
        return line
    head, tail = line[: close + 1], line[close + 1 :].lstrip()
    return f"{head}  {tail}"


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
        # Stata prints each restriction ABOVE the statistic::
        #
        #    ( 1)  x1 = 0
        #    ( 2)  x2 = 0
        #
        #          F(  2,   297) =  156.36
        #               Prob > F =    0.0000
        #
        # ``constraint`` ALREADY carries the numbering and one restriction per line --
        # wrapping it in another ``( n)`` printed "( 1) ( 1) x1 = 0".  It was simply
        # never rendered, so the table showed a statistic with no statement of what was
        # being tested.
        lines: list[str] = []
        text = str(self.constraint or "").strip()
        if text:
            # Stata indents the restriction by ONE space and leaves TWO after the ``( 1)``
            # marker: " ( 1)  k = .1". Measured from a live ``test k = 0.1`` on
            # cobb_douglas.dta -- the earlier comment claimed two leading spaces, and with
            # them the line sat a character right of Stata's.
            lines.extend(
                " " + _space_after_marker(line.strip())
                for line in text.splitlines()
                if line.strip()
            )
            lines.append("")
        # Measured from a live ``test temp=0``::
        #
        #           ( 1)  temp = 0
        #                 F(  1,    26) =   60.25
        #                      Prob > F =    0.0000
        #
        # The F label starts at column 6 and pads BOTH bracketed fields ("  1", "    26"),
        # the value is right-aligned in 8 after " = ", and the probability line starts at
        # column 11 with the value right-aligned in 10. The old code printed no indentation,
        # no padding, and aligned Prob with a hard-coded run of spaces inside the label.
        # Indentation measured from a live ``test k = 0.1``:
        #
        #   '       F(  1,    21) =    1.98'    <- F ends at column 8, value ends at 26
        #   '            Prob > F =    0.1744'  <- Prob ends at column 18, value at 34
        #
        # Seven leading spaces before F and twelve before Prob. The earlier version used six
        # and eleven, which put both lines one character left of Stata's.
        if self.distribution == "F":
            head = (
                f"       F({self.df:>3.0f},{self.df_denom:>6.0f}) = "
                f"{self.statistic:>7.2f}"
            )
            tail = f"            Prob > F = {self.pvalue:>9.4f}"
        else:
            head = (
                f"       chi2({self.df:>3.0f}) = {self.statistic:>7.2f}"
            )
            tail = f"            Prob > chi2 = {self.pvalue:>9.4f}"
        lines.append(head)
        lines.append(tail)
        return chr(10).join(lines)

    def to_dict(self) -> dict[str, float]:
        """Flatten to the ``r()`` names Stata returns for a Wald test.

        ``regress_postestimation.sthlp:770-785`` documents two shapes:

        * F inference -- ``r(F)``, ``r(df)``, ``r(df_r)``, ``r(p)``;
        * chi2 inference -- ``r(chi2)``, ``r(df)``, ``r(p)``.

        The statistic travels under the name of its OWN distribution, so a caller
        reading ``r(F)`` after a likelihood test finds nothing -- which is correct,
        because Stata stores ``r(chi2)`` there.
        """

        table: dict[str, float] = {}
        if self.distribution == "F":
            table["F"] = float(self.statistic)
            if self.df_denom is not None:
                table["df_r"] = float(self.df_denom)
        else:
            table["chi2"] = float(self.statistic)
        table["df"] = float(self.df)
        table["p"] = float(self.pvalue)
        return table


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
        # EXACT MATCH FIRST. A factor level's name is `C(foreign)[T.Foreign]` -- the brackets are part of the
        # literal name, not a wildcard. Testing for wildcard characters before testing for an exact match sent
        # that name to fnmatch, which reads `[T.Foreign]` as a character class and matches nothing, so every
        # factor variable was unreachable through test()/lincom even though its name was in `available`.
        if ref in available:
            positions.append(available.index(ref))
            continue
        if wildcards and any(ch in ref for ch in "*?"):
            matches = [
                i for i, name in enumerate(available) if fnmatch.fnmatchcase(name, ref)
            ]
            if not matches:
                raise KeyError(
                    f"no coefficient matches {ref!r}; available: {available}"
                )
            positions.extend(matches)
        else:
            raise KeyError(f"coefficient {ref!r} not found; available: {available}")
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
            # Stata drops the leading zero on the right-hand side: "k = .1", not "k = 0.1".
            # ``:g`` alone gives "0.1"; the same stripping used by %9.0g is applied here.
            _rhs = f"{r[i]:g}"
            if _rhs.startswith("0."):
                _rhs = _rhs[1:]
            elif _rhs.startswith("-0."):
                _rhs = "-" + _rhs[2:]
            lines.append(f"({i + 1:2d}) {expression} = {_rhs}")
    return "\n".join(lines)
