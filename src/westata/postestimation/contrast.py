"""ANOVA-style contrasts and pairwise comparisons of factor levels.

Targets Stata's ``[R] contrast`` and ``[R] pwcompare``.  Both commands operate
on *estimated marginal means* of a factor variable, so on a plain linear model
each reported quantity is a linear combination of the coefficients:

    contrast r.  level_j - reference          (j runs over non-reference levels)
    contrast g.  level_j - unweighted mean over all levels (j runs over all)
    contrast a.  level_{j+1} - level_j        (j runs over consecutive pairs)
    pwcompare    level_j - level_i            (every pair with j > i)

Each row is evaluated with :func:`westata.postestimation.lincom.run_lincom`, so
the standard error is ``sqrt(L V L')``, the statistic is ``(L b) / se`` and the
interval uses the model's own reference distribution (``t`` with ``e(df_r)``, or
``z``).

Factor columns are recognised by their patsy name.  Treatment coding ends in
``[T.<level>]`` (``C(g, Treatment(reference='a'))[T.b]``, ``g[T.b]``) and
cell-means coding -- a model without an intercept, ``y ~ 0 + C(g)`` -- ends in
``[<level>]`` and estimates the level mean directly.  The reference level of a
treatment-coded factor has no column of its own: its coefficient is zero by
construction, so it is added back with weight zero.  That makes a reference
contrast reduce to the level's own coefficient, exactly as in Stata.

Deliberately not implemented, raising :class:`UnsupportedError` instead of
approximating: Helmert (``h.``) and multiple-comparison (``m.``) contrasts,
every ``pwcompare, adjust()`` correction (Bonferroni, Šidák, Tukey, Scheffé,
Dunnett, ...), contrasts of interaction terms, and factors coded with sum,
Helmert or any other non-treatment contrast (whose coefficients are already
contrasts of level means rather than level means).  A level set that cannot be
enumerated completely is an error too, never a silently shortened table.

Authoritative reference: ``[R] contrast`` and ``[R] pwcompare``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from westata.errors import UnsupportedError
from westata.postestimation.lincom import LincomResult, run_lincom

__all__ = ["contrast", "pwcompare"]

#: ``C(g, Treatment(reference='a'))[T.b]`` / ``g[T.b]`` -> base + level.  The
#: greedy ``.+?`` plus the anchored ``]`` rejects interaction columns such as
#: ``C(g)[T.b]:x``, whose tail after the bracket is not the end of the name.
_FACTOR_COLUMN_RE = re.compile(r"^(?P<base>.+?)\[(?P<level>[^\]]+)\]$")

#: patsy prefixes its non-treatment coding columns with a one-letter code
#: (``[S.a]`` for sum, ``[H.b]`` for Helmert, ...).  Such a column is a contrast
#: of level means, not a level mean, so it cannot be weighted directly.
_CODING_PREFIX_RE = re.compile(r"^[A-Za-z]\.")

#: patsy spells the reference level inside the contrast call; used only as a
#: fallback when the estimation sample is not retained on the result.
_REFERENCE_RE = re.compile(
    r"reference\s*=\s*"
    r"(?P<reference>"
    r"'(?:[^'\\]|\\.)*'"
    r'|"(?:[^"\\]|\\.)*"'
    r"|[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r")"
)

_CALL_RE = re.compile(r"^C\((?P<inner>.+)\)$", re.S)

#: Stata's short names for the contrasts this module can compute.
_METHODS: dict[str, str] = {"r": "reference", "g": "grand mean", "a": "adjacent"}

#: Stata spellings that exist but are out of scope, with the reason.
_KNOWN_UNSUPPORTED_METHODS: dict[str, str] = {
    "h": "Helmert",
    "m": "multiple-comparison",
}


@dataclass(frozen=True)
class _Factor:
    """A factor variable resolved against a result's design matrix."""

    variable: str
    design_base: str
    levels: list[str]
    columns: Mapping[str, int]
    #: ``None`` when no level is omitted (no reference level available).
    reference: str | None
    #: Whether ``levels`` is known to list *every* level of the factor.
    complete: bool
    #: The design columns for each level, by name (for readable expressions).
    column_names: Mapping[str, str]


# --------------------------------------------------------------------- term
def _variable_of(term: str) -> str:
    """Return the data variable behind a term such as ``C(g)`` or ``g``."""

    text = term.strip()
    match = _CALL_RE.match(text)
    if match is None:
        return text
    return match.group("inner").split(",", 1)[0].strip()


def _reference_literal(base: str) -> str | None:
    """Return the reference level literal encoded in a patsy contrast call."""

    match = _REFERENCE_RE.search(base)
    if match is None:
        return None
    literal = match.group("reference")
    if literal[0] in {"'", '"'}:
        return literal[1:-1]
    return literal


def _parse_factor_column(name: str) -> tuple[str, str] | None:
    """Split a factor design column into ``(base, level)``.

    Treatment columns end in ``[T.<level>]``; cell-means columns (a model
    without an intercept, ``y ~ 0 + C(g)``) end in ``[<level>]`` and estimate
    the level mean directly.  Every other coding (``[S.a]``, ``[H.b]``,
    ``C(g, Sum)[mean]``, ...) estimates a *contrast* of level means, so those
    names are rejected here and reported by :func:`_unsupported_coding`.
    """

    match = _FACTOR_COLUMN_RE.match(name.strip())
    if match is None:
        return None
    base = match.group("base").strip()
    level = match.group("level")
    if not base or "[" in level or "]" in level:
        return None
    # Treatment coding first: its "[T." marker must not be read as a coding
    # letter.  After that, any other coding is a contrast, not a level mean.
    if level.startswith("T."):
        level = level[2:]
    elif _CODING_PREFIX_RE.match(level):
        return None
    if _coding_of(base) is not None:
        return None
    return base, level


def _coding_of(base: str) -> str | None:
    """Name of the contrast coding encoded in a column's base, if not treatment."""

    match = _CALL_RE.match(base)
    if match is None:
        return None
    inner = match.group("inner")
    if "," not in inner:
        return None
    rest = inner.split(",", 1)[1].strip()
    if rest.startswith("Treatment("):
        return None
    return rest.split("(", 1)[0].strip() or rest


def _unsupported_coding(name: str) -> str | None:
    """Return the coding name if ``name`` is a non-treatment factor column."""

    match = _FACTOR_COLUMN_RE.match(name.strip())
    if match is None:
        return None
    level = match.group("level")
    if not level.startswith("T."):
        coded = _CODING_PREFIX_RE.match(level)
        if coded is not None:
            return level[0]
    # ``C(g, Sum)``-style calls name the coding in the base, e.g.
    # ``C(g, Sum)[mean]`` or ``C(g, Helmert)[H.b]``.
    return _coding_of(match.group("base").strip())


def _mentions_factor(name: str, variable: str) -> bool:
    """Whether ``name`` uses ``variable`` in a factor (bracketed) position."""

    if "[" not in name:
        return False
    head = name.split("[", 1)[0]
    return any(
        _variable_of(token.strip()) == variable for token in re.split(r"[:*#]", head)
    )


def _observed_levels(model_data: Any, variable: str) -> list[str] | None:
    """Level labels of ``variable`` in the estimation sample, in report order.

    patsy orders string levels by sort order and categorical levels by their
    ``categories``; both orders are what Stata's ``contrast`` table follows.
    """

    if model_data is None:
        return None
    columns = getattr(model_data, "columns", None)
    if columns is None or variable not in columns:
        return None

    series = model_data[variable]
    if isinstance(series.dtype, pd.CategoricalDtype):
        labels = [str(value) for value in series.cat.categories]
        return labels or None

    try:
        values = sorted(pd.unique(series.dropna()))
    except TypeError:  # mixed, unsortable levels: no trustworthy order
        return None
    return [str(value) for value in values]


def _resolve_factor(result: Any, term: str) -> _Factor:
    """Resolve ``term`` to factor columns, levels and the reference level."""

    if not isinstance(term, str) or not term.strip():
        raise ValueError(f"term must be a non-empty string, got {term!r}")

    names = list(getattr(result, "param_names", []) or [])
    term_variable = _variable_of(term)

    matched: list[tuple[int, str, str]] = []
    for position, name in enumerate(names):
        parsed = _parse_factor_column(name)
        if parsed is None:
            continue
        base, level = parsed
        if _variable_of(base) != term_variable:
            continue
        matched.append((position, base, level))

    if not matched:
        coded = next(
            (
                coding
                for name in names
                if _mentions_factor(name, term_variable)
                for coding in [_unsupported_coding(name)]
                if coding is not None
            ),
            None,
        )
        if coded is not None:
            raise UnsupportedError(
                f"term {term!r} is coded with a non-treatment contrast "
                f"([{coded}.<level>] columns), whose coefficients are already "
                "contrasts of level means rather than level means; "
                "contrast/pwcompare cannot weight them. Refit with treatment "
                f"(dummy) coding: C({term_variable}, base=...) or a plain "
                f"C({term_variable})."
            )
        if any(_mentions_factor(name, term_variable) for name in names):
            raise UnsupportedError(
                f"terms such as {term!r} that only appear in interactions are not "
                "supported by contrast/pwcompare; refit the model with the factor "
                "as a main effect"
            )
        raise KeyError(
            f"term {term!r} does not match any factor column; "
            f"available coefficients: {names}"
        )

    matched.sort(key=lambda item: item[0])
    design_base = matched[0][1]
    columns: dict[str, int] = {}
    column_names: dict[str, str] = {}
    for position, base, level in matched:
        if base != design_base:
            raise UnsupportedError(
                f"term {term!r} matches columns built from two different factor "
                f"specifications ({design_base!r} and {base!r}); "
                "contrast/pwcompare needs a single factor term"
            )
        if level in columns:
            raise UnsupportedError(
                f"the design matrix has more than one column for level {level!r} "
                f"of {term_variable!r}; contrasts of duplicated factor columns are "
                "not implemented"
            )
        columns[level] = position
        column_names[level] = names[position]

    observed = _observed_levels(getattr(result, "model_data", None), term_variable)
    if observed is not None:
        levels = list(observed)
        for level in columns:
            if level not in levels:
                levels.append(level)
        missing = [level for level in levels if level not in columns]
        if len(missing) == 1:
            reference: str | None = missing[0]
        elif not missing:
            # Every level has a column: nothing was omitted, so there is no
            # reference level and a reference contrast is undefined.
            reference = None
        else:
            raise UnsupportedError(
                f"cannot tell which level of {term_variable!r} is the reference "
                f"level; levels without a design column: {missing}. Only treatment "
                "(dummy) coding with exactly one reference level is supported."
            )
        complete = True
    else:
        # Without the estimation sample the only evidence left is patsy's own
        # spelling of the reference level.  If that is absent too, the omitted
        # level is simply unknown: reporting a contrast table over an incomplete
        # level set would silently drop comparisons, so refuse instead.
        levels = list(columns)
        literal = _reference_literal(design_base)
        if literal is not None and literal not in columns:
            levels = [literal] + levels
            reference = literal
            complete = True
        else:
            reference = None
            complete = False

    return _Factor(
        variable=term_variable,
        design_base=design_base,
        levels=levels,
        columns=columns,
        reference=reference,
        complete=complete,
        column_names=column_names,
    )


# ------------------------------------------------------------------ weights
def _unit_weights(factor: _Factor, n_params: int, level: str) -> np.ndarray:
    """Weight vector selecting one level's coefficient (zeros if it is the base)."""

    weights = np.zeros(n_params, dtype=float)
    position = factor.columns.get(level)
    if position is not None:
        weights[position] = 1.0
    return weights


def _difference(
    factor: _Factor, n_params: int, high: str, low: str
) -> np.ndarray:
    """Weights for ``level_high - level_low``."""

    return _unit_weights(factor, n_params, high) - _unit_weights(
        factor, n_params, low
    )


def _term_text(factor: _Factor, level: str) -> str:
    """Write one level the way it is estimated.

    A level with its own design column is written as that column's name.  The
    reference level of a treatment-coded factor has no column (its coefficient
    is identically zero), so it is written as ``base(...)`` rather than
    pretending a column exists.
    """

    column = factor.column_names.get(level)
    if column is not None:
        return column
    return f"base({factor.variable}[{level}])"


def _require_complete_levels(factor: _Factor, command: str) -> None:
    """Refuse to contrast a level set that is known to be incomplete.

    Dropping an unlisted level would silently shrink a ``pwcompare`` table and
    bias a grand-mean contrast, so a partial level set is a loud error.
    """

    if factor.complete:
        return
    raise UnsupportedError(
        f"{command}: the complete level set of {factor.variable!r} is unknown. "
        "The result carries no estimation sample and the design columns do not "
        "record which level was used as the reference, so the omitted level "
        "cannot be listed. Refit with an explicit reference "
        f"(C({factor.variable}, base=...)) or pass a result that retains "
        "model_data."
    )


# ------------------------------------------------------------------ methods
def _normalize_method(method: Any) -> str:
    if not isinstance(method, str):
        raise UnsupportedError(
            f"contrast: method must be one of {sorted(_METHODS)}, got {method!r}"
        )
    key = method.strip().lower().rstrip(".")
    if key in _METHODS:
        return key
    if key in _KNOWN_UNSUPPORTED_METHODS:
        raise UnsupportedError(
            f"contrast: the {_KNOWN_UNSUPPORTED_METHODS[key]} contrast {method!r} "
            f"is not implemented; supported methods are "
            f"{sorted(_METHODS)} (reference, grand mean, adjacent)"
        )
    raise UnsupportedError(
        f"contrast: unknown method {method!r}; supported methods are "
        f"{sorted(_METHODS)} (reference, grand mean, adjacent)"
    )


def _check_adjust(adjust: Any) -> None:
    """Reject every multiple-comparison adjustment instead of ignoring it."""

    if adjust is None:
        return
    raise UnsupportedError(
        "pwcompare: multiple-comparison adjustment is not implemented; the only "
        f"supported value is adjust=None (Stata's adjust(none)), got {adjust!r}. "
        "Refusing to report unadjusted p-values as if they had been corrected."
    )


def _run(
    result: Any,
    comparisons: list[tuple[str, str, np.ndarray]],
    *,
    level: float,
) -> list[LincomResult]:
    params = np.asarray(result.params, dtype=float).reshape(-1)
    cov = np.asarray(result.cov_params(), dtype=float)
    dist = "z" if str(getattr(result, "dist", "t")) == "z" else "t"
    df_resid = float(result.df_resid)

    return [
        run_lincom(
            params,
            cov,
            weights,
            dist=dist,
            df_resid=df_resid,
            expression=expression,
            level=level,
            label=label,
        )
        for label, expression, weights in comparisons
    ]


# -------------------------------------------------------------------- API
def contrast(
    result: Any,
    term: str,
    method: str = "r",
    *,
    level: float = 0.95,
) -> list[LincomResult]:
    """ANOVA-style contrasts of a factor variable's estimated means.

    Parameters
    ----------
    result:
        Any fitted result carrying ``params``, ``cov_params()``, ``param_names``,
        ``dist`` and ``df_resid``.
    term:
        The factor's design-matrix prefix, e.g. ``"C(g)"`` or ``"g"``.  It is
        matched on the data variable, so ``"C(g)"`` also finds the columns of a
        model fitted with ``C(g, base='a')``.
    method:
        ``"r"`` reference (each non-reference level minus the reference level),
        ``"g"`` grand mean (each level minus the unweighted average of all
        levels), or ``"a"`` adjacent (``level[i+1] - level[i]``).  Anything else
        raises :class:`UnsupportedError`.
    level:
        Confidence level as a fraction (Stata's ``level(95)`` -> ``0.95``).

    Returns
    -------
    list[LincomResult]
        One row per comparison, in level order: ``L - 1`` rows for ``"r"`` and
        ``"a"``, ``L`` rows for ``"g"``.
    """

    method_key = _normalize_method(method)
    factor = _resolve_factor(result, term)
    _require_complete_levels(factor, "contrast")
    n_params = np.asarray(result.params, dtype=float).reshape(-1).size
    levels = factor.levels

    comparisons: list[tuple[str, str, np.ndarray]] = []

    if method_key == "r":
        reference = factor.reference
        if reference is None:
            raise UnsupportedError(
                f"contrast: cannot determine the reference level of "
                f"{factor.variable!r} (no level is omitted from the design "
                "matrix); use method='g', method='a', or pwcompare"
            )
        for name in levels:
            if name == reference:
                continue
            comparisons.append(
                (
                    f"{name} vs {reference}",
                    f"{_term_text(factor, name)} - {_term_text(factor, reference)}",
                    _difference(factor, n_params, name, reference),
                )
            )
    elif method_key == "a":
        # Stata's adjacent contrast reports ``low vs high``: the *earlier* level
        # minus the later one, so its sign is the opposite of the reference
        # contrast.  Verified against StataNow 19.5, where ``contrast a.g`` on a
        # 3-level factor prints "(0 vs 1) -.8134023" while ``contrast r.g``
        # prints "(1 vs 0) .8134023" for the same pair.
        for low, high in zip(levels, levels[1:]):
            comparisons.append(
                (
                    f"{low} vs {high}",
                    f"{_term_text(factor, low)} - {_term_text(factor, high)}",
                    _difference(factor, n_params, low, high),
                )
            )
    else:  # "g": every level against the unweighted grand mean.
        count = float(len(levels))
        for name in levels:
            weights = _unit_weights(factor, n_params, name)
            for other in levels:
                weights = weights - _unit_weights(factor, n_params, other) / count
            comparisons.append(
                (
                    f"{name} vs grand mean",
                    f"{_term_text(factor, name)} - grand mean of "
                    f"{factor.design_base}",
                    weights,
                )
            )

    return _run(result, comparisons, level=level)


def pwcompare(
    result: Any,
    term: str,
    *,
    level: float = 0.95,
    adjust: str | None = None,
) -> list[LincomResult]:
    """All pairwise differences between the levels of a factor variable.

    Row ``(i, j)`` with ``j > i`` reports ``level_j - level_i``, so a factor with
    ``L`` levels yields ``L * (L - 1) / 2`` rows in that order.

    Parameters
    ----------
    result, term, level:
        As in :func:`contrast`.
    adjust:
        Multiple-comparison adjustment.  Only ``None`` (Stata's ``adjust(none)``,
        also accepted spelled out) is implemented: every real correction raises
        :class:`UnsupportedError` rather than being silently dropped.

    Returns
    -------
    list[LincomResult]
        ``L * (L - 1) / 2`` comparisons.
    """

    _check_adjust(adjust)
    factor = _resolve_factor(result, term)
    _require_complete_levels(factor, "pwcompare")
    n_params = np.asarray(result.params, dtype=float).reshape(-1).size

    comparisons: list[tuple[str, str, np.ndarray]] = []
    for index, low in enumerate(factor.levels):
        for high in factor.levels[index + 1 :]:
            comparisons.append(
                (
                    f"{high} vs {low}",
                    f"{_term_text(factor, high)} - {_term_text(factor, low)}",
                    _difference(factor, n_params, high, low),
                )
            )

    return _run(result, comparisons, level=level)
