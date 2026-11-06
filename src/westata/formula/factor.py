"""Factor handling and design-matrix construction via patsy.

Stata counterpart: the factor-variable notation of ``[R] regress``
(``i.x`` for "factor x", ``ib#.x`` for "factor x with base level #").  The
Python-facing spelling is ``C(x, base=...)``, which is translated to patsy's
``C(x, Treatment(reference=...))`` before the formula is evaluated.

The actual model-matrix construction is delegated to :func:`patsy.dmatrices`
so that the design matrix is identical to ``statsmodels.formula.api``.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import patsy

from westata.errors import FormulaError

# ``C(<plain data expression>, base=<string literal or number>)`` ->
# ``C(<data>, Treatment(reference=<literal>))``.  The reference literal is
# captured verbatim (quotes included) so that the resulting patsy column names
# stay deterministic, e.g. ``C(x, Treatment(reference='b'))[T.a]``.
_BASE_TERM_RE = re.compile(
    r"""
    (?<![\w.])
    C\(
        \s*(?P<data>[^,()]+?)\s*,\s*
        base\s*=\s*
        (?P<reference>
            '(?:[^'\\]|\\.)*'
            | "(?:[^"\\]|\\.)*"
            | [-+]?(?:\d+(?:\.\d*)?|\.\d+)
        )
        \s*\)
    """,
    re.VERBOSE,
)


def _replace_base_term(match: re.Match[str]) -> str:
    """Rewrite one ``C(x, base=v)`` call into explicit patsy contrast syntax."""
    data = match.group("data").strip()
    reference = match.group("reference")
    return f"C({data}, Treatment(reference={reference}))"


def _translate_base_terms(formula: str) -> str:
    """Translate every supported ``C(x, base=v)`` term in ``formula``."""
    return _BASE_TERM_RE.sub(_replace_base_term, formula)


_RESERVED_NAMES = frozenset(
    {
        "inf", "nan", "None", "True", "False", "C", "I", "Q", "and", "or", "not", "in",
        "is", "if", "else", "lambda", "sum", "min", "max", "abs", "len", "list", "dict",
        "set", "all", "any", "float", "int", "str", "bool", "complex", "round", "pow",
    }
)


class _EscapedFormula:
    """Keeps the rename mapping so the result can be restored to the user's spelling."""

    def __init__(self, formula: str, data: Any, aliases: dict[str, str]) -> None:
        self.formula = formula
        self.data = data
        self._aliases = aliases

    def restore(self, name: str) -> str:
        for original, alias in self._aliases.items():
            if alias in name:
                name = name.replace(alias, original)
        return name


def _escape_reserved_names(formula: str, data: Any) -> Any:
    """Rename columns that collide with names the formula evaluator resolves first.

    Returns ``None`` when nothing needs escaping, so the common path is untouched.
    """

    import re

    present = [column for column in data.columns if str(column) in _RESERVED_NAMES]
    if not present:
        return None

    aliases: dict[str, str] = {}
    renamed = data.copy()
    used = {str(column) for column in data.columns}
    for column in present:
        alias = f"__ws_{column}__"
        while alias in used:
            alias = "_" + alias
        used.add(alias)
        aliases[str(column)] = alias
        renamed = renamed.rename(columns={column: alias})

    escaped = formula
    for original, alias in aliases.items():
        escaped = re.sub(rf"(?<![\w.]){re.escape(original)}(?![\w])", alias, escaped)
    return _EscapedFormula(escaped, renamed, aliases)


def build_design(
    formula: str,
    data: pd.DataFrame,
) -> tuple[np.ndarray, list[str], pd.Index]:
    """Build the model matrix for ``formula`` evaluated on ``data``.

    Stata counterpart: ``[R] regress`` model specification with factor
    variables (``i.x`` / ``ib#.x``).  ``C(x, base=v)`` selects the reference
    level, where ``v`` may be a quoted string (``base='b'``, ``base="b"``) or a
    number (``base=2``).

    Parameters
    ----------
    formula:
        Two-sided patsy/statsmodels formula, for example
        ``"y ~ x1 + C(region, base='north') + x1:C(region)"``.
    data:
        DataFrame holding the variables referenced by ``formula``.

    Returns
    -------
    X:
        ``(n_kept, k)`` float ndarray of the design matrix.
    column_names:
        Column names exactly as patsy names them, for example
        ``["Intercept", "C(x, Treatment(reference='b'))[T.a]"]``.
    kept_index:
        Index labels of the observations that survived listwise deletion of
        missing values, for aligning the dependent variable.

    Notes
    -----
    Collinear columns are deliberately *not* dropped here; the estimator is
    responsible for detecting and reporting them as omitted.  Invalid formulas
    raise :class:`patsy.PatsyError` instead of returning an empty matrix.  An
    empty ``DataFrame`` also raises :class:`patsy.PatsyError`: patsy itself
    happily returns a ``0 x k`` matrix for empty numeric data, which would let
    an empty sample leak into estimation.
    """
    if len(data) == 0:
        raise FormulaError(
            "cannot build a design matrix from an empty DataFrame"
        )

    translated = _translate_base_terms(formula)

    # A column named after a Python builtin cannot be used: patsy evaluates the formula in a
    # namespace where "inf" resolves to FLOAT INFINITY before it ever looks at the data, so
    # ``inf`` as a response raised "numbers besides '0' and '1' are only allowed with **".
    # "inf" is the natural name for an inflation series -- macro_swatson.dta uses it -- so this
    # is not a corner case.
    #
    # The columns are renamed to a collision-free alias for the duration of the parse and the
    # names are mapped back afterwards, so the user's formula and the resulting column names both
    # keep the spelling they wrote.
    escapes = _escape_reserved_names(translated, data)
    if escapes:
        translated = escapes.formula
        data = escapes.data
    try:
        _, exog = patsy.dmatrices(translated, data, return_type="dataframe")
    except patsy.PatsyError as exc:
        # Wrap the error so the traceback does not retain patsy's evaluation
        # namespace (see FormulaError's docstring), and add the available
        # column names to make the mistake obvious.
        raise FormulaError(
            f"cannot evaluate formula {formula!r}: {exc}\n"
            f"available columns: {list(data.columns)}"
        ) from None

    X = np.asarray(exog, dtype=float)
    column_names = list(exog.design_info.column_names)
    kept_index = exog.index
    if escapes:
        column_names = [escapes.restore(name) for name in column_names]
    return X, column_names, kept_index
