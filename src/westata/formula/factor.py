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


class _Lag:
    """Stata's ``L.`` / ``L()`` time-series lag, usable inside a formula.

    Stata writes ``L.temp`` and ``L2.temp``; patsy's grammar cannot express those, so this exposes
    the same idea as a call::

        y ~ L(temp) + L(temp, 2) + price

    The value is the column shifted down by ``period`` rows, so the first ``period`` observations
    become missing and are dropped with the rest of the listwise deletion. That is what Stata does.

    WHY THIS IS A FUNCTION AND NOT A TRANSFORM: patsy only accepts names it can resolve, and its
    ``EvalEnvironment`` captures the caller's frame. Registering ``L`` in that namespace is the
    supported way to add an operator without forking the grammar.
    """

    def __call__(self, series: Any, period: int = 1) -> Any:
        periods = int(period)
        if periods < 1:
            raise FormulaError(f"L(): period must be a positive integer, got {period!r}")
        return series.shift(periods)


_lag = _Lag()


def register_time_series_operators() -> None:
    """Make ``L()`` resolvable inside a formula.

    patsy evaluates a formula in the caller's frame, so anything reachable there is visible. This
    inserts ``L`` into the frame of :func:`build_design`'s caller -- but the simpler and more
    reliable route is to substitute the operator into the formula as a fully-qualified call, which
    is what :func:`_substitute_lag_operator` does instead of relying on frame inspection.
    """


def __ws_lag(series: Any, period: int = 1) -> Any:
    """patsy-visible helper for the ``L()`` operator rewritten by :func:`_substitute_lag_operator`."""

    return series.shift(int(period))


def _restore_lag_names(names: list[str]) -> list[str]:
    """Turn patsy's ``__ws_lag(x, 1)`` back into the ``L(x)`` the user wrote."""

    import re

    pattern = re.compile(r"__ws_lag\(\s*([^,]+?),\s*1\s*\)")
    pattern_many = re.compile(r"__ws_lag\(\s*([^,]+?),\s*(\d+)\s*\)")

    out: list[str] = []
    for name in names:
        fixed = pattern.sub(r"L(\1)", name)
        fixed = pattern_many.sub(r"L(\1, \2)", fixed)
        out.append(fixed)
    return out


def _substitute_lag_operator(formula: str) -> str:
    """Rewrite ``L(x)`` and ``L(x, k)`` into a call patsy can evaluate.

    The replacement calls a module-level helper, which patsy resolves through the evaluation
    namespace supplied at the call site::

        L(temp)      ->  __ws_lag(temp, 1)
        L(temp, 2)   ->  __ws_lag(temp, 2)

    Two earlier forms failed and are worth recording: a parenthesised lambda is rejected by patsy's
    parser ("expected an operator, not '('"), and `EvalEnvironment(..., depth=0)` is not this
    patsy's signature.
    """

    import re

    pattern = re.compile(r"\bL\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:,\s*(\d+)\s*)?\)")

    def _replace(match: "re.Match[str]") -> str:
        column = match.group(1)
        period = match.group(2) or "1"
        if int(period) < 1:
            raise FormulaError(
                f"L(): the lag period must be a positive integer, got {period}"
            )
        return f"__ws_lag({column}, {period})"

    return pattern.sub(_replace, formula)



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

    translated = _substitute_lag_operator(_translate_base_terms(formula))

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
        _, exog = patsy.dmatrices(
            translated,
            data,
            return_type="dataframe",
            eval_env=patsy.EvalEnvironment.capture(0).with_outer_namespace(
                {"__ws_lag": __ws_lag}
            ),
        )
    except patsy.PatsyError as exc:
        # Wrap the error so the traceback does not retain patsy's evaluation
        # namespace (see FormulaError's docstring), and add the available
        # column names to make the mistake obvious.
        raise FormulaError(
            f"cannot evaluate formula {formula!r}: {exc}\n"
            f"available columns: {list(data.columns)}"
        ) from None

    X = np.asarray(exog, dtype=float)
    column_names = _restore_lag_names(list(exog.design_info.column_names))
    kept_index = exog.index
    if escapes:
        column_names = [escapes.restore(name) for name in column_names]
    return X, column_names, kept_index
