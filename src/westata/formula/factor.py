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
        raise patsy.PatsyError(
            "cannot build a design matrix from an empty DataFrame"
        )

    translated = _translate_base_terms(formula)
    _, exog = patsy.dmatrices(translated, data, return_type="dataframe")

    X = np.asarray(exog, dtype=float)
    column_names = list(exog.design_info.column_names)
    kept_index = exog.index
    return X, column_names, kept_index
