"""The single place that maps a parameter name onto Stata's printed spelling.

Rounds 123-124 found the SAME rename implemented twice -- once in ``vce.py`` and once
in ``vif.py`` -- and they had drifted: the VIF copy handled the constant but not the
interaction separator, so one analysis printed ``x1#x2`` in the coefficient table and
``x1:x2`` in the VIF table.

A per-module copy is exactly the setup that lets this happen again, so the mapping
lives here and the postestimation modules import it.

What the rename covers, and why each part is not a guess:

* ``Intercept`` -> ``_cons``: patsy's name against Stata's, visible in every captured
  table (``p10_outputs.txt:19`` ends with ``_cons``).
* ``a:b`` -> ``a#b``: Stata separates interaction terms with ``#``.  The ORDER does not
  change, only the separator (``base.py:378`` makes the same point for ``e(b)``).
* a leading ``/`` is DROPPED: ``e(b)`` stores ``cut1``/``lnalpha`` without the slash
  (``ologit.json``, ``nbreg.json``), and ``estat vce``/``estat vif`` print the stored
  names.
"""

from __future__ import annotations

#: Parameter names that patsy, statsmodels or this package use for the constant term.
#:
#: This is the ONE definition (R6.1).  Before it, four different sets were in use and
#: two of them were wrong in ways that mattered:
#:
#: * ``modeltests.py`` listed ``"Intercept"`` (capitalised) in a set it then compared
#:   against ``name.strip().lower()``, so that entry could never match;
#: * ``panel_nonlinear.py`` added ``"1"`` -- the intercept is sometimes named ``1``
#:   there because a formula with an explicit intercept term produces that column.
#:
#: Both spellings are folded in here, and every caller lower-cases before comparing
#: (see :func:`is_constant_name`), so the case bug cannot come back.
CONSTANT_SPELLINGS = frozenset({"intercept", "intercept_", "const", "_cons", "1"})

#: Separator differences between the two vocabularies.
PATSY_INTERACTION = ":"
STATA_INTERACTION = "#"


def is_constant_name(name: str) -> bool:
    """Whether a parameter name denotes the constant term."""

    return str(name).strip().lower() in CONSTANT_SPELLINGS


def stata_display_name(name: str) -> str:
    """Map a stored parameter name to the spelling Stata PRINTS.

    This is for TABLES (coefficient table, ``estat vce``, ``estat vif``).  It is not
    the same function as ``stata_param_names``, which produces the names ``e(b)``
    STORES -- that one keeps the ordering logic and the refusal for ambiguous factor
    terms, and deliberately strips the slash because ``e(b)`` has no slash.
    """

    text = str(name)
    if is_constant_name(text):
        return "_cons"
    if text.startswith("/"):
        # Auxiliary parameters: stored as ``lnsigma``, printed the same way by the
        # matrix commands.  The coefficient TABLE adds the slash back for display,
        # which is a separate decision made where that table is drawn.
        return text[1:].replace(PATSY_INTERACTION, STATA_INTERACTION)
    return text.replace(PATSY_INTERACTION, STATA_INTERACTION)


def require_iv_structure(result: object, command: str) -> dict:
    """Return the IV structure of ``result`` or explain why the command needs it.

    Four modules (``estat endogenous``, ``estat firststage``, ``estat overid``,
    ``estat weakrobust``) each carried their own copy of this check (R6.1).  The four
    differed only in the command name inside the message, which is exactly the kind of
    duplication that drifts: one copy would gain a clarification and the other three
    would not.  The command name is a parameter now.
    """

    structure = getattr(result, "iv_structure", None)
    if structure is None:
        raise NotImplementedError(
            f"{command} is only defined after an instrumental-variables fit: Stata "
            f"reaches it through ivregress_estat, and it needs the split between "
            f"included exogenous regressors, endogenous regressors and excluded "
            f"instruments; got {getattr(result, 'model_name', None)!r}"
        )
    return dict(structure)
