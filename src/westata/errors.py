"""Exception hierarchy for westata.

Every failure mode that could otherwise produce silently wrong numbers has a
dedicated exception type.  Callers can catch :class:`WestataError` to handle
all westata-specific failures at once.
"""

from __future__ import annotations


class WestataError(Exception):
    """Base class for all westata errors."""


class ConsistencyError(WestataError):
    """Raised when a result cannot be guaranteed to be Stata-conformant.

    This is deliberately loud: westata never silently returns a number that it
    knows may diverge from Stata.  The message must say what diverged and which
    evidence level the quantity has.
    """


class ConvergenceError(WestataError):
    """Raised when an iterative estimator fails to converge under ``strict=True``."""


class WeightError(WestataError):
    """Raised for missing/unsupported/ambiguous Stata weight specifications."""


class FormulaError(WestataError, ValueError):
    """Raised when a model formula cannot be evaluated on the given data.

    This deliberately wraps ``patsy.PatsyError``.  Patsy evaluates formulas
    through its own namespace object held in the traceback frame, and a raw
    ``PatsyError`` therefore keeps that frame alive; several exception
    formatters (including pytest's) then fail while introspecting it and the
    user sees an internal rendering error instead of the actual message.
    Re-raising a plain exception with the original text keeps the diagnostic.
    """


class UnsupportedError(WestataError, NotImplementedError):
    """Raised when a capability exists in Stata but is not implemented yet."""
