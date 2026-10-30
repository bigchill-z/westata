"""Exception hierarchy for westata.

Every failure mode that could otherwise produce silently wrong numbers has a
dedicated exception type.  Callers can catch :class:`WestataError` to handle
all westata-specific failures at once.
"""

from __future__ import annotations


class WestataError(Exception):
    """Base class for all westata errors."""


class ConsistencyError(WestataError, ValueError):
    """Raised when a result cannot be guaranteed to be Stata-conformant.

    This is deliberately loud: westata never silently returns a number that it
    knows may diverge from Stata.  The message must say what diverged and which
    evidence level the quantity has.

    Also a ``ValueError`` for the same reason ``FormulaError`` is: the invalid input is a
    value (too few clusters, a degenerate design) and callers legitimately write
    ``except ValueError``.  Adopting both bases lets a caller catch the specific class OR
    the broad one, and -- more importantly -- lets ``except WestataError`` finally catch
    the VCE refusals, which it did not.

    Why that mattered: a single-cluster clustered VCE raised this from the logit path but
    a plain ``ValueError`` from OLS/IV, so one library inconsistency produced two
    different exception types for the same user error.  With both bases in place the sites
    could be unified without breaking ``except ValueError``.
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
