"""Results containers and Stata-style summaries.

Public surface:

- :class:`StataResults` — statsmodels-compatible results object.
- :class:`EReturn` — Stata ``e()`` scalar mapping.
"""

from __future__ import annotations

from westata.results.base import StataResults
from westata.results.stata_return import EReturn

__all__ = ["StataResults", "EReturn"]
