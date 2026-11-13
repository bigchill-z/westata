"""Results containers, Stata-style summaries and regression tables.

Public surface:

- :class:`StataResults` -- statsmodels-compatible results object.
- :class:`EReturn` -- Stata ``e()`` scalar mapping.
- :func:`render_estimates_table` -- Stata's ``estimates table`` layout.
- :func:`render_esttab` / :func:`etable_as_stata` -- the ``esttab`` layout.

``__all__`` names only what this module actually IMPORTS.  It used to list
``render_estimates_table`` while importing only ``etable_as_stata`` and
``render_esttab``, so ``from westata.results import *`` raised ``AttributeError``
and ``westata.results.render_estimates_table`` did not exist (R5.1).
"""

from __future__ import annotations

from westata.results.base import StataResults
from westata.results.esttab import etable_as_stata, render_esttab
from westata.results.estimates_table import render_estimates_table
from westata.results.stata_return import EReturn

__all__ = [
    "EReturn",
    "StataResults",
    "etable_as_stata",
    "render_esttab",
    "render_estimates_table",
]
