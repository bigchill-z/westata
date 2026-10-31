"""Reusable harness for replicating published Stata estimates with westata.

The three modules are deliberately separable so a replication can use only what
it needs:

* :mod:`replication.lib.stata_session` -- one process-wide embedded Stata, with
  the missing-value and working-directory hazards closed.
* :mod:`replication.lib.compare` -- name-based coefficient alignment and a loud
  failure when the two parameter vectors cannot be paired.
* :mod:`replication.lib.report` -- the two markdown tables a replication log
  needs, written to disk.

Nothing here imports ``westata`` itself: the comparison accepts any
statsmodels-shaped result, which keeps the harness usable for estimators the
package does not implement yet.
"""

from __future__ import annotations

from .compare import AlignmentError, Record, Row, align_parameters, compare_capture, verdict_for
from .report import Published, difference_table, replication_table, write_markdown
from .stata_session import MISSING_SENTINEL, Capture, StataError, StataSession

__all__ = [
    "MISSING_SENTINEL",
    "AlignmentError",
    "Capture",
    "Published",
    "Record",
    "Row",
    "StataError",
    "StataSession",
    "align_parameters",
    "compare_capture",
    "difference_table",
    "replication_table",
    "verdict_for",
    "write_markdown",
]
