"""Significance stars -- the ONE implementation (R6.1).

Three modules drew stars and all three agreed only by luck:

* ``results/estimates_table.py`` took a configurable threshold list (Stata's
  ``star(0.1 0.05 0.01)``);
* ``report/etable.py`` and ``results/esttab.py`` each hard-coded 5%/1%/0.1%.

The three tables therefore printed identical stars for the default settings and
would have diverged the moment one of them was corrected -- and they were three
places to correct.  The configurable rule now lives here; the fixed-default callers
pass :data:`ESTTAB_DEFAULT_THRESHOLDS`.
"""

from __future__ import annotations

from typing import Sequence

#: ``esttab``'s own default: ``* p<0.05, ** p<0.01, *** p<0.001``.
#:
#: Deliberately NOT ``estimates table``'s ``star(0.1 0.05 0.01)`` (10%/5%/1%): the
#: two Stata commands ship different defaults, and ESTTAB_SPEC.md records the
#: measurement.
ESTTAB_DEFAULT_THRESHOLDS: tuple[float, ...] = (0.05, 0.01, 0.001)


def stars(pvalue: float, thresholds: Sequence[float] = ESTTAB_DEFAULT_THRESHOLDS) -> str:
    """The star run for one coefficient, given thresholds in any order.

    Stata's ``star(0.1 0.05 0.01)`` yields ``***`` when p < .01, ``**`` when p < .05 and
    ``*`` when p < .1.  The sorted list is walked ascending, so the largest threshold
    gets one star.
    """

    if pvalue != pvalue:  # NaN
        return ""
    ordered = sorted(float(level) for level in thresholds)
    for index, level in enumerate(ordered):
        if pvalue < level:
            return "*" * (len(ordered) - index)
    return ""
