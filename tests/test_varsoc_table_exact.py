"""arsoc's table is byte-identical to Stata's -- 14 lines, all four rounds of it.

C14-04 scored 0.1429. The criteria values were right from round 79 onward; the frame, the footnotes
and the column anchoring took rounds 94-103.

THE ANCHORING RULE, which took four attempts to state correctly because each wrong version matched
most rows:

  * LL, LR, df, p and FPE are the same width on every row, so start and end move together;
  * AIC, HQIC and SBIC have their END fixed for unstarred rows (56/66/76) and one column further
    for starred ones (57/67/77), while their START stays at 49/59/69 either way;
  * the short value is what exposes it: lag 2 prints 5.574 starting at 51 where lag 1 prints
    5.55368 starting at 49, and both end at 56.

The implementation right-aligns to the end column but refuses to start before the measured start
column, which is the only formulation that satisfied all three rows at once.

THE FPE VALUES came from two separate errors, recorded in round 97: the wrong parameter count in the
formula, and the wrong sample window.
"""

STATA_LINES = [
    '',
    'Lag-order selection criteria',
    '',
    '   Sample: 3 thru 200                                      Number of obs = 198',
    '  +---------------------------------------------------------------------------+',
    '  | Lag |    LL      LR      df    p     FPE       AIC      HQIC      SBIC    |',
    '  |-----+---------------------------------------------------------------------|',
    '  |   0 | -544.467                     .855661*  5.51987*  5.53332*  5.55309* |',
    '  |   1 | -543.814  1.3069    4  0.860 .885083   5.55368   5.59401   5.65332  |',
    '  |   2 | -541.826  3.9766    4  0.409 .903268     5.574   5.64122   5.74007  |',
    '  +---------------------------------------------------------------------------+',
    '   * optimal lag',
    '   Endogenous: y1 y2',
    '    Exogenous: _cons',
]


import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def rendered():
    rng = np.random.default_rng(11)
    n = 200
    frame = pd.DataFrame({"y1": rng.normal(size=n), "y2": rng.normal(size=n)})
    return str(ws.varsoc(frame[["y1", "y2"]], maxlag=2))


def test_every_line_matches_stata(rendered) -> None:
    lines = rendered.splitlines()
    assert len(lines) == len(STATA_LINES)
    problems = [
        (index, theirs, ours)
        for index, (theirs, ours) in enumerate(zip(STATA_LINES, lines))
        if theirs != ours
    ]
    assert not problems, problems[:2]


def test_the_bordered_lines_are_seventy_nine_wide(rendered) -> None:
    for line in rendered.splitlines():
        if line.startswith("  |") or line.startswith("  +"):
            assert len(line) == 79, repr(line)
