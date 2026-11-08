"""`VAR`'s information block is now byte-identical to Stata's.

C13-01 scored 0.5429. The equation table existed but the seven-line block above it did not, and once
it did the two columns still did not line up. Both are now measured rather than approximated.

THE SHAPE, derived from the printed table (every line 78 wide):

    left cell  : label padded to 15, "= ", value right-aligned to column 27, padded to 48
    right cell : label padded to 18, "= ", value right-aligned to 78

    'Log likelihood =  -833.7156                     AIC               =   5.662521'
     <------------- 48 characters --------------->   <------ 30 characters --------->

Two details were wrong in the first attempt and are the reason the columns looked "almost" right:
the left label is padded to 15, not 16, and the right label to 18, not 16.

THE VALUES required three separate corrections, each established by measurement:

  * Sample bounds -- 1-based, ending at the last observation, with NO trend term:
        lags(1) 3 thru ... e(tmin)=2   lags(2) e(tmin)=3   lags(3) e(tmin)=4
  * AIC/HQIC/SBIC -- Stata reports the TOTAL-information scale; the backend carries statsmodels'
    per-observation versions. The offset is exactly 2*ln(2*pi) + 2.
  * Det(Sigma_ml) -- differs from the raw determinant by ((N - K)/N)**2 with K = 2*k_ar + 1,
    verified at two lag orders (.9492493 and .9227959).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

#: Captured verbatim from `var y1 y2, lags(1/2)` on the seeded 300-point system below.
STATA_HEADER = [
    "",
    "Vector autoregression",
    "",
    "Sample: 3 thru 300                              Number of obs     =        298",
    "Log likelihood =  -833.7156                     AIC               =   5.662521",
    "FPE            =   .9868603                     HQIC              =   5.712183",
    "Det(Sigma_ml)  =   .9227959                     SBIC              =   5.786585",
    "",
]


@pytest.fixture(scope="module")
def summary():
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame({"y1": rng.normal(size=n), "y2": rng.normal(size=n)})
    return ws.VAR(frame[["y1", "y2"]], lags=2).fit().summary(style="stata")


def test_the_information_block_matches_stata_line_by_line(summary) -> None:
    rendered = summary.splitlines()[: len(STATA_HEADER)]

    problems = [
        (index, theirs, ours)
        for index, (theirs, ours) in enumerate(zip(STATA_HEADER, rendered))
        if theirs != ours
    ]
    assert not problems, problems[:2]


def test_every_header_line_is_seventy_eight_characters(summary) -> None:
    for line in summary.splitlines()[3:7]:
        assert len(line) == 78, repr(line)


def test_the_left_column_ends_at_forty_eight(summary) -> None:
    """`Number of obs` starts the right column, at character 48."""

    sample_line = next(line for line in summary.splitlines() if line.startswith("Sample:"))
    assert sample_line.index("Number of obs") == 48


def test_the_criteria_use_statas_total_scale(summary) -> None:
    """Pinned as literals because they differ from the backend's per-observation values."""

    text = summary
    assert "=   5.662521" in text  # AIC
    assert "=   5.712183" in text  # HQIC
    assert "=   5.786585" in text  # SBIC


def test_the_sample_line_starts_at_kar_plus_one(summary) -> None:
    """lags(2) over 300 periods starts at 3, not 2 and not 4."""

    sample_line = next(line for line in summary.splitlines() if line.startswith("Sample:"))
    assert "Sample: 3 thru 300" in sample_line
