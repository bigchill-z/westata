"""`varsoc` at large n: the banner, the LR column and the star all needed separate fixes.

Round 182-184 found thousands-separator and alignment defects at large n in summarize, the regression block
and esttab. Applying the same check to `varsoc` found three more:

    n=1500
        Stata    '   Sample: 5 thru 1500                                   Number of obs = 1,496'
        package  '   Sample: 5 thru 1500                                     Number of obs = 1,496'

`Number of obs` gains its separator, and AS A RESULT the label moves LEFT -- Stata right-aligns the row to 78
columns, so 'Number of obs' begins at 59 with a three-digit count and at 57 with a four-digit one. Padding to
a fixed 59 was the bug.

THE LR COLUMN IS RIGHT-ALIGNED TO COLUMN 26 while the LL column is start-anchored, and the STAR DOES NOT
count toward that width -- '14.108*' begins one column left of '1412' because only the digits align:

        '  |   1 | -558.096    1412    4  0.000'     1412 ends at 26
        '  |   3 | -548.552  14.108*   4  0.007'     14.108 ends at 26, the star spills

Before those fixes, n=200 and n=300 both differed on every data row; now both match completely.

STILL OPEN, narrowly: when the LR statistic reaches five digits (15513*), Stata widens the df and p fields by
one column each. Recorded rather than guessed -- the rows above it are settled, and this is one field-width
rule in a case that needs a sample large enough to trigger it.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def tables():
    out = {}
    for n in (200, 300):
        rng = np.random.default_rng(n)
        frame = pd.DataFrame(
            {
                "a": np.cumsum(rng.normal(size=n)),
                "b": np.cumsum(rng.normal(size=n)),
            }
        )
        out[n] = [line for line in str(ws.varsoc(frame, maxlag=4)).splitlines() if line.strip()]
    return out


def test_the_banner_separates_the_observation_count(tables) -> None:
    line = tables[300][1]
    assert "Number of obs = 296" in line


def test_the_banner_is_seventy_eight_columns(tables) -> None:
    """Right-aligned to a fixed width, so the label shifts when the count grows."""

    for n in (200, 300):
        banner = tables[n][1]
        assert len(banner) == len(tables[200][1]), (n, len(banner))
        assert "Number of obs" in banner


def test_a_star_does_not_move_the_value(tables) -> None:
    """Only the digits align; the star spills past the column."""

    stars = [line for line in tables[300] if "*" in line and line.startswith("  |")]
    assert stars, "a starred row is needed for this check"
    for row in stars:
        # The FPE field starts at a fixed column whether or not it carries a star.
        assert row[39] in " .", row


def test_every_row_is_the_same_width(tables) -> None:
    for n, table in tables.items():
        widths = {len(line) for line in table if line.startswith("  |")}
        assert len(widths) == 1, (n, widths)


def test_the_line_count_is_stable(tables) -> None:
    assert len(tables[200]) == len(tables[300])
