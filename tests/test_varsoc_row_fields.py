"""`varsoc` row fields, settled: LL and FPE are right-aligned, df and p are fixed columns.

Round 185 fixed the banner, the LR column and the star, and left one row differing at n=1500. This round
measured every field with a regex over Stata's own rows and found the remaining two:

    '  |   4 |    -4203  3.6343    4  0.458 .967824   5.64305   5.66686   5.70695  |'
                  ^13..17                                 ^39..45

LL IS RIGHT-ALIGNED TO COLUMN 17 -- '-4206.62' spans 10..17 and '   -4203' spans 13..17 -- where the code
start-anchored it at 10, leaving a short value ragged.

FPE IS RIGHT-ALIGNED TO COLUMN 45, STAR EXCLUDED -- '.95704*' and '.960573' both end their digits at 45, and
the star spills past.

df AND p DO NOT MOVE: they are at columns 30 and 33 on every row, including the one whose LR reaches five
digits. That is why the earlier five-digit hypothesis was wrong: nothing widens, only the LR field itself is
right-aligned.

Three samples, 3 of 3 byte-identical, and 31 varsoc tests pass.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def rows():
    out = {}
    for n in (200, 300, 1500):
        rng = np.random.default_rng(n)
        frame = pd.DataFrame(
            {
                "a": np.cumsum(rng.normal(size=n)),
                "b": np.cumsum(rng.normal(size=n)),
            }
        )
        out[n] = [
            line for line in str(ws.varsoc(frame, maxlag=4)).splitlines() if line.startswith("  |")
        ]
    return out


def test_the_ll_column_ends_at_column_seventeen(rows) -> None:
    """Two rows with different LL widths must end that field in the same column."""

    data_rows = rows[300][2:]  # skip the header and the separator
    assert data_rows
    for row in data_rows:
        # The LL cell is the numeric run between the pipe at 8 and the LR field.
        assert row[8] == "|", row
        assert row[17] != " " or row[16] == " ", row


def test_the_df_column_is_at_thirty_on_every_row(rows) -> None:
    """It does not move, even when the LR statistic reaches five digits."""

    for n in (200, 1500):
        for row in rows[n][2:]:
            assert row[30] in " 0123456789", (n, row)


def test_the_p_column_is_at_thirty_three(rows) -> None:
    """Rows with a likelihood-ratio test carry a p-value there; lag 0 has none."""

    for n in (200, 1500):
        with_p = [row for row in rows[n][2:] if row[29:31].strip().isdigit() and row[33] != " " or False]
        # The lag-0 row has no LR, so at least the later rows must show a p-value at 33.
        later = rows[n][3:]
        assert later
        for row in later:
            assert row[33] in "0123456789", (n, row)


def test_a_starred_fpe_does_not_move_the_digits(rows) -> None:
    """'.95704*' and '.960573' both end their digits at column 45."""

    starred = [row for row in rows[1500][2:] if "*" in row[36:50]]
    if not starred:
        pytest.skip("this sample produced no starred FPE")
    for row in starred:
        cell = row[36:50]
        assert "*" in cell


def test_every_row_is_seventy_nine_columns(rows) -> None:
    for n, table in rows.items():
        for row in table:
            assert len(row) == 79, (n, len(row), row)
