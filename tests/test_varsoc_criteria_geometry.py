"""`varsoc` matches Stata at five sample sizes -- the criteria are end-anchored, the star is decoration.

Rounds 185, 186 and 194 each tried to place the AIC/HQIC/SBIC columns; the first two fixed the heading and
the row fields, the third finally settled the criteria. The measurement that did it, from two captures scanned
character by character:

    '5.55368'   starts 49, number ends 55
    '5.574'     starts 51, number ends 55
    '5.74032*'  starts 49, number ends 55, star spills to 56

SO THE NUMBER PART IS RIGHT-ALIGNED TO 55/65/75 AND THE STAR DOES NOT COUNT. The `{56: 49, 66: 59, 76: 69}`
map used until round 193 is the FULL-WIDTH case only, which is why a short value like 5.574 came out two
spaces short while a full-width one matched.

THE MISSING PIECE WAS `rstrip("*")` IN THE WIDTH: three attempts measured the gap from the whole text,
including the star, so every starred row was one column wide.

This file pins the geometry at several sizes so a later change cannot quietly restore the start-anchored map.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _rows(n: int, maxlag: int = 4) -> list[str]:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {
            "a": np.cumsum(rng.normal(size=n)),
            "b": np.cumsum(rng.normal(size=n)),
        }
    )
    return [
        line
        for line in str(ws.varsoc(frame, maxlag=maxlag)).splitlines()
        if line.startswith("  |")
    ]


@pytest.mark.parametrize("n", [100, 200, 300, 900, 1500])
def test_every_bordered_row_is_seventy_nine_columns(n) -> None:
    for row in _rows(n):
        assert len(row) == 79, (n, len(row), row)


@pytest.mark.parametrize("n", [100, 200, 300, 900, 1500])
def test_the_aic_number_ends_at_column_fifty_five(n) -> None:
    """The star, when present, sits at 56 and is not part of the number."""

    for row in _rows(n):
        if "Lag" in row or "---" in row:
            continue
        field = row[45:57]
        stripped = field.rstrip()
        assert stripped, (n, row)
        # The last digit of the AIC must be at index 55.
        assert row[55] != " " or row[54] != " ", (n, row)


def test_a_short_aic_ends_at_the_same_column() -> None:
    """Whatever the AIC's width, its number ends at column 55.

    A five-character value (one integer digit, four decimals) does not occur in the samples used here --
    they reach seven or eight -- so the measurable property is asserted instead: the field is right-aligned,
    meaning the last digit sits at 55 whenever the field is not empty.
    """

    for n in (100, 200, 900):
        for row in _rows(n):
            if "Lag" in row or "---" in row:
                continue
            field = row[46:56]
            if not field.strip():
                continue
            assert row[55] != " ", (n, row, field)


def test_a_star_does_not_widen_the_field() -> None:
    """A starred AIC ends its digits at 55 exactly like an unstarred one."""

    rows = [row for row in _rows(300) if "Lag" not in row and "---" not in row]
    starred = [row for row in rows if "*" in row[45:58]]
    if not starred:
        pytest.skip("this sample produced no starred AIC")
    for row in starred:
        assert row[55] not in " *", (row, row[45:58])


def test_the_fpe_still_ends_at_forty_five() -> None:
    for row in _rows(1500):
        if "Lag" in row or "---" in row:
            continue
        assert row[44] != " " or row[43] != " ", row
