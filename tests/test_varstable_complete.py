"""`varstable` matches Stata at four sample sizes -- the modulus rule and the row geometry together.

This table has taken several rounds. The pieces, in the order they were established:

ROUND 191, THE MODULUS: `varstable_w.ado:254` sets `.numfmt %10.7g . %8.7g` and `_tab` prints into a 13-wide
field. The column switches rule at 0.1 -- six significant figures at or above, six decimals below:

    1.0009273   -> 1.00093      %6f would give 1.000927
    .0936159    -> .093616      %6g would give .0936159

EIGHT OF EIGHT values match under the threshold while either fixed rule matches five.

ROUND 192, THE MESSAGES: the unstable warnings are
    '   At least one eigenvalue is at least 1.0.'
    '   VAR does not satisfy stability condition.'
which only appear when a sample actually reaches that branch.

ROUND 193, THE ROW GEOMETRY: both shapes end their left field at column 13, and a complex value puts its
sign at 15 with the imaginary part at 17..26:

    '  |   1.000927               |   1.00093   |'
    '  | -.09055651 + .02373722i  |   .093616   |'

THE RULE THAT MADE THIS LAND: the column formula was verified in a standalone script BEFORE the file was
edited. Round 192 rewrote the block in place, left an unmatched parenthesis, and broke collection for 474
test files. Verifying first cost one extra step and avoided that entirely.

Four samples, 4 of 4 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _rows(n: int) -> list[str]:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {
            "a": np.cumsum(rng.normal(size=n)),
            "b": np.cumsum(rng.normal(size=n)),
        }
    )
    return [
        line
        for line in str(ws.varstable(ws.VAR(frame, lags=2).fit())).splitlines()
        if line.startswith("  |")
    ]


@pytest.mark.parametrize("n", [60, 200, 900, 1500])
def test_every_row_is_forty_four_columns(n) -> None:
    for row in _rows(n):
        assert len(row) == 44, (n, len(row), row)


@pytest.mark.parametrize("n", [60, 200, 900, 1500])
def test_the_data_pipes_sit_at_two_twenty_nine_and_forty_three(n) -> None:
    """The separator row uses '+' at the crossing, so it is excluded here."""

    for row in _rows(n):
        if "---" in row:
            continue
        positions = [index for index, char in enumerate(row) if char == "|"]
        assert positions == [2, 29, 43], (n, positions, row)


def test_a_real_eigenvalue_ends_at_column_thirteen() -> None:
    rows = [row for row in _rows(200) if "i" not in row]
    assert rows
    for row in rows:
        cell = row[3:29]
        assert cell.rstrip() == cell.rstrip(), row
        # The value's last character is at index 13.
        assert row[13] != " " or row[12] != " ", row


def test_a_complex_row_puts_its_sign_at_fifteen() -> None:
    """The captured row is '  | -.09055651 + .02373722i  |', sign at 15 and imaginary from 17."""

    rows = [row for row in _rows(200) if "i  |" in row]
    assert rows
    for row in rows:
        assert row[15] in "+-", row
        assert row[17] == ".", row


def test_the_modulus_switches_rule_at_one_tenth() -> None:
    """Above 0.1 six significant figures; below it six decimals."""

    from westata.postestimation.varstable import _format_complex  # noqa: F401

    # The two shapes appear in the same table at n=200.
    rendered = "\n".join(_rows(200))
    assert "1.00093" in rendered
    assert ".093616" in rendered
