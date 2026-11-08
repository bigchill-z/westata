"""`varwle` drops the leading zero on small chi-squares, which only a second sample reveals.

Round 162 matched all twenty-one lines of `varwle` on one sample, where every Wald statistic was in the tens:

    72.72164    741.8916    796.2418

Checking at other sizes found a lag-2 statistic of 0.1923409, and there the package printed the zero Stata
omits:

    'Stata    '  |   2 |  .1923409     2     0.908    |'
    'package  '  |   2 | 0.1923409     2     0.908    |'

SMALL CHI-SQUARES ARE ORDINARY IN THIS TEST -- a Wald test on a null lag lands below 1 often -- so the first
sample was unlucky in a way that hid the rule.

Verified at n = 100, 300, 900: 3 of 3 byte-identical.
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
        for line in str(ws.varwle(ws.VAR(frame, lags=2).fit())).splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("n", [100, 300, 900])
def test_every_chi2_cell_drops_the_leading_zero(n) -> None:
    """No '0.' may appear where the statistic is below one."""

    for row in _rows(n):
        if not row.strip().startswith("|"):
            continue
        field = row[10:21]
        assert "0." not in field, (n, row, field)


def test_a_small_statistic_is_rendered_without_its_zero() -> None:
    rows = _rows(300)
    small = [row for row in rows if row[10:21].strip().startswith(".")]
    if not small:
        pytest.skip("this sample produced no sub-one statistic")
    for row in small:
        assert row[10:21].strip().startswith("."), row


@pytest.mark.parametrize("n", [100, 300, 900])
def test_the_table_has_twenty_one_lines(n) -> None:
    assert len(_rows(n)) == 21


@pytest.mark.parametrize("n", [100, 300, 900])
def test_the_bordered_rows_share_one_width(n) -> None:
    """The box is 40 wide at these samples; the equation headings are shorter."""

    widths = {len(row) for row in _rows(n) if row.startswith("  |")}
    assert widths == {40}, (n, widths)
