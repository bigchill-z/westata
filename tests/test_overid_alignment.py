"""`estat overid` right-aligns its statistic to column 34, which a sub-one Basmann shows.

Round 161 verified the Sargan and Basmann statistics matched Stata to every printed digit. Checking at a
larger sample found the numbers still matched but their placement did not:

    Stata    '  Basmann chi2(1)        =   .14737  (p = 0.7011)'
    package  '  Basmann chi2(1)        =  .14737  (p = 0.7011)'

THE GAP AFTER "=" IS NOT FIXED: the statistic is right-aligned to column 34, so a five-character value gets
three spaces and a seven-character value gets one. Measured on three rows from two samples:

    '  Sargan (score) chi2(1) =  1.20073  (p = 0.2732)'    value 28..34
    '  Basmann chi2(1)        =   1.1923  (p = 0.2749)'    value 29..34
    '  Basmann chi2(1)        =   .14737  (p = 0.7011)'    value 29..34

A fixed two spaces matched the first sample because every statistic there was narrow enough to fill the field.

Verified at n = 200, 800, 1500: 3 of 3 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _rows(n: int) -> list[str]:
    rng = np.random.default_rng(n)
    exogenous = rng.normal(size=n)
    common = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "y": 0.5 * common + rng.normal(size=n),
            "w": exogenous,
            "x": common + rng.normal(size=n) * 0.8,
            "z1": common + rng.normal(size=n) * 0.5,
            "z2": common + rng.normal(size=n) * 0.5,
        }
    )
    fitted = ws.IV2SLS.from_formula("y ~ w + [x ~ z1 + z2]", frame).fit()
    return [
        line for line in str(ws.estat_overid(fitted)).splitlines() if line.strip()
    ]


@pytest.mark.parametrize("n", [200, 800])
def test_the_statistic_ends_at_column_thirty_four(n) -> None:
    rows = [row for row in _rows(n) if "chi2(" in row and "=" in row]
    assert rows
    for row in rows:
        equals = row.index("=")
        value_start = row.index(row[equals + 1 :].strip().split("  ")[0], equals)
        value = row[equals + 1 :].strip().split("  ")[0]
        end = value_start + len(value) - 1
        assert end == 34, (n, row, value, end)


@pytest.mark.parametrize("n", [200, 800])
def test_the_two_rows_are_present(n) -> None:
    rows = _rows(n)
    assert any("Sargan" in row for row in rows)
    assert any("Basmann" in row for row in rows)


def test_a_value_below_one_still_aligns() -> None:
    """The case that exposed the defect: a five-character statistic."""

    for n in (200, 800, 1500):
        for row in _rows(n):
            if "chi2(" not in row or "=" not in row:
                continue
            value = row.split("=", 1)[1].strip().split("  ")[0]
            if not value.startswith("."):
                continue
            start = row.index(value)
            assert start + len(value) - 1 == 34, (n, row, value)
