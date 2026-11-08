"""`estat ic` after an absorbed fit: three fields, three conventions.

Round 201 fixed the likelihood; this round the other two. Against `areg y x, absorb(id)`:

    Stata    '           . |      1,500  -2454.368  -2453.021       2   4910.041   4920.668'
    package  '           . |       1500  -2454.368  -2453.021       1   4908.041   4913.355'

THE df COLUMN IS `df_m + 1` FOR THESE FITS. `areg` counts the constant in `e(rank)` -- rank 2 for one
regressor -- while the covariance matrix it exposes is 1x1, so a rank-of-V rule yields 1. The two commands mean
different things by rank: `reghdfe` reports the absorbed effects too (rank 21 here for nineteen groups plus the
regressor).

THE COUNT IS SEPARATED HERE, unlike the `dwatson` line where it is not. That is now four conventions for one
field across the package, each measured rather than assumed.

Verified at n = 200 and 1500: 2 of 2 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _row(n: int) -> str:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {
            "y": np.cumsum(rng.normal(size=n)),
            "x": rng.normal(size=n),
            "id": np.arange(n) // 10,
        }
    )
    fitted = ws.FEOLS.from_formula("y ~ x | id", frame).fit()
    rendered = str(ws.estat_ic(fitted))
    return next(line for line in rendered.splitlines() if line.strip().startswith("."))


def test_the_df_counts_the_constant() -> None:
    """One regressor plus the constant makes 2, not the 1x1 covariance's rank."""

    row = _row(200)
    assert row.split("|")[1].split()[3] == "2", row


def test_the_likelihood_is_the_fixed_effects_only_one() -> None:
    """-353.98962 for the 200-row sample; the intercept-only value would be -761.417."""

    row = _row(200)
    assert "-353.9896" in row, row


def test_a_four_digit_count_is_separated() -> None:
    row = _row(1500)
    assert "1,500" in row, row


def test_a_three_digit_count_needs_no_separator() -> None:
    row = _row(200)
    assert " 200" in row
    assert "200" in row


def test_the_information_criteria_follow_the_likelihood() -> None:
    row = _row(200)
    assert "708.0597" in row
    assert "714.6563" in row
