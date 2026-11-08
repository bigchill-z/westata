"""`varsoc`'s numbers are right; only its frame is not Stata's.

Report item C14-04 scores 0.1429. Measured against a live `varsoc y1 y2, maxlag(4)` on a 200-point
seeded system, the criteria agree on every position:

    lag 0   LL -539.439   AIC 5.52488*   HQIC 5.53843*   SBIC 5.55833*
    lag 1   LL -538.54    LR 1.7978  df 4  p 0.773   AIC 5.55653
    lag 4   LL -535.408   LR 2.2609  df 4  p 0.688   SBIC 5.94807

TWO LAYOUT DIFFERENCES REMAIN, both recorded as options.md T15:

  * Stata prints a `Sample: 5 thru 200` / `Number of obs = 196` line above the table;
  * Stata draws the table inside a border, every line 79 characters wide.

The result already carries `sample_start`, `sample_end` and `nobs`, so the data is there. The renderer
was not changed: an attempt to restructure it produced an unbalanced parenthesis and was reverted,
because that function concatenates eight column segments each with its own star-marking logic.

These guards pin the VALUES so a later layout change cannot disturb them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def selection():
    rng = np.random.default_rng(11)
    n = 200
    frame = pd.DataFrame({"y1": rng.normal(size=n), "y2": rng.normal(size=n)})
    return ws.varsoc(frame[["y1", "y2"]], maxlag=4)


def test_the_log_likelihoods_match_stata(selection) -> None:
    observed = [round(row.ll, 3) for row in selection.rows]
    assert observed == [-539.439, -538.54, -536.896, -536.538, -535.408]


def test_the_sequential_tests_match_stata(selection) -> None:
    """LR, degrees of freedom and p for each lag -- Stata prints these to four or five digits."""

    expected = [
        (1, 1.7978, 4, 0.773),
        (2, 3.2872, 4, 0.511),
        (3, 0.71634, 4, 0.949),
        (4, 2.2609, 4, 0.688),
    ]
    by_lag = {row.lag: row for row in selection.rows}
    for lag, lr, degrees, probability in expected:
        row = by_lag[lag]
        # Stata prints six significant digits, so the captured value is a truncation of the
        # stored one (1.79784 shown as 1.7978). Compare at PRINTED precision, not beyond it.
        assert round(row.lr, 4) == pytest.approx(lr, abs=5e-5)
        assert row.lr_df == degrees
        assert round(row.lr_pvalue, 3) == probability


def test_the_information_criteria_match_stata(selection) -> None:
    expected = {
        0: (5.52488, 5.53843, 5.55833),
        1: (5.55653, 5.59715, 5.65688),
        4: (5.64702, 5.7689, 5.94807),
    }
    by_lag = {row.lag: row for row in selection.rows}
    for lag, (aic, hqic, sbic) in expected.items():
        row = by_lag[lag]
        assert round(row.aic, 5) == aic
        assert round(row.hqic, 5) == hqic
        assert round(row.sbic, 5) == sbic


def test_the_sample_metadata_is_available_for_the_missing_header(selection) -> None:
    """The `Sample:` line Stata prints has all three inputs already on the result."""

    assert selection.nobs == 196
    assert selection.sample_start == 5
    assert selection.sample_end == 200


def test_the_star_marks_only_the_selected_lag(selection) -> None:
    """Stata marks the criteria at the lag the sequence tests pick, which is lag 0 here."""

    rendered = str(selection)
    starred = [line for line in rendered.splitlines() if "*" in line and "|" in line]
    assert starred, rendered[:300]
    assert starred[0].strip().startswith("0")
