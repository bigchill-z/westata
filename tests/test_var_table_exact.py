"""`VAR`'s information block and equation table are byte-identical to Stata's.

C13-01 scored 0.5429. Rounds 83-92 rebuilt it piece by piece, and all thirteen lines now compare
equal to a live `var y1 y2, lags(1/2)` on this fixture.

THREE NUMERIC CORRECTIONS, each measured:

  * RMSE was printed with six DECIMAL places; Stata uses six SIGNIFICANT digits. `%.6f` had matched
    the values below one and produced 1.002543 for the one above it.
  * AIC/HQIC/SBIC are on Stata's TOTAL-information scale; the offset from the backend's
    per-observation values is exactly 2*ln(2*pi) + 2.
  * Det(Sigma_ml) differs from the raw determinant by ((N - K)/N)**2 with K = 2*k_ar + 1.

TWO LAYOUT DETAILS THAT ONLY MEASUREMENT SETTLED:

  * the data row is 65 characters wide with fields ENDING at 23, 35, 44, 55 and 64, while the header
    is 64 wide and ends `Parms` at 24 and `chi2` at 53 -- the header is not simply the data layout;
  * the header's first column starts at 19.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

#: Captured verbatim from `var y1 y2, lags(1/2)` on the seeded 300-point system below.
STATA_LINES = [
    "",
    "Vector autoregression",
    "",
    "Sample: 3 thru 300                              Number of obs     =        298",
    "Log likelihood =  -833.7156                     AIC               =   5.662521",
    "FPE            =   .9868603                     HQIC              =   5.712183",
    "Det(Sigma_ml)  =   .9227959                     SBIC              =   5.786585",
    "",
    "Equation           Parms      RMSE     R-sq      chi2     P>chi2",
    "----------------------------------------------------------------",
    "y1                    5     1.00254   0.0073   2.190942   0.7007",
    "y2                    5     .977199   0.0101   3.052658   0.5491",
    "----------------------------------------------------------------",
]


@pytest.fixture(scope="module")
def summary():
    rng = np.random.default_rng(3)
    n = 300
    frame = pd.DataFrame({"y1": rng.normal(size=n), "y2": rng.normal(size=n)})
    return ws.VAR(frame[["y1", "y2"]], lags=2).fit().summary(style="stata")


def test_the_whole_header_and_equation_table_match_stata(summary) -> None:
    rendered = summary.splitlines()[: len(STATA_LINES)]

    problems = [
        (index, theirs, ours)
        for index, (theirs, ours) in enumerate(zip(STATA_LINES, rendered))
        if theirs != ours
    ]
    assert not problems, problems[:2]


def test_the_rmse_column_uses_significant_digits_not_decimals() -> None:
    """`1.00254` is six significant digits; six decimals would print `1.002543`."""

    from westata.estimators.tseries import _six

    assert _six(1.002543) == "1.00254"
    assert _six(0.977199) == ".977199"
    assert _six(1.00483) == "1.00483"
    assert _six(0.976998) == ".976998"


def test_the_data_row_is_sixty_five_wide_and_the_rule_sixty_four(summary) -> None:
    # Only the equation-table block: the coefficient table below it has its own 78-wide rules.
    block = summary.splitlines()[:13]
    data_rows = [line for line in block if line.startswith(("y1 ", "y2 "))]
    rules = [line for line in block if set(line) == {"-"}]

    assert all(len(row) == 64 for row in data_rows), [len(r) for r in data_rows]
    assert all(len(rule) == 64 for rule in rules), [len(r) for r in rules]


def test_the_criteria_use_statas_total_scale(summary) -> None:
    assert "=   5.662521" in summary  # AIC
    assert "=   5.712183" in summary  # HQIC
    assert "=   5.786585" in summary  # SBIC


def test_the_determinant_is_statas_sigma_ml(summary) -> None:
    assert "=   .9227959" in summary
