"""The discrete-model header puts "=" in column 70, which a four-digit sample count exposes.

Round 203 found this without fixing it; this round located the constant. At n=1500 a Logit header read:

    Stata    'Logistic regression                                     Number of obs =  1,500'
    package  'Logistic regression                                     Number of obs    =  1,500'

All six values were right; only the label padding was not. The renderer used a 17-wide label field, which put
the "=" at 73 and made the row 81 characters where Stata's is 78. A 14-wide field lands it at 70.

THE BRANCH IS SHARED, so the change was checked against every estimator that flows through it rather than only
Logit: 910 logit/probit/tobit/xtreg/panel tests pass, and both Logit and Probit compare equal to a live capture
at n=1500.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _header(n: int, estimator: str) -> list[str]:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x": rng.normal(size=n)}
    )
    frame["hi"] = (frame["y"] > 0).astype(float)
    fitted = getattr(ws, estimator).from_formula("hi ~ x", frame).fit()
    return [
        line
        for line in str(fitted.summary(style="stata")).splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("estimator", ["Logit", "Probit"])
def test_the_equals_sits_at_column_seventy(estimator) -> None:
    lines = _header(1500, estimator)
    count_row = next(line for line in lines if "Number of obs" in line)
    assert count_row.index("=") == 70, count_row


@pytest.mark.parametrize("estimator", ["Logit", "Probit"])
def test_the_header_row_is_seventy_eight_wide(estimator) -> None:
    lines = _header(1500, estimator)
    count_row = next(line for line in lines if "Number of obs" in line)
    assert len(count_row) == 78, (len(count_row), count_row)


def test_the_count_is_separated() -> None:
    lines = _header(1500, "Logit")
    count_row = next(line for line in lines if "Number of obs" in line)
    assert "1,500" in count_row


def test_every_right_hand_row_shares_the_equals_column() -> None:
    """The LR chi2 and Prob rows must land on 70 as well, not only the count."""

    lines = _header(1500, "Logit")
    for line in lines:
        if "= " in line and "LR chi2" in line:
            assert line.index("=") == 70, line
        if "Prob > chi2" in line:
            assert line.index("=") == 70, line


def test_a_three_digit_count_also_lands_on_seventy() -> None:
    lines = _header(200, "Logit")
    count_row = next(line for line in lines if "Number of obs" in line)
    assert count_row.index("=") == 70, count_row
