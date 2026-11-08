"""`vargranger`'s chi2 column matches Stata, which took a format change rather than a value change.

C13-05 scores 1.0 and the NUMBERS were already right; the column width was not. Measured on a
150-point seeded system with `var y1 y2, lags(1/2)`:

    Granger causality Wald tests
        y1 -> y2      chi2 1.0532   df 2   p 0.591
        y2 -> y1      chi2 .56897   df 2   p 0.752

The package printed 1.053 and 0.569 -- three decimals, which cannot express either: the first needs
four places and the second five. Stata renders the statistic at five SIGNIFICANT figures with the
leading zero dropped, and both now match.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def granger():
    rng = np.random.default_rng(3)
    n = 200
    frame = pd.DataFrame(
        {
            "y1": np.cumsum(rng.normal(size=n)),
            "y2": np.cumsum(rng.normal(size=n)),
        }
    )
    fitted = ws.VAR(frame[["y1", "y2"]], lags=2).fit()
    return str(ws.vargranger(fitted))


def test_the_chi2_uses_five_significant_figures(granger) -> None:
    """1.0532 needs four decimals; three would print 1.053."""

    assert "1.0532" in granger
    assert "1.053 " not in granger


def test_the_leading_zero_is_dropped(granger) -> None:
    """Stata writes .56897, not 0.56897."""

    assert ".56897" in granger
    assert "0.56897" not in granger


def test_the_probability_column_keeps_three_decimals(granger) -> None:
    """The p-value is a separate field and stays at three places."""

    assert "0.591" in granger
    assert "0.752" in granger
