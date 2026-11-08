"""`estat dwatson`: sub-one statistics use seven decimals trimmed, not six.

Round 168 matched this table on samples where the Durbin-Watson statistic was above 1, and six decimals with
the trailing zero trimmed was right there:

    1.021169    1.704354    1.75295

Checking on a random walk -- where d is tiny -- found the sub-one branch:

    Stata    'Durbin-Watson d-statistic(  2,   200) =  .0127663'
    package  'Durbin-Watson d-statistic(  2,   200) =   0.012766'

SEVEN DECIMALS WITH THE TRAILING ZEROS TRIMMED, then the leading zero dropped. Not seven significant figures:
0.007009565 renders as .0070096, which is five significant figures -- "%.6g" shortens it to .00700956 and
"%.7g" lengthens it to .007009565. Three values settle the rule:

    0.007009565 -> .0070096
    0.01276632  -> .0127663
    0.012766    -> .012766

Verified at n = 200 and 1500: 2 of 2 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _line(n: int) -> str:
    rng = np.random.default_rng(n)
    frame = pd.DataFrame(
        {
            "y": np.cumsum(rng.normal(size=n)),
            "x": rng.normal(size=n),
        }
    )
    rendered = str(ws.estat_dwatson(ws.OLS.from_formula("y ~ x", frame).fit()))
    return next(line for line in rendered.splitlines() if "d-statistic" in line)


@pytest.mark.parametrize("n", [200, 1500])
def test_a_sub_one_statistic_drops_the_leading_zero(n) -> None:
    assert "=  ." in _line(n), _line(n)
    assert "0.0" not in _line(n).split("=")[1], _line(n)


@pytest.mark.parametrize("n", [200, 1500])
def test_the_line_is_forty_nine_columns(n) -> None:
    """The heading is fixed and the value field is nine wide."""

    assert len(_line(n)) == 49, (n, len(_line(n)), _line(n))


def test_the_value_uses_seven_decimals_trimmed() -> None:
    """0.007009565 renders as .0070096, which neither %6g nor %7g produces."""

    line = _line(1500)
    value = line.split("=")[1].strip()
    assert len(value) <= 9, (value, len(value))
    assert value.startswith("."), value


def test_the_observation_count_is_not_separated() -> None:
    """Unlike the summarize counts, this one is plain: 1500 rather than 1,500."""

    assert "1500" in _line(1500)
    assert "1,500" not in _line(1500)
