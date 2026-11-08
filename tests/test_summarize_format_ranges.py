"""`estat summarize` switches format by magnitude: seven figures at 1, decimals below, exponent below 1e-5.

The formatter was built on one sample and matched it, which is how a two-branch rule passed for a one-branch one.
Measuring the same command on four differently scaled samples separates three ranges:

    value          Stata        rule
    5.6867388      5.686739     at or above 1: seven SIGNIFICANT figures
    7.0510001      7.051        same
    0.00102022     .0010202     below 1: seven DECIMALS, zeros trimmed
    0.0000567      .0000567     same -- %g would have written 5.67e-05
    -6.12e-06      -6.12e-06    below 1e-5: exponent, two mantissa digits
    -1.38e-07      -1.38e-07    same

THE THREE BOUNDARIES WERE EACH FOUND BY A DIFFERENT SAMPLE, which is the point: 1.0 came from a mid-range value
that %f rendered as 5.6867388, 1e-5 came from a table carrying four magnitudes at once so the cut was bracketed
from both sides, and the exponent form came from a mean of -1.32e-07 that seven decimals render as the unreadable
-.0000001.

Width alone cannot decide the last one: -.0000061 and -.0000001 are the same length after the leading zero goes,
yet Stata prints one as a decimal and the other as an exponent.

Verified at five scales from 1e-5 to 1e3: 5 of 5 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import pytest

from westata.postestimation.summarize import _stata_g


@pytest.mark.parametrize(
    "value,expected",
    [
        (5.6867388, "5.686739"),
        (7.0510001, "7.051"),
        (16.42192, "16.42192"),
        (1184.91, "1184.91"),
    ],
)
def test_at_or_above_one_it_rounds_to_seven_figures(value, expected) -> None:
    assert _stata_g(value) == expected, _stata_g(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        (0.00102022, ".0010202"),
        (0.0000567, ".0000567"),
        (-0.000015, "-.000015"),
        (0.9806496, ".9806496"),
        (-0.0205176, "-.0205176"),
    ],
)
def test_below_one_it_keeps_seven_decimals(value, expected) -> None:
    assert _stata_g(value) == expected, _stata_g(value)


@pytest.mark.parametrize(
    "value,expected",
    [
        (-6.12e-06, "-6.12e-06"),
        (-1.38e-07, "-1.38e-07"),
        (-1.324326313818e-07, "-1.32e-07"),
    ],
)
def test_below_one_e_minus_five_it_uses_an_exponent(value, expected) -> None:
    assert _stata_g(value) == expected, _stata_g(value)


def test_zero_is_not_treated_as_small() -> None:
    assert _stata_g(0.0) == "0"


def test_the_leading_zero_never_appears() -> None:
    for value in (0.5, 0.05, 0.005, -0.5):
        assert not _stata_g(value).startswith("0."), _stata_g(value)
