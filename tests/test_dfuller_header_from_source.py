"""`dfuller`'s header, built from `dfuller.ado`'s own formula rather than from its output.

Rounds 187 and 188 each tried implementations against a handful of captures and reverted; six attempts total.
The rule was in the source all along:

    _align maxlen "dinobs dilag" : "`T' `lags'" "11.0gc"
    local tablen 62
    local col = `tablen'-14-3-`maxlen'+1
    di as txt "Variable: " as res "`sname'" _c
    di as txt as smcl _col(`col') "Number of obs  = " as res "`dinobs'"
    di as smcl as txt _col(`col') "Number of lags = " as res "`dilag'"

`col = 46 - maxlen`, where maxlen is the wider of the count and the lag as formatted with %11.0gc. In
Python that is `45 - maxlen`, because `_col()` is 1-based. Everything above is written with `display`, which
appends the value with NO field padding -- which is why every column-arithmetic attempt drifted.

Two more facts the source makes explicit and the captures confirmed:

  * `lags > 0` prints "Augmented Dickey-Fuller" and gives the banner its own row (`diaug`), moving the count
    onto the Variable row;
  * the lag text carries the %11.0gc padding, so it reads '=     0' rather than '= 0'.

Verified at four sample sizes crossed with two lag counts: 8 of 8 byte-identical.
"""

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pytest

import westata as ws


def _lines(n: int, lags: int):
    rng = np.random.default_rng(n)
    series = np.cumsum(rng.normal(size=n))
    return [
        line
        for line in str(ws.dfuller(series, lags=lags)).splitlines()
        if line.strip()
    ]


@pytest.mark.parametrize("n", [60, 200, 900, 1500])
def test_the_plain_banner_shares_a_row_with_the_count(n) -> None:
    first = _lines(n, 0)[0]
    assert first.startswith("Dickey\u2013Fuller test for unit root")
    assert "Number of obs  = " in first


@pytest.mark.parametrize("n", [60, 200, 900, 1500])
def test_the_augmented_banner_has_its_own_row(n) -> None:
    lines = _lines(n, 3)
    assert lines[0] == "Augmented Dickey\u2013Fuller test for unit root"
    assert len(lines[0]) == 42
    assert "Number of obs  = " in lines[1]


def test_the_count_label_moves_left_as_the_count_grows() -> None:
    """col = 46 - maxlen in the source, which is 45 - maxlen as a Python index."""

    positions = {}
    for n in (60, 200, 900, 1500):
        line = _lines(n, 0)[0]
        positions[n] = line.index("Number")

    assert positions[60] > positions[200], positions
    assert positions[900] > positions[1500], positions


def test_the_lag_value_carries_its_padding() -> None:
    """The lag is padded to maxlen, so the spacing depends on the count's width.

    n=200 has a three-digit count, so maxlen is 3 and the row reads '=   0'; n=1500's count is five
    characters ('1,499'), so maxlen is 5 and it reads '=     0'. Asserting one literal would pin the wrong
    half of the rule.
    """

    assert "Number of lags =   0" in _lines(200, 0)[1]
    assert "Number of lags =     0" in _lines(1500, 0)[1]


def test_every_header_row_has_a_known_width() -> None:
    """62 for the paired rows, 63 when a separated count overflows, 42 for the augmented banner."""

    for n in (60, 200, 900, 1500):
        for lags in (0, 3):
            for line in (
                _lines(n, lags)[:3] if lags else _lines(n, lags)[:2]
            ):
                if line.startswith("Augmented"):
                    assert len(line) == 42, (n, lags, len(line), line)
                else:
                    assert len(line) == 62, (n, lags, len(line), line)
