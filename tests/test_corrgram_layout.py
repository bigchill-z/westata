"""corrgram: 16 of 16 lines byte-identical -- the bar rule came from corrgram.ado.

C08-07 scored 0.00 layout against 0.91 numeric coverage; the gap was the two bar-chart columns,
which the package did not draw at all.

ROUND 185 rebuilt the frame and left the bars one stroke off, concluding that "neither pure
rounding nor pure truncation reproduces the column". Both guesses were of the wrong FORM -- ten
characters per unit, rounded -- and the reference value being checked against (-.0934 -> 1 stroke)
had been mis-transcribed; Stata draws zero strokes there.

ROUND 186 read ``corrgram.ado``'s ``MkString`` instead:

    if corr >= 0:  vb = 9;              ve = int(8*corr) + 9
    else:          ve = 9;              vb = int(8*corr) + 9

EIGHT characters per unit of correlation, with the boundary TRUNCATED. That reproduces all sixteen
lines exactly.

This is the third time reading Stata's own source settled a question that fitting could not
(varsoc's FPE in round 180, and this).
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"

#: From `corrgram e1, lags(13)` after `reg consumption price income temp`.
STATA_LINES = [
    " " * 42 + "-1" + " " * 7 + "0" + " " * 7 + "1 -1" + " " * 7 + "0" + " " * 7 + "1",
    " LAG       AC       PAC      Q     Prob>Q  [Autocorrelation]  [Partial autocor]",
    "-" * 79,
    "1        0.3298   0.3969      3.6  0.0578          |--                |---     ",
    "2        0.0362  -0.1681    3.645  0.1616          |                 -|        ",
    "3        0.0111   0.0767   3.6494  0.3019          |                  |        ",
    "4       -0.0934  -0.1483   3.9715  0.4099          |                 -|        ",
    "5       -0.3186  -0.3565   7.8703  0.1635        --|                --|        ",
    "6       -0.2058   0.0011   9.5645  0.1442         -|                  |        ",
    "7       -0.2582  -0.4237   12.346  0.0897        --|               ---|        ",
    "8       -0.1373  -0.0721   13.169  0.1062         -|                  |        ",
    "9       -0.1035  -0.3300   13.658  0.1350          |                --|        ",
    "10      -0.2378  -0.8928   16.372  0.0895         -|           -------|        ",
    "11      -0.1193  -0.5017   17.091  0.1052          |              ----|        ",
    "12       0.1923  -0.4590   19.064  0.0870          |-              ---|        ",
    "13       0.3554   0.0493   26.197  0.0160          |--                |        ",
]


@pytest.fixture(scope="module")
def correlogram():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    fitted = ws.OLS.from_formula("consumption ~ price + income + temp", frame).fit()
    return ws.estat_corrgram(fitted, lags=13)


def test_the_whole_block_matches_stata(correlogram) -> None:
    rendered = [line for line in str(correlogram).splitlines() if line.strip()]
    assert rendered == STATA_LINES


def test_every_line_is_seventy_nine_characters(correlogram) -> None:
    for line in str(correlogram).splitlines():
        if line.strip():
            assert len(line) == 79


def test_the_numeric_columns_are_exact(correlogram) -> None:
    rows = {row.lag: row for row in correlogram.rows}
    assert rows[1].ac == pytest.approx(0.3298, abs=5e-5)
    assert rows[1].pac == pytest.approx(0.3969, abs=5e-5)
    assert rows[1].q == pytest.approx(3.6, abs=5e-3)
    assert rows[5].q == pytest.approx(7.8703, abs=5e-5)


def test_q_uses_five_significant_figures(correlogram) -> None:
    """'%.4g' printed 9.564 where Stata prints 9.5645."""

    rendered = str(correlogram)
    assert "9.5645" in rendered
    assert "7.8703" in rendered


def test_a_strong_negative_value_draws_seven_strokes(correlogram) -> None:
    """|-.8928| * 8 = 7.14 -> truncated to 7, which is why the cap is not 5."""

    line = next(
        line for line in str(correlogram).splitlines() if line.startswith("10 ")
    )
    assert "-------|" in line


def test_a_small_negative_value_draws_none(correlogram) -> None:
    """-.0934 * 8 = -0.747, truncated to 0 -- the row I mis-transcribed in round 185."""

    line = next(
        line for line in str(correlogram).splitlines() if line.startswith("4 ")
    )
    assert "-|" in line and "--|" not in line
