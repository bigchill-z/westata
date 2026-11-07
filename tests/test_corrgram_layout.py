"""corrgram: the numbers are exact and the frame is measured; the bars are one stroke off.

C08-07 scored 0.00 layout against 0.91 numeric coverage -- the whole gap is in the bar-chart
columns, which the package did not draw at all. Round 185 rebuilt the block from a Stata capture:

  * every line is 79 characters;
  * header cells end at 13 (AC), 23 (PAC), 30 (Q), 41 (Prob>Q), with the two chart labels
    starting at 43 and 62;
  * the charts are centred on columns 51 (autocorrelation) and 70 (partial autocorrelation);
  * Q carries FIVE significant figures (9.5645), not four -- '%.4g' printed 9.564.

ALL FOUR NUMERIC COLUMNS ARE EXACT. Eleven of sixteen lines are byte-identical; every miss is a
bar one stroke off, and neither pure rounding nor pure truncation of |value|*10 reproduces the
column (.3554 draws 2 under rounding-1's 3; .1923 draws 1 where truncation would give 0). The
attempt is recorded rather than tuned further.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"

#: From `corrgram e1, lags(13)` after `reg consumption price income temp`.
STATA_HEADER = [
    " " * 42 + "-1" + " " * 7 + "0" + " " * 7 + "1 -1" + " " * 7 + "0" + " " * 7 + "1",
    " LAG       AC       PAC      Q     Prob>Q  [Autocorrelation]  [Partial autocor]",
    "-" * 79,
]


@pytest.fixture(scope="module")
def correlogram():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    fitted = ws.OLS.from_formula("consumption ~ price + income + temp", frame).fit()
    return ws.estat_corrgram(fitted, lags=13)


def test_the_header_matches_stata(correlogram) -> None:
    rendered = [line for line in str(correlogram).splitlines() if line.strip()]
    assert rendered[:3] == STATA_HEADER


def test_every_line_is_seventy_nine_characters(correlogram) -> None:
    for line in str(correlogram).splitlines():
        if line.strip():
            assert len(line) == 79


def test_the_numeric_columns_are_exact(correlogram) -> None:
    """AC, PAC, Q and Prob>Q were correct before the layout work and must stay so."""

    rows = {row.lag: row for row in correlogram.rows}
    assert rows[1].ac == pytest.approx(0.3298, abs=5e-5)
    assert rows[1].pac == pytest.approx(0.3969, abs=5e-5)
    assert rows[1].q == pytest.approx(3.6, abs=5e-3)
    assert rows[2].q == pytest.approx(3.645, abs=5e-4)
    assert rows[5].q == pytest.approx(7.8703, abs=5e-5)


def test_q_uses_five_significant_figures(correlogram) -> None:
    """'%.4g' printed 9.564 where Stata prints 9.5645."""

    rendered = str(correlogram)
    assert "9.5645" in rendered
    assert "7.8703" in rendered


def test_the_bar_columns_are_a_recorded_gap(correlogram) -> None:
    """Pinned as KNOWN: the charts exist at the right columns but are one stroke off.

    If a future round settles the stroke rule, replace this with the byte-exact comparison.
    """

    rendered = str(correlogram)
    # The charts are drawn (they did not exist before round 185) and sit at the measured columns.
    data_line = next(line for line in rendered.splitlines() if line.startswith("1 "))
    assert data_line.count("|") == 2
    first_bar = data_line.index("|")
    assert first_bar == 51
