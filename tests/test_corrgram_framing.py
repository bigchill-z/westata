"""`corrgram`'s framing: the blank lines that wrap the printed block.

The table body was already byte-identical to Stata; what differed was where the block STARTS and
ENDS, which only shows up when the output is diffed against a real Stata log rather than read.

Both shapes were measured from StataNow 19.5:

    no gap                          one interior gap
    ------------------------------  ---------------------------------
    ''                              '(note: time series has 1 gap)'
    '            -1  0  1 ...'      ''
    ' LAG    AC   PAC ...'          '            -1  0  1 ...'
    '---...'                        ' LAG    AC   PAC ...'
    (rows)                          '---...'
    ''                              (rows)
                                    ''

So: the note comes FIRST when there is one, a single blank separates it from the scale row, and a
single blank closes the block.
"""

import contextlib
import io
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, "replication")

import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"


def _ar_frame(n: int = 60, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    values = np.zeros(n)
    for index in range(1, n):
        values[index] = 0.6 * values[index - 1] + rng.normal()
    return pd.DataFrame({"t": range(1, n + 1), "y": values})


@pytest.fixture(scope="module")
def plain():
    frame = _ar_frame()
    return ws.OLS.from_formula("y ~ 1", frame).fit()


def _stata_lines(frame: pd.DataFrame, command: str) -> list[str]:
    """Run the equivalent command in live Stata; skip when it is unavailable."""

    try:
        from lib.stata_session import StataSession
    except Exception:  # noqa: BLE001
        pytest.skip("live Stata is not available")

    try:
        with StataSession() as stata:
            stata.push(frame)
            with contextlib.redirect_stdout(io.StringIO()):
                stata.run("tsset t")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                stata.run(command)
            return out.getvalue().split("\n")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live Stata unavailable: {exc}")


# ---------------------------------------------------------------- framing
def test_block_opens_with_a_blank_line(plain) -> None:
    lines = str(ws.estat_corrgram(plain, lags=8, time="t")).split("\n")
    assert lines[0] == ""
    assert "-1       0       1" in lines[1]


def test_block_closes_with_a_blank_line(plain) -> None:
    lines = str(ws.estat_corrgram(plain, lags=8, time="t")).split("\n")
    assert lines[-1] == ""
    assert lines[-2].lstrip().startswith(("7", "8"))


def _gapped_frame() -> pd.DataFrame:
    """Random values with one interior gap; `np.arange` would make the PAC degenerate."""

    rng = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "t": [1, 2, 3, 5, 6, 7, 8, 9, 10],
            "y": [float(rng.normal()) for _ in range(9)],
        }
    )


def test_a_gapped_series_puts_the_note_above_the_blank_line() -> None:
    """Measured: the note is the FIRST line, then the blank, then the scale row."""

    frame = _gapped_frame()
    fit = ws.OLS.from_formula("y ~ 1", frame).fit()
    lines = str(ws.estat_corrgram(fit, lags=2, time="t")).split("\n")
    assert lines[0] == "(note: time series has 1 gap)"
    assert lines[1] == ""
    assert "-1       0       1" in lines[2]
    assert lines[-1] == ""


# ---------------------------------------------------------------- byte equality
def test_whole_block_matches_stata_without_a_gap(plain) -> None:
    lines = _stata_lines(_ar_frame(), "corrgram y, lags(8)")
    ours = str(ws.estat_corrgram(plain, lags=8, time="t")).split("\n")
    assert ours == lines


def test_whole_block_matches_stata_with_a_gap() -> None:
    frame = _gapped_frame()
    fit = ws.OLS.from_formula("y ~ 1", frame).fit()
    lines = _stata_lines(frame, "corrgram y, lags(2)")
    ours = str(ws.estat_corrgram(fit, lags=2, time="t")).split("\n")
    assert ours == lines


# ---------------------------------------------------------------- numbers
def test_numbers_match_stata_at_every_lag(plain) -> None:
    """AC/PAC/Q/Prob>Q were already right; keep them pinned so a framing fix cannot drift them."""

    lines = _stata_lines(_ar_frame(), "corrgram y, lags(8)")
    stata_rows = [line for line in lines if line[:1].isdigit()]
    ours = [line for line in str(ws.estat_corrgram(plain, lags=8, time="t")).split("\n")
            if line[:1].isdigit()]
    assert len(stata_rows) == len(ours) == 8
    for stata_row, our_row in zip(stata_rows, ours):
        assert our_row.split()[:4] == stata_row.split()[:4]


def test_every_body_line_is_79_characters(plain) -> None:
    lines = str(ws.estat_corrgram(plain, lags=8, time="t")).split("\n")
    body = [line for line in lines if line and line[:1].isdigit() or line.startswith(" LAG")]
    assert body
    for line in body:
        assert len(line) == 79, repr(line)


def test_lags_option_is_honoured(plain) -> None:
    for order in (3, 5, 12):
        rows = str(ws.estat_corrgram(plain, lags=order, time="t")).split("\n")
        body = [line for line in rows if line[:1].isdigit()]
        assert len(body) == order
