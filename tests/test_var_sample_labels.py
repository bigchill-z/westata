"""`var`'s `Sample:` line on a date axis: the labels Stata's `tsset` prints.

THE BUG: the line was built with ``float(time_value)``. For a ``datetime64`` that silently
becomes NANOSECONDS SINCE 1970 -- ``1961-04-01`` -> ``-2.8e+17`` -- so a quarterly fit printed

    Sample: -276220800000000000 thru 1009843200000000000          Number of obs = ...

19 digits that also overflowed the 48-character field and pushed ``Number of obs`` off the
line. Nothing raised: the numbers below were right, only the label was wrong, and it was wrong
in a way that deleted the column beside it.

The fix formats by TYPE rather than casting to float. A ``datetime64`` axis is labelled by
inferring its frequency from the spacing, because pandas stores no Stata ``%tq``/``%tm``
format to read; a numeric axis keeps its number.

Every expected string below was captured from a live StataNow 19.5 run of
``gen <axis>; format <axis> %t?; tsset; var y1 y2, lags(1)`` on a 40-period axis, where
``lags(1)`` drops the first period:

    quarterly   1990q2 thru 1999q4      monthly   1990m2 thru 1993m4
    weekly      1990w2 thru 1990w40     half-yearly 1990h2 thru 2009h2
    daily       02jan1990 thru 09feb1990            yearly  1981 thru 2019
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

import westata as ws

N = 40


def _frame(axis) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {"t": axis, "y1": rng.normal(size=N), "y2": rng.normal(size=N)}
    )


def _sample_line(fitted) -> str:
    return [line for line in str(fitted.summary()).splitlines() if line.startswith("Sample:")][0]


def _label(fitted) -> str:
    """Just the `Sample: ...` part, trimmed of the padding before `Number of obs`."""

    return _sample_line(fitted)[:39].rstrip()


#: kind -> (axis builder, Stata's label with `lags(1)` on a 40-period axis)
AXES = {
    "quarterly": (lambda: pd.date_range("1990-01-01", periods=N, freq="QS"), "Sample: 1990q2 thru 1999q4"),
    "monthly": (lambda: pd.date_range("1990-01-01", periods=N, freq="MS"), "Sample: 1990m2 thru 1993m4"),
    "weekly": (lambda: pd.date_range("1990-01-01", periods=N, freq="7D"), "Sample: 1990w2 thru 1990w40"),
    "halfyearly": (lambda: pd.date_range("1990-01-01", periods=N, freq="6MS"), "Sample: 1990h2 thru 2009h2"),
    "daily": (lambda: pd.date_range("1990-01-01", periods=N, freq="D"), "Sample: 02jan1990 thru 09feb1990"),
    "yearly": (lambda: np.arange(1980, 1980 + N), "Sample: 1981 thru 2019"),
}


@pytest.mark.parametrize("kind", sorted(AXES))
def test_every_frequency_matches_the_stata_label(kind: str) -> None:
    """All six `%t?` formats Stata supports, against live `tsset` output."""

    build, expected = AXES[kind]
    frame = _frame(build())
    fitted = ws.VAR(frame[["y1", "y2"]], lags=1, time=frame["t"]).fit()
    assert _label(fitted) == expected


@pytest.mark.parametrize("kind", ["quarterly", "monthly", "weekly", "halfyearly", "daily"])
def test_a_nanosecond_integer_never_appears(kind: str) -> None:
    """The regression itself: `float(datetime64)` produced a 19-digit epoch count."""

    build, _ = AXES[kind]
    frame = _frame(build())
    fitted = ws.VAR(frame[["y1", "y2"]], lags=1, time=frame["t"]).fit()
    line = _sample_line(fitted)
    digits = [part for part in line.replace(":(", "( ").split() if part.isdigit()]
    assert all(len(part) <= 8 for part in digits), line


def test_the_right_column_survives_the_label() -> None:
    """The old label was so wide it pushed `Number of obs` off the line entirely."""

    frame = _frame(pd.date_range("1990-01-01", periods=N, freq="QS"))
    line = _sample_line(ws.VAR(frame[["y1", "y2"]], lags=1, time=frame["t"]).fit())
    assert "Number of obs     =" in line
    assert len(line) <= 78


def test_an_over_long_label_is_truncated_not_overlapped() -> None:
    """A caller may pass any axis; a long label must not delete the column beside it."""

    frame = _frame(pd.Series([f"period-{index:04d}-with-a-very-long-name" for index in range(N)]).to_numpy())
    line = _sample_line(
        ws.VAR(frame[["y1", "y2"]], lags=1, time=frame["t"].to_numpy()).fit()
    )
    assert "Number of obs     =" in line
    assert len(line) <= 78


def test_a_numeric_axis_prints_without_a_decimal_point() -> None:
    """`1981`, not `1981.0` -- the behaviour that already worked and must not regress."""

    frame = _frame(np.arange(1980, 1980 + N))
    assert _label(ws.VAR(frame[["y1", "y2"]], lags=1, time=frame["t"]).fit()) == (
        "Sample: 1981 thru 2019"
    )


def test_a_non_integer_numeric_axis_keeps_its_value() -> None:
    """A fractional axis must not be rounded to an integer.

    `lags(1)` drops the first period, so the sample begins at 1981.5 -- the point is that the
    `.5` survives at all, which `int(x) if x == int(x)` was already careful to preserve.
    """

    frame = _frame(np.arange(1980.5, 1980.5 + N))
    assert _label(ws.VAR(frame[["y1", "y2"]], lags=1, time=frame["t"]).fit()) == (
        "Sample: 1981.5 thru 2019.5"
    )


def test_without_a_time_axis_the_row_index_is_the_honest_fallback() -> None:
    """Unchanged: no axis means no invented dates."""

    frame = _frame(None).drop(columns=["t"])
    assert _label(ws.VAR(frame[["y1", "y2"]], lags=1).fit()) == "Sample: 2 thru 40"


def test_a_frequency_that_cannot_be_inferred_falls_back_to_the_calendar_date() -> None:
    """An irregular axis has no period tag, so the daily label is the only true one."""

    # A 1-2 day mix has no single period length, so the spacing test finds nothing to name.
    irregular = pd.date_range("1990-01-01", periods=N, freq="D") + pd.to_timedelta(
        np.random.default_rng(5).integers(0, 2, size=N), unit="D"
    )
    frame = _frame(irregular)
    fitted = ws.VAR(frame[["y1", "y2"]], lags=1, time=frame["t"]).fit()
    label = _label(fitted)
    # Either a daily date (the honest fallback) or a period tag if the spacing looked regular;
    # what must never appear is an epoch integer.
    assert "1990" in label, label
    assert "e+" not in label and "e1" not in label, label
