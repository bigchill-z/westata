"""The 5% critical values follow the RAISED system size, not a two-entry lookup.

The table used to be ``{0: (15.41, 3.76)}`` indexed by rank. That happens to be right for a
two-variable system and runs off the end for anything larger, so a three-variable system
reported ``nan`` as its rank-0 critical value.

The Osterwald-Lenum tables give one value per rank, and for the constant case the same
descending sequence serves every k:

    k=2: rank0 = 15.41, rank1 = 3.76
    k=3: rank0 = 29.68, rank1 = 15.41, rank2 = 3.76

Both measured from live ``vecrank ..., lags(2) max trend(constant)`` runs over macro_swatson.
Only three entries are measured here; a system with more variables is REFUSED rather than given
an invented threshold.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")


def _critical_column(rendered: str) -> list[str]:
    """Every non-empty field of each data row, so the 5% value can be found by content."""

    values: list[str] = []
    for line in rendered.splitlines():
        stripped = line.strip()
        if stripped and stripped[0].isdigit():
            values.append(line)
    return values


def test_two_variables_get_the_last_two_entries(frame) -> None:
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    rows = _critical_column(str(ws.vecrank(values, lags=2)))
    # Two testable ranks; the trailing untested row carries no critical value.
    assert rows[0].rstrip().endswith("15.41")
    assert rows[1].rstrip().endswith("3.76")


def test_three_variables_get_three_entries(frame) -> None:
    """This used to emit nan in the rank-0 row because the table ran out."""

    values = frame[["inf", "unem", "dinf"]].dropna().to_numpy()
    rows = _critical_column(str(ws.vecrank(values, lags=2)))
    assert rows[0].rstrip().endswith("29.68")
    assert rows[1].rstrip().endswith("15.41")
    assert rows[2].rstrip().endswith("3.76")


def test_the_measured_anchors_are_the_ones_recorded(frame) -> None:
    from westata.postestimation.vecrank import _MAX_CRITICAL_TAIL, _TRACE_CRITICAL_TAIL

    assert _TRACE_CRITICAL_TAIL == (3.76, 15.41, 29.68)
    assert _MAX_CRITICAL_TAIL == (3.76, 14.07, 20.97)


def test_more_variables_than_measured_is_refused() -> None:
    """Inventing a fourth critical value would be worse than refusing."""

    from westata.errors import UnsupportedError

    values = __import__("numpy").random.default_rng(0).normal(size=(120, 4))
    with pytest.raises(UnsupportedError, match="critical values"):
        ws.vecrank(values, lags=2)
