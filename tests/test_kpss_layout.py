"""kpss: X01 scored 0.12 because the NUMBERS were right and the layout was not.

Round 146 captured a live ``kpss`` output and fixed four things, all measurable:

  1. The Maxlag line says "chosen by Schwert criterion" ONLY when the lag count was automatic.
     An explicit ``maxlag(4)`` prints just ``Maxlag = 4``. The old renderer always added the
     phrase, so every explicit-maxlag call was wrong.
  2. Statistics are THREE SIGNIFICANT FIGURES with no leading zero: ``2.14``, ``1.16``, ``.811``,
     ``.0146``. The old ``%.4g`` produced ``0.01459`` against Stata's ``.0146``.
  3. Critical values sit on ONE compact line (``10%: 0.119  5% : 0.146  2.5%: 0.176  1% : 0.216``),
     not four indented lines.
  4. Stata's blank lines are a SINGLE SPACE, not an empty string.

A new optional ``name=`` argument supplies the variable name Stata prints in the title and H0
line; a bare ndarray keeps the generic wording rather than inventing a name.

VERIFIED byte-for-byte in all four cases against live Stata on macro_swatson.dta:
    kpss inf                  16/16 lines
    kpss inf, maxlag(4)       16/16 lines
    kpss dinf, maxlag(4)      16/16 lines
    kpss inf, notrend         16/16 lines

The statistics themselves were already correct (0.014588 -> .0146, 0.02941 -> .0294), so this
was a pure presentation defect -- safe to fix without touching any numeric code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

#: Captured from live Stata on macro_swatson.dta.
INF_DEFAULT = [
    " ",
    "KPSS test for inf",
    " ",
    "Maxlag = 4 chosen by Schwert criterion",
    "Autocovariances weighted by Bartlett kernel",
    " ",
    "Critical values for H0: inf is trend stationary",
    " ",
    "10%: 0.119  5% : 0.146  2.5%: 0.176  1% : 0.216",
    " ",
    "Lag order    Test statistic",
    "    0           2.14",
    "    1           1.16",
    "    2           .811",
    "    3           .625",
    "    4           .514",
]

INF_EXPLICIT = [
    " ",
    "KPSS test for inf",
    " ",
    "Maxlag = 4",
    "Autocovariances weighted by Bartlett kernel",
    " ",
    "Critical values for H0: inf is trend stationary",
    " ",
    "10%: 0.119  5% : 0.146  2.5%: 0.176  1% : 0.216",
    " ",
    "Lag order    Test statistic",
    "    0           2.14",
    "    1           1.16",
    "    2           .811",
    "    3           .625",
    "    4           .514",
]


@pytest.fixture(scope="module")
def frame():
    try:
        # NOTE: drop NaNs on `inf` ALONE. The Stata reference was `kpss inf` on the full inf
        # series (167 rows); adding dinf to the subset silently shortens it and changes every
        # statistic. This is the same "wrong sample looks like a numeric bug" trap as round 129.
        return pd.read_stata(DATA).dropna(subset=["inf"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")


def test_default_maxlag_matches_stata_line_for_line(frame) -> None:
    result = ws.kpss(frame["inf"].to_numpy(float), name="inf")
    assert str(result).splitlines() == INF_DEFAULT


def test_explicit_maxlag_matches_stata_line_for_line(frame) -> None:
    result = ws.kpss(frame["inf"].to_numpy(float), name="inf", maxlag=4)
    assert str(result).splitlines() == INF_EXPLICIT
    assert result.maxlag_explicit is True


def test_dinf_with_explicit_maxlag_matches_stata() -> None:
    """The small-value case: three significant figures, no leading zero.

    dinf has its OWN missing-value pattern, so it must be dropped on dinf alone -- reusing the
    `inf` fixture's rows produced different statistics and looked like a layout regression.
    """

    frame = pd.read_stata(DATA).dropna(subset=["dinf"])
    result = ws.kpss(frame["dinf"].to_numpy(float), name="dinf", maxlag=4)
    lines = str(result).splitlines()
    assert "KPSS test for dinf" in lines
    assert "    0          .0146" in lines
    assert "    2          .0294" in lines
    assert "    4          .0283" in lines


def test_notrend_uses_the_level_stationary_wording(frame) -> None:
    result = ws.kpss(frame["inf"].to_numpy(float), name="inf", trend=False)
    lines = str(result).splitlines()
    assert "Critical values for H0: inf is level stationary" in lines
    assert "inf is trend stationary" not in str(result)


def test_a_bare_array_keeps_generic_wording(frame) -> None:
    """No name to print, so do not invent one."""

    result = ws.kpss(frame["inf"].to_numpy(float))
    assert result.name == ""
    assert "KPSS test for trend stationary" in str(result)
    assert "series is trend stationary" in str(result)


def test_the_underlying_statistics_are_unchanged(frame) -> None:
    """Guard against a 'layout fix' that quietly changed the numbers."""

    result = ws.kpss(frame["dinf"].to_numpy(float), maxlag=4)
    assert [round(row.statistic, 6) for row in result.rows] == [
        0.014588, 0.019251, 0.029410, 0.027786, 0.028251
    ]
