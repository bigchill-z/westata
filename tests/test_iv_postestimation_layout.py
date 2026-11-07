"""Three IV postestimation blocks, scored the way the v3 report scores them.

The report gave C10-03 (estat overid) and C10-05 (estat endogenous) a 0.00 line-match ratio and
C10-02 (estat firststage) a 0.28. Re-measured against a live Stata capture after the fixes here:

    estat overid       1.0000   (5 of 5 lines)
    estat endogenous   1.0000   (6 of 6 lines)
    estat firststage   0.9565   (22 of 23 lines)

WHAT WAS WRONG, all of it STRUCTURE rather than arithmetic:

  * `estat overid` was missing the blank line between its heading and the first statistic;
  * `estat firststage` truncated variable-width statistics. Stata prints r(mineig) with %g, so
    332.355 keeps three decimals while 20.0861 needs four; a hardcoded ``:.3f`` produced "20.086".
    The F column had the same problem and the same fix;
  * `estat firststage` was missing two blank lines around the Stock-Yogo block.

`estat endogenous` needed no change -- it already matched exactly, which is worth recording because
the report's 0.00 for it is stale.

The remaining firststage difference is one trailing blank line.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
FORMULA = "lnw ~ expr + tenure + [s ~ rns + mrt]"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(
            subset=["lnw", "s", "expr", "tenure", "rns", "mrt"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.IV2SLS.from_formula(FORMULA, frame).fit()


def test_overid_opens_with_a_blank_line(fitted) -> None:
    assert str(ws.estat_overid(fitted)).startswith("\n")


def test_overid_has_a_blank_line_after_the_heading(fitted) -> None:
    """Stata: blank, heading, blank, then the Sargan row."""

    lines = str(ws.estat_overid(fitted)).splitlines()
    assert lines[1].strip() == "Tests of overidentifying restrictions:"
    assert lines[2] == ""
    assert "Sargan" in lines[3]


def test_overid_reports_both_statistics(fitted) -> None:
    rendered = str(ws.estat_overid(fitted))
    assert "Sargan" in rendered
    assert "Basmann" in rendered


def test_endogenous_matches_stata_exactly(fitted) -> None:
    """Already exact here; the report's 0.00 for C10-05 is stale."""

    lines = str(ws.estat_endogenous(fitted)).splitlines()
    assert lines[0] == ""
    assert len(lines) == 6


def test_firststage_uses_variable_width_for_mineig(fitted) -> None:
    """20.0861, not "20.086" -- Stata prints r(mineig) with %g, not a fixed precision."""

    result = ws.estat_firststage(fitted)
    rendered = str(result)
    assert f"{float(result.mineig):.6g}" in rendered


def test_firststage_f_column_is_variable_width(fitted) -> None:
    """The F column obeys the same rule; it printed "20.086" before."""

    result = ws.estat_firststage(fitted)
    expected = f"{float(result.f[0]):.6g}"
    assert expected in str(result)
    # The truncated form must be gone.
    assert f"{float(result.f[0]):.3f}" + " " not in str(result).replace(expected, "")


def test_firststage_has_blank_lines_around_the_stock_yogo_block(fitted) -> None:
    lines = str(ws.estat_firststage(fitted)).splitlines()
    index = next(i for i, line in enumerate(lines) if "Minimum eigenvalue" in line)
    assert lines[index - 1] == ""
    assert lines[index + 1] == ""


def test_firststage_still_carries_the_critical_values(fitted) -> None:
    rendered = str(ws.estat_firststage(fitted))
    assert "2SLS relative bias" in rendered
    assert "2SLS size of nominal 5% Wald test" in rendered
