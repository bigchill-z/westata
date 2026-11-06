"""estat overid prints SIX SIGNIFICANT DIGITS, not six decimals.

C10-03 scores 0.00 on line agreement with 1.00 numeric coverage -- the numbers were right and the
format wrong. The code used "%.6f", so a statistic of 14.533466 printed in full where Stata shows
14.5335.

Four fits on one dataset pin the rule, and they contradict the comment that used to sit above the
formatter ("BOTH statistics print SIX decimals"):

    instruments      Sargan      Basmann
    mrt iq           14.5335     14.6807      <- six sig figs = four decimals
    mrt med          11.7433     11.818       <- THREE decimals
    iq kww           .028687     .028424      <- leading zero dropped
    age mrt iq       14.5004     14.6272

"11.818" alone rules out any fixed-decimal format, and ".028687" rules out any leading zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
COLUMNS = ["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "iq", "med", "kww", "age"]


@pytest.fixture(scope="module")
def frame():
    try:
        data = pd.read_stata(DATA).dropna(subset=COLUMNS)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return data


def _overid(frame, instruments):
    result = ws.IV2SLS.from_formula(
        f"lnw ~ expr + tenure + rns + smsa + [s ~ {instruments}]", frame
    ).fit()
    return str(result.estat_overid())


def test_a_two_digit_statistic_keeps_four_decimals(frame) -> None:
    rendered = _overid(frame, "mrt + iq")
    assert "Sargan (score) chi2(1) =  14.5335  (p = 0.0001)" in rendered
    assert "Basmann chi2(1)        =  14.6807  (p = 0.0001)" in rendered


def test_a_statistic_near_ten_drops_to_three_decimals(frame) -> None:
    """The row that rules out any fixed-width format: Stata prints 11.818, not 11.8180."""

    rendered = _overid(frame, "mrt + med")
    assert "11.818 " in rendered or rendered.rstrip().endswith("11.818")
    assert "11.8180" not in rendered


def test_a_sub_unit_statistic_drops_the_leading_zero(frame) -> None:
    rendered = _overid(frame, "iq + kww")
    assert ".028687" in rendered
    assert "0.028687" not in rendered


def test_the_two_rows_use_the_same_format(frame) -> None:
    """Sargan and Basmann must not disagree in precision, as they once did."""

    rendered = _overid(frame, "mrt + iq")
    for line in rendered.splitlines():
        if "Sargan" in line or "Basmann" in line:
            assert "=  " in line
            assert "(p = " in line


def test_the_number_itself_is_unchanged(frame) -> None:
    """Formatting fix only -- the statistic still matches Stata's 14.533466."""

    result = ws.IV2SLS.from_formula(
        "lnw ~ expr + tenure + rns + smsa + [s ~ mrt + iq]", frame
    ).fit()
    outcome = result.estat_overid()
    assert float(outcome.statistic) == pytest.approx(14.5335, abs=5e-4)
    assert float(outcome.basmann) == pytest.approx(14.6807, abs=5e-4)
