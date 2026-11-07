"""lincom and contrast: the ninth and tenth blocks found by the 189 prefix survey.

Round 195 extended the survey past the `estat` family to the postestimation renderers that print as
text. Both were missing only structure -- the numbers and the column geometry were already exact.

LINCOM printed SEVEN lines where Stata prints EIGHT: the "( 1)  2*s - expr = 0" marker has a blank
line above it (the raw capture starts with a newline after the command's own output).

CONTRAST printed NINE where Stata prints TWELVE: a leading blank, then a blank after "Contrasts of
marginal linear predictions" and another after "Margins: asbalanced".

Both blocks' values -- .1756518 / .0121317 / 14.48 for lincom, F 1.46 / p .2314 / df 66 for
contrast -- matched before and after.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

GRILIC = r"D:\myCodeProject\test_westata\data\grilic.dta"
AUTO = "tests/data/auto.dta"


@pytest.fixture(scope="module")
def lincom_fit():
    try:
        frame = pd.read_stata(GRILIC).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.OLS.from_formula("lnw ~ s + expr + tenure", frame).fit()


@pytest.fixture(scope="module")
def contrast_fit():
    try:
        frame = pd.read_stata(AUTO).dropna(subset=["mpg", "weight", "foreign", "rep78"])
    except FileNotFoundError:  # pragma: no cover - the file ships with the repo
        pytest.skip("auto.dta is missing")
    frame = frame.reset_index(drop=True)
    frame["fx"] = (frame["foreign"] == "Foreign").astype(float)
    return ws.OLS.from_formula("mpg ~ weight + C(fx)", frame).fit()


def test_lincom_opens_with_a_blank_line(lincom_fit) -> None:
    lines = str(lincom_fit.lincom("2*s - expr")).splitlines()
    assert len(lines) == 8
    assert lines[0] == ""
    assert lines[1] == " ( 1)  2*s - expr = 0"


def test_lincom_values_are_unchanged(lincom_fit) -> None:
    rendered = str(lincom_fit.lincom("2*s - expr"))
    assert ".1756518" in rendered
    assert ".0121317" in rendered


def test_contrast_opens_with_a_blank_line(contrast_fit) -> None:
    lines = str(contrast_fit.contrast("C(fx)")).splitlines()
    assert len(lines) == 12
    assert lines[0] == ""
    assert lines[1] == "Contrasts of marginal linear predictions"
    assert lines[2] == ""
    assert lines[3] == "Margins: asbalanced"
    assert lines[4] == ""


def test_contrast_values_are_unchanged(contrast_fit) -> None:
    rendered = str(contrast_fit.contrast("C(fx)"))
    assert "1.46" in rendered
    assert "0.2314" in rendered
    assert "66" in rendered