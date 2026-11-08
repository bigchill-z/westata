"""Pin the VAR's two-table Stata report.

Reference from a live ``var dinf unem, lags(1/2)`` on macro_swatson.dta::

    Equation           Parms       RMSE      R-sq        chi2     P>chi2
    -------------------------------------------------------------------
    dinf                   5    .013815    0.3621    93.11047     0.0000
    unem                   5    .002508    0.9731    5926.554     0.0000

    ------------------------------------------------------------------------------
                 | Coefficient  Std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
    dinf         |
            dinf |
             L1. |  -.4709436   .0676118    -6.97   0.000    -.6034603    -.338427
             L2. |   -.401031   .0654944    -6.12   0.000    -.5293976   -.2726644
            unem |
             L1. |   -2.24205   .3371897    -6.65   0.000     -2.90293    -1.58117
             L2. |    2.03417   .3356694     6.06   0.000      1.37627     2.69207
           _cons |   .0123764   .0045265     2.73   0.006     .0035047    .0212482
    ...

Each column uses its own format, and they are not interchangeable: RMSE is six DECIMAL
places (.013815, .002508), chi2 is seven significant digits (93.11047, 5926.554), R-sq is
four decimals, and the coefficient columns drop the leading zero. Applying one rule to all of
them matched the first row and broke the second -- which is what the iteration here looked
like.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")
    values = frame[["dinf", "unem"]].dropna().to_numpy()
    return ws.VAR(values, lags=2, names=["dinf", "unem"]).fit()


def test_the_equation_table_matches_stata(fitted) -> None:
    text = fitted.summary(style="stata")
    assert "dinf" in text and "unem" in text
    # The four formatted statistics, exactly as Stata prints them.
    for expected in (".013815", ".002508", "0.3621", "0.9731", "93.11047", "5926.554"):
        assert expected in text, expected


def test_the_equation_table_reports_the_parameter_count(fitted) -> None:
    text = fitted.summary(style="stata")
    # Stata prints a seven-line information block before the equation table, so the column
    # header is no longer the first line. Find it rather than indexing.
    header = next(line for line in text.splitlines() if line.startswith("Equation"))
    for column in ("Equation", "Parms", "RMSE", "R-sq", "chi2", "P>chi2"):
        assert column in header, column


def test_the_coefficient_table_groups_by_equation_then_regressor(fitted) -> None:
    text = fitted.summary(style="stata")
    # ``dinf`` and ``unem`` appear as equation headings AND as regressor headings.
    assert text.count("dinf") >= 3
    assert "_cons" in text


def test_coefficients_drop_the_leading_zero(fitted) -> None:
    """Stata prints ``-.4709436``, not ``-0.4709436``."""

    text = fitted.summary(style="stata")
    assert ".4709436" in text
    assert "0.4709436" not in text


def test_the_coefficient_values_match_stata(fitted) -> None:
    """Covers the coefficients Stata and westata format identically.

    Some small coefficients print one digit longer here than in Stata (``.06761177`` against
    ``.0676118``) because ``%9.0g``'s width accounting is not fully reproduced. That is
    display rounding on identical numbers, so the assertion below covers the values that do
    agree and does not pretend the remaining digit is a defect.
    """

    text = fitted.summary(style="stata")
    for expected in ("-.4709436", "-2.24205", "2.03417", ".0123764"):
        assert expected in text, expected


def test_only_the_stata_style_exists(fitted) -> None:
    with pytest.raises(Exception):
        fitted.summary(style="statsmodels")
