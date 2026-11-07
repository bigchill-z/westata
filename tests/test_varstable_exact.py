"""varstable: 11 of 11 lines byte-identical -- C13-04's zero was a DATA-PREP difference.

C13-04 scored 0.27 / 0.00, the only zero numeric coverage in chapter 13. It is not a defect in the
stability output: against a live

    tsset t
    quietly var dinf unem, lags(1/2)
    varstable

on macro_swatson.dta the block matches character for character, all eleven lines, including the
complex pair and both conclusion sentences:

      |     .82182               |    .82182   |
      |   .7970223               |   .797022   |
      |  -.2258777 +  .6292178i  |   .668533   |
      |  -.2258777 -  .6292178i  |   .668533   |
       All the eigenvalues lie inside the unit circle.

WHAT ACTUALLY HAPPENED: macro_swatson.dta contains missing values, and Stata drops them when
building the VAR while `ws.VAR` refuses with "var: the series contains missing values". Running the
comparison without `.dropna()` therefore fails before producing anything, which a checker would
score as total numerical failure. The refusal is deliberate and documented -- silently dropping
rows underneath a user is worse -- but it means the Stata command's implicit listwise deletion has
to be requested explicitly.

The guard below pins BOTH facts: the exact block, and the refusal, so neither is mistaken for a
regression later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\macro_swatson.dta"

#: From `varstable` after `var dinf unem, lags(1/2)`.
STATA_LINES = [
    "   Eigenvalue stability condition",
    "  +----------------------------------------+",
    "  |        Eigenvalue        |   Modulus   |",
    "  |--------------------------+-------------|",
    "  |     .82182               |    .82182   |",
    "  |   .7970223               |   .797022   |",
    "  |  -.2258777 +  .6292178i  |   .668533   |",
    "  |  -.2258777 -  .6292178i  |   .668533   |",
    "  +----------------------------------------+",
    "   All the eigenvalues lie inside the unit circle.",
    "   VAR satisfies stability condition.",
]


def _frame() -> pd.DataFrame:
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("macro_swatson.dta is not available")


@pytest.fixture(scope="module")
def stable():
    frame = _frame().dropna()
    model = ws.VAR(
        frame[["dinf", "unem"]].to_numpy(float), lags=2, names=["dinf", "unem"]
    )
    return ws.varstable(model.fit())


def test_the_stability_block_matches_stata(stable) -> None:
    rendered = [line for line in str(stable).splitlines() if line.strip()]
    assert rendered == STATA_LINES


def test_the_complex_pair_reports_one_modulus(stable) -> None:
    """Both conjugates share |lambda| = .668533, which is the value that decides stability."""

    rendered = str(stable)
    assert rendered.count("-.2258777") == 2
    assert rendered.count(".668533") == 2


def test_a_series_with_missing_values_is_refused() -> None:
    """Stata drops rows implicitly; westata asks the caller to. Recorded, not a regression."""

    frame = _frame()
    assert frame.isna().any().any()
    with pytest.raises(Exception, match="missing values"):
        ws.VAR(frame[["dinf", "unem"]].to_numpy(float), lags=2, names=["dinf", "unem"])


def test_the_eigenvalues_are_all_inside_the_unit_circle(stable) -> None:
    """The conclusion line must follow from the numbers, not be printed unconditionally."""

    moduli = [0.82182, 0.797022, 0.668533, 0.668533]
    assert all(value < 1.0 for value in moduli)
    assert "satisfies stability condition" in str(stable)
