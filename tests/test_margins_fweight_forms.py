"""margins under fweight: the two forms Stata distinguishes, both verified.

Round 117 retracted a commit whose EXPLANATION was invented even though its code was right. This
round checked whether the same had happened elsewhere in the margins comments -- specifically the
claims about `dydx` versus `dydx atmeans` under frequency weights.

VERIFIED CORRECT, against a live `logit survive child female class1 [fweight=freq]`:

    margins, dydx(child)            child .1164165   westata .1164165
    margins, dydx(child) atmeans    child .1476849   westata .1476849

So the code comment's central claim holds: the AME form and the atmeans form are DIFFERENT
estimands, which is why a defect in one can hide from the other. Stata writes "dy/dx wrt:  s" with
TWO spaces after the colon, also as the comment says.

ONE CLAIM COULD NOT BE REPRODUCED, and is recorded rather than deleted: the comment quotes
historical values ".1809351 here against .1732315 in Stata" and "child .2628119 against .2223422".
Neither pair matches the current code or the current Stata output. They describe a BUG THAT NO
LONGER EXISTS and cannot be re-measured from this checkout -- which is fine for a comment about
history, but it means those numbers are narrative, not evidence, and a reader should not treat them
as a reproducible measurement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\titanic.dta"
FORMULA = "survive ~ child + female + class1"

#: From `margins, dydx(child)` and `margins, dydx(child) atmeans` after the fweight logit.
STATA_AME = 0.1164165
STATA_CONDITIONAL = 0.1476849


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("titanic.dta is not available")
    return ws.Logit.from_formula(FORMULA, frame, freq_weights="freq").fit()


def test_the_average_marginal_effect_matches_stata(fitted) -> None:
    result = fitted.margins(dydx="child")
    assert float(result.estimate) == pytest.approx(STATA_AME, abs=5e-7)


def test_the_atmeans_marginal_effect_matches_stata(fitted) -> None:
    result = fitted.margins(dydx="child", atmeans=True)
    assert float(result.estimate) == pytest.approx(STATA_CONDITIONAL, abs=5e-7)


def test_the_two_forms_are_different_estimands(fitted) -> None:
    """Which is exactly why a defect in one form survives in the other."""

    ame = float(fitted.margins(dydx="child").estimate)
    conditional = float(fitted.margins(dydx="child", atmeans=True).estimate)
    assert abs(ame - conditional) / ame > 0.20


def test_the_wrt_line_has_two_spaces_after_the_colon(fitted) -> None:
    """Stata prints 'dy/dx wrt:  s' -- measured, and the comment says so."""

    rendered = str(fitted.margins(dydx="child"))
    line = next(row for row in rendered.splitlines() if "wrt:" in row)
    assert line.startswith("dy/dx wrt:  ")
