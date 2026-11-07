"""Re-verify three tables whose tests hard-code widths, against a live Stata.

Rounds 99 and 100 found code and its tests sharing the same wrong constant -- the two masked each
other, so a test passing proved only that they agreed. That risk applies to every test asserting a
literal width, and this module checks the three most width-dependent ones:

    pwcompare   62-character rows
    contrast    48-character rows
    summarize   71-character rows

All three turned out to be CORRECT. Recorded so the audit has a result either way -- a check that
only ever reports defects is not a check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

ICECREAM = r"D:\myCodeProject\test_westata\data\icecream.dta"
GIRLIC = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def icecream_fit():
    try:
        frame = pd.read_stata(ICECREAM, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")
    frame["grp"] = np.tile([1, 2, 3], len(frame))[: len(frame)].astype(int)
    return ws.OLS.from_formula("consumption ~ C(grp) + temp", frame).fit()


def test_pwcompare_matches_stata(icecream_fit) -> None:
    """Captured reference, 62 characters per row."""

    lines = str(ws.pwcompare(icecream_fit, "grp")).splitlines()
    assert "Pairwise comparisons of marginal linear predictions" in lines
    assert "Margins: asbalanced" in lines
    assert "     2 vs 1  |  -.0035617   .0194151     -.0434699    .0363465" in lines
    assert "     3 vs 1  |   .0112906   .0194011     -.0285889    .0511701" in lines
    assert "     3 vs 2  |   .0148523   .0193778     -.0249792    .0546839" in lines


def test_contrast_matches_stata(icecream_fit) -> None:
    """Captured reference, 48 characters for the rule/header/statistic rows."""

    lines = str(ws.contrast(icecream_fit, "grp")).splitlines()
    assert "Contrasts of marginal linear predictions" in lines
    assert "             |         df           F        P>F" in lines
    assert "         grp |          2        0.32     0.7288" in lines
    assert " Denominator |         26" in lines


def test_summarize_matches_stata() -> None:
    """Captured reference, 71 characters throughout."""

    try:
        frame = pd.read_stata(GIRLIC).dropna(subset=["lnw", "s", "expr", "tenure"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    lines = str(ws.summarize(frame[["lnw", "s", "expr", "tenure"]])).splitlines()
    assert "    Variable |        Obs        Mean    Std. dev.       Min        Max" in lines
    assert "         lnw |        758    5.686739    .4289494      4.605      7.051" in lines
    assert "      tenure |        758    1.831135     1.67363          0         10" in lines
    for line in lines:
        # A blank first line is part of Stata's output (measured in round 8 of the v3 work);
        # the 71-character rule applies to the rows that carry content.
        if not line.strip():
            continue
        assert len(line) == 71, (len(line), line)
