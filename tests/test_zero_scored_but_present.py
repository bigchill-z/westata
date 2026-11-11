"""Rows the report scored 0.00 that are in fact implemented -- verified one by one.

The report's score column is a snapshot, not a work list, and this session has repeatedly found
0.00 rows that were already present. Round 212 (C20-02, the exporters) and round 213 (C11-08,
`predict p`) are the two clearest. The guards below pin the ones that can be checked without
guessing an API, so a future round does not re-implement what already exists.

C11-08  predict p / pr            WORKS, two spellings plus the default
C20-02  to_markdown/to_latex      WORKS (round 212)

STILL GENUINELY ABSENT, and confirmed absent rather than assumed:
C12-11     xtserial               no such symbol
C15-03     caliper                no such argument

SINCE IMPLEMENTED (both were 0.00 in the original snapshot):
C18-01     synth                  landed 2026-11-09 (default + extended, E2)
C18-02     synth2                 landed 2026-11-11 (non-graphics, E2)

STILL OPEN, not re-audited this round:
C12-09     xtreg ..., fe i.year   per-year coefficient rows are absorbed by PanelOLS
C15-04     predict, ps            psmatch refuses a continuous treatment with a clear error
                                  ("treatment must be coded 0/1"), which is correct behaviour for
                                  the call I made rather than proof of a gap -- needs the right
                                  fixture before it can be judged.
"""

from __future__ import annotations

import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"


@pytest.fixture(scope="module")
def logit_fit():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["rns", "s", "expr"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")
    return ws.Logit.from_formula("rns ~ s + expr", frame).fit()


def test_predict_p_returns_probabilities(logit_fit) -> None:
    """C11-08 scored 0.00/0.00. The alias works and agrees with the default prediction."""

    alias = logit_fit.predict(kind="p")
    explicit = logit_fit.predict(kind="pr")
    default = logit_fit.predict()
    assert float(alias[0]) == pytest.approx(float(explicit[0]), abs=1e-12)
    assert float(alias[0]) == pytest.approx(float(default[0]), abs=1e-12)
    # A probability, not a log-odds.
    assert 0.0 < float(alias[0]) < 1.0


def test_predict_p_matches_the_manual_link_inverse(logit_fit) -> None:
    """The number is the fitted probability, so it must equal expit(xb)."""

    import numpy as np

    xb = logit_fit.predict(kind="xb")
    expected = 1.0 / (1.0 + np.exp(-np.asarray(xb, dtype=float)))
    got = np.asarray(logit_fit.predict(kind="p"), dtype=float)
    assert float(np.max(np.abs(expected - got))) < 1e-12


def test_the_absent_features_are_still_absent() -> None:
    """Pinned so that 'not implemented' is a checked statement rather than an assumption.

    ``synth`` was removed on 2026-11-09 (B-grade default path) and ``synth2``
    on 2026-11-11 (non-graphics feature set); both are now covered by their
    own tests and E2 fixture comparisons.  Only ``xtserial`` remains absent.
    """

    for name in ("xtserial",):
        assert not hasattr(ws, name), f"{name} now exists -- update the report row"
    assert callable(ws.synth)
    assert callable(ws.synth2)
    assert callable(ws.synth)
