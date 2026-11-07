"""teffects nnmatch: the ATE is 7% off and the matching rule is not yet understood.

Rounds 201-203 fixed the missing intercept in three places in `treatreg_effects.py`, after which
`teffects ipw`, `aipw` and `ra` all match Stata exactly. `teffects nnmatch` does NOT, and its
structure is different enough that the same fix does not apply -- it is a MATCHING estimator, with no
regression to carry a constant:

    teffects nnmatch (re78 <covariates>) (treat)

        Stata        1906.126     (ATET 2453.076)
        westata      2039.4582    (ATET 2277.0212)
        hand-rolled  1879.2743    (sample-covariance Mahalanobis, 1 neighbour, ATE)

THREE different numbers, so this is not one rule being misapplied: the matching SET or the
tie/weighting convention differs from Stata's. `teffects_ra/ipw/aipw` share an estimating-equation
structure that could be checked against a live fit in one step; nnmatch needs its match set
inspected neighbour by neighbour, which is a separate piece of work.

WHAT THIS FILE PINS, so the gap cannot be mistaken for agreement:
  * the estimator runs and produces a finite ATE;
  * it does NOT match Stata, at the tolerance Stata prints;
  * and the three-way disagreement above is recorded in the docstring of the guard.

The correct next step is to export Stata's own match set (`teffects nnmatch ..., generate()` writes
the matched-observation indices the way `teffects psmatch` does) and compare it neighbour by
neighbour, rather than continuing to compare only the final estimate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

STATA_NNMATCH_ATE = 1906.126
STATA_NNMATCH_ATET = 2453.076


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")


def test_the_estimator_produces_a_finite_ate(frame) -> None:
    fit = ws.NearestNeighbourMatching(
        frame["re78"].to_numpy(float),
        frame["treat"].to_numpy(float),
        frame[COVARIATES].to_numpy(float),
    ).fit()
    assert np.isfinite(float(fit.estimate))
    assert 0.0 < float(fit.estimate) < 10000.0


def test_it_does_not_yet_match_stata(frame) -> None:
    """Recorded as an OPEN gap. Replace with the equality assertion when the match set is aligned."""

    fit = ws.NearestNeighbourMatching(
        frame["re78"].to_numpy(float),
        frame["treat"].to_numpy(float),
        frame[COVARIATES].to_numpy(float),
    ).fit()
    gap = abs(float(fit.estimate) - STATA_NNMATCH_ATE)
    assert gap > 10.0, "if this fails, the matching was fixed -- pin the equality instead"
    assert gap < 500.0, "a gap this large would mean the estimator broke, not just disagreed"


def test_the_ate_is_high_and_the_atet_is_low(frame) -> None:
    """The two estimands move in OPPOSITE directions relative to Stata, which I first got wrong.

    I assumed both sat below Stata's and wrote the test that way; it failed on ATE (2039.46 is
    ABOVE 1906.126). Opposite directions rule out a single tie-handling slip -- a rule that
    admitted extra matches would shift both the same way, because both arms would gain neighbours.
    Something about WHICH observations get matched differs.
    """

    y = frame["re78"].to_numpy(float)
    treated = frame["treat"].to_numpy(float)
    x = frame[COVARIATES].to_numpy(float)
    ate = float(ws.NearestNeighbourMatching(y, treated, x).fit().estimate)
    atet = float(ws.NearestNeighbourMatching(y, treated, x, estimand="atet").fit().estimate)
    assert ate > STATA_NNMATCH_ATE
    assert atet < STATA_NNMATCH_ATET