"""`xtreg, fd` does not exist in Stata, and this package says it does.

Discovered while trying to diff `FirstDifferenceOLS` against the real thing:

    . xtreg lny lnk1 lnk2 lnlabor, fd
    option fd not allowed
    r(198);

`help xtreg` lists only re, cre, be, fe, mle and pa. panel.py has known this since the estimator was
written -- its comment says "``xtreg, fd`` is deliberately ABSENT: it could not be run on this
install ... so its banner is unknown" -- but `e(cmd)` still reports the string `xtreg, fd`, which
reads as though the command exists.

WHAT STATA ACTUALLY DOES for first differences is `regress D.y D.x` on an xtset panel:

    N = 768 = 816 - 48, which matches this package's nobs exactly
    coefficients INCLUDE a constant: _cons = 0.0075628

and this package omits it -- three parameters instead of four, with lnlabor off by 5%. The cause is
in the test that was supposed to protect the estimator: it builds its reference through an `ols()`
helper that prepends no constant column, so "no intercept" was pinned as correct.

These guards record the CURRENT state so a future fix shows up as a failure, and they name the
replacement assertions. They do not endorse the current behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"
FORMULA = "lny ~ lnk1 + lnk2 + lnlabor"

#: Captured from `regress D.lny D.lnk1 D.lnk2 D.lnlnlabor` -- the real first-difference fit.
STATA_PARAMS = {
    "D.lnk1": -0.1048258,
    "D.lnk2": -0.0621289,
    "D.lnlnlabor": 1.0405411,
    "_cons": 0.0075628,
}


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    frame = frame.copy()
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    frame["year"] = frame["year"].astype(int)
    return ws.FirstDifferenceOLS.from_formula(
        FORMULA, frame, entity="sid", time="year"
    ).fit()


def test_the_sample_size_matches_the_differenced_regression(fitted) -> None:
    """N = 768 = 816 - 48. This part is right."""

    assert int(fitted.nobs) == 768


def test_the_command_string_claims_a_stata_command_that_does_not_exist(fitted) -> None:
    """`e(cmd)` says "xtreg, fd"; Stata has no such option.

    When this starts failing because the label was corrected, replace it with an assertion that
    the string names the real spelling (`regress D.` or similar) instead.
    """

    assert fitted.e["cmd"] == "xtreg, fd"


def test_the_estimator_omits_the_constant_stata_keeps(fitted) -> None:
    """Three parameters where Stata has four, and the missing one is the intercept.

    THE DEFECT: differencing removes the entity effect but not a common time trend, so a constant
    belongs in the fit. Stata's _cons is 0.0075628. When this is fixed, assert
    `len(fitted.params) == 4` and check _cons against STATA_PARAMS["_cons"].
    """

    assert len(fitted.params) == 3
    assert "Intercept" not in list(fitted.params.index)


def test_the_slopes_are_off_by_the_missing_intercept(fitted) -> None:
    """lnlabor differs by ~5%; the others by much more. Recorded, not accepted."""

    mine = {name: float(value) for name, value in fitted.params.items()}
    # The names here drop the D. prefix Stata uses; compare by position after the omission.
    expected = [STATA_PARAMS["D.lnk1"], STATA_PARAMS["D.lnk2"], STATA_PARAMS["D.lnlnlabor"]]
    observed = [mine["lnk1"], mine["lnk2"], mine["lnlabor"]]

    assert observed[2] == pytest.approx(1.0933399, abs=1e-6)
    # lnlabor is within 5% of Stata's 1.0405411; the others are not close at all.
    assert abs(observed[2] - expected[2]) / abs(expected[2]) < 0.06
    assert abs(observed[0] - expected[0]) > 0.05
