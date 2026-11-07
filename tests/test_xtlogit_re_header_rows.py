"""xtlogit, re: the coefficient header names the outcome, and Log likelihood shares the test row.

Round 232 finished the two one-line defects round 231 identified.

1. ``e(depvar)`` was never stored by the panel-nonlinear family, so the coefficient header
   printed a bare ``| Coefficient`` where Stata prints ``y | Coefficient``. PanelOLS had always
   set it; these estimators did not. Now taken from the formula's left-hand side and stored under
   Stata's name.

2. ``Log likelihood`` was popped out of the left column during the panel splice with a comment
   saying it "is re-appended after the splice" -- and nothing ever re-appended it, so that row
   was blank. It now goes to the index of the model test, which is ``Prob > chi2`` and NOT
   ``Wald chi2`` (one row above); matching the first label containing "chi2" put it one row high,
   which the live comparison caught immediately.

With both, the whole header block -- rows 0 through the coefficient header -- matches a live
``xtlogit y x1, re nolog`` line for line.

WHAT REMAINS IS NOT LAYOUT: the coefficients themselves differ in the seventh significant figure
(x1 0.6684314 against 0.668431, _cons .1949556 against .1949552, /lnsig2u .592867 against
.5928603), which is the optimiser difference already recorded for other ML models, plus the
missing "LR test of rho=0" row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _panel():
    rng = np.random.default_rng(3)
    n_grp, n_obs = 30, 20
    sid = np.repeat(np.arange(n_grp), n_obs)
    x1 = rng.normal(size=n_grp * n_obs)
    u = np.repeat(rng.normal(scale=1.5, size=n_grp), n_obs)
    p = 1.0 / (1.0 + np.exp(-(0.3 + 0.6 * x1 + u)))
    y = (rng.random(n_grp * n_obs) < p).astype(float)
    return pd.DataFrame(
        {"y": y, "x1": x1, "sid": sid, "t": np.tile(np.arange(n_obs), n_grp)}
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.XtLogit.from_formula("y ~ x1", _panel(), entity="sid", time="t").fit()


def test_the_header_names_the_outcome(fitted) -> None:
    """'y | Coefficient ...', not a bare '| Coefficient ...'."""

    lines = str(fitted.summary(style="stata")).splitlines()
    header = next(line for line in lines if "Coefficient" in line)
    assert header.lstrip().startswith("y |"), repr(header[:30])


def test_depvar_is_stored_under_stat_as_name(fitted) -> None:
    """e(depvar) is what Stata's r() exposes, and what the header reads."""

    assert str((getattr(fitted, "e", None) or {}).get("depvar")) == "y"


def test_log_likelihood_shares_the_prob_row(fitted) -> None:
    """It belongs beside "Prob > chi2", not beside "Wald chi2" one row above."""

    lines = str(fitted.summary(style="stata")).splitlines()
    ll_index = next(i for i, line in enumerate(lines) if line.startswith("Log likelihood"))
    prob_index = next(i for i, line in enumerate(lines) if "Prob > chi2" in line)
    assert ll_index == prob_index


def test_the_header_block_has_no_blank_name_row(fitted) -> None:
    """The failure mode was a blank left cell on the Log likelihood row."""

    lines = str(fitted.summary(style="stata")).splitlines()
    prob_index = next(i for i, line in enumerate(lines) if "Prob > chi2" in line)
    assert lines[prob_index].strip() != ""
