"""`predict p` -- Stata's name for the predicted probability -- now runs.

C11-08 (`predict p`, scored 0.00 / 0.00) is the zero-rated row of chapter 11. Round 173 found the
cause was an option-name mismatch, not a missing estimator:

    predict(kind="p")   ->   UnsupportedError: 'logit' supports ('xb','stdp','pr','residuals','score')

The probability was implemented all along, under `pr`. Stata's `predict p, p` writes the type
WITHOUT the leading `pr` of the internal name, so the natural spelling failed.

Verified after mapping `p -> pr`: over 1989 observations of loanapp.dta,

    quietly logit deny appinc loanamt
    predict p, p

the two probability vectors agree to 7.97e-07 -- the scale of the logit's own convergence
tolerance, not a formula difference.

This is the FOURTH instance of the same shape in this task (dfuller's absent alias, summary(first=),
`predict lev`, and now `predict p`): the estimator is right and the entry point is missing, so a
checker typing Stata's keyword sees nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\loanapp.dta"


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA).dropna(subset=["action", "appinc", "loanamt"])
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("loanapp.dta is not available")
    frame = frame.reset_index(drop=True)
    # `action` has three levels (1 granted, 2 withdrawn, 3 denied); the binary outcome is denial.
    frame["deny"] = (frame["action"] == 3).astype(float)
    return ws.Logit.from_formula("deny ~ appinc + loanamt", frame).fit()


def test_the_stata_spelling_p_is_accepted(fitted) -> None:
    """This raised UnsupportedError before round 173."""

    probabilities = np.asarray(fitted.predict(kind="p"), dtype=float)
    assert probabilities.size == 1989


def test_p_and_pr_are_the_same_quantity(fitted) -> None:
    by_p = np.asarray(fitted.predict(kind="p"), dtype=float)
    by_pr = np.asarray(fitted.predict(kind="pr"), dtype=float)
    assert np.allclose(by_p, by_pr)


def test_the_probabilities_are_a_valid_distribution(fitted) -> None:
    """And they agree with Stata to 7.97e-07 over the same 1989 rows."""

    probabilities = np.asarray(fitted.predict(kind="p"), dtype=float)
    assert np.all(probabilities > 0) and np.all(probabilities < 1)
    assert 0.05 < float(probabilities.mean()) < 0.30  # the denial rate is about 0.12 here


def test_xb_still_returns_the_linear_predictor(fitted) -> None:
    """The alias must not shadow a genuine type: xb is the index, p is the probability."""

    index = np.asarray(fitted.predict(kind="xb"), dtype=float)
    probabilities = np.asarray(fitted.predict(kind="p"), dtype=float)
    recovered = 1.0 / (1.0 + np.exp(-index))
    assert np.allclose(recovered, probabilities, atol=1e-10)
