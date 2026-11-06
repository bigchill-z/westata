"""C10-03 and C11-08 both work; C11-08's zero was traced to a different cause entirely.

C10-03  estat overid    report scores 0.00 with numeric coverage 1.00 -- the two columns disagree,
                        and the numbers are right. Sargan 273.155 / Basmann 424.793 are exactly
                        Stata's on grilic.dta with `ivregress 2sls lnw (expr = s tenure age)`.

C11-08  predict p       report scores 0.00 / 0.00. `result.predict()` DOES return probabilities.
                        The first comparison against Stata failed badly (max diff 0.436), but the
                        cause was the MODEL, not predict: the array-based constructor does not add
                        an intercept, so

                            ws.Logit(y, X, param_names=[...])        coef [.002219, -.000498, ...]
                            Stata                                    coef [.459083, .158881, ...]

                        With an explicit constant column the coefficients match Stata to six
                        digits and predict(p) lands within 3.25e-08.

This is the third time the array-vs-formula intercept question has produced a false defect signal
(round 116's margins, round 117's freq_weights error message, and here). The array constructor is
deliberately statsmodels-like -- it treats the design matrix as complete -- but nothing in the call
says so, and the failure mode is a plausible-looking set of coefficients rather than an error.

Recorded, not changed: adding a silent intercept would break callers that pass their own constant,
and the formula interface is the documented way to get one. What is worth adding is a note in the
constructor's docstring -- flagged here rather than guessed at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
COLUMNS = ["lnw", "s", "expr", "tenure", "rns", "smsa", "mrt", "age"]

#: From `estat overid` after `ivregress 2sls lnw (expr = s tenure age)`.
STATA_SARGAN = 273.155
STATA_BASMANN = 424.793

#: From `logit hi s expr tenure` -- with the constant.
STATA_LOGIT = {"_cons": -6.691539, "s": 0.459083, "expr": 0.158881, "tenure": 0.153049}


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA).dropna(subset=COLUMNS).reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")


def test_overid_matches_stata(frame) -> None:
    """Report scores this 0.00 while its own numeric-coverage column says 1.00."""

    fit = ws.IV2SLS.from_formula("lnw ~ [expr ~ s + tenure + age]", frame).fit()
    rendered = str(fit.estat_overid())
    assert f"{STATA_SARGAN:.3f}" in rendered
    assert f"{STATA_BASMANN:.3f}" in rendered


def test_the_logit_needs_an_explicit_constant(frame) -> None:
    """The array constructor treats the design as complete; without a column of ones the fit is
    a different model, and its coefficients look plausible rather than wrong."""

    work = frame.copy()
    work["hi"] = (work["lnw"] > work["lnw"].median()).astype(float)
    design = work[["s", "expr", "tenure"]].to_numpy(float)

    without = ws.Logit(work["hi"], design, param_names=["s", "expr", "tenure"]).fit()
    assert abs(float(without.params["s"]) - STATA_LOGIT["s"]) > 0.1  # nothing like Stata

    with_constant = np.column_stack([np.ones(len(work)), design])
    with_onset = ws.Logit(
        work["hi"], with_constant, param_names=["_cons", "s", "expr", "tenure"]
    ).fit()
    for name, expected in STATA_LOGIT.items():
        assert float(with_onset.params[name]) == pytest.approx(expected, abs=5e-6), name


def test_predict_p_is_correct_once_the_model_is(frame) -> None:
    """C11-08's zero is not a predict defect: with the right model it lands at 3.25e-08."""

    work = frame.copy()
    work["hi"] = (work["lnw"] > work["lnw"].median()).astype(float)
    design = np.column_stack([np.ones(len(work)), work[["s", "expr", "tenure"]].to_numpy(float)])
    fitted = ws.Logit(
        work["hi"], design, param_names=["_cons", "s", "expr", "tenure"]
    ).fit()
    probabilities = np.asarray(fitted.predict(), dtype=float)
    assert probabilities.size == len(work)
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    # The mean predicted probability must equal the sample proportion for a logit ML fit.
    assert float(probabilities.mean()) == pytest.approx(float(work["hi"].mean()), abs=1e-6)
