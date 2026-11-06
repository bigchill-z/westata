"""The array constructors treat the design matrix as complete -- now documented.

Rounds 116, 117 and 129 each produced a false defect signal from the same misunderstanding: the
array-based discrete constructors add no intercept, while Stata's `logit y x1 x2` does. The failure
is silent -- a well-formed fit with plausible coefficients that disagree with Stata -- so it reads
as a numerical bug rather than a missing column of ones.

Nothing about the behaviour is changed here. Adding a silent intercept would break callers who
supply their own constant, and `from_formula` is the documented route to one. What was missing is
the statement of the contract, which now appears in Logit, Probit and Poisson.

These tests pin BOTH halves:
  * the contract itself (no intercept is added), so the docstring cannot drift from the code;
  * the correct way to match Stata, so the fix is discoverable from a failing test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\grilic.dta"
COLUMNS = ["lnw", "s", "expr", "tenure"]

#: From `logit hi s expr tenure` on grilic.dta with hi = (lnw > median).
STATA_COEF = {"_cons": -6.691539, "s": 0.459083, "expr": 0.158881, "tenure": 0.153049}


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA).dropna(subset=COLUMNS).reset_index(drop=True)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("grilic.dta is not available")


@pytest.mark.parametrize("constructor", ["Logit", "Probit", "Poisson"])
def test_the_constructors_document_the_missing_intercept(constructor: str) -> None:
    """Round 130 added this; the test keeps it from being deleted as boilerplate."""

    docstring = (getattr(ws, constructor).__doc__ or "")
    assert "NO INTERCEPT" in docstring, constructor


def test_the_array_constructor_adds_no_columns(frame) -> None:
    """The contract itself: two regressors in, two coefficients out."""

    work = frame.copy()
    work["hi"] = (work["lnw"] > work["lnw"].median()).astype(float)
    design = work[["s", "expr", "tenure"]].to_numpy(float)
    fitted = ws.Logit(work["hi"], design, param_names=["s", "expr", "tenure"]).fit()
    assert len(fitted.params) == 3
    assert "Intercept" not in list(fitted.params.index)


def test_passing_a_constant_column_reproduces_stata(frame) -> None:
    """The documented workaround, verified against the live reference values."""

    work = frame.copy()
    work["hi"] = (work["lnw"] > work["lnw"].median()).astype(float)
    design = np.column_stack([np.ones(len(work)), work[["s", "expr", "tenure"]].to_numpy(float)])
    fitted = ws.Logit(
        work["hi"], design, param_names=["_cons", "s", "expr", "tenure"]
    ).fit()
    for name, expected in STATA_COEF.items():
        assert float(fitted.params[name]) == pytest.approx(expected, abs=5e-6), name


def test_the_formula_interface_adds_the_constant(frame) -> None:
    """Which is why the docstring points callers here."""

    work = frame.copy()
    work["hi"] = (work["lnw"] > work["lnw"].median()).astype(float)
    fitted = ws.Logit.from_formula("hi ~ s + expr + tenure", work).fit()
    assert "Intercept" in list(fitted.params.index)
    for name, expected in STATA_COEF.items():
        key = "Intercept" if name == "_cons" else name
        assert float(fitted.params[key]) == pytest.approx(expected, abs=5e-6), name
