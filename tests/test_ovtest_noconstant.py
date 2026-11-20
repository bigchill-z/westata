"""`estat ovtest` after `regress, noconstant`: Stata refuses, and so must this.

Measured on a live StataNow 19.5 run (2026-11-15), both paths:

    . reg y x1 x2, noconstant
    . estat ovtest
    not appropriate after regress, nocons
    r(301);

    . estat ovtest, rhs
    not appropriate after regress, nocons
    r(301);

The ado says so in its FIRST executable line, before it reads any option: `ovtest.ado:7`
is `_isfit cons anovaok`, and `cons` is exactly this refusal. The package had no such
guard, and the DEFAULT path -- powers of the rescaled fitted values -- returned

    F(3, 195) = 64.86, p = 3.9e-29

on a model Stata rejects outright. That is a catastrophic false rejection (p < 1e-28 on
data with no functional-form problem), not a rounding difference, and it is the worst
failure shape this project has recorded: a finite, plausible-looking, five-star number
for a test that is not defined.

For contrast, `estat imtest` DOES accept a noconstant fit (also measured), so this is not
a blanket rule for the `estat` family -- it is `ovtest`'s and `hettest`'s own precondition.

`estat hettest` already carried the guard (hettest.py, "not appropriate after a
noconstant fit"); ovtest was the one that did not.
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 200, seed: int = 20261115) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return pd.DataFrame({"y": 1.0 + 2.0 * x1 + 0.3 * x2 + rng.normal(size=n), "x1": x1, "x2": x2})


@pytest.fixture(scope="module")
def noconstant_fit():
    return ws.Regress.from_formula("y ~ x1 + x2 - 1", _frame()).fit()


def test_the_default_path_refuses_a_noconstant_fit(noconstant_fit) -> None:
    """The dangerous path: it used to answer F(3, 195) = 64.86, p = 3.9e-29."""

    with pytest.raises(ValueError, match="noconstant"):
        ws.estat_omitted_variables_test(noconstant_fit)


def test_the_rhs_path_refuses_a_noconstant_fit(noconstant_fit) -> None:
    with pytest.raises(ValueError, match="noconstant"):
        ws.estat_omitted_variables_test(noconstant_fit, rhs=True)


def test_the_message_names_statas_own_diagnosis(noconstant_fit) -> None:
    """A reader must be able to map the exception onto the r(301) Stata gives."""

    with pytest.raises(ValueError) as caught:
        ws.estat_omitted_variables_test(noconstant_fit)
    message = str(caught.value)
    assert "not appropriate after regress, nocons" in message
    assert "r(301)" in message
    # And the ADO line that makes it a precondition rather than an option check.
    assert "ovtest.ado:7" in message


def test_the_refusal_precedes_the_power_arithmetic(noconstant_fit) -> None:
    """It is a precondition, not a post-hoc sanity check on the result.

    If the guard ran AFTER the auxiliary regression, a singular design could raise
    something else first (or return), and the exception type would depend on the data.
    The guard sits with the other eligibility checks, before `resid` is even read.
    """

    import inspect

    import westata.postestimation.ovtest as module

    source = inspect.getsource(module.estat_omitted_variables_test)
    guard = source.index("not appropriate after a noconstant fit")
    assert guard < source.index("resid = np.asarray(result.resid"), (
        "the noconstant guard must run before the residuals are read"
    )


def test_a_model_with_a_constant_is_unaffected() -> None:
    """The guard must not misfire: an ordinary fit still returns three df."""

    fitted = ws.Regress.from_formula("y ~ x1 + x2", _frame()).fit()
    assert ws.estat_omitted_variables_test(fitted).df == 3
    assert ws.estat_omitted_variables_test(fitted, rhs=True).df == 6


def test_the_guard_reads_the_design_not_the_formula() -> None:
    """A design whose constant column is ABSENT is what triggers it.

    `- 1` is how the formula spells it, but the guard must key off the retained design
    so an estimator that builds its own intercept-free design is covered too.
    """

    fitted = ws.Regress.from_formula("y ~ x1 + x2 - 1", _frame()).fit()
    names = [str(name) for name in fitted.design_matrix[1]]
    assert "Intercept" not in names, names
    with pytest.raises(ValueError, match="noconstant"):
        ws.estat_omitted_variables_test(fitted)
