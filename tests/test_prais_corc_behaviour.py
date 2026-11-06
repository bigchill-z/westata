"""prais, corc: the option exists and behaves as Stata does, including when it fails.

The report's C08-05 reads "westata is the default PW, the book is `, corc`" -- i.e. two different
estimators, not a defect. Checking against a live Stata on icecream.dta confirmed more than that:

    prais consumption price income, corc
        rho |   .9724118
        convergence not achieved
        r(430);

Stata ITSELF fails to converge on this data. westata does the same, and reaches a nearly identical
value: at maxiter=500, tolerance=1e-6 it returns rho = .9724596 against Stata's .9724118, a 0.005%
difference on a non-converged estimate.

One real improvement came out of the comparison. Stata prints the rho it reached BEFORE reporting
r(430); westata raised a bare ConvergenceError with no value, so a caller could not tell .97 that
is still creeping from one oscillating wildly without re-running at a looser tolerance. The
exception now carries `.rho` and prints it in the message.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\icecream.dta"

#: Captured from `prais consumption price income, corc` -- Stata's own non-converged value.
STATA_CORC_RHO = 0.9724118


@pytest.fixture(scope="module")
def frame():
    try:
        return pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("icecream.dta is not available")


def test_corc_is_an_accepted_option(frame) -> None:
    """It exists; the report's phrasing ("missing") is about the default, not the option."""

    with pytest.raises(ws.ConvergenceError):
        ws.prais(frame, "consumption ~ price + income", corc=True)


def test_the_failure_carries_the_value_it_reached(frame) -> None:
    """Stata prints rho before r(430); the exception should too."""

    with pytest.raises(ws.ConvergenceError) as caught:
        ws.prais(frame, "consumption ~ price + income", corc=True)
    reached = getattr(caught.value, "rho", None)
    assert reached is not None
    assert reached == pytest.approx(STATA_CORC_RHO, abs=5e-4)


def test_the_message_names_the_last_value(frame) -> None:
    with pytest.raises(ws.ConvergenceError) as caught:
        ws.prais(frame, "consumption ~ price + income", corc=True)
    assert "last value" in str(caught.value)


def test_a_looser_tolerance_reproduces_statas_value(frame) -> None:
    """Stata's .9724118; ours .9724596 at maxiter=500, tolerance=1e-6."""

    result = ws.prais(
        frame, "consumption ~ price + income", corc=True, maxiter=500, tolerance=1e-6
    )
    assert float(result.rho) == pytest.approx(STATA_CORC_RHO, abs=5e-4)


def test_pw_and_corc_give_different_estimates(frame) -> None:
    """Which is the whole point of the report's C08-05 remark -- not a defect."""

    pw = ws.prais(frame, "consumption ~ price + income")
    corc = ws.prais(
        frame, "consumption ~ price + income", corc=True, maxiter=500, tolerance=1e-6
    )
    assert abs(float(pw.rho) - float(corc.rho)) > 0.01
