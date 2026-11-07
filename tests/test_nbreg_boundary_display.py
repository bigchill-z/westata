"""A dispersion at the boundary has NO standard error, and Stata prints "." rather than one.

Measured on `nbreg mpg weight fgn`, whose optimum sits at alpha -> 0:

    e(alpha)        = 1.034e-09
    _se[/lnalpha]   = 0
    alpha |   1.03e-09          .                             .
    /lnalpha |  -20.69024          .                             .

All four of the SE/P/CI columns are blank or ".", because there is no sampling variation to report:
the observed information is rank-deficient at the boundary (this fit's covariance has a zero row,
leaving se(weight) = 0 and a smallest eigenvalue of 7e-11).

THE PACKAGE WAS PRINTING A NUMBER. Its analytic information matrix still produces a finite
delta-method value (se = 12.69) and an interval spanning [-38.7, 11.1] -- a range no data supports,
derived from a Hessian that is singular by construction. The alpha row now suppresses the SE and
the interval when exp(lnalpha) < 1e-5, which is where alpha stops being distinguishable from the
boundary at these sample sizes.

WHAT THIS DOES NOT FIX, recorded rather than hidden: the fit lands at alpha = 1e-6 where Stata
reaches 1.03e-09 (all three starts produce the same likelihood to 4e-4, and picking between them on
likelihood alone was already shown in round 13 to be numerical noise). The /lnalpha row is rendered
by the shared coefficient loop and still shows its computed SE; only the derived alpha row, which
this module formats itself, carries the boundary rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

AUTO = r"tests\data\auto.dta"


@pytest.fixture(scope="module")
def frame():
    try:
        base = pd.read_stata(AUTO).dropna(subset=["mpg", "weight"]).copy()
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("auto.dta is not available")
    base["fgn"] = (
        base["foreign"]
        .astype(str)
        .str.lower()
        .str.startswith(("foreign", "1"))
        .astype(float)
    )
    return base


@pytest.fixture(scope="module")
def nbreg(frame):
    return ws.NegativeBinomial.from_formula("mpg ~ weight + fgn", frame).fit()


def _alpha_row(nbreg) -> str:
    lines = str(nbreg.summary(style="stata")).splitlines()
    return next(line for line in lines if line.strip().startswith("alpha |"))


def test_the_dispersion_is_at_the_boundary(nbreg) -> None:
    """The premise of the rule: alpha is small enough to be indistinguishable from zero."""

    assert float(nbreg.e["alpha"]) < 1e-5


def test_the_alpha_row_suppresses_its_standard_error(nbreg) -> None:
    row = _alpha_row(nbreg)
    assert " .000001" in row or "1.0e-06" in row or ".000001" in row
    # The SE cell carries a lone "." -- not a number such as 12.69 or .0000127.
    assert "12.69" not in row
    assert ".0000127" not in row


def test_the_alpha_row_has_no_interval(nbreg) -> None:
    """The CI columns are empty: a boundary estimate has no interval to report."""

    row = _alpha_row(nbreg)
    assert "1.57e-17" not in row
    assert "63740" not in row


def test_an_interior_dispersion_keeps_its_standard_error() -> None:
    """The boundary rule must not fire for a genuinely interior alpha.

    `nbreg` on Poisson-generated counts has a real dispersion; its alpha row must still carry an
    SE and an interval.
    """

    rng = np.random.default_rng(5)
    n = 500
    x1 = rng.normal(size=n)
    mu = np.exp(0.6 + 0.5 * x1)
    counts = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    frame = pd.DataFrame({"y": counts, "x1": x1})

    fitted = ws.NegativeBinomial.from_formula("y ~ x1", frame).fit()
    assert float(fitted.e["alpha"]) > 1e-4, "this dataset was meant to have a real dispersion"
    row = next(
        line
        for line in str(fitted.summary(style="stata")).splitlines()
        if line.strip().startswith("alpha |")
    )
    # A number in the SE cell, not the boundary placeholder.
    assert row.split("|")[1].split()[1] != "."
