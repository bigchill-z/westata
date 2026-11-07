"""`LR test of alpha=0` compares nbreg against POISSON, not against the constant-only model.

Measured on `nbreg mpg weight fgn` against a live Stata:

    LR test of alpha=0: chibar2(01) = 0.00                 Prob >= chibar2 = 1.000

and the arithmetic behind it:

    ll(nbreg)    = -196.130285
    ll(poisson)  = -196.129885
    2*(difference) = -0.0008, which Stata clamps to 0.00

This package used `e(ll_0)` -- the CONSTANT-ONLY likelihood, which belongs to the header's model
test -- and therefore printed 69.88: a model test wearing the dispersion test's label.

`e(ll_0)` is still recorded and still drives the header, so the two quantities now live under
separate names: `e(ll_0)` for the constant-only comparison, `e(ll_poisson_only)` for this one.
`xtpoisson`/`tnbreg` have no `ll_poisson_only` and keep e(ll_0), which is the comparison their own
help files describe.
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


def test_the_poisson_likelihood_is_recorded(nbreg) -> None:
    value = float(nbreg.e["ll_poisson_only"])
    assert np.isfinite(value)
    assert value == pytest.approx(-196.129885, abs=1e-4)


def test_the_constant_only_likelihood_is_a_different_number(nbreg) -> None:
    """If these ever coincide, the two comparisons have been conflated again."""

    assert float(nbreg.e["ll_0"]) != pytest.approx(
        float(nbreg.e["ll_poisson_only"]), rel=1e-6
    )


def test_the_alpha_test_prints_zero(nbreg) -> None:
    """Stata: chibar2(01) = 0.00 -- the negative likelihood difference is clamped."""

    tail = str(nbreg.summary(style="stata")).splitlines()[-1]
    assert "LR test of alpha=0" in tail
    assert "chibar2(01) = 0.00" in tail


def test_the_alpha_test_is_not_the_model_test(nbreg) -> None:
    """69.88 is the MODEL statistic; it must not appear on the dispersion line."""

    tail = str(nbreg.summary(style="stata")).splitlines()[-1]
    assert "chibar2(01) = 69" not in tail


def test_the_header_still_carries_the_model_test(nbreg) -> None:
    """`LR chi2(1) = 69.88` belongs in the header, and stays there."""

    rendered = str(nbreg.summary(style="stata"))
    assert "LR chi2(1)" in rendered
    assert "69.88" in rendered


def test_poisson_itself_is_unaffected(frame) -> None:
    """No k_aux, so no ll_poisson_only and no alpha test line."""

    fitted = ws.Poisson.from_formula("mpg ~ weight + fgn", frame).fit()
    assert "ll_poisson_only" not in (fitted.e or {})
    assert "LR test of alpha=0" not in str(fitted.summary(style="stata"))
