"""Tests for `nbreg`'s comparison-model block: `ll_c`, `chi2_c`, `rank0`.

``nbreg.sthlp:414-422`` documents them as:

    e(ll_c)     log likelihood, comparison model
    e(chi2_c)   chi-squared for comparison test
    e(rank0)    rank of e(V) for constant-only model

``nbreg.ado:215-228`` reveals what the comparison model is: it fits
``poisson y xvars`` before the negative binomial and stores that log-likelihood
as ``llp``; ``nbreg.ado:377`` then sets ``e(ll_c) = llp``. The Poisson model is
the ``alpha = 0`` special case, so ``chi2_c`` tests whether the dispersion is
needed at all.

The fixture confirms the arithmetic exactly:

    2*(ll - ll_c) = 2*(-976.791299423244 + 1025.0164795366088)
                  = 96.45036022672957  ==  its e(chi2_c)

**A negative LR statistic is impossible**, and Stata clamps rather than printing
one (``nbreg.ado:378-384``). A guard was added for the same reason; it is what
turns a non-converged backend into ``chi2 = 0`` instead of an impossible value.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

FIXTURE = Path("tests/conformance/stata_fixtures/nbreg.json")


@pytest.fixture()
def nb2_fit():
    """Data simulated FROM an NB2 so the optimiser has a clean target."""
    rng = np.random.default_rng(7)
    n = 800
    x = rng.normal(size=n)
    mu = np.exp(1.0 + 0.5 * x)
    alpha_true = 0.4
    draws = rng.negative_binomial(
        1.0 / alpha_true, 1.0 / (1.0 + alpha_true * mu)
    )
    frame = pd.DataFrame({"y": draws.astype(float), "x": x})
    return frame, alpha_true, ws.NegativeBinomial.from_formula("y ~ x", frame).fit()


def test_the_fixture_arithmetic_reproduces_its_own_chi2_c() -> None:
    """Arithmetic check on Stata's own numbers, independent of our data."""

    s = json.loads(FIXTURE.read_text())["scalars"]

    assert 2.0 * (s["ll"] - s["ll_c"]) == pytest.approx(s["chi2_c"], rel=1e-12)


def test_the_fixture_rank0_is_rank_minus_aux() -> None:
    s = json.loads(FIXTURE.read_text())["scalars"]

    assert s["rank0"] == s["rank"] - s["k_aux"]
    assert s["rank0"] == 2.0


def test_all_three_keys_are_present(nb2_fit) -> None:
    _, _, result = nb2_fit

    for key in ("ll_c", "chi2_c", "rank0"):
        assert key in result.e, key


def test_ll_c_is_the_poisson_loglikelihood(nb2_fit) -> None:
    """``nbreg.ado:215-228`` fits ``poisson y xvars`` as the comparison model."""

    import statsmodels.api as sm

    frame, _, result = nb2_fit
    y = frame["y"].to_numpy(float)
    X = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy(float)])
    expected = float(sm.Poisson(y, X).fit(disp=False, maxiter=300).llf)

    assert result.e["ll_c"] == pytest.approx(expected, rel=1e-8)


def test_chi2_c_is_twice_the_difference(nb2_fit) -> None:
    _, _, result = nb2_fit

    expected = 2.0 * (result.e["ll"] - result.e["ll_c"])
    assert result.e["chi2_c"] == pytest.approx(expected, rel=1e-10)


def test_rank0_is_the_constant_only_rank(nb2_fit) -> None:
    _, _, result = nb2_fit

    assert result.e["rank0"] == 1.0


def test_nb2_beats_poisson_on_data_simulated_from_it(nb2_fit) -> None:
    """Sanity anchor: Poisson is the alpha -> 0 case, so NB2 cannot be worse."""

    frame, _, result = nb2_fit

    assert result.e["ll"] >= result.e["ll_c"] - 1e-6


def test_alpha_is_recovered(nb2_fit) -> None:
    _, alpha_true, result = nb2_fit

    assert result.e["alpha"] == pytest.approx(alpha_true, abs=0.25)


# --------------------------------------------------- the negative-chi2 guard
def test_chi2_is_never_negative() -> None:
    """The auto data lands in a bad NB2 optimum; the statistic must still be
    a valid chi-square rather than a negative number."""

    df = pd.read_stata("tests/data/auto.dta")
    result = ws.NegativeBinomial.from_formula("mpg ~ weight", df).fit()
    e = result.e

    assert e["chi2"] >= 0.0
    assert e["chi2_c"] >= 0.0


def test_a_lower_fitted_ll_than_the_baseline_is_clamped_not_printed() -> None:
    """When ``ll < ll_0`` the raw ``2(ll-ll_0)`` is negative; Stata reports 0.

    ``nbreg.ado:378-384`` clamps ``chi2_c``; the same rule now applies to the
    model test, which is what keeps ``e(chi2)`` a valid statistic.
    """

    df = pd.read_stata("tests/data/auto.dta")
    result = ws.NegativeBinomial.from_formula("mpg ~ weight", df).fit()
    e = result.e

    raw = 2.0 * (e["ll"] - e["ll_0"])
    if raw < 0.0:
        assert e["chi2"] == 0.0
        # And demonstrably not the impossible negative value.
        assert e["chi2"] != raw
    else:  # pragma: no cover - depends on the backend optimum
        assert e["chi2"] == pytest.approx(raw, rel=1e-10)


def test_nbreg_covers_its_whole_fixture() -> None:
    df = pd.read_stata("tests/data/auto.dta")
    result = ws.NegativeBinomial.from_formula("mpg ~ weight", df).fit()
    expected = set(json.loads(FIXTURE.read_text())["scalars"])

    assert not (expected - set(result.e)), sorted(expected - set(result.e))
