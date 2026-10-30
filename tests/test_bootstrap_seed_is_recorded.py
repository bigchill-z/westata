"""``e(seed)`` -- the bootstrap seed must be recorded, or an unseeded fit is unreproducible.

``quantile_boot.py``'s own comment promised that when no seed is given the code would
"pick a seed now and RECORD it, rather than silently fixing a constant (which would make
every unseeded run identical, unlike Stata)". It picked one, but the value lived in a
local variable and vanished: nothing recorded it, so the numbers from an unseeded
``bsqreg``/``sqreg`` fit could not be obtained a second time by anyone, including the
person who had just run it.

Stata's discipline is ``set seed`` + ``c(seed)``: a draw is reproducible from the seed the
session held. Recording ``e(seed)`` gives the same guarantee here -- pass it back to
``fit(seed=...)`` and the bootstrap replicates repeat exactly.

The distinction that matters, and the reason "it is random anyway" is not an excuse: a
draw that is random but RECORDED is reproducible; a draw that is random and unrecorded is
not. Only the second one loses information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 140, seed: int = 7001):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    return pd.DataFrame({"y": 1.0 + 0.8 * covariate + rng.normal(size=n), "x": covariate})


# --------------------------------------------------------------- the recording
def test_bsqreg_records_a_seed() -> None:
    fitted = ws.Bsqreg.from_formula("y ~ x", _frame()).fit(reps=15)

    assert "seed" in fitted.e
    assert isinstance(fitted.e["seed"], int)


def test_sqreg_records_a_seed() -> None:
    fitted = ws.Sqreg.from_formula("y ~ x", _frame(), quantiles=(0.25, 0.5)).fit(reps=10)

    assert "seed" in fitted.e


def test_the_recorded_seed_is_an_integer() -> None:
    """It has to be passable back to ``fit(seed=...)``, so an int not a numpy scalar."""

    fitted = ws.Bsqreg.from_formula("y ~ x", _frame()).fit(reps=10)

    assert type(fitted.e["seed"]) is int


# ------------------------------------------------------------ reproducibility
def test_the_recorded_seed_reproduces_the_fit() -> None:
    """THE point of recording it: replaying the seed gives identical standard errors."""

    frame = _frame()
    model = ws.Bsqreg.from_formula("y ~ x", frame)

    first = model.fit(reps=15)
    replay = model.fit(reps=15, seed=first.e["seed"])

    assert np.allclose(
        np.asarray(first.bse, dtype=float), np.asarray(replay.bse, dtype=float)
    )


def test_the_replay_reproduces_the_covariance_not_just_the_se() -> None:
    frame = _frame()
    model = ws.Bsqreg.from_formula("y ~ x", frame)

    first = model.fit(reps=15)
    replay = model.fit(reps=15, seed=first.e["seed"])

    assert np.allclose(
        np.asarray(first.cov_params(), dtype=float),
        np.asarray(replay.cov_params(), dtype=float),
    )


def test_two_unseeded_fits_differ_but_are_each_reproducible() -> None:
    """Unseeded draws must still be DIFFERENT from each other -- Stata's behaviour.

    A drawer that silently fixed a constant would satisfy the reproducibility test above
    while making every unseeded run identical, which is the failure the original comment
    warned against. This asserts both halves: different seeds, and each replayable.
    """

    frame = _frame()
    model = ws.Bsqreg.from_formula("y ~ x", frame)

    first = model.fit(reps=15)
    second = model.fit(reps=15)

    assert first.e["seed"] != second.e["seed"]
    # And each replays.
    assert np.allclose(
        np.asarray(model.fit(reps=15, seed=first.e["seed"]).bse, dtype=float),
        np.asarray(first.bse, dtype=float),
    )


def test_an_explicit_seed_is_kept_verbatim() -> None:
    """``e(seed)`` reports what the caller asked for, not a re-derived value."""

    fitted = ws.Bsqreg.from_formula("y ~ x", _frame()).fit(reps=10, seed=12345)

    assert fitted.e["seed"] == 12345


# ------------------------------------------------------------ not collateral
def test_reps_is_still_recorded_alongside_it() -> None:
    fitted = ws.Bsqreg.from_formula("y ~ x", _frame()).fit(reps=12)

    assert fitted.e["reps"] == 12


def test_the_estimates_are_unchanged_in_shape() -> None:
    """Recording a seed must not disturb the fit itself."""

    fitted = ws.Bsqreg.from_formula("y ~ x", _frame()).fit(reps=10)

    assert len(fitted.params) == 2
    assert np.all(np.isfinite(np.asarray(fitted.params, dtype=float)))
    assert np.all(np.isfinite(np.asarray(fitted.bse, dtype=float)))