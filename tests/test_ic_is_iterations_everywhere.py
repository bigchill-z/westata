"""``e(ic)`` is the iteration count in EVERY estimator, not just the ones fixed first.

``logit.sthlp:257`` (and the ML family generally) says::

    {synopt:{cmd:e(ic)}}number of iterations

``xtlogit.sthlp:587`` says the same for the panel command. The ``poisson_sim`` fixture
settles it numerically: ``ic = 1.0`` with TWO parameters, so a ``k + 1`` formula (which
would give 3) cannot produce it, while "iterations" describes both it and ``logit_sim``'s
4.0.

**The wrong reading was copied, not invented once.** Round 238 corrected
``estimators/discrete.py``. Round 253 found the same wrong comment AND formula in
``estimators/panel_nonlinear.py``. Round 254 found it a third time, twice, in
``estimators/limited.py`` -- once for ``heckman, method(ml)`` and once for
``etregress, method(ml)``. Each copy passed its own tests because those tests had copied
the same wrong expectation.

That is why this file sweeps every estimator that stores ``ic`` instead of checking one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws

BASE = Path(r"D:\Program Files\StataNow19\ado\base")


def _frame(n: int = 400, seed: int = 8300):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    instrument = rng.normal(size=n)
    selected = (0.5 + 0.8 * instrument + rng.normal(size=n) > 0).astype(int)
    treated = (0.3 + 0.9 * instrument + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(
        {
            "y": np.where(
                selected == 1, 1.0 + 0.7 * covariate + 0.5 * rng.normal(size=n), np.nan
            ),
            "y2": 1.0 + 0.8 * covariate + 1.5 * treated + rng.normal(size=n),
            "yb": (rng.uniform(size=n) < 0.5).astype(int),
            "x": covariate,
            "z": instrument,
            "s": selected,
            "t": treated,
            "id": np.repeat(np.arange(40), n // 40),
        }
    )


@pytest.fixture(scope="module")
def frame():
    return _frame()


BUILDERS = {
    "heckman_ml": lambda frame: ws.Heckman.from_formula(
        "y ~ x + select(s ~ z)", frame
    ).fit(method="ml"),
    "etregress_ml": lambda frame: ws.Etregress.from_formula(
        "y2 ~ x + treat(t ~ z)", frame
    ).fit(method="ml"),
    "xtlogit_fe": lambda frame: ws.XtLogit.from_formula(
        "yb ~ x", frame, entity="id", method="fe"
    ).fit(),
    "logit": lambda frame: ws.Logit.from_formula("yb ~ x", frame).fit(),
}


# --------------------------------------------------------------- the sweep
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_ic_is_a_positive_number(frame, name: str) -> None:
    e = BUILDERS[name](frame).e

    assert float(e["ic"]) >= 1.0, name


def _parameter_count(e: dict) -> float:
    """``e(k)`` where the command stores it, else ``e(rank)``.

    ``xtlogit`` records ``rank`` but not ``k``, so the comparison has to use whichever the
    command provides rather than assuming one.
    """

    if "k" in e:
        return float(e["k"])
    return float(e["rank"])


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_ic_is_not_the_parameter_penalty(frame, name: str) -> None:
    """THE check: ``ic != parameters + 1`` for every estimator that stores it.

    A copy of the old formula anywhere in the codebase fails here, which is the point of
    sweeping rather than spot-checking.
    """

    e = BUILDERS[name](frame).e
    parameters = _parameter_count(e)

    assert e["ic"] != parameters + 1, (
        f"{name}: e(ic) = {e['ic']} equals parameters+1 ({parameters} + 1) -- the "
        "penalty-term reading the fixture refutes"
    )


def test_at_least_one_estimator_disagrees_with_the_penalty(frame) -> None:
    """Guard: if every ``ic`` happened to equal ``k + 1``, the test above would be
    vacuous on this data."""

    disagreements = []
    for name in BUILDERS:
        e = BUILDERS[name](frame).e
        if e["ic"] != _parameter_count(e) + 1:
            disagreements.append(name)

    assert disagreements, "every estimator's ic equals k+1; the sweep cannot discriminate"


# ------------------------------------------------------- the help files agree
@pytest.mark.parametrize(
    "relative", ["l/logit.sthlp", "x/xtlogit.sthlp", "p/poisson.sthlp"]
)
def test_the_help_files_say_iterations(relative: str) -> None:
    text = (BASE / relative).read_text(encoding="utf-8", errors="ignore")

    assert "number of iterations" in text
    assert "{cmd:e(ic)}" in text


# ---------------------------------------------------- the fixture that decides
def test_the_poisson_fixture_refutes_k_plus_1() -> None:
    """``ic = 1.0`` with two parameters -- a numeric proof, not an interpretation."""

    import json

    fixture = (
        Path(__file__).resolve().parent
        / "conformance"
        / "stata_fixtures"
        / "poisson_sim.json"
    )
    scalars = json.loads(fixture.read_text(encoding="utf-8"))["scalars"]

    assert scalars["ic"] == 1.0
    assert scalars["k"] == 2.0
    assert scalars["ic"] != scalars["k"] + 1


# ------------------------------------------------------------ not collateral
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_the_information_criteria_are_still_there(frame, name: str) -> None:
    e = BUILDERS[name](frame).e

    assert np.isfinite(float(e["aic"])), name
    assert np.isfinite(float(e["bic"])), name