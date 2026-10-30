"""``heckman``'s five selection statistics.

``heckman.ado:491``:

    est local marginsok default XB PSel YCond YExpected Mills NShazard

``heckman_postestimation.sthlp:201-221`` defines them:

    psel       probability of selection, Phi(z gamma)
    mills      nonselection hazard phi(z gamma)/Phi(z gamma) -- the inverse Mills ratio
    nshazard   a SYNONYM of mills, same numbers
    ycond      E(y | observed) = xb + lambda * mills
    yexpected  E(y*) with an unselected case counted as 0

Only ``xb``/``stdp``/``stdf``/``fitted`` were registered here, so the macro advertised
five capabilities that did not exist.

**Two shapes matter and they differ.**

``psel``/``mills``/``nshazard`` belong to the SELECTION equation, which is fitted on the
full sample -- 400 rows in the reference fit. ``ycond``/``yexpected`` predict the OUTCOME
equation, which only exists on the selected rows -- 256. Returning either set on the other
length would silently misalign every downstream computation, so the lengths are asserted
here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws


def _frame(n: int = 400, seed: int = 5700):
    rng = np.random.default_rng(seed)
    covariate = rng.normal(size=n)
    instrument = rng.normal(size=n)
    selected = (0.5 + 0.8 * instrument + rng.normal(size=n) > 0).astype(int)
    outcome = np.where(
        selected == 1, 1.0 + 0.7 * covariate + 0.5 * rng.normal(size=n), np.nan
    )
    return pd.DataFrame(
        {"y": outcome, "x": covariate, "z": instrument, "s": selected}
    )


@pytest.fixture(scope="module")
def fitted():
    return ws.Heckman.from_formula("y ~ x + select(s ~ z)", _frame()).fit()


# ------------------------------------------------------------- the registration
@pytest.mark.parametrize(
    "kind", ["psel", "mills", "nshazard", "ycond", "yexpected"]
)
def test_the_five_statistics_are_registered(kind: str) -> None:
    from westata.postestimation.predict import supported_types

    assert kind in supported_types("heckman")


def test_the_selection_varlist_is_recorded(fitted) -> None:
    """The probit design must be rebuildable; nothing else records which columns it used."""

    recorded = str(fitted.e.get("select_varlist", "")).split()

    assert recorded == ["Intercept", "z"]


# --------------------------------------------------------------------- shapes
@pytest.mark.parametrize("kind", ["psel", "mills", "nshazard"])
def test_the_selection_statistics_span_the_full_sample(fitted, kind: str) -> None:
    """The probit was fitted on every row, so these are defined on every row."""

    produced = np.asarray(fitted.predict(kind=kind), dtype=float)

    assert produced.size == len(fitted.model_data)


@pytest.mark.parametrize("kind", ["ycond", "yexpected"])
def test_the_outcome_statistics_span_the_selected_rows(fitted, kind: str) -> None:
    """The outcome equation only exists where the outcome is observed."""

    produced = np.asarray(fitted.predict(kind=kind), dtype=float)

    assert produced.size == int(fitted.nobs)
    assert produced.size < len(fitted.model_data)


# --------------------------------------------------------------------- values
def test_psel_is_a_probability(fitted) -> None:
    produced = np.asarray(fitted.predict(kind="psel"), dtype=float)

    assert np.all(produced >= 0.0)
    assert np.all(produced <= 1.0)


def test_mills_and_nshazard_are_the_same_numbers(fitted) -> None:
    """``heckman_postestimation.sthlp:214``: "``nshazard`` and ``mills`` are synonyms"."""

    assert np.allclose(
        np.asarray(fitted.predict(kind="mills"), dtype=float),
        np.asarray(fitted.predict(kind="nshazard"), dtype=float),
        rtol=1e-12,
    )


def test_the_hazard_is_nonnegative(fitted) -> None:
    """``phi/Phi`` cannot be negative -- a wrong tail (``Phi(-z)``) would break this."""

    produced = np.asarray(fitted.predict(kind="mills"), dtype=float)

    assert np.all(produced >= 0.0)


def test_the_hazard_falls_as_selection_becomes_more_likely(fitted) -> None:
    """A basic property of the inverse Mills ratio: high ``psel`` means low hazard."""

    probability = np.asarray(fitted.predict(kind="psel"), dtype=float)
    hazard = np.asarray(fitted.predict(kind="mills"), dtype=float)

    correlation = float(np.corrcoef(probability, hazard)[0, 1])
    assert correlation < -0.9


def test_ycond_equals_xb_plus_lambda_times_the_hazard(fitted) -> None:
    """THE identity: ``E(y | observed) = xb + lambda * mills``.

    ``lambda`` is ``rho * sigma`` and appears as its own parameter in the outcome
    equation, which is what makes this checkable rather than merely plausible.
    """

    # The hazard is defined on the full sample; take the selected rows, which are the
    # ones the outcome equation covers.
    selected = np.asarray(_frame()["s"], dtype=bool)
    hazard = np.asarray(fitted.predict(kind="mills"), dtype=float)[selected]
    linear = np.asarray(fitted.predict(kind="xb"), dtype=float)
    coefficient = float(fitted.e["lambda"])

    assert np.allclose(
        np.asarray(fitted.predict(kind="ycond"), dtype=float),
        linear + coefficient * hazard,
        rtol=1e-8,
    )


def test_yexpected_never_exceeds_the_unconditional_path(fitted) -> None:
    """``E(y*)`` folds in the probability of selection, so it is bounded by ``ycond``
    where ``psel < 1``... and may be smaller again once the 0 substitution applies."""

    conditional = np.asarray(fitted.predict(kind="ycond"), dtype=float)
    expected = np.asarray(fitted.predict(kind="yexpected"), dtype=float)

    # Both are on the selected rows.  With psel <= 1 and the 0-mass substitution, the
    # unconditional expectation cannot exceed the conditional one when ycond >= 0.
    positive = conditional >= 0.0
    assert np.all(expected[positive] <= conditional[positive] + 1e-9)


def test_ycond_and_yexpected_are_distinct(fitted) -> None:
    """Guard: if one delegated to the other, the identity tests would still pass."""

    conditional = np.asarray(fitted.predict(kind="ycond"), dtype=float)
    expected = np.asarray(fitted.predict(kind="yexpected"), dtype=float)

    assert not np.allclose(conditional, expected, rtol=1e-6)


# ----------------------------------------------------------- nothing regressed
def test_xb_still_works(fitted) -> None:
    assert np.asarray(fitted.predict(kind="xb")).size == int(fitted.nobs)


def test_the_registration_keeps_the_linear_types(fitted) -> None:
    from westata.postestimation.predict import supported_types

    offered = set(supported_types("heckman"))

    assert {"xb", "stdp", "stdf", "fitted"} <= offered