"""``estat vif`` takes exactly two options, and they interact.

``vif.ado:4``: ``syntax [, UNCentered NOSORT]``.

Round 144 established the practice for multi-option commands: test the cross-product,
not just each option alone. Two things here only appear in combination:

* ``uncentered`` ADDS the ``_cons`` row, so it changes the row count that
  ``Mean VIF`` divides by (``vif.ado:145`` divides by the number of REPORTED rows);
* the sorting rule (``vif.ado:121-123``: ``gsort -vvv nms``) reorders ALL rows including
  ``_cons``, so ``uncentered`` + sorted puts the constant in the middle rather than at
  the end.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def collinear_result():
    """VIFs far apart, so the sorted order is unambiguous.

    ``x1``/``x2`` are near-duplicates; ``x3`` is independent. With two equal VIFs the
    sorted and unsorted orders could coincide and a broken sort would pass -- the trap
    from round 33.
    """

    rng = np.random.default_rng(709)
    n = 400
    x3 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    x2 = x1 * 0.95 + rng.normal(size=n) * 0.05
    frame = pd.DataFrame({"y": rng.normal(size=n), "x3": x3, "x1": x1, "x2": x2})
    return ws.OLS.from_formula("y ~ x3 + x1 + x2", frame).fit()


COMBINATIONS = list(itertools.product([False, True], [False, True]))


@pytest.mark.parametrize("nosort,uncentered", COMBINATIONS)
def test_the_combination_renders(collinear_result, nosort, uncentered) -> None:
    rendered = collinear_result.estat_vif(nosort=nosort, uncentered=uncentered)

    assert len(rendered.names) == len(rendered.vif)
    assert np.all(np.isfinite(rendered.vif))
    assert "Variable" in str(rendered)


@pytest.mark.parametrize("nosort", [False, True])
def test_uncentered_adds_exactly_one_row(collinear_result, nosort) -> None:
    """``uncentered`` appends ``_cons``; it does not replace anything.

    ``vif.ado:115-118`` writes ``_cons`` into the name vector when the option is
    present, so every original regressor keeps its row.
    """

    centred = collinear_result.estat_vif(nosort=nosort)
    uncentred = collinear_result.estat_vif(nosort=nosort, uncentered=True)

    assert "_cons" not in centred.names
    assert "_cons" in uncentred.names
    assert len(uncentred.names) == len(centred.names) + 1
    assert set(centred.names).issubset(set(uncentred.names))


def test_sorted_order_treats_cons_like_any_other_row(collinear_result) -> None:
    """``vif.ado:121-123`` sorts by ``-vvv`` over ALL rows, ``_cons`` included.

    So the constant can appear anywhere in the sorted table -- its position is decided
    by its VIF, not by a special case.
    """

    rendered = collinear_result.estat_vif(uncentered=True)
    values = list(rendered.vif)

    assert values == sorted(values, reverse=True)


def test_unsorted_order_keeps_the_model_order_and_appends_cons(collinear_result) -> None:
    """Without sorting the rows follow ``e(b)`` with the constant last."""

    model_order = [
        str(name) for name in collinear_result.param_names if str(name) != "Intercept"
    ]

    rendered = collinear_result.estat_vif(nosort=True, uncentered=True)

    assert list(rendered.names) == model_order + ["_cons"]


def test_mean_vif_divides_by_the_reported_row_count(collinear_result) -> None:
    """``vif.ado:145``: ``Mean VIF`` is the sum over the REPORTED rows.

    Adding ``_cons`` therefore changes the mean twice over -- one more term and a larger
    denominator. A mean computed from the regressor rows alone would drift silently.
    """

    for uncentered in (False, True):
        rendered = collinear_result.estat_vif(uncentered=uncentered)
        assert rendered.mean_vif == pytest.approx(
            float(np.mean(rendered.vif)), rel=1e-12
        )


def test_the_two_options_really_interact(collinear_result) -> None:
    """Guard: if ``uncentered`` changed nothing, the cross-product is pointless."""

    plain = collinear_result.estat_vif()
    uncentred = collinear_result.estat_vif(uncentered=True)

    assert list(plain.names) != list(uncentred.names)
    assert plain.mean_vif != pytest.approx(uncentred.mean_vif, rel=1e-6)


def test_tolerance_stays_the_reciprocal_in_every_combination(collinear_result) -> None:
    """The identity must hold regardless of which options are on."""

    for nosort, uncentered in COMBINATIONS:
        rendered = collinear_result.estat_vif(nosort=nosort, uncentered=uncentered)
        for name, value, tolerance in zip(rendered.names, rendered.vif, rendered.tolerance):
            assert tolerance == pytest.approx(1.0 / value, rel=1e-12), (
                nosort,
                uncentered,
                name,
            )