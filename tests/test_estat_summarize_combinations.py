"""Every combination of ``estat summarize`` options must work together.

Round 143 shipped a real bug that no per-option test could have caught: a local
variable named ``labels`` inside ``estat_summarize`` held the CLUSTER vector, so passing
``cluster=`` shadowed the ``labels=`` mapping and the constructor tried to ``dict()`` a
numpy array. Both options worked alone; they only broke together.

The lesson is structural: options that are tested one at a time leave the cross-product
untested, and the cross-product is where shared state shows up. This file exercises all
eight combinations of ``noweights`` / ``noheader`` / ``labels`` with ``cluster=``
attached, and asserts the independent effects survive together.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import westata as ws


@pytest.fixture(scope="module")
def weighted_clustered():
    rng = np.random.default_rng(701)
    n = 140
    weights = rng.integers(1, 4, size=n).astype(float)
    frame = pd.DataFrame(
        {"y": rng.normal(size=n), "x1": rng.normal(size=n)}
    )
    result = ws.OLS.from_formula(
        "y ~ x1", frame, weights=weights, weight_type="fweight"
    ).fit()
    groups = (np.arange(n) % 7).astype(int)
    return frame, weights, groups, result


COMBINATIONS = list(
    itertools.product([False, True], [False, True], [None, {"y": "Outcome"}])
)


@pytest.mark.parametrize("noweights,noheader,labels", COMBINATIONS)
def test_the_combination_renders(weighted_clustered, noweights, noheader, labels) -> None:
    """No combination may raise.

    ``cluster=`` is deliberately always present: it is the argument that used to
    clobber ``labels``.
    """

    frame, weights, groups, result = weighted_clustered

    kwargs = {
        "cluster": groups,
        "cluster_name": "grp",
        "noweights": noweights,
        "noheader": noheader,
    }
    if labels is not None:
        kwargs["labels"] = labels

    rendered = result.estat_summarize(**kwargs)

    # The table exists, has a row per regressor, and renders as text.
    assert len(rendered.table) == 2
    assert "Variable" in str(rendered)


@pytest.mark.parametrize("noweights,noheader,labels", COMBINATIONS)
def test_each_switch_keeps_its_own_effect(
    weighted_clustered, noweights, noheader, labels
) -> None:
    """Independence: one option must not cancel another.

    Each effect is asserted in the presence of the others, which is exactly what the
    single-option tests could not do.
    """

    frame, weights, groups, result = weighted_clustered

    kwargs = {
        "cluster": groups,
        "cluster_name": "grp",
        "noweights": noweights,
        "noheader": noheader,
    }
    if labels is not None:
        kwargs["labels"] = labels

    rendered = result.estat_summarize(**kwargs)
    text = str(rendered)

    # ``noheader`` removes the obs line, and the column headings survive it.
    assert ("Number of obs" not in text) if noheader else ("Number of obs" in text)
    assert "Variable" in text

    # ``labels`` adds the column and its text.
    if labels is not None:
        assert "Label" in text
        assert "Outcome" in text
    else:
        assert "Label" not in text

    # ``noweights`` restores the row count in Obs.
    expected_obs = len(frame) if noweights else weights.sum()
    assert rendered.table.loc["y", "Obs"] == pytest.approx(expected_obs)

    # The cluster block is never affected by any of them.
    assert rendered.clusters is not None
    assert rendered.clusters["n"] == 7.0


def test_the_combination_count_is_complete() -> None:
    """Guard: the parametrisation really covers 2 x 2 x 2 distinct cases."""

    assert len(COMBINATIONS) == 8
    # ``labels`` may be ``None`` or a dict, so hash the repr rather than the value.
    distinct = {(bool(a), bool(b), repr(c)) for a, b, c in COMBINATIONS}
    assert len(distinct) == 8