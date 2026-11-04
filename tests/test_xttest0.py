"""Pin ``xttest0`` against Stata on a BALANCED panel.

Reference values from a live run on a synthetic 12-entity x 10-period panel
(``quietly xtreg y x1 x2, re`` then ``xttest0``)::

    y | 3.557623
    e | .7086426
    u | 1.471772
    chibar2(01) = 252.58
    Prob > chibar2 = 0.0000

Only the STATISTIC and ``var_y`` are pinned here. ``e``/``u`` come from the random-effects
fit's own variance components, and westata's ``xtreg, re`` still differs from Stata there
(1.424235 / .8542778 against 1.471772 / .7086426) -- that is a separate estimation gap, not
part of this test, and asserting it here would hide one bug behind another.

The earlier attempt at this test failed because it used the WRONG DATA: an unbalanced panel
(eleven counties with 262 periods, one with 118) where the balanced-panel formula does not
apply. Stata reported 2232.70 there and this implementation gives 2289.52. It therefore
refuses unbalanced panels outright rather than reporting a number that is 2.5% wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.postestimation.xttest0 import xttest0


@pytest.fixture(scope="module")
def balanced_panel() -> pd.DataFrame:
    rng = np.random.default_rng(20261105)
    entities, periods = 12, 10
    n = entities * periods
    ids = np.repeat(np.arange(entities), periods)
    times = np.tile(np.arange(periods), entities)
    effect = rng.normal(scale=1.2, size=entities)[ids]
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.7 * x1 - 0.4 * x2 + effect + rng.normal(scale=0.8, size=n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2, "id": ids, "t": times})


def test_chibar2_matches_stata(balanced_panel) -> None:
    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", balanced_panel, entity="id", time="t"
    ).fit()
    test = xttest0(result, entity=balanced_panel["id"])
    assert test.chibar2 == pytest.approx(252.58, abs=5e-3)
    assert test.pvalue == pytest.approx(0.0, abs=5e-5)


def test_var_y_matches_stata(balanced_panel) -> None:
    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", balanced_panel, entity="id", time="t"
    ).fit()
    test = xttest0(result, entity=balanced_panel["id"])
    assert test.var_y == pytest.approx(3.557623, abs=1e-6)


def test_the_statistic_uses_pooled_not_random_effects_residuals(balanced_panel) -> None:
    """Random-effects residuals give 0.27 here; pooled residuals give Stata's 252.58.

    The ``re`` residuals are nearly orthogonal to the group means by construction, so using
    them destroys the between-group signal the test measures. This is the single detail that
    makes the difference between 252.58 and 0.27.
    """

    result = ws.RandomEffects.from_formula(
        "y ~ x1 + x2", balanced_panel, entity="id", time="t"
    ).fit()
    test = xttest0(result, entity=balanced_panel["id"])
    assert test.chibar2 > 100.0, test.chibar2


def test_unbalanced_panels_are_refused() -> None:
    """A 2.5%-wrong number is worse than an error, so the unbalanced case raises."""

    rng = np.random.default_rng(5)
    sizes = [6, 6, 4]
    ids = np.concatenate([np.full(size, i) for i, size in enumerate(sizes)])
    n = ids.size
    x1 = rng.normal(size=n)
    y = 1.0 + 0.5 * x1 + rng.normal(size=n)
    frame = pd.DataFrame({"y": y, "x1": x1, "id": ids, "t": np.concatenate([np.arange(s) for s in sizes])})
    result = ws.RandomEffects.from_formula(
        "y ~ x1", frame, entity="id", time="t"
    ).fit()
    with pytest.raises(Exception):
        xttest0(result, entity=frame["id"])
