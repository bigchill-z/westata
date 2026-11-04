"""``xtreg, fe`` must drop absorbed regressors instead of refusing, and hausman then works.

Stata's behaviour on a regressor that is constant within every entity::

    . xtreg rug_after maize prerebel, fe
        maize |  37.43478 ...
    prerebel |  (omitted)

westata raised ``AbsorbingEffectError`` instead, so an ordinary specification became a hard
failure -- and the downstream ``hausman`` could not be computed at all.

The reference values come from a live run on cao_chen.dta (20 counties x ~260 periods,
``prerebel`` constant within county)::

    hausman fe     chi2(1) = 0.77331812   p = 0.37919252

Two things had to be right for that to reproduce:

* the absorbed column is dropped, but the ``Intercept`` is NOT -- linearmodels needs it to
  recognise a fixed-effect regression, and dropping it silently changed every R-squared in
  the suite (the first attempt at this fix broke 17 tests);
* the group labels come from the panel index's levels, not from named columns, because the
  index is a ``(entity, time)`` MultiIndex.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\cao_chen.dta"


@pytest.fixture(scope="module")
def panel_slice():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("cao_chen.dta is not available")
    frame = frame.dropna(subset=["rug_after", "maize", "prerebel", "year", "county"])
    counts = frame["county"].value_counts()
    keep = counts[counts >= 10].index[:20]
    return frame[frame["county"].isin(keep)].reset_index(drop=True)


def test_prerebel_really_is_constant_within_county(panel_slice) -> None:
    """The precondition for the whole test: it is absorbed, not merely collinear."""

    per_county = panel_slice.groupby("county")["prerebel"].nunique()
    assert int(per_county.max()) == 1


def test_the_fit_succeeds_and_drops_only_the_absorbed_column(panel_slice) -> None:
    result = ws.PanelOLS.from_formula(
        "rug_after ~ maize + prerebel", panel_slice, entity="county", time="year"
    ).fit()
    # The intercept stays (the backend needs it); only the absorbed slope is gone.
    assert [name for name in result.params.index if name != "Intercept"] == ["maize"]


def test_the_surviving_coefficient_matches_stata(panel_slice) -> None:
    result = ws.PanelOLS.from_formula(
        "rug_after ~ maize + prerebel", panel_slice, entity="county", time="year"
    ).fit()
    assert float(result.params["maize"]) == pytest.approx(37.43478, abs=5e-4)


def test_hausman_matches_stata(panel_slice) -> None:
    fe = ws.PanelOLS.from_formula(
        "rug_after ~ maize + prerebel", panel_slice, entity="county", time="year"
    ).fit()
    re = ws.RandomEffects.from_formula(
        "rug_after ~ maize + prerebel", panel_slice, entity="county", time="year"
    ).fit()
    result = ws.hausman(fe, re)
    assert float(result.statistic) == pytest.approx(0.77331812, abs=5e-6)
    assert int(result.df) == 1
    assert float(result.pvalue) == pytest.approx(0.37919252, abs=5e-6)


def test_ordinary_columns_are_never_dropped() -> None:
    """Guard the regression that broke 17 tests: only genuinely absorbed columns go."""

    rng = np.random.default_rng(20261022)
    entities, periods = 5, 4
    n = entities * periods
    entity = np.repeat(np.arange(entities), periods)
    time = np.tile(np.arange(periods), entities)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    effect = rng.normal(scale=1.0, size=entities)[entity]
    y = 2.0 + 1.5 * x1 - 0.7 * x2 + effect + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "firm": entity, "year": time})
    result = ws.PanelOLS.from_formula(
        "y ~ x1 + x2", frame, entity="firm", time="year"
    ).fit()
    assert set(result.params.index) == {"Intercept", "x1", "x2"}
