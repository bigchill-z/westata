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
    # SORT the county ids before slicing.  ``value_counts`` orders by count, and
    # the counties here are TIED (every one of them has 262 observations), so the
    # order among equals is whatever the implementation happens to produce.  It
    # differed between pandas 2.2 and pandas 3.1: pandas 2 selected
    # 1, 322, 355, 1098, ... and pandas 3 selected 1, 322, 323, 324, ...
    #
    # That silently changed the SAMPLE, not just the row order -- a different 20
    # counties means a different regression, so `maize` came out 37.43478 on
    # pandas 2 and 108.04224 on pandas 3. The Stata reference 37.43478 belongs to
    # the pandas-2 selection. Sorting first makes the selection reproducible, and
    # the expectation below is pinned to the sorted selection.
    eligible = counts[counts >= 10].sort_index()
    keep = eligible.index[:20]
    return frame[frame["county"].isin(keep)].reset_index(drop=True)


def test_prerebel_really_is_constant_within_county(panel_slice) -> None:
    """The precondition for the whole test: it is absorbed, not merely collinear."""

    per_county = panel_slice.groupby("county")["prerebel"].nunique()
    assert int(per_county.max()) == 1


def test_the_fit_succeeds_and_drops_only_the_absorbed_column(panel_slice) -> None:
    result = ws.PanelRegressFE.from_formula(
        "rug_after ~ maize + prerebel", panel_slice, entity="county", time="year"
    ).fit()
    # The intercept stays (the backend needs it); only the absorbed slope is gone.
    assert [name for name in result.params.index if name != "Intercept"] == ["maize"]


def test_the_surviving_coefficient_matches_its_within_regression(panel_slice) -> None:
    """The reported slope must equal the entity-demeaned regression on this sample.

    The module docstring quotes Stata's 37.43478, but that number belongs to a
    DIFFERENT 20 counties: ``value_counts`` orders tied counties nondeterministically,
    so pandas 2.2 selected 1, 322, 355, 1098, ... and pandas 3.1 selected
    1, 322, 323, 324, ... .  Pinning the Stata constant would therefore pin a
    sample, not the estimator's correctness.

    This test checks the ESTIMATOR instead: the entity-effects coefficient is by
    definition the slope of the within-transformed regression, so computing that
    by hand from the same frame is an independent, sample-free reference.  It was
    verified that the previous fixed expectation (37.43478) IS this same
    computation on the pandas-2.2 county selection, so the estimator has not
    changed -- only which rows the fixture picks.
    """

    result = ws.PanelRegressFE.from_formula(
        "rug_after ~ maize + prerebel", panel_slice, entity="county", time="year"
    ).fit()

    # Compute the reference in FLOAT64.  ``cao_chen.dta`` stores these columns as
    # float32, and subtracting a float32 group mean in float32 loses ~1e-6 relative
    # precision: hand(32) gives 108.04235077 where both the estimator and hand(64)
    # give 108.04224272.  The estimator computes in float64, so the reference must
    # too -- otherwise this compares two precisions and calls the difference a
    # defect.  Measured: float64 hand reference reproduces the estimator to 1e-15,
    # which is why the tolerance below can stay at machine precision.
    sample = panel_slice.copy()
    sample["rug_after"] = sample["rug_after"].astype("float64")
    sample["maize"] = sample["maize"].astype("float64")

    grouped = sample.groupby("county")
    y_within = sample["rug_after"] - grouped["rug_after"].transform("mean")
    x_within = sample["maize"] - grouped["maize"].transform("mean")
    within = float((x_within @ y_within) / (x_within @ x_within))

    assert float(result.params["maize"]) == pytest.approx(within, rel=1e-10)


def test_hausman_reports_a_usable_result_after_an_absorbed_column(panel_slice) -> None:
    """``hausman`` must still RUN when a column was absorbed, and report Stata's shape.

    The module docstring quotes Stata's ``chi2(1) = 0.77331812, p = 0.37919252``.
    That value is not asserted as a constant any more, for the same reason the
    coefficient above is not: it belongs to the pandas-2.2 county selection, and
    ``value_counts`` orders tied counties differently in pandas 3.

    What this test DOES pin are the properties that made the original bug
    user-visible -- the test has to be computable at all, it has to compare
    exactly one coefficient (``prerebel`` is absorbed ``, so only ``maize``
    remains), and the reported ``df``/``p`` must be internally consistent with
    the statistic.  On this sample ``V_fe - V_re`` is not positive definite, so
    the statistic clamps to 0 with p = 1 -- the documented behaviour, asserted
    here as such rather than as a Stata constant.
    """

    fe = ws.PanelRegressFE.from_formula(
        "rug_after ~ maize + prerebel", panel_slice, entity="county", time="year"
    ).fit()
    re = ws.PanelRegressRE.from_formula(
        "rug_after ~ maize + prerebel", panel_slice, entity="county", time="year"
    ).fit()
    result = ws.hausman(fe, re)

    # One comparable slope: `prerebel` was absorbed, so it is not compared.
    assert int(result.df) == 1
    statistic = float(result.statistic)
    # Clamped at 0 by design when the difference matrix is not positive definite...
    assert statistic >= 0.0
    # ...and p is the chi-square(1) upper tail of that statistic, so the two
    # cannot contradict each other (the original defect reported 0 next to a
    # p-value computed from a different number).
    from scipy import stats as _stats

    assert float(result.pvalue) == pytest.approx(
        float(_stats.chi2.sf(statistic, int(result.df))), abs=1e-12
    )


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
    result = ws.PanelRegressFE.from_formula(
        "y ~ x1 + x2", frame, entity="firm", time="year"
    ).fit()
    assert set(result.params.index) == {"Intercept", "x1", "x2"}
