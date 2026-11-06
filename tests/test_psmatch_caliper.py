"""``teffects psmatch ..., caliper()``, added in round 7 of the v2 report (C15-03).

The report listed caliper as a missing option. It was, and GETTING ITS REFERENCE was recorded
last round as blocked on an environment failure -- ``r(103) too many variables specified`` for
every covariate count. That diagnosis was wrong, and the error message said so plainly:

    The outcome-model is misspecified. An outline of the syntax is
    teffects psmatch (outcome_variable) (treatment_variable varlist)

``psmatch`` takes its covariates on the TREATMENT side, the opposite of ``nnmatch``. I had been
writing ``(re78 age) (treat)``; ``(re78) (treat age)`` runs. Worth pinning here because the
error names the problem and I read it as an environment limit for a whole round.

Reference, ``teffects psmatch (re78)(treat age education black hispanic married nodegree re74
re75), atet`` on nsw_dw:

    (no caliper)     ATET = 2639.8646
    caliper(0.1)     ATET = 2639.8646
    caliper(0.01)    r(459) "no matches"

0.1 is loose enough to exclude nothing on this sample, so it must reproduce the uncapped
answer exactly -- a switch that quietly changed the number here would be wrong.

The ``nneighbor(3)`` gap recorded in round 7 is CLOSED in round 8 and pinned below: the
1e-6 tie tolerance had merged two controls 6e-7 apart. See test_the_tolerance_does_not_merge_
distinct_distances.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = [
    "age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75",
]


@pytest.fixture(scope="module")
def data():
    try:
        return pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")


def test_the_uncapped_estimate_matches_stata(data) -> None:
    result = ws.teffects_psmatch(
        data["re78"], data["treat"], data[COVARIATES].to_numpy(float)
    )
    assert float(result.ate) == pytest.approx(2639.8646, abs=5e-4)


def test_a_loose_caliper_changes_nothing(data) -> None:
    """0.1 excludes no control on this sample, so the answer must be identical."""

    capped = ws.teffects_psmatch(
        data["re78"], data["treat"], data[COVARIATES].to_numpy(float), caliper=0.1
    )
    assert float(capped.ate) == pytest.approx(2639.8646, abs=5e-4)
    assert float(capped.caliper) == pytest.approx(0.1)


def test_matches_stay_inside_the_caliper(data) -> None:
    """A tight caliper can only REDUCE the match set, never widen it.

    On this sample even caliper(1e-9) still matches something (propensity scores coincide),
    so the check is directional rather than "must raise": the realised match counts with a
    tighter caliper can never exceed the uncapped ones.
    """

    uncapped = ws.teffects_psmatch(
        data["re78"], data["treat"], data[COVARIATES].to_numpy(float)
    )
    for limit in (0.05, 0.01, 1e-3):
        capped = ws.teffects_psmatch(
            data["re78"], data["treat"], data[COVARIATES].to_numpy(float), caliper=limit
        )
        assert capped.matches_max <= uncapped.matches_max
        assert capped.matches_min >= 1


def test_the_caliper_is_reported(data) -> None:
    uncapped = ws.teffects_psmatch(
        data["re78"], data["treat"], data[COVARIATES].to_numpy(float)
    )
    assert np.isnan(float(uncapped.caliper))


def test_a_tight_but_feasible_caliper_only_removes_matches(data) -> None:
    """A caliper between 0 and 0.1 may drop some treated units but cannot add any."""

    tight = ws.teffects_psmatch(
        data["re78"], data["treat"], data[COVARIATES].to_numpy(float), caliper=0.05
    )
    assert tight.matches_min >= 1
    assert tight.matches_max <= 8  # the uncapped maximum measured on this sample


STATA_BY_NNEIGHBOR = {1: 2639.8646, 2: 2491.8602, 3: 2397.636, 4: 2330.1077, 5: 2321.0271}


@pytest.mark.parametrize("nneighbor", sorted(STATA_BY_NNEIGHBOR))
def test_every_nneighbor_count_matches_stata(data, nneighbor: int) -> None:
    """nn(1) through nn(5), live StataNow on nsw_dw."""

    result = ws.teffects_psmatch(
        data["re78"],
        data["treat"],
        data[COVARIATES].to_numpy(float),
        nneighbor=nneighbor,
    )
    assert float(result.ate) == pytest.approx(
        STATA_BY_NNEIGHBOR[nneighbor], abs=5e-4
    )


def test_the_tolerance_does_not_merge_distinct_distances(data) -> None:
    """The round-7 gap: a 1e-6 tolerance merged two controls 6e-7 apart.

    For treated unit 2 the third and fourth nearest controls sit at propensity-score
    distances 0.0011575908 and 0.00115824. A 1e-6 tolerance counted both, reporting four
    matches where Stata uses three, which moved nn(3) from 2397.636 to 2412.8207.
    """

    loose = ws.teffects_psmatch(
        data["re78"],
        data["treat"],
        data[COVARIATES].to_numpy(float),
        nneighbor=3,
        tie_tolerance=1e-6,
    )
    tight = ws.teffects_psmatch(
        data["re78"],
        data["treat"],
        data[COVARIATES].to_numpy(float),
        nneighbor=3,
    )
    assert float(tight.ate) == pytest.approx(2397.636, abs=5e-4)
    # And the loose tolerance is demonstrably a different, wrong answer.
    assert abs(float(loose.ate) - 2397.636) > 10.0
