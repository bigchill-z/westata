"""teffects nnmatch: TIES were dropped, and keeping them makes both estimands exact.

Round 204 recorded a 7% gap and left a note: export Stata's own match set rather than comparing
final estimates. Round 207 did that, and the exported set explained everything at once:

    teffects nnmatch (re78 <covariates>) (treat), generate(q_)

writes EIGHT variables -- q_2 populated for 90 rows, q_3 for 42, q_4 for 19, q_8 for 1 -- so
distances genuinely tie and Stata keeps every tied neighbour. The package took only the k-th
nearest, which silently dropped real matches:

    Stata        1906.126  (ATET 2453.076)
    westata      2039.4582 (ATET 2277.0212)   <- before
    westata      1906.1262 (ATET 2453.0763)   <- after

Averaging each unit's matched set equally, with ties included, is what reproduces Stata -- verified
by recomputing from Stata's OWN q_* columns before touching any code.

THE GUARD THAT ROUND 204 WROTE IS GONE ON PURPOSE. It asserted the estimate did NOT match Stata
(`gap > 10.0`) with the note "if this fails, the matching was fixed -- pin the equality instead".
It failed exactly as designed, which is the useful kind of stale test: it told us when to replace it
rather than quietly passing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

STATA_NNMATCH_ATE = 1906.126
STATA_NNMATCH_ATET = 2453.076


@pytest.fixture(scope="module")
def data():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")
    return (
        frame["re78"].to_numpy(float),
        frame["treat"].to_numpy(float),
        frame[COVARIATES].to_numpy(float),
    )


def test_the_ate_matches_stata(data) -> None:
    y, treated, x = data
    fit = ws.NearestNeighbourMatching(y, treated, x).fit()
    assert float(fit.estimate) == pytest.approx(STATA_NNMATCH_ATE, abs=5e-3)


def test_the_atet_matches_stata(data) -> None:
    y, treated, x = data
    fit = ws.NearestNeighbourMatching(y, treated, x, estimand="atet").fit()
    assert float(fit.estimate) == pytest.approx(STATA_NNMATCH_ATET, abs=5e-3)


def test_tied_neighbours_are_kept(data) -> None:
    """Some units match more than one control at the same distance; all of them must be used.

    Stata's own generate() wrote eight match variables on this data, so ties are not hypothetical.
    """

    y, treated, x = data
    model = ws.NearestNeighbourMatching(y, treated, x)
    distances = model._distances()
    treated_index = np.flatnonzero(treated == 1.0)
    control_index = np.flatnonzero(treated == 0.0)
    matched = model._neighbours(distances, control_index, treated_index)
    sizes = [row.size for row in matched]
    assert max(sizes) > 1, "no ties found -- the tie tolerance or the metric changed"
    # And one unit matched a single neighbour, so the ragged case is exercised too.
    assert min(sizes) >= 1
