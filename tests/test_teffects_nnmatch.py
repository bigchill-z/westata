"""Pin ``teffects nnmatch``: Mahalanobis ATET and the tie rule that decides it.

Reference from a live ``teffects nnmatch (re78 <8 covariates>) (treat), atet`` on nsw_dw.dta:

    nneighbor(1) : ATET = 2453.0763
    nneighbor(4) : ATET = 2060.4859

Report case C15-03 described this as "NNMATCH has no caliper parameter", implying the
estimator existed. It did not -- there was no nnmatch implementation at all.

THE TIE RULE is what makes the number reproducible. Matching uses the Mahalanobis distance with
the inverse sample covariate covariance (``e(metric)`` reads "mahalanobis"), and when several
controls sit at the same minimum distance Stata uses ALL of them:

    first tied control only : 2277.0212
    all tied controls       : 2453.0763   <- Stata

47 of the 185 treated units have ties, so this is not a rounding difference.

A NOTE ON THE GAP REPORT. It lists "ATET 2277.021 vs Stata 2639.865" for ``teffects psmatch``.
2277.0212 is exactly Mahalanobis matching with the first-tie-only rule -- an nnmatch computation,
not a broken psmatch. The two cases had been cross-wired, which is why fixing psmatch (round 52)
never explained the 2277 figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVS = [
    "age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75",
]


@pytest.fixture(scope="module")
def sample():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")
    return frame["re78"].to_numpy(float), frame["treat"].to_numpy(float), frame[COVS].to_numpy(float)


def test_nneighbor_1_matches_stata(sample) -> None:
    y, d, x = sample
    assert ws.teffects_nnmatch(y, d, x, nneighbor=1).ate == pytest.approx(2453.0763, abs=5e-4)


def test_nneighbor_4_matches_stata(sample) -> None:
    y, d, x = sample
    assert ws.teffects_nnmatch(y, d, x, nneighbor=4).ate == pytest.approx(2060.4859, abs=5e-4)


def test_the_tie_rule_is_all_tied_controls(sample) -> None:
    """The first-tie-only value must NOT be what we return."""

    y, d, x = sample
    result = ws.teffects_nnmatch(y, d, x, nneighbor=1)
    assert abs(result.ate - 2277.0212) > 100
    assert result.matches_max > 1


def test_the_match_counts_match_psmatchs(sample) -> None:
    """Both estimators use all tied neighbours, so the counts agree."""

    y, d, x = sample
    one = ws.teffects_nnmatch(y, d, x, nneighbor=1)
    four = ws.teffects_nnmatch(y, d, x, nneighbor=4)
    assert (one.matches_min, one.matches_max) == (1, 8)
    assert (four.matches_min, four.matches_max) == (4, 11)


def test_the_metric_is_mahalanobis(sample) -> None:
    y, d, x = sample
    assert ws.teffects_nnmatch(y, d, x).metric == "mahalanobis"


def test_a_caliper_that_excludes_everything_raises(sample) -> None:
    """A treated unit with no control inside the caliper must not borrow a distant one."""

    y, d, x = sample
    with pytest.raises(Exception, match="caliper|no matches"):
        ws.teffects_nnmatch(y, d, x, nneighbor=1, caliper=1e-9)


def test_a_generous_caliper_leaves_the_estimate_alone(sample) -> None:
    y, d, x = sample
    wide = ws.teffects_nnmatch(y, d, x, nneighbor=1, caliper=1e6)
    assert wide.ate == pytest.approx(2453.0763, abs=5e-4)
    assert wide.caliper == pytest.approx(1e6)
