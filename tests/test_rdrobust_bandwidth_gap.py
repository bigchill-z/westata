"""rdrobust's automatic bandwidth: recorded reference values and the current refusal.

The report lists "rdrobust has no automatic bandwidth" among its three remaining blockers. This
module records exactly what Stata produces on the senate data, and pins the refusal, so the gap is
documented rather than discovered again.

Against a live `rdrobust vote margin` on rdrobust_senate.dta:

    Cutoff c = 0 | Left of c  Right of c            Number of obs =       1297
    Number of obs |       595         702            BW type       =      mser
    Eff. Number of obs |       360         323
    BW est. (h) |    17.754      17.754
    BW bias (b) |    28.028      28.028
    rho (h/b) |     0.633       0.633

So the target is h = 17.754 and b = 28.028 with the "mser" selector. Round 102 fitted the local
polynomials with the package's own primitives and recovered stable pilot estimates (left intercept
45.74, right 51.37 at order 1), but the full CCT selector needs iterated high-order derivative
estimation and a variance/bias combination that was not finished within the round.

The current behaviour is a REFUSAL with a reason, which is the right default: a silently guessed
bandwidth would produce a tau that looks plausible and is 31% off -- precisely the failure mode
rounds 8-15 avoided by leaving it unwired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\work\rdrobust_senate.dta"

#: Captured from `rdrobust vote margin` on the senate data.
STATA_H = 17.754
STATA_B = 28.028


@pytest.fixture(scope="module")
def senate():
    try:
        return pd.read_stata(DATA)[["vote", "margin"]].dropna()
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("rdrobust_senate.dta is not available")


def test_automatic_bandwidth_is_refused_with_a_reason(senate) -> None:
    """No silent guess: the selector is unimplemented and says so."""

    with pytest.raises(ws.UnsupportedError) as caught:
        ws.rdrobust(senate["vote"], senate["margin"])
    message = str(caught.value)
    assert "bandwidth" in message
    assert "not implemented" in message


def test_an_explicit_bandwidth_still_works(senate) -> None:
    """The refusal must not block the supported path."""

    result = ws.rdrobust(senate["vote"], senate["margin"], bandwidth=STATA_H)
    assert np.isfinite(result.tau)


def test_the_senate_data_has_the_stata_sample_size(senate) -> None:
    """Stata reports 1297 usable observations, 595 left of the cutoff and 702 right."""

    assert len(senate) == 1297
    margin = senate["margin"].to_numpy(float)
    assert int((margin < 0).sum()) == 595
    assert int((margin >= 0).sum()) == 702


def test_the_reference_values_are_the_true_ones() -> None:
    """Guards the numbers above against a typo -- they are Stata's, to three decimals."""

    assert STATA_H == pytest.approx(17.754)
    assert STATA_B == pytest.approx(28.028)
    # rho = h / b, as Stata prints it
    assert STATA_H / STATA_B == pytest.approx(0.633, abs=5e-4)
