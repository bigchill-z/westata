"""PSMATCH: the ATET is exact, the variance is not implemented, and the flag says so.

C15-01 scores `teffects psmatch, atet` at 0.8333. Measured on nsw_dw.dta:

    teffects psmatch (re78) (treat age education black hispanic married nodegree re74 re75), atet
        ATET   Stata 2639.86455455    package 2639.864555
        se     Stata 714.77582515     package nan

THE POINT ESTIMATE MATCHES TO EIGHT DECIMALS; the standard error is genuinely absent rather than
wrong. `variance_matches_stata` is False and `se` is NaN, which is the honest shape for an
unimplemented Abadie-Imbens variance -- a plausible number there would be worse than none, because a
caller cannot tell an approximation from a derivation.

That gap is recorded in options.md. This file pins what IS true so it cannot regress: the estimate,
the flag, and the fact that the package refuses to invent a standard error.
"""

import math

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = "age education black hispanic married nodegree re74 re75".split()


@pytest.fixture(scope="module")
def matched():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nsw_dw.dta is not available")
    return ws.teffects_psmatch(
        frame["re78"], frame["treat"], frame[COVARIATES]
    )


def test_the_atet_matches_stata(matched) -> None:
    assert float(matched.ate) == pytest.approx(2639.86455455, abs=1e-6)


def test_the_variance_is_marked_as_not_matching_stata(matched) -> None:
    """The flag exists so a caller cannot mistake an absent variance for a derived one."""

    assert matched.variance_matches_stata is False


def test_no_standard_error_is_invented(matched) -> None:
    """NaN is the correct answer here; a plausible number would be indistinguishable from a fit."""

    assert math.isnan(float(matched.se))
    assert math.isnan(float(matched.pvalue))


def test_the_match_counts_are_reported(matched) -> None:
    """Stata allows several controls to share a nearest neighbour, so the range is not always 1-1."""

    assert matched.matches_min >= 1
    assert matched.matches_max >= matched.matches_min
