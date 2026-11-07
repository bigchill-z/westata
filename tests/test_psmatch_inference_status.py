"""teffects psmatch: the point estimate is exact and the inference is honestly absent.

The report's section 7.3 says PSM has "point estimates exact, inference empty". Verified against a
live Stata on nsw_dw.dta:

    Treatment-effects estimation                   Number of obs      =        445
    ATET         |
    (1 vs 0)  |   2639.865   714.7758     3.69   0.000      1238.93    4040.799

and the package returns ate = 2639.864554546459 with se/z/p/ci all NaN and
`variance_matches_stata = False`.

That is the correct shape for this situation, though the REASON needed correcting in round 157. An
earlier version of this docstring said the Abadie-Imbens variance "needs a local-linear estimator
rather than a closed form". That understates it: Abadie and Imbens (2006, 2011, 2012) give an
explicit estimator, and `teffects_psmatch.sthlp:210-233` points at the formula (noting that
`vce(robust)` uses TWO matches by default). What is missing is the IMPLEMENTATION -- the matching
sets, the covariate distance matrix, a local-linear sigma^2, and a four-term decomposition -- not
the formula. `teffects psmatch` also has no `vce(iid)` fallback (Stata rejects it with r(198)), so
there is one route, not two. A guessed standard error would still look authoritative while being
wrong, which is why the fields stay NaN. The flag exists so a caller can tell "not computed" from
"computed as zero".

WHAT IS PINNED HERE:
  * the point estimate, to the printed precision;
  * that the inference fields are NaN rather than plausible-looking numbers;
  * that the flag reports False, so silence cannot be mistaken for agreement.

The module docstring records the detail that decides the estimate: distances are measured on the
propensity score, and ALL controls tied at the minimum distance are used (Stata's generate(m)
produced eight match variables for nneighbor(1)); using a single nearest control gives 2569.745
instead of 2639.865.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]

#: From `teffects psmatch (re78) (treat <8 covariates>), atet`.
STATA_ATET = 2639.865
STATA_ATET_SE = 714.7758


@pytest.fixture(scope="module")
def fitted():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")
    return ws.teffects_psmatch(frame["re78"], frame["treat"], frame[COVARIATES])


def test_the_point_estimate_matches_stata(fitted) -> None:
    assert float(fitted.ate) == pytest.approx(STATA_ATET, abs=5e-4)


def test_the_sample_size_is_statas(fitted) -> None:
    assert int(fitted.nobs) == 445


def test_the_inference_is_nan_rather_than_invented(fitted) -> None:
    """NaN is the honest answer while the AI-robust variance is unimplemented."""

    assert np.isnan(float(fitted.se))
    assert np.isnan(float(fitted.z))
    assert np.isnan(float(fitted.pvalue))
    assert np.isnan(float(fitted.ci_lower))
    assert np.isnan(float(fitted.ci_upper))


def test_the_flag_reports_that_the_variance_does_not_match(fitted) -> None:
    """So silence cannot be read as agreement with Stata's 714.7758."""

    assert fitted.variance_matches_stata is False
    assert not np.isclose(float(fitted.se), STATA_ATET_SE, equal_nan=False)


def test_the_module_says_what_is_actually_missing() -> None:
    """Not "no closed form" -- the formula is published; the implementation is not written.

    Round 157 found the earlier wording understated the problem in a way that made the gap look
    unclosable. Abadie-Imbens (2006, 2011, 2012) give an explicit estimator, and Stata's own help
    points at it (teffects_psmatch.sthlp:210-233, which also notes vce(robust) defaults to TWO
    matches). teffects psmatch has no vce(iid) fallback either.
    """

    import inspect

    import westata.estimators.psmatch as module

    doc = inspect.getdoc(module) or ""
    assert "Abadie" in doc
    assert "explicit one" in doc or "explicit estimator" in doc
    assert "not written" in doc or "IMPLEMENTATION" in doc
    assert "r(198)" in doc  # the vce(iid) route is closed, recorded so it is not re-tried
