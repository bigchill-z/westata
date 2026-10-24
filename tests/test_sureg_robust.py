"""Tests for sureg's robust VCE, pinned against a live Stata session.

sureg ..., vce(robust) was refused with UnsupportedError. reg3.ado line 533 delegates it to _reg3robust,
whose GetSandwich builds a standard sandwich with a detail that took three attempts to get right:

    omega     = the OLS-based Sigma       (equals Stata e(Sigma) exactly)
    score     = the SUR residuals         (from _predict after the fit, GetUlist2)
    psi_t[i]  = x_i(t) * sum_j precision[i,j] e_j(t)
    V         = (X Omega^-1 X)^-1 [ sum_t psi psi ] (X Omega^-1 X)^-1

Using OLS residuals in the score gave a 2-4 percent error on every standard error; using SUR residuals in omega
made Sigma itself wrong. Both are called "the residuals" in the ado file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import westata as ws
from westata.errors import UnsupportedError

AUTO = Path(__file__).resolve().parents[1] / "tests" / "data" / "auto.dta"

STATA_ROBUST_SE = np.array(
    [
        7.59937450e-01,
        9.17032670e01,
        4.05305586e03,
        1.09265040e00,
        5.31370000e-04,
        1.74656612e00,
    ]
)


def _system(seed: int = 20261024, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    shared = rng.normal(size=n)
    e1 = shared
    e2 = 0.6 * shared + 0.8 * rng.normal(size=n)
    return pd.DataFrame(
        {
            "y1": 1.0 + 0.8 * x1 - 0.4 * x2 + e1,
            "y2": 2.0 - 0.5 * x1 + 0.3 * x2 + e2,
            "x1": x1,
            "x2": x2,
        }
    )


def test_sureg_robust_matches_stata_on_the_auto_data():
    auto = pd.read_stata(AUTO)
    result = ws.Sureg.from_formula(
        ["price ~ weight + mpg", "mpg ~ weight + foreign"], auto
    ).fit(cov_type="robust")
    np.testing.assert_allclose(
        np.asarray(result.bse, dtype=float), STATA_ROBUST_SE, rtol=1e-6
    )


def test_sureg_robust_differs_from_unadjusted():
    frame = _system()
    adjusted = ws.Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x1 + x2"], frame).fit()
    robust = ws.Sureg.from_formula(["y1 ~ x1 + x2", "y2 ~ x1 + x2"], frame).fit(
        cov_type="robust"
    )
    assert not np.allclose(
        np.asarray(adjusted.bse, dtype=float),
        np.asarray(robust.bse, dtype=float),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(adjusted.params, dtype=float),
        np.asarray(robust.params, dtype=float),
        rtol=1e-12,
    )


def test_other_system_commands_still_reject_robust():
    frame = _system()
    with pytest.raises(UnsupportedError, match="cov_type"):
        ws.Mvreg.from_formula("y1 y2 = x1 + x2", frame).fit(cov_type="robust")

