"""``rdrobust covs()``: covariate-adjusted RD (v2 report section 7.3 class 2).

The report noted rdrobust lacked ``covs``. It now has it, transcribed from
``rdrobust.ado:761-841`` and verified on a live Stata run.

Reference, ``rdrobust vote margin, h(17.75) b(28.03)`` on rdrobust_senate, with
``cov1 = standardised margin^2``:

    (no covs)     tau = 7.414370281708
    covs(cov1)    tau = 7.490617475565   e(coef_covs) = -94.34797652
                                          e(N_h_l) = 360, e(N_h_r) = 323

All four reproduce exactly.

TWO LAYOUT DETAILS cost a round each and are pinned by the tests below:

* ``beta_p`` has one row per POLYNOMIAL ORDER (2x2 for a linear fit with one covariate), not
  one row per outcome. Round 25 had collapsed it and got tau 6720.51.
* ``s_Y' * beta_p[deriv+1, ]`` with deriv=0 reads ROW 0, the intercept row. Reading row 1 gave
  0.4238 against Stata's 7.4906.

and one implementation bug: the per-side mask must be applied BEFORE the kernel, or both "sides"
contain every observation and the estimate collapses to 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\work\rdrobust_senate.dta"
H = 17.75
B = 28.03


@pytest.fixture(scope="module")
def senate():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False).dropna(
            subset=["vote", "margin"]
        )
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("rdrobust_senate.dta is not available")
    frame = frame.copy()
    squared = frame["margin"] ** 2
    frame["cov1"] = (squared - squared.mean()) / squared.std()
    return frame


def test_without_covariates_is_unchanged(senate) -> None:
    result = ws.rdrobust(senate["vote"], senate["margin"], bandwidth=H, bias_bandwidth=B)
    assert float(result.tau) == pytest.approx(7.414370281708, abs=1e-9)
    assert result.coef_covs is None


def test_with_covariates_matches_stata(senate) -> None:
    result = ws.rdrobust(
        senate["vote"], senate["margin"], bandwidth=H, bias_bandwidth=B, covs=senate["cov1"]
    )
    assert float(result.tau) == pytest.approx(7.490617475565, abs=1e-9)


def test_the_joint_covariate_coefficient_matches_stata(senate) -> None:
    """One coefficient across both sides -- this is what e(coef_covs) reports."""

    result = ws.rdrobust(
        senate["vote"], senate["margin"], bandwidth=H, bias_bandwidth=B, covs=senate["cov1"]
    )
    coefficient = float(np.asarray(result.coef_covs).ravel()[0])
    assert coefficient == pytest.approx(-94.34797652, abs=1e-6)


def test_the_effective_samples_match_stata(senate) -> None:
    """360 / 323; getting this wrong is the symptom of a missing side mask."""

    result = ws.rdrobust(
        senate["vote"], senate["margin"], bandwidth=H, bias_bandwidth=B, covs=senate["cov1"]
    )
    assert (result.nobs_effective_left, result.nobs_effective_right) == (360, 323)


def test_the_covariate_really_changes_the_estimate(senate) -> None:
    """Guard against a silently ignored argument."""

    plain = ws.rdrobust(senate["vote"], senate["margin"], bandwidth=H, bias_bandwidth=B)
    adjusted = ws.rdrobust(
        senate["vote"], senate["margin"], bandwidth=H, bias_bandwidth=B, covs=senate["cov1"]
    )
    assert abs(float(plain.tau) - float(adjusted.tau)) > 0.05


def test_missing_covariate_values_are_refused(senate) -> None:
    covariates = senate["cov1"].to_numpy().copy()
    covariates[0] = np.nan
    with pytest.raises(Exception, match="missing"):
        ws.rdrobust(
            senate["vote"], senate["margin"], bandwidth=H, bias_bandwidth=B, covs=covariates
        )
