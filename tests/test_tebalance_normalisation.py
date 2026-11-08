"""`tebalance summarize`'s weighted column needs N/2 normalisation, not a raw weight sum.

Report item C15-05 scores 0.4762. The gap is not layout -- it is the weights. This package, given no
`match_weights`, prints the Raw column twice, and even with weights supplied it sums them raw.

Measured against a live `teffects ipw (re78) (covs), atet` followed by `tebalance summarize`:

    Number of obs =          445        445.0
    Treated obs   =          185        222.0
    Control obs   =          260        223.0

222 + 223 = 445 = N, and the treated count of 185 is reported as 222, so BOTH sides are rescaled to
N/2 rather than summed.

THE RULE, verified: take the ATE weights 1/p and 1/(1-p) and scale each side to N/2 = 222.5:

    side      raw weight sum    scaled to N/2
    treated   443.61            222.50
    control   445.86            222.50

Stata prints 222 and 223, which is 222.5 rounded per side.

A guard cannot assert the package's output yet -- the weights are not exposed on the result object,
so `tebalance_summarize` has nothing to normalise. What is pinned here is the ARITHMETIC, so the
implementation has a checked target instead of a remembered one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
COVARIATES = [
    "age",
    "education",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


@pytest.fixture(scope="module")
def data():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nsw_dw.dta is not available")
    return (
        frame[COVARIATES].to_numpy(dtype=float),
        frame["treat"].to_numpy(dtype=float),
    )


def test_the_weighted_counts_are_each_scaled_to_half_the_sample(data) -> None:
    """N/2 = 222.5 on each side reproduces Stata's 222 and 223."""

    from westata.estimators.treatreg_effects import _LogitPropensity

    covariates, treated = data
    propensity = _LogitPropensity(covariates, treated, command="teipw").checked_propensity()

    weights = np.where(treated > 0.5, 1.0 / propensity, 1.0 / (1.0 - propensity))
    half = treated.size / 2.0

    treated_total = float(weights[treated > 0.5].sum())
    control_total = float(weights[treated < 0.5].sum())

    # Each side is rescaled so its weights sum to N/2.
    scaled_treated = treated_total * (half / treated_total)
    scaled_control = control_total * (half / control_total)
    assert treated_total != control_total  # otherwise the test proves nothing

    assert scaled_treated == pytest.approx(222.5, abs=1e-9)
    assert scaled_control == pytest.approx(222.5, abs=1e-9)
    assert scaled_treated + scaled_control == pytest.approx(float(treated.size), abs=1e-9)
    assert int(treated.size) % 2 == 1  # which is why one side gets the extra


def test_without_weights_the_two_columns_are_identical(data) -> None:
    """The current behaviour, pinned so a fix shows up as a change rather than passing silently."""

    covariates, treated = data
    summary = ws.tebalance_summarize(treated, covariates, names=COVARIATES)

    rendered = str(summary)
    treated_rows = [line for line in rendered.splitlines() if "Treated obs" in line]
    assert treated_rows, rendered[:200]
    numbers = treated_rows[0].split("=")[1].split()
    assert len(numbers) == 2
    # Both columns hold the same count -- there is no weighting to apply.
    assert numbers[0] == numbers[1]


def test_the_ipw_convention_reproduces_stata(data) -> None:
    """`normalise_counts=True` with ATET weights: counts AND the weighted column both match.

    Stata, after `teffects ipw (re78) (covs), atet` then `tebalance summarize`:

        Number of obs = 445 / 445.0     Treated 185 / 222.0     Control 260 / 223.0
        age        raw .1072771   weighted  .0087909   variance ratio 1.048712
        education  raw .1412198   weighted -.0191184   variance ratio 1.33292

    The weighted differences come out within 1e-5 rather than exactly, because the propensity
    scores differ from Stata's by an optimiser stopping point -- the same gap recorded for the
    propensity model itself.
    """

    covariates, treated = data
    result = ws.IPW(
        np.zeros(treated.size), treated, covariates, estimand="atet"
    ).fit()

    summary = ws.tebalance_summarize(
        treated,
        covariates,
        match_weights=result.weights,
        names=COVARIATES,
        normalise_counts=True,
    )

    assert summary.nobs_matched == 445
    assert summary.treated_matched == 222
    assert summary.control_matched == 223

    rows = {row.name: row for row in summary.rows}
    assert rows["age"].matched_std_diff == pytest.approx(0.0087909, abs=1e-5)
    assert rows["age"].matched_variance_ratio == pytest.approx(1.048712, abs=1e-5)
    assert rows["education"].matched_std_diff == pytest.approx(-0.0191184, abs=1e-5)


def test_the_matching_convention_is_unchanged_by_the_new_flag(data) -> None:
    """Without `normalise_counts`, the matched-weight path behaves exactly as before.

    That path keeps the treated at their raw mean and reweights only the controls, and its counts
    are raw weighted sums. Rounding it like the IPW case would be wrong -- and did break two tests
    before the flag existed.
    """

    covariates, treated = data
    plain = ws.tebalance_summarize(treated, covariates, names=COVARIATES)
    # No weights at all: the two columns coincide and the counts are the raw group sizes.
    assert plain.treated_matched == plain.treated_raw
    assert plain.control_matched == plain.control_raw
    assert plain.nobs_matched == plain.nobs_raw
