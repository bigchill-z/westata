"""Three teffects estimators, three different levels of agreement with Stata.

Measured on nsw_dw.dta with the eight standard covariates:

    command   Stata ATE   Stata SE      westata ATE   westata SE
    ra        1621.583    679.0541      1621.5831     679.0541    EXACT
    ipw       1641.315    671.0452      1641.3152     672.6727    SE 0.24% high
    aipw      1621.583    679.0541      1619.0528     673.1762    ATE 2.53 LOW

THE `ra` RESULT IS EXACT on both statistics. That supersedes the module's own note that says the
robust standard errors are not reproduced: the note was written for `teffects_ra`'s convenience
function, whose docstring records six influence-function forms tried and none shipped, while the
`RegressionAdjustment` CLASS reproduces Stata to the printed digit. Worth separating, because the
docstring reads as though nothing about ra's variance works.

THE `aipw` DISAGREEMENT IS NOT JUST THE VARIANCE. Stata's aipw and ra give the SAME point estimate
on this data (1621.583), which says the propensity-score correction is essentially zero here; this
package's aipw gives 1619.05, so its correction is not. That is a point-estimate defect rather than
a variance one and needs its own investigation.

THE `ipw` STANDARD ERROR is 0.24% high -- the same class of influence-function approximation the
`nnmatch` code documents, and below the threshold that changes any printed digit of z or p.

This pins what works so a future change cannot silently break the exact `ra` agreement.
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

#: Captured from live `teffects ra (re78 ...) (treat)`: ATE 1621.583, SE 679.0541.
STATA_RA = (1621.583, 679.0541)
#: `teffects ipw (re78) (treat ...)`: 1641.315 / 671.0452.
STATA_IPW = (1641.315, 671.0452)
#: `teffects aipw (re78 ...) (treat)`: 1621.583 / 679.0541.
STATA_AIPW = (1621.583, 679.0541)


@pytest.fixture(scope="module")
def data():
    try:
        frame = pd.read_stata(DATA)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nsw_dw.dta is not available")
    return (
        frame[COVARIATES].to_numpy(dtype=float),
        frame["re78"].to_numpy(dtype=float),
        frame["treat"].to_numpy(dtype=float),
    )


def test_regression_adjustment_point_estimate_is_exact(data) -> None:
    x, y, treated = data
    fitted = ws.RegressionAdjustment(y, treated, x).fit()
    assert float(fitted.estimate) == pytest.approx(STATA_RA[0], abs=1e-3)


def test_regression_adjustment_standard_error_is_exact(data) -> None:
    """The class reproduces Stata's influence-function variance to the printed digit.

    The module docstring says the robust standard errors are NOT reproduced; that note was
    written for the `teffects_ra` convenience function, not for this class.
    """

    x, y, treated = data
    fitted = ws.RegressionAdjustment(y, treated, x).fit()
    assert float(fitted.std_err) == pytest.approx(STATA_RA[1], abs=1e-3)


def test_ipw_point_estimate_is_exact(data) -> None:
    x, y, treated = data
    fitted = ws.IPW(y, treated, x).fit()
    assert float(fitted.estimate) == pytest.approx(STATA_IPW[0], abs=1e-3)


def test_ipw_standard_error_is_within_a_quarter_percent(data) -> None:
    """Not exact -- the approximation is documented -- but tight enough not to move z or p."""

    x, y, treated = data
    fitted = ws.IPW(y, treated, x).fit()
    relative = abs(float(fitted.std_err) - STATA_IPW[1]) / STATA_IPW[1]
    assert relative < 0.005, relative


def test_ra_and_aipw_should_agree_but_do_not(data) -> None:
    """Pins a KNOWN defect so it cannot be mistaken for coverage.

    Stata's aipw and ra give the same point estimate on this data (1621.583), which means the
    propensity correction is essentially zero at this sample. This package's aipw gives 1619.05,
    so its correction is not zero. When this test starts failing, the aipw point estimate was
    fixed and the assertion should become an equality against STATA_AIPW[0].
    """

    x, y, treated = data
    ra = float(ws.RegressionAdjustment(y, treated, x).fit().estimate)
    aipw = float(ws.AIPW(y, treated, x).fit().estimate)
    assert abs(aipw - ra) > 1.0, f"aipw now agrees with ra ({aipw}); update this test"
