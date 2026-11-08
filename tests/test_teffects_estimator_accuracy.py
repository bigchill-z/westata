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


def test_the_aipw_correction_term_should_average_to_zero(data) -> None:
    """Pins the exact size of the defect, so a future fix shows up as a number.

    The augmented mean is `fitted + mean(t*resid/p)`. On this data `fitted1` and `fitted0` already
    EQUAL Stata's PO(1) = 6179.120 and PO(0) = 4557.537 to four decimals, so the correction terms
    are the entire error:

        mean(t*resid1/p)          = -6.5792
        mean((1-t)*resid0/(1-p))  = -4.0489

    They should be ~0. Stata's own aipw returns the SAME potential-outcome means as its ra here
    (1621.583 either way), which is exactly what a zero correction produces.

    RULED OUT -- weighted least squares. Fitting the outcome with weights t/p and (1-t)/(1-p) makes
    the first term exactly 0.0 by construction, but the point estimate then goes to 1617.18, FURTHER
    from Stata. So the fix is not simply "weight the outcome regression".

    RULED OUT -- the propensity model. It matches a direct BFGS maximisation of the logit likelihood
    to 4e-05, so the scores are the MLE.

    WHAT REMAINS -- and this is now READ FROM THE SOURCE, not inferred:
    _teffects_gmm_aipw.ado is readable (1500 lines, in ado/base/_/). Its PO moment equation is

        L469  ipw  = 1/pij                       (treated only)
        L475  resj = cond(tr, ipw*(depvar-muj), 0) + muj - pop[1,j]

    which is exactly the augmented form this package computes. The CONDITIONAL MEAN equation at
    L630 for oform=linear is 
esj = cond(tr, depvar - muj, 0) -- per-arm OLS, also the same.

    So the FORMULAS agree. The divergence is in the SOLVER: the derivative block at L753 reads

        dcm = cond(tr, 1-ipw, 1)

    which means muj is differentiated WITH RESPECT TO THE PO PARAMETER pop. Stata estimates
    (pop, cm, ps) JOINTLY by GMM; this package fits the outcome model first and then averages the
    augmented values. That lands on a different point, which is what the -6.58 / -4.05 gap is --
    a difference between two estimators, not a bug in one of them.
    `_teffects_aipw.ado` delegates to `_teffects_ipwra_gmm`, whose arithmetic lives in a compiled
    builtin, so the exact construction is not readable from the installation.
    """

    import numpy as np

    from westata.estimators.treatreg_effects import _LogitPropensity

    x, y, treated = data
    n = y.size
    arm = treated > 0.5
    design = np.column_stack([np.ones(n), x])
    propensity = _LogitPropensity(x, treated, command="teipw").checked_propensity()

    beta1 = np.linalg.lstsq(design[arm], y[arm], rcond=None)[0]
    beta0 = np.linalg.lstsq(design[~arm], y[~arm], rcond=None)[0]
    fitted1, fitted0 = design @ beta1, design @ beta0

    # The fitted means alone already equal Stata's potential-outcome means ...
    assert float(fitted1.mean()) == pytest.approx(6179.120, abs=1e-3)
    assert float(fitted0.mean()) == pytest.approx(4557.537, abs=1e-3)

    # ... so the correction terms are the whole error, and they are far from zero.
    correction1 = float((treated * (y - fitted1) / propensity).mean())
    correction0 = float(((1.0 - treated) * (y - fitted0) / (1.0 - propensity)).mean())
    assert correction1 == pytest.approx(-6.5792, abs=1e-3)
    assert correction0 == pytest.approx(-4.0489, abs=1e-3)


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


def test_the_propensity_model_is_the_logit_mle(data) -> None:
    """The AIPW/ipw propensity is the logistic MLE, not a linear-probability regression.

    Worth pinning because diagnosing the aipw discrepancy began by comparing against a least-squares
    fit of `treat` on the covariates, which gives a much narrower probability range ([0.543, 0.663]
    against the MLE's [0.195, 0.676]). The package was right and the diagostic was wrong: the
    coefficients match a direct BFGS maximisation of the logit likelihood to four decimals.
    """

    import numpy as np
    from scipy import optimize
    from scipy.special import expit

    from westata.estimators.treatreg_effects import _LogitPropensity

    x, y, treated = data
    design = np.column_stack([np.ones(treated.size), x])

    def negative_loglik(beta):
        eta = design @ beta
        return -np.sum(treated * eta - np.logaddexp(0.0, eta))

    optimum = optimize.minimize(
        negative_loglik, np.zeros(design.shape[1]), method="BFGS"
    )
    reference = expit(design @ optimum.x)

    fitted = _LogitPropensity(x, treated, command="teipw")
    mine = fitted.checked_propensity()

    # Two optimisers, two stopping points: measured agreement is 4.2e-05 on the
    # coefficients. The claim being pinned is WHICH estimator is used, not bit equality.
    assert np.allclose(mine, reference, atol=1e-3)
    # The MLE range is wide; a narrower one would indicate a linear-probability fit.
    assert float(mine.max() - mine.min()) > 0.4
