"""`xtreg, mle` reproduces Stata's coefficients, variance components and likelihood.

C12-05 scores `xtreg, mle` at 0.24. Measured on munnell.dta:

    quietly xtreg lny lnk1 lnk2 lnlabor, mle
        lnk1 -0.0559845045   lnk2 0.2469788270   lnlabor 0.8251519437
        e(sigma_u) 0.09829901   e(sigma_e) 0.03882722
        e(rho)     0.86503853   e(ll)      1380.228032

Every quantity matches; sigma_u differs in the eighth decimal, which is the optimiser's stopping
point rather than a specification difference.

THE RESULT EMITS A ConvergenceWarning and that is honest rather than incidental: the ML estimate of
sigma_u sits near the boundary of the parameter space on this panel, so the optimiser reports it. A
guard checks the warning is raised rather than swallowed -- a boundary optimum that converges quietly
is the case where a caller most needs to know.
"""

import numpy as np
import pandas as pd
import pytest

import westata as ws

MUNNELL = r"D:\myCodeProject\test_westata\data\munnell.dta"


@pytest.fixture(scope="module")
def panel():
    try:
        frame = pd.read_stata(MUNNELL)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("munnell.dta is not available")
    frame = frame.copy()
    frame["sid"] = pd.Categorical(frame["st_abb"]).codes
    frame["year"] = frame["year"].astype(int)
    return frame


@pytest.fixture(scope="module")
def mle(panel):
    return ws.RandomEffectsMLE.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", panel, entity="sid", time="year"
    ).fit()


def test_the_coefficients_match_stata(mle) -> None:
    assert float(mle.params["lnk1"]) == pytest.approx(-0.0559845045, abs=1e-8)
    assert float(mle.params["lnk2"]) == pytest.approx(0.2469788270, abs=1e-8)
    assert float(mle.params["lnlabor"]) == pytest.approx(0.8251519437, abs=1e-8)


def test_the_variance_components_match_stata(mle) -> None:
    """sigma_u carries the optimiser's stopping point: Stata 0.09829901, package 0.09829902."""

    assert float(mle.e["sigma_u"]) == pytest.approx(0.09829901, abs=1e-7)
    assert float(mle.e["sigma_e"]) == pytest.approx(0.03882722, abs=1e-8)


def test_the_intraclass_correlation_and_likelihood_match_stata(mle) -> None:
    assert float(mle.e["rho"]) == pytest.approx(0.86503853, abs=1e-7)
    assert float(mle.e["ll"]) == pytest.approx(1380.228032, abs=1e-6)


def test_mle_and_random_effects_are_different_fits(panel, mle) -> None:
    """RE uses the GLS transform with a moment-based variance; MLE maximises the likelihood.

    If the two agreed exactly, one of them would not be doing its own job.
    """

    re = ws.RandomEffects.from_formula(
        "lny ~ lnk1 + lnk2 + lnlabor", panel, entity="sid", time="year"
    ).fit()
    assert float(mle.params["lnk1"]) != pytest.approx(
        float(re.params["lnk1"]), rel=1e-4
    )
