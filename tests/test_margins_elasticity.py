"""Pin margins elasticities: the average of per-observation elasticities.

Reference from a live ``logit treat age education re74 re75`` then ``margins, eyex(*)`` on
nsw_dw.dta::

    Average marginal effects                                   Number of obs = 445
    ey/ex wrt:  age education re74 re75

             |            Delta-method
             |      ey/ex   std. err.      z    P>|z|     [95% conf. interval]
    -------------+----------------------------------------------------------------
         age |   .1992726   .1973257     1.01   0.313    -.1874787    .5860239
   education |   .4779253   .3206997     1.49   0.136    -.1506345    1.106485
        re74 |  -.0241888   .0325458    -0.74   0.457    -.0879774    .0395998
        re75 |    .033811   .0281863     1.20   0.230    -.0214332    .0890552

THE BUG. The elasticity was computed as ``mean(dy_i) * mean(factor_i)`` instead of
``mean(dy_i * factor_i)``. They differ whenever the derivative and the factor covary, and here
both carry the fitted probability:

    Stata                      .1992726
    mean(dy_i)*mean(x_i/p_i)   .1996857   <- 0.2% high

The point estimates now match all four rows exactly. The STANDARD ERRORS still do not
(.1986018 against .1973257 for age): the delta-method gradient still uses the averaged factor,
and pinning that is left open rather than approximated -- the tests below assert the point
estimates tightly and do NOT assert the standard errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import westata as ws

DATA = r"D:\myCodeProject\test_westata\data\nsw_dw.dta"
STATA_POINTS = {
    "age": 0.1992726,
    "education": 0.4779253,
    "re74": -0.0241888,
    "re75": 0.033811,
}


@pytest.fixture(scope="module")
def elasticities():
    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover - data lives outside the repo
        pytest.skip("nsw_dw.dta is not available")
    fit = ws.Logit.from_formula("treat ~ age + education + re74 + re75", frame).fit()
    return ws.margins(fit, eyex=["age", "education", "re74", "re75"])


def test_every_elasticity_matches_stata(elasticities) -> None:
    got = {row.label: row.estimate for row in elasticities}
    for name, expected in STATA_POINTS.items():
        assert got[name] == pytest.approx(expected, abs=5e-7), name


def test_the_column_is_labelled_ey_ex(elasticities) -> None:
    rendered = str(elasticities)
    assert "ey/ex wrt:" in rendered
    assert "ey/ex   std. err." in rendered


def test_it_is_not_the_product_of_averages(elasticities) -> None:
    """The old formula is 0.2% high; that difference must remain visible."""

    old = 0.1996857
    assert abs(STATA_POINTS["age"] - old) > 3e-4
    got = {row.label: row.estimate for row in elasticities}
    assert abs(got["age"] - old) > 3e-4


def test_the_plain_average_marginal_effect_is_unaffected() -> None:
    """dydx must not be touched by the elasticity fix."""

    try:
        frame = pd.read_stata(DATA, convert_categoricals=False)
    except FileNotFoundError:  # pragma: no cover
        pytest.skip("nsw_dw.dta is not available")
    fit = ws.Logit.from_formula("treat ~ age + education + re74 + re75", frame).fit()
    plain = ws.margins(fit, dydx=["age"])
    # Stata: margins, dydx(age) -> .0032721  .0032599  1.00  0.316
    assert float(plain.estimate) == pytest.approx(0.0032721, abs=5e-8)
    assert float(plain.std_err) == pytest.approx(0.0032599, abs=5e-8)
