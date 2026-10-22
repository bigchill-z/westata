"""Estimator classes.

Each class targets one Stata command:

===============  ==========================================
westata          Stata
===============  ==========================================
``OLS``          ``regress``
``IV2SLS``       ``ivregress 2sls``
``IVLIML``       ``ivregress liml``
``IVGMM``        ``ivregress gmm``
``FEOLS``        ``reghdfe`` / ``areg``
``PanelOLS``     ``xtreg, fe``
``RandomEffects````xtreg, re``
``BetweenOLS``   ``xtreg, be``
``FirstDifferenceOLS``  ``xtreg, fd``
===============  ==========================================
"""

from westata.estimators.hdfe import FEOLS
from westata.estimators.iv import IV2SLS, IVGMM, IVLIML
from westata.estimators.linear import OLS

__all__ = [
    "OLS",
    "IV2SLS",
    "IVLIML",
    "IVGMM",
    "FEOLS",
]
