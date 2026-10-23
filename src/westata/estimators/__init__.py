"""Estimator classes.

Each class targets one Stata command:

=========================  ========================
westata                    Stata
=========================  ========================
``OLS``                    ``regress``
``IV2SLS``                 ``ivregress 2sls``
``IVLIML``                 ``ivregress liml``
``IVGMM``                  ``ivregress gmm``
``FEOLS``                  ``reghdfe`` / ``areg``
``PanelOLS``               ``xtreg, fe``
``RandomEffects``          ``xtreg, re``
``BetweenOLS``             ``xtreg, be``
``FirstDifferenceOLS``     ``xtreg, fd``
=========================  ========================
"""

from westata.estimators.discrete import (
    Logit,
    NegativeBinomial,
    OrderedLogit,
    OrderedProbit,
    Poisson,
    Probit,
)
from westata.estimators.hdfe import FEOLS
from westata.estimators.iv import IV2SLS, IVGMM, IVLIML
from westata.estimators.counts import (
    TruncatedNegativeBinomial,
    TruncatedPoisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from westata.estimators.limited import (
    Etregress,
    Heckman,
    Intreg,
    Truncreg,
)
from westata.estimators.linear import OLS
from westata.estimators.panel_dynamic import XtAbond
from westata.estimators.panel_nonlinear import (
    XtLogit,
    XtNbreg,
    XtPoisson,
    XtProbit,
    XtTobit,
)
from westata.estimators.panel_iv import XtGls, XtIvreg, XtPcse, XtUnitroot
from westata.estimators.did import Didregress, Xtdidregress
from westata.estimators.treatreg_effects import (
    AIPW,
    IPW,
    NearestNeighbourMatching,
    RegressionAdjustment,
    TreatmentEffectResult,
)
from westata.estimators.nonlinear import Frontier, Nl
from westata.estimators.resample import Bootstrap, Jackknife, Permute
from westata.estimators.xtunitroot_ips import IpsResult, im_pesaran_shin
from westata.estimators.survey import (
    SurveyDesign,
    SvyLogit,
    SvyMean,
    SvyProportion,
    SvyRegress,
    SvyTotal,
)
from westata.estimators.systems import Eivreg, Manova, ManovaResult, Mvreg, Sureg
from westata.estimators.quantile import QuantileRegression, RobustRegression
from westata.estimators.quantile_boot import Bsqreg, Sqreg
from westata.estimators.panel import (
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    RandomEffects,
)
from westata.estimators.tobit import Tobit
from westata.estimators.tseries import (
    ARIMA,
    GARCH,
    VAR,
    VECM,
    UnitRootResult,
    unitroot,
)

__all__ = [
    "IpsResult",
    "im_pesaran_shin",
    "Permute",
    "Bootstrap",
    "Jackknife",
    "SvyLogit",
    "SvyRegress",
    "SvyProportion",
    "SvyTotal",
    "SvyMean",
    "SurveyDesign",
    "TreatmentEffectResult",
    "NearestNeighbourMatching",
    "AIPW",
    "IPW",
    "RegressionAdjustment",
    "Xtdidregress",
    "Didregress",
    "Frontier",
    "Nl",
    "ManovaResult",
    "Manova",
    "Eivreg",
    "Mvreg",
    "Sureg",
    "XtAbond",
    "XtNbreg",
    "XtPoisson",
    "XtTobit",
    "XtProbit",
    "XtLogit",
    "XtUnitroot",
    "XtPcse",
    "XtGls",
    "XtIvreg",
    "Bsqreg",
    "Sqreg",
    "RobustRegression",
    "QuantileRegression",
    "TruncatedNegativeBinomial",
    "TruncatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
    "Etregress",
    "Heckman",
    "Intreg",
    "Truncreg",
    "OLS",
    "IV2SLS",
    "IVLIML",
    "IVGMM",
    "FEOLS",
    "PanelOLS",
    "RandomEffects",
    "BetweenOLS",
    "FirstDifferenceOLS",
    "Logit",
    "Probit",
    "Poisson",
    "NegativeBinomial",
    "OrderedLogit",
    "OrderedProbit",
    "Tobit",
    "ARIMA",
    "VAR",
    "VECM",
    "GARCH",
    "UnitRootResult",
    "unitroot",
]
