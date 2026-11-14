"""Estimator classes.

Each class targets one Stata command:

=========================  ========================
westata                    Stata
=========================  ========================
``OLS``                    ``regress``
``IV2SLS``                 ``ivregress 2sls``
``IVLIML``                 ``ivregress liml``
``IVGMM``                  ``ivregress gmm``
``FEOLS``                  ``reghdfe``
``Areg``                   ``areg``  (absorb= required)
``PanelOLS``               ``xtreg, fe``
``RandomEffects``          ``xtreg, re``
``BetweenOLS``             ``xtreg, be``
``FirstDifferenceOLS``     ``xtreg, fd``
=========================  ========================
"""

from westata.estimators.discrete import (
    Logit,
    NegativeBinomial,
    NegativeBinomialNB1,
    OrderedLogit,
    MultinomialLogit,
    OrderedProbit,
    Poisson,
    Probit,
)
from westata.estimators.hdfe import Areg, FEOLS
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
from westata.estimators.newey import Newey, NeweyWestResult, check_time_spacing, newey
from westata.estimators.prais import Prais, PraisResult, prais, prais_winsten
from westata.estimators.rdrobust import RdRobustResult, rdrobust
from westata.estimators.teffects_ra import TefectsRaResult, teffects_ra
from westata.estimators.psmatch import PsmatchResult, teffects_psmatch
from westata.estimators.nnmatch import NnmatchResult, teffects_nnmatch
from westata.estimators.panel_dynamic import XtAbond
from westata.estimators.panel_nonlinear import (
    XtLogit,
    XtNbreg,
    XtPoisson,
    XtProbit,
    XtTobit,
)
from westata.estimators.panel_iv import (
    PanelUnitRootResult,
    XtGls,
    XtIvreg,
    XtPcse,
    XtUnitroot,
)
from westata.estimators.did import Didregress, Xtdidregress, estat_ptrends
from westata.estimators.treatreg_effects import (
    AIPW,
    IPW,
    NearestNeighbourMatching,
    RegressionAdjustment,
    TreatmentEffectResult,
)
from westata.estimators.nonlinear import Frontier, Nl
from westata.estimators.resample import Bootstrap, Jackknife, Permute
from westata.estimators.xtunitroot_llc import LlcResult, levin_lin_chu
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
    RandomEffectsMLE,
    BetweenOLS,
    FirstDifferenceOLS,
    PanelOLS,
    RandomEffects,
)
from westata.estimators.tobit import Tobit
from westata.estimators.synth import SynthResult, synth
from westata.estimators.synth2 import Synth2Result, synth2
from westata.estimators.tseries import (
    ARIMA,
    GARCH,
    JohansenResult,
    VAR,
    VARResults,
    VECM,
    VECMResults,
    UnitRootResult,
    unitroot,
    dfuller,
    pperron,
)

__all__ = [
    "Prais",
    "Newey",
    "estat_ptrends",
    "JohansenResult",
    "VECMResults",
    "VARResults",
    "PanelUnitRootResult",
    "LlcResult",
    "levin_lin_chu",
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
    "newey",
    "NeweyWestResult",
    "PraisResult",
    "prais",
    "prais_winsten",
    "RdRobustResult",
    "rdrobust",
    "TefectsRaResult",
    "teffects_ra",
    "PsmatchResult",
    "teffects_psmatch",
    "NnmatchResult",
    "teffects_nnmatch",
    "check_time_spacing",
    "IV2SLS",
    "IVLIML",
    "IVGMM",
    "Areg",
    "FEOLS",
    "PanelOLS",
    "RandomEffects",
    "RandomEffectsMLE",
    "BetweenOLS",
    "FirstDifferenceOLS",
    "Logit",
    "Probit",
    "Poisson",
    "NegativeBinomial",
    "OrderedLogit",
    "MultinomialLogit",
    "OrderedProbit",
    "Tobit",
    "SynthResult",
    "synth",
    "Synth2Result",
    "synth2",
    "ARIMA",
    "VAR",
    "VECM",
    "GARCH",
    "UnitRootResult",
    "unitroot",
]
