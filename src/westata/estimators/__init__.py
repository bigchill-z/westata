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
from westata.estimators.hdfe import Areg, RegressHighDimFixedEffects
from westata.estimators.iv import IV2SLS, IVGMM, IVLIML
from westata.estimators.counts import (
    TruncatedNegativeBinomial,
    TruncatedPoisson,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
)
from westata.estimators.limited import (
    EndogenousTreatmentRegress,
    Heckman,
    IntervalRegress,
    TruncatedRegress,
)
from westata.estimators.linear import Regress
from westata.estimators.newey import Newey, NeweyWestResults, check_time_spacing, newey_west
from westata.estimators.prais import Prais, PraisResults, prais, prais_winsten
from westata.estimators.rdrobust import RdRobustResults, RDRobust, rd_robust
from westata.estimators.teffects_ra import TreatmentEffectsRaResults, TreatmentEffectsRa, treatment_effects_ra
from westata.estimators.psmatch import PsmatchResults, TreatmentEffectsPsmatch, treatment_effects_psmatch
from westata.estimators.nnmatch import NnmatchResults, TreatmentEffectsNnmatch, treatment_effects_nnmatch
from westata.estimators.panel_dynamic import PanelAbond
from westata.estimators.panel_nonlinear import (
    PanelLogit,
    PanelNbRegress,
    PanelPoisson,
    PanelProbit,
    PanelTobit,
)
from westata.estimators.panel_iv import (
    PanelUnitRootResults,
    PanelGls,
    PanelIVRegress,
    PanelPcse,
    PanelUnitroot,
)
from westata.estimators.did import DIDRegress, PanelDIDRegress, estat_parallel_trends
from westata.estimators.treatreg_effects import (
    AIPW,
    IPW,
    NearestNeighbourMatching,
    RegressionAdjustment,
    TreatmentEffectResults,
)
from westata.estimators.nonlinear import Frontier, Nl
from westata.estimators.resample import Bootstrap, Jackknife, Permute
from westata.estimators.xtunitroot_llc import LlcResults, levin_lin_chu
from westata.estimators.xtunitroot_ips import IpsResults, im_pesaran_shin
from westata.estimators.survey import (
    SurveyDesign,
    SurveyLogit,
    SurveyMean,
    SurveyProportion,
    SurveyRegress,
    SurveyTotal,
)
from westata.estimators.systems import ErrorsInVariablesRegress, Manova, ManovaResults, MultivariateRegress, SeeminglyUnrelatedRegress
from westata.estimators.quantile import QuantileRegress, RobustRegress
from westata.estimators.quantile_boot import BootstrappedQuantileRegress, SimultaneousQuantileRegress
from westata.estimators.panel import (
    PanelRegressMLE,
    PanelRegressBE,
    PanelRegressFD,
    PanelRegressFE,
    PanelRegressRE,
)
from westata.estimators.tobit import Tobit
from westata.estimators.synth import Synth, SynthResults, synth
from westata.estimators.synth2 import Synth2, Synth2Results, synth2
from westata.estimators.tseries import (
    ARIMA,
    GARCH,
    JohansenResults,
    VAR,
    VARResults,
    VECM,
    VECMResults,
    UnitRootResults,
    unitroot,
    dfuller,
    pperron,
)

__all__ = [
    "Synth2",
    "Synth",
    "RDRobust",
    "TreatmentEffectsNnmatch",
    "TreatmentEffectsPsmatch",
    "TreatmentEffectsRa",
    "Prais",
    "Newey",
    "estat_parallel_trends",
    "JohansenResults",
    "VECMResults",
    "VARResults",
    "PanelUnitRootResults",
    "LlcResults",
    "levin_lin_chu",
    "IpsResults",
    "im_pesaran_shin",
    "Permute",
    "Bootstrap",
    "Jackknife",
    "SurveyLogit",
    "SurveyRegress",
    "SurveyProportion",
    "SurveyTotal",
    "SurveyMean",
    "SurveyDesign",
    "TreatmentEffectResults",
    "NearestNeighbourMatching",
    "AIPW",
    "IPW",
    "RegressionAdjustment",
    "PanelDIDRegress",
    "DIDRegress",
    "Frontier",
    "Nl",
    "ManovaResults",
    "Manova",
    "ErrorsInVariablesRegress",
    "MultivariateRegress",
    "SeeminglyUnrelatedRegress",
    "PanelAbond",
    "PanelNbRegress",
    "PanelPoisson",
    "PanelTobit",
    "PanelProbit",
    "PanelLogit",
    "PanelUnitroot",
    "PanelPcse",
    "PanelGls",
    "PanelIVRegress",
    "BootstrappedQuantileRegress",
    "SimultaneousQuantileRegress",
    "RobustRegress",
    "QuantileRegress",
    "TruncatedNegativeBinomial",
    "TruncatedPoisson",
    "ZeroInflatedNegativeBinomial",
    "ZeroInflatedPoisson",
    "EndogenousTreatmentRegress",
    "Heckman",
    "IntervalRegress",
    "TruncatedRegress",
    "Regress",
    "newey_west",
    "NeweyWestResults",
    "PraisResults",
    "prais",
    "prais_winsten",
    "RdRobustResults",
    "rd_robust",
    "TreatmentEffectsRaResults",
    "treatment_effects_ra",
    "PsmatchResults",
    "treatment_effects_psmatch",
    "NnmatchResults",
    "treatment_effects_nnmatch",
    "check_time_spacing",
    "IV2SLS",
    "IVLIML",
    "IVGMM",
    "Areg",
    "RegressHighDimFixedEffects",
    "PanelRegressFE",
    "PanelRegressRE",
    "PanelRegressMLE",
    "PanelRegressBE",
    "PanelRegressFD",
    "Logit",
    "Probit",
    "Poisson",
    "NegativeBinomial",
    "OrderedLogit",
    "MultinomialLogit",
    "OrderedProbit",
    "Tobit",
    "SynthResults",
    "synth",
    "Synth2Results",
    "synth2",
    "ARIMA",
    "VAR",
    "VECM",
    "GARCH",
    "UnitRootResults",
    "unitroot",
]
