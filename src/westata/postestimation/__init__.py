"""Postestimation: hypothesis tests, linear combinations, marginal effects.

Targets Stata's ``test``, ``lincom`` and ``margins`` commands.
"""

from __future__ import annotations

from westata.postestimation.api import (
    lincom_as_stata,
    statsmodels_wald_test,
    test_as_stata,
)
from westata.postestimation.contrast import ContrastResult, PwCompareTable, contrast, pairwise_compare
from westata.postestimation.lincom import LincomResults, run_lincom
from westata.postestimation.modeltests import (
    HausmanResults,
    LRTestResults,
    hausman,
    lrtest,
)
from westata.postestimation.nonlinear import nlcom, testnl
from westata.postestimation.margins import MarginsResults, MarginsTable, margins
from westata.postestimation.predict import SUPPORTED_TYPES, predict, supported_types
from westata.postestimation.vargranger import VarGrangerResults, var_granger
from westata.postestimation.ereturn import format_ereturn_list, standardized_coefficients
from westata.postestimation.rcm import RegressionControlMethod, RcmResults, rcm
from westata.postestimation.tebalance import (
    BalanceRow,
    BalanceSummary,
    treatment_balance_summarize,
)
from westata.postestimation.tests import (
    WaldTest,
    build_contrast,
    describe_constraint,
    resolve_names,
    run_wald_test,
)
from westata.postestimation.classification import (
    ClassificationResults,
    estat_classification,
)
from westata.postestimation.endogenous import EndogenousResults, estat_endogenous
from westata.postestimation.firststage import FirstStageResults, estat_first_stage
from westata.postestimation.gof import GofResults, LfitResults, estat_gof
from westata.postestimation.hettest import HetTestResults, estat_heteroskedasticity_test
from westata.postestimation.ic import IcResults, estat_ic
from westata.postestimation.imtest import ImTestResults, estat_information_matrix_test
from westata.postestimation.xttest0 import PanelTest0Results, panel_test0
from westata.postestimation.xtoverid import PanelOveridResults, panel_overid
from westata.postestimation.xtserial import PanelSerialResults, panel_serial
from westata.postestimation.varsoc import VarSocResults, var_select_order
from westata.postestimation.varstable import VarStableResults, var_stable
from westata.postestimation.irf_graph import (
    IRF_GRAPH_STATISTICS,
    IRF_STATISTICS,
    IrfSeries,
    irf_cgraph,
    irf_graph,
    irf_ograph,
    irf_series,
)
from westata.postestimation.varwle import VarWleResults, var_wle
from westata.postestimation.bgodfrey import BgTestResults, estat_bgodfrey
from westata.postestimation.dfgls import DFGLSResults, df_gls
from westata.postestimation.kpss import KpssResults, kpss
from westata.postestimation.vecrank import VecRankResults, vec_rank
from westata.postestimation.vecstable import VecStableResults, vec_stable
from westata.postestimation.varlmar import LmarResults, var_lmar, vec_lmar
from westata.postestimation.serial import (
    Correlogram,
    DurbinWatsonResults,
    PortmanteauResults,
    estat_correlogram,
    estat_dwatson,
    wntestq,
)
from westata.postestimation.ovtest import OvTestResults, estat_omitted_variables_test
from westata.postestimation.mtest import MTEST_METHODS, MTestResults, adjust_pvalues
from westata.postestimation.overid import OverIdResults, estat_overidentification
from westata.postestimation.summarize import SummarizeResults, estat_summarize
from westata.postestimation.summary_stats import SummarizeTable, summarize
from westata.postestimation.vce import VceMatrix, estat_vce
from westata.postestimation.vif import VifResults, estat_vif
from westata.postestimation.auc import AucResults, estat_auc
from westata.postestimation.endog_regf import RegFResults, endog_regf_test
from westata.postestimation.endog_score import EndogScoreResults, endog_score_test
from westata.postestimation.esize import EffectSizeResults, estat_effect_size
from westata.postestimation.linktest import LinkTestResults, linktest
from westata.postestimation.moran import MoranResults, estat_moran
from westata.postestimation.mundlak import MundlakResults, estat_mundlak
from westata.postestimation.suest import SuestResults, suest
from westata.postestimation.szroeter import SzroeterResults, SzroeterTable, estat_szroeter
from westata.postestimation.weakrobust import WeakRobustResults, estat_weak_instrument_robust

__all__ = [
    "RegressionControlMethod",
    "endog_score_test",
    "endog_regf_test",
    "treatment_balance_summarize",
    "rcm",
    "var_granger",
    "suest",
    "linktest",
    "estat_weak_instrument_robust",
    "estat_szroeter",
    "estat_mundlak",
    "estat_moran",
    "estat_effect_size",
    "estat_auc",
    "PwCompareTable",
    "ContrastResult",
    "WeakRobustResults",
    "VarGrangerResults",
    "SzroeterTable",
    "SzroeterResults",
    "SuestResults",
    "RcmResults",
    "MundlakResults",
    "MoranResults",
    "LinkTestResults",
    "LfitResults",
    "EffectSizeResults",
    "EndogScoreResults",
    "RegFResults",
    "AucResults",
    "HausmanResults",
    "LRTestResults",
    "VceMatrix",
    "estat_vce",
    "GofResults",
    "estat_gof",
    "VifResults",
    "estat_vif",
    "HetTestResults",
    "estat_heteroskedasticity_test",
    "OvTestResults",
    "estat_omitted_variables_test",
    "IcResults",
    "estat_ic",
    "FirstStageResults",
    "estat_first_stage",
    "OverIdResults",
    "estat_overidentification",
    "EndogenousResults",
    "estat_endogenous",
    "ClassificationResults",
    "estat_classification",
    "ImTestResults",
    "estat_information_matrix_test",
    "PanelTest0Results",
    "panel_test0",
    "PanelOveridResults",
    "panel_overid",
    "PanelSerialResults",
    "panel_serial",
    "VarSocResults",
    "var_select_order",
    "VarStableResults",
    "var_stable",
    "IRF_STATISTICS",
    "IRF_GRAPH_STATISTICS",
    "IrfSeries",
    "irf_cgraph",
    "irf_graph",
    "irf_ograph",
    "irf_series",
    "VarWleResults",
    "var_wle",
    "Correlogram",
    "DurbinWatsonResults",
    "PortmanteauResults",
    "estat_correlogram",
    "estat_dwatson",
    "wntestq",
    "BgTestResults",
    "estat_bgodfrey",
    "DFGLSResults",
    "df_gls",
    "KpssResults",
    "kpss",
    "VecRankResults",
    "vec_rank",
    "VecStableResults",
    "vec_stable",
    "LmarResults",
    "var_lmar",
    "vec_lmar",
    "SummarizeResults",
    "estat_summarize",
    "summarize",
    "SummarizeTable",
    "MTEST_METHODS",
    "MTestResults",
    "adjust_pvalues",
    "hausman",
    "lrtest",
    "testnl",
    "nlcom",
    "pairwise_compare",
    "contrast",
    "WaldTest",
    "LincomResults",
    "MarginsResults",
    "MarginsTable",
    "lincom_as_stata",
    "margins",
    "predict",
    "supported_types",
    "test_as_stata",
]
