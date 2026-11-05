"""Postestimation: hypothesis tests, linear combinations, marginal effects.

Targets Stata's ``test``, ``lincom`` and ``margins`` commands.
"""

from __future__ import annotations

from westata.postestimation.api import (
    lincom_as_stata,
    statsmodels_wald_test,
    test_as_stata,
)
from westata.postestimation.contrast import contrast, pwcompare
from westata.postestimation.lincom import LincomResult, run_lincom
from westata.postestimation.modeltests import (
    HausmanResult,
    LRTestResult,
    hausman,
    lrtest,
)
from westata.postestimation.nonlinear import nlcom, testnl
from westata.postestimation.margins import MarginsResult, MarginsTable, margins
from westata.postestimation.predict import SUPPORTED_TYPES, predict, supported_types
from westata.postestimation.tests import (
    WaldTest,
    build_contrast,
    describe_constraint,
    resolve_names,
    run_wald_test,
)
from westata.postestimation.classification import (
    ClassificationResult,
    estat_classification,
)
from westata.postestimation.endogenous import EndogenousResult, estat_endogenous
from westata.postestimation.firststage import FirstStageResult, estat_firststage
from westata.postestimation.gof import GofResult, estat_gof
from westata.postestimation.hettest import HetTestResult, estat_hettest
from westata.postestimation.ic import IcResult, estat_ic
from westata.postestimation.imtest import ImTestResult, estat_imtest
from westata.postestimation.xttest0 import XtTest0Result, xttest0
from westata.postestimation.varsoc import VarSocResult, varsoc
from westata.postestimation.varstable import VarStableResult, varstable
from westata.postestimation.varwle import VarWleResult, varwle
from westata.postestimation.bgodfrey import BgTestResult, estat_bgodfrey
from westata.postestimation.dfgls import DFGLSResult, dfgls
from westata.postestimation.kpss import KpssResult, kpss
from westata.postestimation.vecrank import VecRankResult, vecrank
from westata.postestimation.vecstable import VecStableResult, vecstable
from westata.postestimation.varlmar import LmarResult, varlmar, veclmar
from westata.postestimation.serial import (
    Correlogram,
    DurbinWatsonResult,
    PortmanteauResult,
    estat_corrgram,
    estat_dwatson,
    wntestq,
)
from westata.postestimation.ovtest import OvTestResult, estat_ovtest
from westata.postestimation.mtest import MTEST_METHODS, MTestResult, adjust_pvalues
from westata.postestimation.overid import OverIdResult, estat_overid
from westata.postestimation.summarize import SummarizeResult, estat_summarize
from westata.postestimation.vce import VceMatrix, estat_vce
from westata.postestimation.vif import VifResult, estat_vif

__all__ = [
    "HausmanResult",
    "LRTestResult",
    "VceMatrix",
    "estat_vce",
    "GofResult",
    "estat_gof",
    "VifResult",
    "estat_vif",
    "HetTestResult",
    "estat_hettest",
    "OvTestResult",
    "estat_ovtest",
    "IcResult",
    "estat_ic",
    "FirstStageResult",
    "estat_firststage",
    "OverIdResult",
    "estat_overid",
    "EndogenousResult",
    "estat_endogenous",
    "ClassificationResult",
    "estat_classification",
    "ImTestResult",
    "estat_imtest",
    "XtTest0Result",
    "xttest0",
    "VarSocResult",
    "varsoc",
    "VarStableResult",
    "varstable",
    "VarWleResult",
    "varwle",
    "Correlogram",
    "DurbinWatsonResult",
    "PortmanteauResult",
    "estat_corrgram",
    "estat_dwatson",
    "wntestq",
    "BgTestResult",
    "estat_bgodfrey",
    "DFGLSResult",
    "dfgls",
    "KpssResult",
    "kpss",
    "VecRankResult",
    "vecrank",
    "VecStableResult",
    "vecstable",
    "LmarResult",
    "varlmar",
    "veclmar",
    "SummarizeResult",
    "estat_summarize",
    "MTEST_METHODS",
    "MTestResult",
    "adjust_pvalues",
    "hausman",
    "lrtest",
    "testnl",
    "nlcom",
    "pwcompare",
    "contrast",
    "WaldTest",
    "LincomResult",
    "MarginsResult",
    "MarginsTable",
    "SUPPORTED_TYPES",
    "build_contrast",
    "describe_constraint",
    "lincom_as_stata",
    "margins",
    "predict",
    "resolve_names",
    "run_lincom",
    "run_wald_test",
    "statsmodels_wald_test",
    "supported_types",
    "test_as_stata",
]
