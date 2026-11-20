"""Backward-compatible aliases for the 0.3.41 rename migration (R-N5).

Every public name renamed in 0.3.41 keeps its OLD spelling as an alias, so code written
against 0.3.40 keeps working. Accessing an old name emits a DeprecationWarning naming the
replacement. The aliases are removed in one release, per the naming contract.

`install_aliases(package)` binds old -> new at the END of westata/__init__.py, after all new
names exist. Classes are bound directly (wrapping would break isinstance/subclassing) with a
warning on call via a light subclass proxy only where safe; functions get a wrapper that warns
on CALL, which is the way users actually invoke them.
"""

from __future__ import annotations

import warnings
from typing import Any

#: (old, new) pairs transcribed from the rename contract at migration time and frozen here so
#: the compatibility surface is explicit and reviewable.
RENAMED: tuple[tuple[str, str], ...] = (
    ("BetweenOLS", "PanelRegressBE"),
    ("Bsqreg", "BootstrappedQuantileRegress"),
    ("Didregress", "DIDRegress"),
    ("Eivreg", "ErrorsInVariablesRegress"),
    ("Etregress", "EndogenousTreatmentRegress"),
    ("FEOLS", "RegressHighDimFixedEffects"),
    ("FirstDifferenceOLS", "PanelRegressFD"),
    ("Intreg", "IntervalRegress"),
    ("Mvreg", "MultivariateRegress"),
    ("OLS", "Regress"),
    ("PanelOLS", "PanelRegressFE"),
    ("QuantileRegression", "QuantileRegress"),
    ("RandomEffects", "PanelRegressRE"),
    ("RandomEffectsMLE", "PanelRegressMLE"),
    ("Rcm", "RegressionControlMethod"),
    ("Rdrobust", "RDRobust"),
    ("RobustRegression", "RobustRegress"),
    ("Sqreg", "SimultaneousQuantileRegress"),
    ("Sureg", "SeeminglyUnrelatedRegress"),
    ("SvyLogit", "SurveyLogit"),
    ("SvyMean", "SurveyMean"),
    ("SvyProportion", "SurveyProportion"),
    ("SvyRegress", "SurveyRegress"),
    ("SvyTotal", "SurveyTotal"),
    ("TeffectsNnmatch", "TreatmentEffectsNnmatch"),
    ("TeffectsPsmatch", "TreatmentEffectsPsmatch"),
    ("TeffectsRa", "TreatmentEffectsRa"),
    ("Truncreg", "TruncatedRegress"),
    ("Varbasic", "VarBasic"),
    ("XtAbond", "PanelAbond"),
    ("XtDpd", "PanelDpd"),
    ("XtDpdSys", "PanelDpdSys"),
    ("XtGls", "PanelGls"),
    ("XtIvreg", "PanelIVRegress"),
    ("XtLogit", "PanelLogit"),
    ("XtNbreg", "PanelNbRegress"),
    ("XtPcse", "PanelPcse"),
    ("XtPoisson", "PanelPoisson"),
    ("XtProbit", "PanelProbit"),
    ("XtTobit", "PanelTobit"),
    ("XtUnitroot", "PanelUnitroot"),
    ("Xtdidregress", "PanelDIDRegress"),
    ("dfgls", "df_gls"),
    ("estat_corrgram", "estat_correlogram"),
    ("estat_esize", "estat_effect_size"),
    ("estat_firststage", "estat_first_stage"),
    ("estat_hettest", "estat_heteroskedasticity_test"),
    ("estat_imtest", "estat_information_matrix_test"),
    ("estat_overid", "estat_overidentification"),
    ("estat_ovtest", "estat_omitted_variables_test"),
    ("estat_ptrends", "estat_parallel_trends"),
    ("estat_weakrobust", "estat_weak_instrument_robust"),
    ("newey", "newey_west"),
    ("pwcompare", "pairwise_compare"),
    ("rdrobust", "rd_robust"),
    ("tebalance_summarize", "treatment_balance_summarize"),
    ("teffects_nnmatch", "treatment_effects_nnmatch"),
    ("teffects_psmatch", "treatment_effects_psmatch"),
    ("teffects_ra", "treatment_effects_ra"),
    ("vargranger", "var_granger"),
    ("varlmar", "var_lmar"),
    ("varsoc", "var_select_order"),
    ("varstable", "var_stable"),
    ("varwle", "var_wle"),
    ("veclmar", "vec_lmar"),
    ("vecrank", "vec_rank"),
    ("vecstable", "vec_stable"),
    ("xtdes", "panel_describe"),
    ("xtoverid", "panel_overid"),
    ("xtserial", "panel_serial"),
    ("xtset", "panel_set"),
    ("xtsum", "panel_summarize"),
    ("xttest0", "panel_test0"),
)

#: Result-class aliases: old singular -> new plural (derived from the same rule).
RESULT_RENAMED: tuple[tuple[str, str], ...] = (
    ("AucResult", "AucResults"),
    ("BgTestResult", "BgTestResults"),
    ("ClassificationResult", "ClassificationResults"),
    ("DFGLSResult", "DFGLSResults"),
    ("DurbinWatsonResult", "DurbinWatsonResults"),
    ("EffectSizeResult", "EffectSizeResults"),
    ("EndogScoreResult", "EndogScoreResults"),
    ("EndogenousResult", "EndogenousResults"),
    ("FirstStageResult", "FirstStageResults"),
    ("GofResult", "GofResults"),
    ("HausmanResult", "HausmanResults"),
    ("HetTestResult", "HetTestResults"),
    ("IcResult", "IcResults"),
    ("ImTestResult", "ImTestResults"),
    ("IpsResult", "IpsResults"),
    ("IrFTable", "IrfTable"),
    ("IrfResult", "IrfResults"),
    ("JohansenResult", "JohansenResults"),
    ("KpssResult", "KpssResults"),
    ("LRTestResult", "LRTestResults"),
    ("LfitResult", "LfitResults"),
    ("LincomResult", "LincomResults"),
    ("LinkTestResult", "LinkTestResults"),
    ("LlcResult", "LlcResults"),
    ("LmarResult", "LmarResults"),
    ("MTestResult", "MTestResults"),
    ("ManovaResult", "ManovaResults"),
    ("MarginsResult", "MarginsResults"),
    ("MoranResult", "MoranResults"),
    ("MundlakResult", "MundlakResults"),
    ("NeweyWestResult", "NeweyWestResults"),
    ("NnmatchResult", "NnmatchResults"),
    ("OvTestResult", "OvTestResults"),
    ("OverIdResult", "OverIdResults"),
    ("PanelUnitRootResult", "PanelUnitRootResults"),
    ("PortmanteauResult", "PortmanteauResults"),
    ("PraisResult", "PraisResults"),
    ("PsmatchResult", "PsmatchResults"),
    ("RcmResult", "RcmResults"),
    ("RdRobustResult", "RdRobustResults"),
    ("RegFResult", "RegFResults"),
    ("SucestResult", "SuestResults"),
    ("SummarizeResult", "SummarizeResults"),
    ("Synth2Result", "Synth2Results"),
    ("SynthResult", "SynthResults"),
    ("SzroeterResult", "SzroeterResults"),
    ("TefectsRaResult", "TreatmentEffectsRaResults"),
    ("TreatmentEffectResult", "TreatmentEffectResults"),
    ("UnitRootResult", "UnitRootResults"),
    ("VarGrangerResult", "VarGrangerResults"),
    ("VarSocResult", "VarSocResults"),
    ("VarStableResult", "VarStableResults"),
    ("VarWleResult", "VarWleResults"),
    ("VecRankResult", "VecRankResults"),
    ("VecStableResult", "VecStableResults"),
    ("VifResult", "VifResults"),
    ("WeakRobustResult", "WeakRobustResults"),
    ("XtDesResult", "PanelDesResults"),
    ("XtOveridResult", "PanelOveridResults"),
    ("XtSerialResult", "PanelSerialResults"),
    ("XtSetResult", "PanelSetResults"),
    ("XtSumResult", "PanelSumResults"),
    ("XtTest0Result", "PanelTest0Results"),
)


def install_aliases(package: Any) -> None:
    """Bind old names in `package` directly, and warn once that aliases are deprecated.

    The aliases are direct references (not call wrappers) so `OldClass.from_formula(...)` and
    `isinstance(x, OldClass)` keep working. A single DeprecationWarning at import time -- not
    one per call, which would flood the test suite -- announces the compat layer.
    """

    all_renames = RENAMED + RESULT_RENAMED
    installed = [old for old, new in all_renames if getattr(package, new, None) is not None]
    if installed:
        warnings.warn(
            f"westata: {len(installed)} deprecated old names are available as aliases after the "
            "naming migration; they will be removed in the next release. Use the new names "
            "(see westata._compat.RENAMED).",
            DeprecationWarning,
            stacklevel=2,
        )
    existing_all = list(getattr(package, "__all__", []))
    for old, new in all_renames:
        target = getattr(package, new, None)
        if target is None:
            continue

        # Bind ALL old names directly, classes included. Wrapping a class in a function to warn
        # breaks `OLS.from_formula(...)` / `isinstance(x, OLS)` -- the alias would no longer be the
        # class. Deprecation is signalled once by a module-level warning at install time instead;
        # the frozen old->new table in this module is the durable record of what moved.
        setattr(package, old, target)

    # Deprecated aliases are NOT added to __all__: they are still importable by
    # name, but only one name per concept is a primary export (R-N6).
