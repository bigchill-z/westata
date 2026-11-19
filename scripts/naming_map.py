"""Machine-readable rename contract for the westata naming convention.

This is the single source of truth for the *planned* renames documented in
``docs/NAMING_MAP.md``.  ``check_naming.py`` validates this contract against the
rules in ``docs/NAMING_CONVENTIONS.md`` (R-N1..R-N6) and reports migration
progress against the source tree.

Every entry is data, never a side effect.  Keeping it declarative is what lets
the checker catch an inconsistent or colliding mapping BEFORE any code is renamed.

Kinds: "class" (estimator), "result" (result/table/row object), "function".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rename:
    old: str
    new: str
    kind: str  # class | result | function
    note: str = ""


# --------------------------------------------------------------------------- #
# Rule parameters (from docs/NAMING_CONVENTIONS.md R-N1..R-N6)
# --------------------------------------------------------------------------- #

#: Familiar acronyms / command-shaped names that are NOT expanded (R-N1 rule 3,
#: plus the user's later additions DID / IV / RD / VECM).
FAMILIAR = {
    "DID", "IV", "RD", "VECM", "VAR", "ARIMA", "GARCH", "ARCH", "AR", "MA",
    "2SLS", "LIML", "GMM", "FE", "RE", "BE", "FD", "MLE", "NB", "NB1",
    "AUC", "GOF", "IC", "VCE", "VIF",
}

#: Cryptic Stata prefixes that MUST be expanded when they appear as a leading
#: token of a class/function name (R-N1 rule 2).  Value is the expansion.
PREFIX_EXPANSIONS = {
    "Xt": "Panel",
    "Svy": "Survey",
    "Teffects": "TreatmentEffects",
    "TEFFECTS": "TreatmentEffects",
}

#: estat_* suffix expansions: the `estat_` prefix is KEPT, only the suffix expands
#: (R-N5).  Suffixes that are a person's name or a familiar acronym are exempt.
ESTAT_SUFFIX_EXPANSIONS = {
    "ptrends": "parallel_trends",
    "esize": "effect_size",
    "firststage": "first_stage",
    "hettest": "heteroskedasticity_test",
    "imtest": "information_matrix_test",
    "overid": "overidentification",
    "ovtest": "omitted_variables_test",
    "corrgram": "correlogram",
    "weakrobust": "weak_instrument_robust",
}
ESTAT_SUFFIX_KEEP = {
    # person names (user: keep)
    "bgodfrey", "moran", "mundlak", "szroeter", "dwatson",
    # familiar acronyms (user: keep)
    "auc", "gof", "ic", "vce", "vif",
    # already whole words
    "classification", "endogenous", "summarize",
}

#: Result suffix rule (R-N3/R-N6): every singular *Result becomes plural *Results.
#: These two are semantic opposites and each already exists -> keep both (exception).
RESULT_COLLISION_KEEP = {"ContrastResult", "ContrastResults"}


# --------------------------------------------------------------------------- #
# Rename contract — transcribed from docs/NAMING_MAP.md
# --------------------------------------------------------------------------- #

CLASS_RENAMES = [
    # §1 linear / IV / HDFE / panel
    Rename("OLS", "Regress", "class"),
    Rename("FEOLS", "RegressHighDimFixedEffects", "class"),
    Rename("PanelOLS", "PanelRegressFE", "class"),
    Rename("RandomEffects", "PanelRegressRE", "class"),
    Rename("RandomEffectsMLE", "PanelRegressMLE", "class"),
    Rename("BetweenOLS", "PanelRegressBE", "class"),
    Rename("FirstDifferenceOLS", "PanelRegressFD", "class"),
    # §2 discrete / counts / limited / quantile / nonlinear
    Rename("Truncreg", "TruncatedRegress", "class"),
    Rename("Intreg", "IntervalRegress", "class"),
    Rename("Etregress", "EndogenousTreatmentRegress", "class"),
    Rename("QuantileRegression", "QuantileRegress", "class"),
    Rename("RobustRegression", "RobustRegress", "class"),
    Rename("Sqreg", "SimultaneousQuantileRegress", "class"),
    Rename("Bsqreg", "BootstrappedQuantileRegress", "class"),
    # §3 panel / svy / teffects / rd / rcm
    Rename("XtAbond", "PanelAbond", "class"),
    Rename("XtDpd", "PanelDpd", "class"),
    Rename("XtDpdSys", "PanelDpdSys", "class"),
    Rename("XtIvreg", "PanelIVRegress", "class"),
    Rename("XtGls", "PanelGls", "class"),
    Rename("XtPcse", "PanelPcse", "class"),
    Rename("XtUnitroot", "PanelUnitroot", "class"),
    Rename("XtNbreg", "PanelNbRegress", "class"),
    Rename("XtLogit", "PanelLogit", "class"),
    Rename("XtProbit", "PanelProbit", "class"),
    Rename("XtTobit", "PanelTobit", "class"),
    Rename("XtPoisson", "PanelPoisson", "class"),
    Rename("Didregress", "DIDRegress", "class"),
    Rename("Xtdidregress", "PanelDIDRegress", "class"),
    Rename("SvyMean", "SurveyMean", "class"),
    Rename("SvyTotal", "SurveyTotal", "class"),
    Rename("SvyProportion", "SurveyProportion", "class"),
    Rename("SvyRegress", "SurveyRegress", "class"),
    Rename("SvyLogit", "SurveyLogit", "class"),
    Rename("TeffectsRa", "TreatmentEffectsRa", "class"),
    Rename("TeffectsNnmatch", "TreatmentEffectsNnmatch", "class"),
    Rename("TeffectsPsmatch", "TreatmentEffectsPsmatch", "class"),
    Rename("Rdrobust", "RDRobust", "class"),
    Rename("Rcm", "RegressionControlMethod", "class"),
    # §4 systems / varbasic
    Rename("Sureg", "SeeminglyUnrelatedRegress", "class"),
    Rename("Mvreg", "MultivariateRegress", "class"),
    Rename("Eivreg", "ErrorsInVariablesRegress", "class"),
    Rename("Varbasic", "VarBasic", "class"),
]

#: Classes/docstrings explicitly kept as-is, with the reason (so the checker can
#: distinguish "forgotten" from "deliberately kept").  Only the notable ones are
#: listed; the rest are implied by absence from CLASS_RENAMES.
CLASS_KEEP = {
    "IV2SLS": "IV familiar acronym", "IVLIML": "IV familiar acronym", "IVGMM": "IV familiar acronym",
    "Areg": "PascalCase of Stata command areg", "Newey": "PascalCase of command",
    "Prais": "PascalCase of command", "Logit": "command name", "Probit": "command name",
    "Poisson": "command name", "Heckman": "command name", "Tobit": "command name",
    "Frontier": "command name", "Nl": "command name `nl`", "Manova": "command name",
    "ARIMA": "familiar", "VAR": "familiar", "VECM": "familiar", "GARCH": "familiar (Stata arch covers family)",
    "Bootstrap": "command name", "Jackknife": "command name", "Permute": "command name",
    "Synth": "command name", "Synth2": "command name", "SurveyDesign": "design object, not estimator",
    "NegativeBinomial": "already expanded", "NegativeBinomialNB1": "variant, avoid collision",
    "OrderedLogit": "already expanded", "OrderedProbit": "already expanded",
    "MultinomialLogit": "already expanded", "ZeroInflatedPoisson": "already expanded",
    "ZeroInflatedNegativeBinomial": "already expanded", "TruncatedPoisson": "already expanded",
    "TruncatedNegativeBinomial": "already expanded",
    "IPW": "independent teffects estimator", "AIPW": "independent teffects estimator",
    "RegressionAdjustment": "independent teffects estimator (merge deferred)",
    "NearestNeighbourMatching": "independent teffects estimator (merge deferred)",
    "DynamicPanelInstruments": "helper dataclass, no Stata command",
}

FUNCTION_RENAMES = [
    Rename("xtset", "panel_set", "function"),
    Rename("xtdes", "panel_describe", "function"),
    Rename("xtsum", "panel_summarize", "function"),
    Rename("newey", "newey_west", "function"),
    Rename("teffects_ra", "treatment_effects_ra", "function"),
    Rename("teffects_nnmatch", "treatment_effects_nnmatch", "function"),
    Rename("teffects_psmatch", "treatment_effects_psmatch", "function"),
    Rename("rdrobust", "rd_robust", "function"),
    Rename("xtserial", "panel_serial", "function"),
    Rename("xtoverid", "panel_overid", "function"),
    Rename("xttest0", "panel_test0", "function"),
    Rename("varsoc", "var_select_order", "function"),
    Rename("vargranger", "var_granger", "function"),
    Rename("varlmar", "var_lmar", "function"),
    Rename("varstable", "var_stable", "function"),
    Rename("varwle", "var_wle", "function"),
    Rename("veclmar", "vec_lmar", "function"),
    Rename("vecrank", "vec_rank", "function"),
    Rename("vecstable", "vec_stable", "function"),
    Rename("dfgls", "df_gls", "function"),
    Rename("pwcompare", "pairwise_compare", "function"),
    Rename("tebalance_summarize", "treatment_balance_summarize", "function"),
    # estat_* suffix expansions (prefix kept)
    Rename("estat_ptrends", "estat_parallel_trends", "function"),
    Rename("estat_esize", "estat_effect_size", "function"),
    Rename("estat_firststage", "estat_first_stage", "function"),
    Rename("estat_hettest", "estat_heteroskedasticity_test", "function"),
    Rename("estat_imtest", "estat_information_matrix_test", "function"),
    Rename("estat_overid", "estat_overidentification", "function"),
    Rename("estat_ovtest", "estat_omitted_variables_test", "function"),
    Rename("estat_corrgram", "estat_correlogram", "function"),
    Rename("estat_weakrobust", "estat_weak_instrument_robust", "function"),
]

#: estat_* kept as-is (person names / familiar / whole words).
ESTAT_KEEP = {
    "estat_auc", "estat_gof", "estat_ic", "estat_vce", "estat_vif",
    "estat_bgodfrey", "estat_moran", "estat_mundlak", "estat_szroeter",
    "estat_dwatson", "estat_classification", "estat_endogenous", "estat_summarize",
}

#: Functions kept as-is (command-shaped / familiar / descriptive).
FUNCTION_KEEP = {
    "dfuller", "pperron", "kpss", "unitroot", "levin_lin_chu", "im_pesaran_shin",
    "prais", "prais_winsten", "synth", "synth2", "rcm", "linktest", "suest",
    "margins", "contrast", "predict", "summarize", "supported_types",
    "lincom_as_stata", "test_as_stata", "endog_regf_test", "endog_score_test",
    "check_time_spacing", "normalize_weights", "parse_exp_list", "etable", "ssc",
    "wntestq", "evaluate_statistic", "compile_expression", "solve_lad",
    "build_design", "build_instruments", "adjust_pvalues", "format_ereturn_list",
    "first_stage_table", "render_first_stage", "render_estimates_table",
    "render_esttab", "etable_as_stata", "render_esttab_table", "canonical_label",
    "esttab_number", "star_run", "stars", "render_summary", "stata_number",
    "to_latex", "to_markdown", "to_dict", "to_frame",
}

#: Result-class name transformations applied BEFORE the uniform "+s".  The checker
#: discovers every singular *Result from the source AST and applies these to build
#: its rename, so adding a new result class needs no contract edit.
RESULT_NAME_FIXES = {
    "TefectsRaResult": "TreatmentEffectsRaResult",  # spelling + prefix rule 2
    "SucestResult": "SuestResult",                  # spelling defect
    "XtSetResult": "PanelSetResult",                # xt -> Panel
    "XtDesResult": "PanelDesResult",
    "XtSumResult": "PanelSumResult",
    "XtSerialResult": "PanelSerialResult",
    "XtOveridResult": "PanelOveridResult",
    "XtTest0Result": "PanelTest0Result",
    "IrFTable": "IrfTable",  # table/row spelling normalisation (not a *Result)
}


def result_rename(old: str) -> Rename:
    """Derive the contract entry for a singular result class from the rules."""
    base = RESULT_NAME_FIXES.get(old, old)
    if base == "Irftable":  # pragma: no cover - guard
        return Rename(old, "IrfTable", "result")
    if old == "IrFTable":
        return Rename(old, "IrfTable", "result")
    return Rename(old, base + "s", "result")


ALL_RENAMES = CLASS_RENAMES + FUNCTION_RENAMES

