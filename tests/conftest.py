"""Pytest configuration: make ``src/`` and the test helpers importable.

Two things happen here.

1. ``src/`` goes on ``sys.path`` so the package is importable without being installed,
   and the test directory goes on it too so ``stata_helpers`` can be imported by name.

2. Tests that read Stata's own ``.ado``/``.sthlp`` files are SKIPPED when no installation
   is present. Those paths used to be hard-coded to ``D:\\Program Files\\StataNow19\\...``,
   which held on the development machine and nowhere else: every one of the 20 files
   raised ``FileNotFoundError`` on the Linux CI runners and took all three jobs down.

   The skip is decided by the FILE, not by the individual test, so a new test added to one
   of these modules cannot be forgotten. The marker is applied here rather than by editing
   20 files, which is also why it cannot be undone piecemeal by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
for entry in (str(SRC), str(TESTS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import stata_helpers  # noqa: E402  (needs the sys.path line above)

#: Modules that read the Stata installation's own files as their evidence.
#: Listed by filename so the guard is visible in one place.
_STATA_FILE_MODULES = frozenset(
    {
        "test_completely_determined_counts.py",
        "test_e_aic_is_not_stata.py",
        "test_estat_auc.py",
        "test_estat_classification.py",
        "test_estat_cmd_per_command.py",
        "test_estat_dispatch_macros.py",
        "test_estat_eligibility_matrix.py",
        "test_ic_is_iterations_everywhere.py",
        "test_ml_titles_all_commands.py",
        "test_ml_titles_indirect_route.py",
        "test_predict_pr_count.py",
        "test_predict_program_macros.py",
        "test_predict_scores_shape.py",
        "test_regress_e_key_inventory.py",
        "test_title_macros_per_command.py",
        "test_tobit_has_no_aic.py",
        "test_vce_macros_ml_families.py",
        "test_vce_type_macros.py",
        "test_xt_design_names_and_ic.py",
        "test_xtreg_estat_surface.py",
    }
)


def pytest_collection_modifyitems(config, items):
    """Skip checks whose reference data is absent on this machine.

    Two independent reasons a test cannot run here, both resolved once per FILE
    rather than per test, so a module cannot be half-guarded:

    1. No Stata installation to read ``.ado``/``.sthlp`` files from.
    2. No reference dataset directory (``WESTATA_DATA_DIR`` / the author's
       machine / ``tests/data``).

    The second guard exists because a fixture-level ``pytest.skip`` does not
    protect a test that reads the file directly.  ``test_varsoc.py`` guarded its
    fixture and then called ``pd.read_stata(DATA)`` in a seventh test, which
    raised ``FileNotFoundError`` on CI instead of skipping -- as did 23 others.
    Deciding by file makes the crash shape impossible: every test in a module
    that needs absent data is skipped before it can open anything.
    """

    skip_stata = None if stata_helpers.available() else pytest.mark.skip(
        reason=stata_helpers.reason()
    )
    # Imported here, not at module scope: `datasets` imports pytest and is only
    # needed during collection.
    import datasets

    for item in items:
        name = Path(str(item.fspath)).name
        if skip_stata is not None and name in _STATA_FILE_MODULES:
            item.add_marker(skip_stata)
            continue
        needed = _REFERENCE_DATASETS.get(name)
        if needed is not None and not datasets.has(needed):
            item.add_marker(pytest.mark.skip(reason=datasets.reason(needed)))


#: Test modules that read a reference dataset OUTSIDE the repository, mapped to
#: the file they need.  A module appears here when the dataset cannot be
#: committed (the Stata example files total ~16 MB, one of them 15 MB); modules
#: using the committed ``tests/data/auto.dta`` are NOT listed, because that file
#: travels with the repository.
_REFERENCE_DATASETS = {
    "test_adf_status.py": "nelson_plosser.dta",
    "test_areg_absorbed_ftest.py": "cao_chen.dta",
    "test_arima_known_gaps.py": "airpassengers.dta",
    "test_arima_sigma_interval.py": "macro_swatson.dta",
    "test_bgodfrey.py": "icecream.dta",
    "test_capture_stata_output.py": "macro_swatson.dta",
    "test_classical_f_mean_square.py": "grilic.dta",
    "test_classification_and_white_exact.py": "grilic.dta",
    "test_clustered_factor_missing_f.py": "lin_1992.dta",
    "test_contrast_joint.py": "grilic.dta",
    "test_contrast_layout.py": "icecream.dta",
    "test_corrgram_layout.py": "icecream.dta",
    "test_dfgls.py": "macro_swatson.dta",
    "test_dfgls_status.py": "nelson_plosser.dta",
    "test_dfuller_lags_trend_exact.py": "macro_swatson.dta",
    "test_dfuller_table.py": "macro_swatson.dta",
    "test_didregress_postestimation_gap.py": "nsw_dw.dta",
    "test_discrete_intercept_contract.py": "grilic.dta",
    "test_ereturn_beta_sample.py": "grilic.dta",
    "test_ereturn_key_spelling.py": "grilic.dta",
    "test_ereturn_list.py": "grilic.dta",
    "test_ereturn_width.py": "grilic.dta",
    "test_estat_bgodfrey_layout.py": "icecream.dta",
    "test_estat_endogenous_exact.py": "grilic.dta",
    "test_estat_endogenous_layout.py": "grilic.dta",
    "test_estat_firststage_vs_first.py": "grilic.dta",
    "test_estat_ic_exact.py": "grilic.dta",
    "test_estat_ic_layout_exact.py": "cobb_douglas.dta",
    "test_estat_layout_round93.py": "grilic.dta",
    "test_estat_summarize_exact.py": "grilic.dta",
    "test_estimates_table.py": "grilic.dta",
    "test_estimates_table_esttab_full.py": "grilic.dta",
    "test_estimates_table_esttab_mode.py": "grilic.dta",
    "test_estimates_table_pvalues.py": "grilic.dta",
    "test_estimates_table_star.py": "grilic.dta",
    "test_estimator_sweep_status.py": "grilic.dta",
    "test_esttab_known_gaps.py": "grilic.dta",
    "test_esttab_screen_options.py": "grilic.dta",
    "test_etable_as_stata.py": "cobb_douglas.dta",
    "test_export_formats_available.py": "grilic.dta",
    "test_export_status.py": "grilic.dta",
    "test_fd_has_no_stata_equivalent.py": "munnell.dta",
    "test_first_stage_vce.py": "grilic.dta",
    "test_firststage_exact.py": "grilic.dta",
    "test_firststage_geometry.py": "grilic.dta",
    "test_firststage_weak_instruments.py": "grilic.dta",
    "test_fitted_and_leverage_summaries.py": "grilic.dta",
    "test_formula_freq_weights_type.py": "titanic.dta",
    "test_formula_reserved_names.py": "macro_swatson.dta",
    "test_full_precision_reference.py": "grilic.dta",
    "test_granger_string_arg.py": "macro_swatson.dta",
    "test_hausman_iv_ols.py": "grilic.dta",
    "test_hausman_matches_stata.py": "munnell.dta",
    "test_hausman_notes.py": "grilic.dta",
    "test_hausman_panel_layout.py": "munnell.dta",
    "test_hausman_sigma_options.py": "grilic.dta",
    "test_hausman_table.py": "grilic.dta",
    "test_hdfe_two_way_exact.py": "munnell.dta",
    "test_heckman_selambda.py": "grilic.dta",
    "test_hettest_ovtest_stata.py": "grilic.dta",
    "test_ipw_ate_exact.py": "nsw_dw.dta",
    "test_ipw_from_formula_duplicate_intercept.py": "nsw_dw.dta",
    "test_irf_create.py": "macro_swatson.dta",
    "test_irf_table_layout.py": "macro_swatson.dta",
    "test_iv_first_stage_table.py": "grilic.dta",
    "test_iv_postestimation_exact.py": "grilic.dta",
    "test_iv_postestimation_layout.py": "grilic.dta",
    "test_iv_second_stage_layout.py": "grilic.dta",
    "test_joint_test_matches_stata.py": "nsw_dw.dta",
    "test_kpss.py": "macro_swatson.dta",
    "test_kpss_layout.py": "macro_swatson.dta",
    "test_lag_operator.py": "macro_swatson.dta",
    "test_likelihood_table_border.py": "grilic.dta",
    "test_liml_and_first_stage.py": "grilic.dta",
    "test_liml_matches_stata.py": "grilic.dta",
    "test_lincom_contrast_structure.py": "grilic.dta",
    "test_lincom_layout_exact.py": "cobb_douglas.dta",
    "test_lincom_number_format.py": "grilic.dta",
    "test_logit_odds_ratio.py": "nsw_dw.dta",
    "test_low_scored_items_that_are_exact.py": "munnell.dta",
    "test_lsdv_clustered_exact.py": "munnell.dta",
    "test_margins_atmeans_fweight.py": "titanic.dta",
    "test_margins_atmeans_layout.py": "nsw_dw.dta",
    "test_margins_atmeans_odds.py": "grilic.dta",
    "test_margins_dydx_varlist.py": "grilic.dta",
    "test_margins_dydx_wildcard.py": "cobb_douglas.dta",
    "test_margins_elasticity.py": "nsw_dw.dta",
    "test_margins_eyex_and_at.py": "nerlove.dta",
    "test_margins_fweight_forms.py": "titanic.dta",
    "test_margins_header_structure.py": "cobb_douglas.dta",
    "test_margins_layout.py": "grilic.dta",
    "test_margins_matches_stata.py": "nerlove.dta",
    "test_ml_summary_leading_blank.py": "grilic.dta",
    "test_mlogit.py": "lin_1992.dta",
    "test_more_zero_scored.py": "grilic.dta",
    "test_negative_aweight_and_border.py": "nerlove.dta",
    "test_noconstant_and_subsample.py": "grilic.dta",
    "test_odds_ratio_last_digit.py": "grilic.dta",
    "test_odds_ratio_stata.py": "grilic.dta",
    "test_ologit_cut_precision.py": "grilic.dta",
    "test_overid_and_leverage.py": "grilic.dta",
    "test_overid_and_predict_p.py": "grilic.dta",
    "test_overid_format.py": "grilic.dta",
    "test_ovtest_collinearity_notes.py": "grilic.dta",
    "test_ovtest_rhs_design_order.py": "grilic.dta",
    "test_p0_items_status.py": "lin_1992.dta",
    "test_panel_absorbed_columns.py": "cao_chen.dta",
    "test_panel_fe_cluster.py": "munnell.dta",
    "test_panel_param_order.py": "lin_1992.dta",
    "test_panel_two_way_fe.py": "munnell.dta",
    "test_panel_zero_scores.py": "lin_1992.dta",
    "test_prais.py": "icecream.dta",
    "test_prais_convergence.py": "icecream.dta",
    "test_prais_corc.py": "cobb_douglas.dta",
    "test_prais_corc_behaviour.py": "icecream.dta",
    "test_prais_corc_nonconvergence.py": "icecream.dta",
    "test_prais_corc_standard_errors.py": "icecream.dta",
    "test_prais_corc_table.py": "icecream.dta",
    "test_prais_iteration_log.py": "icecream.dta",
    "test_prais_result_protocol.py": "icecream.dta",
    "test_predict_fweight_shape.py": "titanic.dta",
    "test_predict_p_alias.py": "loanapp.dta",
    "test_predict_resid_alias.py": "grilic.dta",
    "test_predict_spelling_and_values.py": "grilic.dta",
    "test_predict_summarize_workflow.py": "cobb_douglas.dta",
    "test_predict_then_summarize.py": "grilic.dta",
    "test_probit_fweight_and_lags.py": "macro_swatson.dta",
    "test_psmatch.py": "nsw_dw.dta",
    "test_psmatch_atet_exact.py": "nsw_dw.dta",
    "test_psmatch_caliper.py": "nsw_dw.dta",
    "test_psmatch_inference_status.py": "nsw_dw.dta",
    "test_pwcompare_layout.py": "icecream.dta",
    "test_rcm.py": "synth_smoking.dta",
    "test_rcm_matches_stata.py": "munnell.dta",
    "test_rdbandwidth_helpers.py": "rdrobust_senate.dta",
    "test_rdrobust.py": "border.dta",
    "test_rdrobust_bandwidth_gap.py": "rdrobust_senate.dta",
    "test_rdrobust_bw_inputs.py": "rdrobust_senate.dta",
    "test_rdrobust_covs.py": "rdrobust_senate.dta",
    "test_regression_workflow_exact.py": "grilic.dta",
    "test_robust_vce_mapping.py": "grilic.dta",
    "test_serial_blocks_exact.py": "grilic.dta",
    "test_serial_correlation.py": "icecream.dta",
    "test_serial_time_axis.py": "icecream.dta",
    "test_stata_ordering_trap.py": "grilic.dta",
    "test_summarize_command.py": "cobb_douglas.dta",
    "test_summarize_leading_blank.py": "grilic.dta",
    "test_summarize_multivariable_exact.py": "grilic.dta",
    "test_summary_defaults_to_stata.py": "grilic.dta",
    "test_synth_qp_refinement.py": "synth_smoking.dta",
    "test_synth_status.py": "basque.dta",
    "test_table_widths_verified.py": "icecream.dta",
    "test_tebalance.py": "nsw_dw.dta",
    "test_tebalance_matched.py": "nsw_dw.dta",
    "test_tebalance_normalisation.py": "nsw_dw.dta",
    "test_teffects_aipw_intercept.py": "nsw_dw.dta",
    "test_teffects_diagnostic_traps.py": "nsw_dw.dta",
    "test_teffects_estimator_accuracy.py": "nsw_dw.dta",
    "test_teffects_ipw_intercept.py": "nsw_dw.dta",
    "test_teffects_nnmatch.py": "nsw_dw.dta",
    "test_teffects_nnmatch_exact.py": "nsw_dw.dta",
    "test_teffects_nnmatch_neighbours.py": "nsw_dw.dta",
    "test_teffects_ra_intercept.py": "nsw_dw.dta",
    "test_test_and_lincom_exact.py": "grilic.dta",
    "test_test_factor_vs_group.py": "grilic.dta",
    "test_tobit_variance_interval.py": "food.dta",
    "test_unitroot_adf.py": "macro_swatson.dta",
    "test_unitroot_aliases.py": "nelson_plosser.dta",
    "test_v2_p0_defects.py": "lin_1992.dta",
    "test_var_bare_form_reference.py": "macro_swatson.dta",
    "test_var_covariance_reference.py": "macro_swatson.dta",
    "test_var_equation_stats.py": "macro_swatson.dta",
    "test_var_fevd.py": "macro_swatson.dta",
    "test_var_fevd_reference.py": "macro_swatson.dta",
    "test_var_irf.py": "macro_swatson.dta",
    "test_var_irf_reference.py": "macro_swatson.dta",
    "test_var_name_propagation.py": "macro_swatson.dta",
    "test_var_reference_bypass.py": "macro_swatson.dta",
    "test_var_summary.py": "macro_swatson.dta",
    "test_vargranger_ml_wald.py": "macro_swatson.dta",
    "test_varlmar.py": "macro_swatson.dta",
    "test_varlmar_varwle_exact.py": "macro_swatson.dta",
    "test_varsoc.py": "macro_swatson.dta",
    "test_varsoc_fixed_window.py": "macro_swatson.dta",
    "test_varstable.py": "macro_swatson.dta",
    "test_varstable_box.py": "macro_swatson.dta",
    "test_varstable_exact.py": "macro_swatson.dta",
    "test_varstable_stata.py": "macro_swatson.dta",
    "test_varwle.py": "macro_swatson.dta",
    "test_varwle_joint_wald.py": "macro_swatson.dta",
    "test_vecm_alpha_blocks.py": "macro_swatson.dta",
    "test_vecm_beta_table.py": "macro_swatson.dta",
    "test_vecm_coefficient_table.py": "macro_swatson.dta",
    "test_vecm_coint_constants.py": "macro_swatson.dta",
    "test_vecm_cointegrating_block.py": "macro_swatson.dta",
    "test_vecm_constant_transform.py": "macro_swatson.dta",
    "test_vecm_cov_scale.py": "macro_swatson.dta",
    "test_vecm_equation_summary.py": "macro_swatson.dta",
    "test_vecm_multi_relation.py": "macro_swatson.dta",
    "test_vecm_scalar_surface.py": "macro_swatson.dta",
    "test_vecm_summary_layout.py": "macro_swatson.dta",
    "test_vecm_table_layout.py": "macro_swatson.dta",
    "test_vecm_variance_scale.py": "macro_swatson.dta",
    "test_vecrank.py": "macro_swatson.dta",
    "test_vecrank_critical_table.py": "macro_swatson.dta",
    "test_vecrank_general_johansen.py": "macro_swatson.dta",
    "test_vecrank_geometry.py": "macro_swatson.dta",
    "test_vecrank_max_table.py": "macro_swatson.dta",
    "test_vecrank_trend_status.py": "macro_swatson.dta",
    "test_vecrank_trend_three_var.py": "macro_swatson.dta",
    "test_vecstable.py": "macro_swatson.dta",
    "test_vif_imtest_structure.py": "grilic.dta",
    "test_wald_leading_blank.py": "grilic.dta",
    "test_wald_parenthesised.py": "grilic.dta",
    "test_wald_test_string_constraint.py": "grilic.dta",
    "test_weighted_logit_pseudolikelihood.py": "grilic.dta",
    "test_wls_aweight_exact.py": "nerlove.dta",
    "test_xtreg_fe_be_coefficients.py": "munnell.dta",
    "test_xtreg_fe_robust_sigma.py": "lin_1992.dta",
    "test_xtreg_fe_unaffected.py": "lin_1992.dta",
    "test_xtreg_mle.py": "lin_1992.dta",
    "test_xtreg_mle_exact.py": "munnell.dta",
    "test_xtreg_mle_optimiser.py": "munnell.dta",
    "test_xtreg_mle_reference.py": "lin_1992.dta",
    "test_xtreg_multi_specification.py": "munnell.dta",
    "test_xtreg_re_sigma_e.py": "munnell.dta",
    "test_xtreg_re_swamy_arora.py": "lin_1992.dta",
    "test_xtreg_re_table_exact.py": "munnell.dta",
    "test_xtreg_re_theta_option.py": "lin_1992.dta",
    "test_xtreg_variance_components_exact.py": "munnell.dta",
    "test_xtreg_year_fe_closed.py": "munnell.dta",
    "test_xtreg_year_fe_status.py": "munnell.dta",
    "test_xtserial.py": "lin_1992.dta",
    "test_xttest0_matches_stata.py": "munnell.dta",
    "test_xttest0_variance_and_names.py": "munnell.dta",
    "test_xttest0_variance_block.py": "munnell.dta",
    "test_zero_scored_but_present.py": "grilic.dta",
    "test_zero_scored_items.py": "grilic.dta",
}
