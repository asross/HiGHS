/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkRustSolve.cpp
 * @brief HEkk::solve delegating to the Rust port (rust/src/simplex/hekk.rs,
 * dual.rs and primal.rs): the set-up of HEkk's vectors, the wrap-up, and
 * the C++ that the Rust calls (logging, analysis reports, the clock, the
 * user interrupt, and rare paths)
 */
#include <cmath>
#include <cstdio>

#include "simplex/HEkk.h"

#ifdef HIGHS_RUST

#include "parallel/HighsParallel.h"
#include "simplex/HEkkRust.h"
#include "simplex/HSimplexDebug.h"

namespace highs_rs {
// Mirrors of the #[repr(C)] structs in rust/src/simplex/dual.rs and
// primal.rs
struct DualState {
  int solve_phase;
  int edge_weight_mode;
  int num_devex_iterations;
  int row_out;
  int variable_out;
  int variable_in;
  int rebuild_reason;
  double delta_primal;
  double theta_primal;
  double theta_dual;
  double alpha_col;
  double alpha_row;
  double numerical_trouble;
};

struct AnalysisData {
  DualState state;
  int iteration_count;
  double factor_pivot_threshold;
  double edge_weight_error;
  double updated_dual_objective_value;
  int num_primal_infeasibilities;
  double sum_primal_infeasibilities;
  int num_dual_infeasibilities;
  double sum_dual_infeasibilities;
  double col_aq_density;
  double row_ep_density;
  double row_ap_density;
  double row_dse_density;
  double col_bfrt_density;
  double primal_col_density;
  double dual_col_density;
  int num_costly_dse_iteration;
  double costly_dse_measure;
};

struct PrimalReport {
  int solve_phase;
  int edge_weight_mode;
  int num_devex_iterations;
  int row_out;
  int variable_out;
  int variable_in;
  int rebuild_reason;
  int reason_for_rebuild;
  double theta_primal;
  double theta_dual;
  double alpha_col;
  double alpha_row;
  double numerical_trouble;
};
}  // namespace highs_rs

static_assert(sizeof(HighsModelStatus) == sizeof(int),
              "model_status_ is shared with Rust as an i32");
static_assert(sizeof(SimplexAlgorithm) == sizeof(int),
              "exit_algorithm_ is shared with Rust as an i32");
static_assert(sizeof(HighsRandom) == sizeof(uint64_t),
              "random_ is shared with Rust as its 64-bit state");
static_assert(sizeof(std::pair<HighsInt, double>) == 16,
              "workData is shared with Rust as #[repr(C)] (i32, f64)");

// The context of the C++ that the Rust solve calls
struct HEkk::RustHost {
  HEkk* ekk;
  const HighsLogOptions* factor_log_options;

  static HEkk& e(void* ctx) { return *static_cast<RustHost*>(ctx)->ekk; }

  static void log(void* ctx, int channel, int type, const char* msg) {
    const RustHost& host = *static_cast<RustHost*>(ctx);
    const HighsLogOptions& log_options = host.ekk->options_->log_options;
    switch (channel) {
      case 0:
        highsLogUser(log_options, static_cast<HighsLogType>(type), "%s", msg);
        break;
      case 1:
        highsLogDev(log_options, static_cast<HighsLogType>(type), "%s", msg);
        break;
      case 2:
        highsLogDev(*host.factor_log_options, static_cast<HighsLogType>(type),
                    "%s", msg);
        break;
      default:
        printf("%s", msg);
        fflush(stdout);
    }
  }

  static double timerRead(void* ctx) { return e(ctx).timer_->read(); }

  // The user interrupt part of HEkk::bailout
  static bool interrupt(void* ctx) {
    HEkk& ekk = e(ctx);
    HighsCallback& callback = *ekk.callback_;
    callback.clearHighsCallbackOutput();
    callback.data_out.simplex_iteration_count = ekk.iteration_count_;
    if (callback.callbackAction(kCallbackSimplexInterrupt,
                                "Simplex interrupt")) {
      highsLogDev(ekk.options_->log_options, HighsLogType::kInfo,
                  "User interrupt\n");
      ekk.solve_bailout_ = true;
      ekk.model_status_ = HighsModelStatus::kInterrupt;
    }
    return ekk.solve_bailout_;
  }

  static void userInvertReport(void* ctx) {
    const bool force = true;
    e(ctx).analysis_.userInvertReport(force);
  }

  // HEkkDual::iterationAnalysisData with the data of the Rust driver, and
  // its iteration (kind 1) or rebuild (kind 2) report
  static void dualReport(void* ctx, int kind, const highs_rs::AnalysisData* s,
                         int reason) {
    HEkk& ekk = e(ctx);
    HighsSimplexAnalysis& analysis = ekk.analysis_;
    const highs_rs::DualState& d = s->state;
    const double cost_scale_factor = pow(2.0, -ekk.options_->cost_scale_factor);
    const HighsSimplexInfo& info = ekk.info_;
    analysis.simplex_strategy = info.simplex_strategy;
    analysis.edge_weight_mode = static_cast<EdgeWeightMode>(d.edge_weight_mode);
    analysis.solve_phase = d.solve_phase;
    analysis.simplex_iteration_count = s->iteration_count;
    analysis.devex_iteration_count = d.num_devex_iterations;
    analysis.pivotal_row_index = d.row_out;
    analysis.leaving_variable = d.variable_out;
    analysis.entering_variable = d.variable_in;
    analysis.rebuild_reason = d.rebuild_reason;
    analysis.reduced_rhs_value = 0;
    analysis.reduced_cost_value = 0;
    analysis.edge_weight = 0;
    analysis.primal_delta = d.delta_primal;
    analysis.primal_step = d.theta_primal;
    analysis.dual_step = d.theta_dual * cost_scale_factor;
    analysis.pivot_value_from_column = d.alpha_col;
    analysis.pivot_value_from_row = d.alpha_row;
    analysis.factor_pivot_threshold = s->factor_pivot_threshold;
    analysis.numerical_trouble = d.numerical_trouble;
    analysis.edge_weight_error = s->edge_weight_error;
    analysis.objective_value = s->updated_dual_objective_value;
    if (d.solve_phase == kSolvePhase2)
      analysis.objective_value *= (HighsInt)ekk.lp_.sense_;
    analysis.num_primal_infeasibility = s->num_primal_infeasibilities;
    analysis.sum_primal_infeasibility = s->sum_primal_infeasibilities;
    analysis.num_dual_infeasibility = s->num_dual_infeasibilities;
    analysis.sum_dual_infeasibility = s->sum_dual_infeasibilities;
    analysis.col_aq_density = s->col_aq_density;
    analysis.row_ep_density = s->row_ep_density;
    analysis.row_ap_density = s->row_ap_density;
    analysis.row_DSE_density = s->row_dse_density;
    analysis.col_basic_feasibility_change_density =
        info.col_basic_feasibility_change_density;
    analysis.row_basic_feasibility_change_density =
        info.row_basic_feasibility_change_density;
    analysis.col_BFRT_density = s->col_bfrt_density;
    analysis.primal_col_density = s->primal_col_density;
    analysis.dual_col_density = s->dual_col_density;
    analysis.num_costly_DSE_iteration = s->num_costly_dse_iteration;
    analysis.costly_DSE_measure = s->costly_dse_measure;
    if (kind == 1) {
      analysis.iterationReport();
    } else if (kind == 2) {
      analysis.rebuild_reason = reason;
      analysis.rebuild_reason_string = ekk.rebuildReason(reason);
      if (ekk.options_->output_flag) analysis.invertReport();
    }
  }

  // HEkkPrimal::iterationAnalysisData with the data of the Rust solver,
  // and its iteration (kind 0) or rebuild (kind 1) report
  static void primalReport(void* ctx, int kind,
                           const highs_rs::PrimalReport* r) {
    HEkk& ekk = e(ctx);
    HighsSimplexAnalysis& analysis = ekk.analysis_;
    const HighsSimplexInfo& info = ekk.info_;
    analysis.simplex_strategy = kSimplexStrategyPrimal;
    analysis.edge_weight_mode = static_cast<EdgeWeightMode>(r->edge_weight_mode);
    analysis.solve_phase = r->solve_phase;
    analysis.simplex_iteration_count = ekk.iteration_count_;
    analysis.devex_iteration_count = r->num_devex_iterations;
    analysis.pivotal_row_index = r->row_out;
    analysis.leaving_variable = r->variable_out;
    analysis.entering_variable = r->variable_in;
    analysis.rebuild_reason = r->rebuild_reason;
    analysis.reduced_rhs_value = 0;
    analysis.reduced_cost_value = 0;
    analysis.edge_weight = 0;
    analysis.primal_delta = 0;
    analysis.primal_step = r->theta_primal;
    analysis.dual_step = r->theta_dual;
    analysis.pivot_value_from_column = r->alpha_col;
    analysis.pivot_value_from_row = r->alpha_row;
    analysis.numerical_trouble = r->numerical_trouble;
    analysis.edge_weight_error = ekk.edge_weight_error_;
    analysis.objective_value = info.updated_primal_objective_value;
    analysis.num_primal_infeasibility = info.num_primal_infeasibilities;
    analysis.num_dual_infeasibility = info.num_dual_infeasibilities;
    analysis.sum_primal_infeasibility = info.sum_primal_infeasibilities;
    analysis.sum_dual_infeasibility = info.sum_dual_infeasibilities;
    if ((analysis.edge_weight_mode == EdgeWeightMode::kDevex) &&
        (r->num_devex_iterations == 0))
      analysis.num_devex_framework++;
    analysis.col_aq_density = info.col_aq_density;
    analysis.row_ep_density = info.row_ep_density;
    analysis.row_ap_density = info.row_ap_density;
    analysis.row_DSE_density = info.row_DSE_density;
    analysis.col_steepest_edge_density = info.col_steepest_edge_density;
    analysis.col_basic_feasibility_change_density =
        info.col_basic_feasibility_change_density;
    analysis.row_basic_feasibility_change_density =
        info.row_basic_feasibility_change_density;
    analysis.col_BFRT_density = info.col_BFRT_density;
    analysis.primal_col_density = info.primal_col_density;
    analysis.dual_col_density = info.dual_col_density;
    analysis.num_costly_DSE_iteration = info.num_costly_DSE_iteration;
    analysis.costly_DSE_measure = info.costly_DSE_measure;
    if (kind == 0) {
      analysis.iterationReport();
    } else if (kind == 1) {
      analysis.rebuild_reason = r->reason_for_rebuild;
      analysis.rebuild_reason_string =
          ekk.rebuildReason(r->reason_for_rebuild);
      if (ekk.options_->output_flag) analysis.invertReport();
    }
  }

  static void chuzcFail(void* ctx, int kind, int work_count,
                        const std::pair<HighsInt, double>* work_data,
                        double select_theta, double remain_theta) {
    HEkk& ekk = e(ctx);
    const std::vector<std::pair<HighsInt, double>> data(work_data,
                                                        work_data + work_count);
    const HighsInt num_tot = ekk.lp_.num_col_ + ekk.lp_.num_row_;
    if (kind == 1) {
      debugDualChuzcFailQuad0(*ekk.options_, work_count, data, num_tot,
                              ekk.info_.workDual_.data(), select_theta,
                              remain_theta, true);
    } else {
      debugDualChuzcFailQuad1(*ekk.options_, work_count, data, num_tot,
                              ekk.info_.workDual_.data(), select_theta, true);
    }
  }

  // The handling of a rank deficient initial basis in
  // initialiseSimplexLpBasisAndFactor
  static void initialRankDeficiency(void* ctx, highs_rs::Slice<double>* saved) {
    HEkk& ekk = e(ctx);
    ekk.simplex_nla_.factor_.pullRustBuildInfo();
    highsLogDev(
        ekk.options_->log_options, HighsLogType::kInfo,
        "HEkk::initialiseSimplexLpBasisAndFactor (%s) Rank_deficiency %d: Id "
        "= "
        "%d; UpdateCount = %d\n",
        ekk.basis_.debug_origin_name.c_str(),
        (int)ekk.simplex_nla_.factor_.rank_deficiency, (int)ekk.basis_.debug_id,
        (int)ekk.basis_.debug_update_count);
    ekk.handleRankDeficiency();
    ekk.updateStatus(LpAction::kNewBasis);
    ekk.setNonbasicMove();
    ekk.status_.has_basis = true;
    ekk.status_.has_invert = true;
    ekk.status_.has_fresh_invert = true;
    *saved = highs_rs::slice(ekk.saved_dual_edge_weight_);
  }

  static void debugCheckInvert(void* ctx) {
    HEkk& ekk = e(ctx);
    ekk.simplex_nla_.factor_.pullRustBuildInfo();
    ekk.debugNlaCheckInvert("HEkk::computeFactor - original",
                            kHighsDebugLevelCostly);
  }
};

// Every strategy (SIP and PAMI run as the serial dual simplex), and no
// simplex analysis or debugging, which Crestline leaves out
bool HEkk::rustSolveEligible() const {
  return !simplex_nla_.update_.valid_;
}

highs_rs::Hekk HEkk::rustHekk(void* host_ctx, const bool draw_random_vectors) {
  using highs_rs::slice;
  highs_rs::Hekk x;
  x.ekk = rustView();
  x.host.ctx = host_ctx;
  x.host.log = RustHost::log;
  x.host.timer_read = RustHost::timerRead;
  x.host.interrupt = RustHost::interrupt;
  x.host.user_invert_report = RustHost::userInvertReport;
  x.host.dual_report = RustHost::dualReport;
  x.host.primal_report = RustHost::primalReport;
  x.host.chuzc_fail = RustHost::chuzcFail;
  x.host.initial_rank_deficiency = RustHost::initialRankDeficiency;
  x.host.debug_check_invert = RustHost::debugCheckInvert;
  HighsSimplexInfo& info = info_;
  HighsSimplexStatus& status = status_;
  const HighsOptions& options = *options_;
  x.iteration_count = &iteration_count_;
  x.model_status = reinterpret_cast<int*>(&model_status_);
  x.solve_bailout = &solve_bailout_;
  x.called_return_from_solve = &called_return_from_solve_;
  x.exit_algorithm = reinterpret_cast<int*>(&exit_algorithm_);
  x.return_primal_solution_status = &return_primal_solution_status_;
  x.return_dual_solution_status = &return_dual_solution_status_;
  x.dual_values_valid = &dual_values_valid_;
  x.dual_values_scaled = &dual_values_scaled_;
  x.dual_values_basis_hash = &dual_values_basis_hash_;
  x.dual_values_cost_hash = &dual_values_cost_hash_;
  x.fresh_unperturbed_dual = &fresh_unperturbed_dual_;
  x.fresh_dual = &fresh_dual_;
  x.fresh_primal = &fresh_primal_;
  x.edge_weight_error = &edge_weight_error_;
  x.dual_simplex_cleanup_level = &dual_simplex_cleanup_level_;
  x.dual_simplex_phase1_cleanup_level = &dual_simplex_phase1_cleanup_level_;
  x.previous_iteration_cycling_detected = &previous_iteration_cycling_detected;
  x.random = reinterpret_cast<uint64_t*>(&random_);
  x.basis_records = basis_records_.p;
  x.nla_build_synthetic_tick = &simplex_nla_.build_synthetic_tick_;
  x.num_invert = &simplex_stats_.num_invert;
  x.debug_solve_call_num = debug_solve_call_num_;
  x.ar_matrix_is_scaled = &ar_matrix_is_scaled_;
  x.random_vectors_drawn = &random_vectors_drawn_for_solve_;
  x.draw_random_vectors = draw_random_vectors;
  x.lp_is_scaled = lp_.is_scaled_;
  x.lp_has_scaling = lp_.scale_.has_scaling;
  x.lp_col_scale = slice(lp_.scale_.col);
  x.lp_row_scale = slice(lp_.scale_.row);
  x.model_name = {const_cast<char*>(lp_.model_name_.data()),
                  (int)lp_.model_name_.size()};
  x.saved_dual_edge_weight = slice(saved_dual_edge_weight_);
  x.saved_dual_edge_weight_taken = false;
  x.dual_ray_index = &dual_ray_record_.index;
  x.dual_ray_sign = &dual_ray_record_.sign;
  x.primal_ray_index = &primal_ray_record_.index;
  x.primal_ray_sign = &primal_ray_record_.sign;
  x.ray_value_clear = 0;
  x.basis_debug_id = &basis_.debug_id;
  x.basis_debug_update_count = &basis_.debug_update_count;
  x.has_invert = &status.has_invert;
  x.has_fresh_invert = &status.has_fresh_invert;
  x.has_fresh_rebuild = &status.has_fresh_rebuild;
  x.has_dual_objective_value = &status.has_dual_objective_value;
  x.has_primal_objective_value = &status.has_primal_objective_value;
  x.has_dual_steepest_edge_weights = &status.has_dual_steepest_edge_weights;
  x.has_ar_matrix = &status.has_ar_matrix;
  SimplexBasis& bt = info.backtracking_basis_;
  x.valid_backtracking_basis = &info.valid_backtracking_basis_;
  x.bt_basic_index = slice(bt.basicIndex_);
  x.bt_nonbasic_flag = slice(bt.nonbasicFlag_);
  x.bt_nonbasic_move = slice(bt.nonbasicMove_);
  x.bt_hash = &bt.hash;
  x.bt_debug_id = &bt.debug_id;
  x.bt_debug_update_count = &bt.debug_update_count;
  x.bt_costs_shifted = &info.backtracking_basis_costs_shifted_;
  x.bt_costs_perturbed = &info.backtracking_basis_costs_perturbed_;
  x.bt_bounds_shifted = &info.backtracking_basis_bounds_shifted_;
  x.bt_bounds_perturbed = &info.backtracking_basis_bounds_perturbed_;
  x.bt_work_shift = slice(info.backtracking_basis_workShift_);
  x.bt_edge_weight = slice(info.backtracking_basis_edge_weight_);
  x.devex_index = slice(info.devex_index_);
  x.num_tot_permutation = slice(info.numTotPermutation_);
  // C++ only permutes the columns if there are some
  x.num_col_permutation = {info.numColPermutation_.data(),
                           lp_.num_col_ ? (int)lp_.num_col_ : 0};
  x.dual_phase1_iteration_count = &info.dual_phase1_iteration_count;
  x.dual_phase2_iteration_count = &info.dual_phase2_iteration_count;
  x.allow_cost_shifting = &info.allow_cost_shifting;
  x.allow_cost_perturbation = &info.allow_cost_perturbation;
  x.store_squared_primal_infeasibility =
      &info.store_squared_primal_infeasibility;
  x.factor_pivot_threshold = &info.factor_pivot_threshold;
  x.col_bfrt_density = &info.col_BFRT_density;
  x.costly_dse_measure = &info.costly_DSE_measure;
  x.costly_dse_frequency = &info.costly_DSE_frequency;
  x.num_costly_dse_iteration = &info.num_costly_DSE_iteration;
  x.average_log_low_dse_weight_error = &info.average_log_low_DSE_weight_error;
  x.average_log_high_dse_weight_error = &info.average_log_high_DSE_weight_error;
  x.simplex_strategy = &info.simplex_strategy;
  x.min_concurrency = &info.min_concurrency;
  x.max_concurrency = &info.max_concurrency;
  x.num_concurrency = &info.num_concurrency;
  x.iteration_count0 = &info.iteration_count0;
  x.dual_phase1_iteration_count0 = &info.dual_phase1_iteration_count0;
  x.dual_phase2_iteration_count0 = &info.dual_phase2_iteration_count0;
  x.primal_phase1_iteration_count0 = &info.primal_phase1_iteration_count0;
  x.primal_phase2_iteration_count0 = &info.primal_phase2_iteration_count0;
  x.primal_bound_swap0 = &info.primal_bound_swap0;
  x.control_iteration_count0 = info.control_iteration_count0;
  x.allow_dual_steepest_edge_to_devex_switch =
      info.allow_dual_steepest_edge_to_devex_switch;
  x.dual_steepest_edge_weight_log_error_threshold =
      info.dual_steepest_edge_weight_log_error_threshold;
  x.dual_edge_weight_strategy = info.dual_edge_weight_strategy;
  x.run_quiet = info.run_quiet;
  HFactor& factor = simplex_nla_.factor_;
  x.hfactor_pivot_threshold = &factor.pivot_threshold;
  x.hfactor_pivot_tolerance = factor.pivot_tolerance;
  x.hfactor_time_limit = factor.time_limit_;
  x.objective_bound = options.objective_bound;
  x.time_limit = options.time_limit;
  x.simplex_iteration_limit = options.simplex_iteration_limit;
  x.simplex_update_limit = options.simplex_update_limit;
  x.max_dual_simplex_cleanup_level = options.max_dual_simplex_cleanup_level;
  x.max_dual_simplex_phase1_cleanup_level =
      options.max_dual_simplex_phase1_cleanup_level;
  x.dual_simplex_pivot_growth_tolerance =
      options.dual_simplex_pivot_growth_tolerance;
  x.simplex_dse_exact_init_max_rows = options.simplex_dse_exact_init_max_rows;
  x.small_matrix_value = options.small_matrix_value;
  x.dual_steepest_edge_weight_error_tolerance =
      options.dual_steepest_edge_weight_error_tolerance;
  x.no_unnecessary_rebuild_refactor = options.no_unnecessary_rebuild_refactor;
  x.rebuild_refactor_solution_error_tolerance =
      options.rebuild_refactor_solution_error_tolerance;
  x.option_simplex_strategy = options.simplex_strategy;
  x.simplex_min_concurrency = options.simplex_min_concurrency;
  x.simplex_max_concurrency = options.simplex_max_concurrency;
  x.allow_unbounded_or_infeasible = options.allow_unbounded_or_infeasible;
  x.less_infeasible_dse_check = options.less_infeasible_DSE_check;
  x.less_infeasible_dse_choose_row = options.less_infeasible_DSE_choose_row;
  x.num_threads = highs::parallel::num_threads();
  const HighsLogOptions& log_options = options.log_options;
  x.output_flag = options.output_flag;
  x.log_dev_level = options.log_dev_level;
  x.dev_level = *log_options.output_flag ? *log_options.log_dev_level : 0;
  x.factor_dev_level =
      *factor.log_options.output_flag ? *factor.log_options.log_dev_level : 0;
  x.dev_log = x.dev_level != 0;
  x.iteration_report =
      *log_options.log_dev_level >= (HighsInt)kIterationReportLogType;
  x.interrupt_callback =
      callback_->user_callback && callback_->active[kCallbackSimplexInterrupt];
  return x;
}

HighsStatus HEkk::solveRust(const bool force_phase2) {
  // The C++ part of initialiseSimplexLpBasisAndFactor, before INVERT:
  // the basis and the simplex NLA
  if (!status_.has_basis) setBasis();
  HighsSparseMatrix* local_scaled_a_matrix = getScaledAMatrixPointer();
  if (status_.has_nla) {
    assert(lpFactorRowCompatible());
    simplex_nla_.setPointers(&lp_, local_scaled_a_matrix,
                             basis_.basicIndex_.data(), options_, timer_,
                             &analysis_);
  } else {
    assert(info_.factor_pivot_threshold >= options_->factor_pivot_threshold);
    simplex_nla_.setup(&lp_, basis_.basicIndex_.data(), options_, timer_,
                       &analysis_, local_scaled_a_matrix,
                       info_.factor_pivot_threshold);
    status_.has_nla = true;
  }
  updateSimplexOptions();
  // Size the vectors that the solve may otherwise resize, so that Rust
  // can work on views of them
  const HighsInt num_col = lp_.num_col_;
  const HighsInt num_row = lp_.num_row_;
  const HighsInt num_tot = num_col + num_row;
  // The random vectors only depend on the LP dimensions: if asked to (as
  // for the LP relaxation of a MIP), or for large LPs, keep them over
  // re-solves
  const bool draw_random_vectors =
      (!options_->simplex_keep_random_vectors &&
       num_row <= options_->simplex_dse_exact_init_max_rows) ||
      !random_vectors_drawn_for_solve_ ||
      static_cast<HighsInt>(info_.numTotRandomValue_.size()) != num_tot ||
      static_cast<HighsInt>(info_.numColPermutation_.size()) != num_col;
  if (draw_random_vectors && num_tot) {
    if (num_col) info_.numColPermutation_.resize(num_col);
    info_.numTotPermutation_.resize(num_tot);
    info_.numTotRandomValue_.resize(num_tot);
  }
  {
    HighsSparseMatrix& ar = ar_matrix_;
    const HighsInt num_nz = lp_.a_matrix_.numNz();
    if (!(status_.has_ar_matrix && ar_matrix_is_scaled_ == lp_.is_scaled_)) {
      ar.format_ = MatrixFormat::kRowwisePartitioned;
      ar.num_col_ = num_col;
      ar.num_row_ = num_row;
    }
    ar.start_.resize(num_row + 1);
    ar.p_end_.resize(num_row);
    ar.index_.resize(num_nz);
    ar.value_.resize(num_nz);
  }
  allocateWorkAndBaseArrays();
  basis_.nonbasicMove_.resize(num_tot);
  if (!status_.has_dual_steepest_edge_weights) {
    dual_edge_weight_.resize(num_row);
    scattered_dual_edge_weight_.resize(num_tot);
  }
  info_.backtracking_basis_edge_weight_.resize(num_tot);
  {
    SimplexBasis& bt = info_.backtracking_basis_;
    bt.basicIndex_.resize(num_row);
    bt.nonbasicFlag_.resize(num_tot);
    bt.nonbasicMove_.resize(num_tot);
    bt.debug_origin_name = basis_.debug_origin_name;
    info_.backtracking_basis_workShift_.resize(num_tot);
  }
  RustHost host{this, &simplex_nla_.factor_.log_options};
  highs_rs::Hekk x = rustHekk(&host, draw_random_vectors);
  const HighsStatus return_status =
      (HighsStatus)highs_rs_ekk_solve(&x, force_phase2);
  // Take what the Rust left for the C++-owned vectors
  simplex_nla_.factor_.pullRustBuildInfo();
  void* records = basis_records_.p;
  {
    bool use;
    const HighsInt *pivot_row, *pivot_var;
    const int8_t *pivot_type, *nonbasic_move;
    int num_pivot, num_move;
    double build_synthetic_tick;
    if (highs_rs_ekk_hot_start(records, &use, &pivot_row, &pivot_var,
                               &pivot_type, &num_pivot, &build_synthetic_tick,
                               &nonbasic_move, &num_move)) {
      RefactorInfo& refactor_info = hot_start_.refactor_info;
      refactor_info.use = use;
      refactor_info.pivot_row.assign(pivot_row, pivot_row + num_pivot);
      refactor_info.pivot_var.assign(pivot_var, pivot_var + num_pivot);
      refactor_info.pivot_type.assign(pivot_type, pivot_type + num_pivot);
      refactor_info.build_synthetic_tick = build_synthetic_tick;
      hot_start_.nonbasicMove.assign(nonbasic_move, nonbasic_move + num_move);
      hot_start_.valid = true;
    }
    const double* values;
    int n;
    if (highs_rs_ekk_primal_phase1_dual(records, &values, &n))
      primal_phase1_dual_.assign(values, values + n);
    highs_rs_ekk_clear_out(records);
  }
  if (x.ray_value_clear & 1) dual_ray_record_.value.clear();
  if (x.ray_value_clear & 2) primal_ray_record_.value.clear();
  if (x.saved_dual_edge_weight_taken) saved_dual_edge_weight_.clear();
  return returnFromEkkSolve(return_status);
}

#endif  // HIGHS_RUST
