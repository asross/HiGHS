/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkRust.cpp
 * @brief HEkk as a shell of the Rust-owned simplex engine
 * (rust/src/simplex/lp_solver.rs): the LP moves, dualization, the views
 * passed to Rust and the C++ the Rust calls (logging, the clock, the user
 * interrupt, the CHUZC failure reports)
 */
#include "simplex/HEkk.h"

#ifdef HIGHS_RUST

#include <cmath>
#include <cstddef>
#include <cstdio>

#include "lp_data/HighsLpSolverObject.h"
#include "simplex/HApp.h"
#include "parallel/HighsParallel.h"
#include "simplex/HSimplex.h"
#include "simplex/HSimplexDebug.h"

static_assert(sizeof(HighsInt) == 4, "the Rust simplex takes 32-bit ints");
static_assert(sizeof(bool) == 1, "Rust bools are bytes");
static_assert(sizeof(HighsSimplexStatus) == 13,
              "HighsSimplexStatus is mirrored by SimplexStatus in ekk.rs");
static_assert(sizeof(HighsModelStatus) == sizeof(int),
              "model_status_ is shared with Rust as an i32");
static_assert(sizeof(SimplexAlgorithm) == sizeof(int),
              "exit_algorithm_ is shared with Rust as an i32");
static_assert(sizeof(HighsBasisStatus) == 1, "basis statuses are bytes");
static_assert(sizeof(std::pair<HighsInt, double>) == 16,
              "workData is shared with Rust as #[repr(C)] (i32, f64)");
static_assert(sizeof(highs_rs::SimplexReport) == 128,
              "SimplexReport is #[repr(C)] in rust/src/simplex/report.rs");
static_assert(sizeof(highs_rs::HEkkInfo) == 88, "SharedInfo in lp_solver.rs");
static_assert(sizeof(highs_rs::HEkkShared) == 144,
              "EkkShared in lp_solver.rs");
static_assert(offsetof(highs_rs::HEkkShared, status) == 124,
              "EkkShared in lp_solver.rs");

// The context of the C++ that the Rust simplex calls
struct HEkk::RustHost {
  HEkk* ekk;

  // The host functions' context is the HEkk
  static HEkk& e(void* ctx) { return *static_cast<HEkk*>(ctx); }

  static void log(void* ctx, int channel, int type, const char* msg) {
    const HEkk& ekk = e(ctx);
    const HighsLogOptions& log_options = ekk.options_->log_options;
    switch (channel) {
      case 0:
        highsLogUser(log_options, static_cast<HighsLogType>(type), "%s", msg);
        break;
      case 1:
        highsLogDev(log_options, static_cast<HighsLogType>(type), "%s", msg);
        break;
      case 2:
        highsLogDev(ekk.factor_log_options_, static_cast<HighsLogType>(type),
                    "%s", msg);
        break;
      default:
        printf("%s", msg);
        fflush(stdout);
    }
  }

  static double timerRead(void* ctx) { return e(ctx).timer_->read(); }

  // The user interrupt part of HEkk::bailout: whether the user interrupts
  static bool interrupt(void* ctx) {
    HEkk& ekk = e(ctx);
    HighsCallback& callback = *ekk.callback_;
    callback.clearHighsCallbackOutput();
    callback.data_out.simplex_iteration_count = ekk.iteration_count_;
    if (callback.callbackAction(kCallbackSimplexInterrupt,
                                "Simplex interrupt")) {
      highsLogDev(ekk.options_->log_options, HighsLogType::kInfo,
                  "User interrupt\n");
      return true;
    }
    return false;
  }

  static void chuzcFail(void* ctx, int kind, int work_count,
                        const std::pair<HighsInt, double>* work_data,
                        double select_theta, double remain_theta) {
    HEkk& ekk = e(ctx);
    const std::vector<std::pair<HighsInt, double>> data(work_data,
                                                        work_data + work_count);
    const HighsInt num_tot = ekk.lpNumCol() + ekk.lpNumRow();
    HighsInt n;
    const double* work_dual = highs_rs_lps_work_dual(ekk.rs_, &n);
    if (kind == 1) {
      debugDualChuzcFailQuad0(*ekk.options_, work_count, data, num_tot,
                              work_dual, select_theta, remain_theta, true);
    } else {
      debugDualChuzcFailQuad1(*ekk.options_, work_count, data, num_tot,
                              work_dual, select_theta, true);
    }
  }
};

HEkk::HEkk()
    : callback_(nullptr),
      options_(nullptr),
      timer_(nullptr),
      lp_name_(""),
      rs_(highs_rs_lps_new()),
      sh_(*highs_rs_lps_shared(rs_)),
      status_(sh_.status),
      info_(sh_.info),
      model_status_(sh_.model_status),
      iteration_count_(sh_.iteration_count),
      exit_algorithm_(sh_.exit_algorithm),
      dual_values_valid_(sh_.dual_values_valid),
      debug_solve_call_num_(sh_.debug_solve_call_num),
      debug_initial_build_synthetic_tick_(
          sh_.debug_initial_build_synthetic_tick),
      nla_lp_(nullptr),
      nla_scale_(nullptr) {
  factor_log_options_.output_flag = &factor_log_data_.output_flag;
  factor_log_options_.log_to_console = &factor_log_data_.log_to_console;
  factor_log_options_.log_dev_level = &factor_log_data_.log_dev_level;
}

HEkk::~HEkk() { highs_rs_lps_free(rs_); }

highs_rs::LpsEnv HEkk::rsEnv(RustHost& host) const {
  // The LP and its name are the engine's (filled by Rust)
  highs_rs::LpsEnv env{};
  const bool nla = sh_.nla_lp_set && nla_lp_ != nullptr;
  env.nla_num_col = nla ? nla_lp_->num_col_ : 0;
  env.nla_num_row = nla ? nla_lp_->num_row_ : 0;
  env.nla_has_scale = nla && nla_scale_ != nullptr;
  env.nla_col_scale = env.nla_has_scale ? rsMut(nla_scale_->col)
                                        : RsMut<double>{nullptr, 0};
  env.nla_row_scale = env.nla_has_scale ? rsMut(nla_scale_->row)
                                        : RsMut<double>{nullptr, 0};
  highs_rs::LpsOptions& o = env.opt;
  o = highs_rs::LpsOptions{};
  if (options_) {
    const HighsOptions& options = *options_;
    o.primal_feasibility_tolerance = options.primal_feasibility_tolerance;
    o.dual_feasibility_tolerance = options.dual_feasibility_tolerance;
    o.time_limit = options.time_limit;
    o.objective_bound = options.objective_bound;
    o.dual_simplex_pivot_growth_tolerance =
        options.dual_simplex_pivot_growth_tolerance;
    o.small_matrix_value = options.small_matrix_value;
    o.dual_steepest_edge_weight_error_tolerance =
        options.dual_steepest_edge_weight_error_tolerance;
    o.rebuild_refactor_solution_error_tolerance =
        options.rebuild_refactor_solution_error_tolerance;
    o.factor_pivot_tolerance = options.factor_pivot_tolerance;
    o.factor_pivot_threshold = options.factor_pivot_threshold;
    o.dual_simplex_cost_perturbation_multiplier =
        options.dual_simplex_cost_perturbation_multiplier;
    o.primal_simplex_bound_perturbation_multiplier =
        options.primal_simplex_bound_perturbation_multiplier;
    o.dual_steepest_edge_weight_log_error_threshold =
        options.dual_steepest_edge_weight_log_error_threshold;
    o.cost_scale_factor = options.cost_scale_factor;
    const HighsLogOptions& log_options = options.log_options;
    o.log_dev_level = *log_options.log_dev_level;
    o.dev_level = *log_options.output_flag ? *log_options.log_dev_level : 0;
    o.simplex_primal_edge_weight_strategy =
        options.simplex_primal_edge_weight_strategy;
    o.simplex_iteration_limit = options.simplex_iteration_limit;
    o.simplex_update_limit = options.simplex_update_limit;
    o.max_dual_simplex_cleanup_level = options.max_dual_simplex_cleanup_level;
    o.max_dual_simplex_phase1_cleanup_level =
        options.max_dual_simplex_phase1_cleanup_level;
    o.simplex_dse_exact_init_max_rows = options.simplex_dse_exact_init_max_rows;
    o.simplex_strategy = options.simplex_strategy;
    o.simplex_min_concurrency = options.simplex_min_concurrency;
    o.simplex_max_concurrency = options.simplex_max_concurrency;
    o.simplex_dual_edge_weight_strategy =
        options.simplex_dual_edge_weight_strategy;
    o.simplex_price_strategy = options.simplex_price_strategy;
    o.random_seed = options.random_seed;
    o.output_flag = options.output_flag;
    o.no_unnecessary_rebuild_refactor = options.no_unnecessary_rebuild_refactor;
    o.allow_unbounded_or_infeasible = options.allow_unbounded_or_infeasible;
    o.less_infeasible_dse_check = options.less_infeasible_DSE_check;
    o.less_infeasible_dse_choose_row = options.less_infeasible_DSE_choose_row;
    o.simplex_keep_random_vectors = options.simplex_keep_random_vectors;
  }
  (void)host;
  env.host.ctx = const_cast<HEkk*>(this);
  env.host.log = RustHost::log;
  env.host.timer_read = RustHost::timerRead;
  env.host.interrupt = RustHost::interrupt;
  env.host.chuzc_fail = RustHost::chuzcFail;
  env.report = const_cast<highs_rs::SimplexReport*>(&analysis_.rs_report_);
  env.num_invert = const_cast<HighsInt*>(&simplex_stats_.num_invert);
  env.num_threads = highs::parallel::num_threads();
  env.interrupt_callback = callback_ && callback_->user_callback &&
                           callback_->active[kCallbackSimplexInterrupt];
  return env;
}

// The factor's log options are a copy of the options' log flags made when
// the simplex NLA is set up (HFactor::setupGeneral), without callbacks
void HEkk::snapshotFactorLog() {
  if (status_.has_nla || !options_) return;
  const HighsLogOptions& log_options = options_->log_options;
  factor_log_data_.output_flag = *log_options.output_flag;
  factor_log_data_.log_to_console = *log_options.log_to_console;
  factor_log_data_.log_dev_level = *log_options.log_dev_level;
  factor_log_options_.log_stream = log_options.log_stream;
}

// What clear() clears on the C++ side
void HEkk::clearCpp() {
  clearEkkLp();
  callback_ = nullptr;
  options_ = nullptr;
  timer_ = nullptr;
  nla_lp_ = nullptr;
  nla_scale_ = nullptr;
  primal_phase1_dual_.clear();
}

void HEkk::clear() {
  clearCpp();
  highs_rs_lps_clear(rs_);
}

void HEkk::clearEkkLp() {
  highs_rs_lps_clear_lp(rs_);
  lp_name_ = "";
}

HighsInt HEkk::lpNumCol() const {
  RsLp v;
  highs_rs_lps_lp_view(rs_, &v);
  return v.num_col;
}

HighsInt HEkk::lpNumRow() const {
  RsLp v;
  highs_rs_lps_lp_view(rs_, &v);
  return v.num_row;
}

void HEkk::clearRayRecords() { highs_rs_lps_clear_ray_records(rs_); }

void HEkk::invalidate() {
  assert(!status_.is_dualized);
  assert(!status_.is_permuted);
  highs_rs_lps_invalidate(rs_);
  simplex_stats_.initialise();
}

void HEkk::updateStatus(LpAction action) {
  assert(!status_.is_dualized);
  assert(!status_.is_permuted);
  assert(action != LpAction::kDelRowsBasisOk);
  if (highs_rs_lps_update_status(rs_, int(action))) clearCpp();
}

void HEkk::setNlaLp(const HighsLp& lp) {
  nla_lp_ = &lp;
  nla_scale_ = lp.scale_.has_scaling && !lp.is_scaled_ ? &lp.scale_ : nullptr;
  highs_rs_lps_set_nla_rust(rs_, false);
}

void HEkk::setNlaEngineLp() {
  nla_lp_ = nullptr;
  nla_scale_ = nullptr;
  highs_rs_lps_set_nla_rust(rs_, true);
}

void HEkk::setNlaPointersForLpAndScale(const HighsLp& lp) {
  assert(status_.has_nla);
  setNlaLp(lp);
}

void HEkk::btran(HVector& rhs, const double expected_density) {
  assert(status_.has_nla);
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  highs_rs::HVecCall v(rhs);
  highs_rs_lps_nla_solve(rs_, &env, v.get(), expected_density, true);
}

void HEkk::ftran(HVector& rhs, const double expected_density) {
  assert(status_.has_nla);
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  highs_rs::HVecCall v(rhs);
  highs_rs_lps_nla_solve(rs_, &env, v.get(), expected_density, false);
}

void HEkk::moveLp(HighsLpSolverObject& solver_object) {
  // Copy the incumbent LP to the engine
  const HighsLp& incumbent_lp = solver_object.lp_;
  const RsLp v = rsLp(incumbent_lp);
  highs_rs_lps_import_lp(rs_, &v, incumbent_lp.model_name_.data(),
                         incumbent_lp.model_name_.size());
  movedLp(solver_object);
}

void HEkk::movedLp(HighsLpSolverObject& solver_object) {
  movedLp(solver_object.callback_, solver_object.options_,
          solver_object.timer_);
}

void HEkk::movedLp(HighsCallback& callback, HighsOptions& options,
                   HighsTimer& timer) {
  this->setPointers(&callback, &options, &timer);
  // The row-wise matrix, the scaled space, and initialiseEkk if this has
  // not been done (it clears the simplex NLA)
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  if (highs_rs_lps_move_lp(rs_, &env)) {
    nla_lp_ = nullptr;
    nla_scale_ = nullptr;
  }
}

void HEkk::setPointers(HighsCallback* callback, HighsOptions* options,
                       HighsTimer* timer) {
  this->callback_ = callback;
  this->options_ = options;
  this->timer_ = timer;
  this->analysis_.timer_ = this->timer_;
}

HighsStatus HEkk::solve(const bool force_phase2) {
  // initialiseAnalysis
  RsLp v;
  highs_rs_lps_lp_view(rs_, &v);
  RsMut<char> name;
  highs_rs_lps_model_name(rs_, &name);
  analysis_.setup(lp_name_, v.num_col, v.num_row,
                  std::string(name.ptr, name.len), *options_,
                  iteration_count_);
  // The solve sets up the simplex NLA for this LP
  snapshotFactorLog();
  setNlaEngineLp();
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  const highs_rs::LpsSolveOut out = highs_rs_lps_solve(rs_, &env, force_phase2);
  takeRustOut();
  return returnFromEkkSolve(HighsStatus(out.status), out);
}

void HEkk::takeRustOut() {
  void* records = highs_rs_lps_records(rs_);
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

HighsStatus HEkk::returnFromEkkSolve(const HighsStatus return_status,
                                     const highs_rs::LpsSolveOut& out) {
  // Saved weights not used by this solve are stale for the next one
  highs_rs_lps_return_from_solve(rs_);
  simplex_stats_.valid = true;
  // Since HEkk::iteration_count_ includes iteration on presolved LP,
  // simplex_stats_.iteration_count is initialised to -
  // HEkk::iteration_count_
  simplex_stats_.iteration_count += iteration_count_;
  simplex_stats_.last_invert_num_el = out.invert_num_el;
  simplex_stats_.last_factored_basis_num_el = out.basis_matrix_num_el;
  const highs_rs::SimplexReport& report = analysis_.rs_report_;
  simplex_stats_.col_aq_density = report.col_aq_density;
  simplex_stats_.row_ep_density = report.row_ep_density;
  simplex_stats_.row_ap_density = report.row_ap_density;
  simplex_stats_.row_DSE_density = report.row_DSE_density;
  return return_status;
}

HighsStatus HEkk::setBasis() {
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  highs_rs_lps_set_basis_logical(rs_, &env);
  return HighsStatus::kOk;
}

HighsStatus HEkk::setBasis(const HighsBasis& highs_basis) {
  // An internal call, so the basis is not checked (debugging is left
  // out of this build)
  assert(highs_basis.debug_origin_name != "");
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  highs_rs_lps_set_basis(
      rs_, &env, reinterpret_cast<const uint8_t*>(highs_basis.col_status.data()),
      reinterpret_cast<const uint8_t*>(highs_basis.row_status.data()),
      highs_basis.debug_id, highs_basis.debug_update_count,
      highs_basis.debug_origin_name.data(),
      highs_basis.debug_origin_name.size());
  return HighsStatus::kOk;
}

void HEkk::putIterate() {
  assert(this->status_.has_invert);
  highs_rs_lps_put_iterate(rs_);
}

HighsStatus HEkk::getIterate() {
  return highs_rs_lps_get_iterate(rs_) ? HighsStatus::kOk
                                       : HighsStatus::kError;
}

void HEkk::addCols(const HighsLp& lp,
                   const HighsSparseMatrix& /*scaled_a_matrix*/) {
  if (this->status_.has_nla) setNlaLp(lp);
  this->updateStatus(LpAction::kNewCols);
}

void HEkk::addRows(const HighsLp& lp,
                   const HighsSparseMatrix& scaled_ar_matrix) {
  // Update the number of rows in the simplex LP so that it's
  // consistent with simplex basis information
  highs_rs_lps_set_lp_num_row(rs_, lp.num_row_);
  // New rows come in with basic logicals, which leaves the DSE weights
  // of the existing rows unchanged: kept over the clear of kNewRows
  highs_rs_lps_add_rows(rs_, lp.num_row_ - scaled_ar_matrix.num_row_,
                        lp.num_row_);
  clearCpp();
}

void HEkk::deleteCols(const HighsIndexCollection& /*index_collection*/) {
  this->updateStatus(LpAction::kDelCols);
}

void HEkk::deleteRows(const HighsIndexCollection& index_collection) {
  // Deleting rows with basic logicals leaves the DSE weights of the
  // remaining rows unchanged: kept over the clear of kDelRows
  const RsIndexCollection ic = rsIndexCollection(index_collection);
  highs_rs_lps_delete_rows(rs_, &ic);
  clearCpp();
}

void HEkk::unscaleSimplex(const HighsLp& incumbent_lp) {
  const RsLp lp = rsLp(incumbent_lp);
  highs_rs_lps_unscale_simplex(rs_, &lp);
}

bool HEkk::proofOfPrimalInfeasibility() {
  assert(sh_.dual_ray_index >= 0);
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  return highs_rs_lps_proof_of_primal_infeasibility(rs_, &env);
}

HighsBasis HEkk::getHighsBasis(HighsLp& use_lp) const {
  HighsBasis highs_basis;
  highs_basis.col_status.resize(use_lp.num_col_);
  highs_basis.row_status.resize(use_lp.num_row_);
  assert(status_.has_basis);
  const RsLp lp = rsLp(use_lp);
  const char* origin;
  size_t origin_len;
  highs_rs_lps_get_highs_basis(
      rs_, &lp, 0,
      reinterpret_cast<uint8_t*>(highs_basis.col_status.data()),
      reinterpret_cast<uint8_t*>(highs_basis.row_status.data()),
      &highs_basis.debug_id, &highs_basis.debug_update_count, &origin,
      &origin_len);
  highs_basis.debug_origin_name.assign(origin, origin_len);
  highs_basis.valid = true;
  highs_basis.alien = false;
  highs_basis.useful = true;
  highs_basis.was_alien = false;
  return highs_basis;
}

double HEkk::computeBasisCondition(const HighsLp& lp, const bool exact,
                                   const bool report) const {
  RustHost host{const_cast<HEkk*>(this)};
  const highs_rs::LpsEnv env = rsEnv(host);
  const RsLp v = rsLp(lp);
  return highs_rs_lps_basis_condition(rs_, &env, &v, lp.model_name_.data(),
                                      lp.model_name_.size(), exact, report);
}

HighsStatus HEkk::initialiseSimplexLpBasisAndFactor(
    const bool only_from_known_basis) {
  // If only_from_known_basis is true, then there should be a simplex
  // basis to use
  if (only_from_known_basis) assert(status_.has_basis);
  // The simplex NLA is set up for this LP
  snapshotFactorLog();
  setNlaEngineLp();
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  const HighsStatus status = HighsStatus(
      highs_rs_lps_initialise_basis_and_factor(rs_, &env, only_from_known_basis));
  takeRustOut();
  return status;
}

bool HEkk::lpFactorRowCompatible() const {
  return lpFactorRowCompatible(lpNumRow());
}

double HEkk::computeBasisCondition() const {
  RustHost host{const_cast<HEkk*>(this)};
  const highs_rs::LpsEnv env = rsEnv(host);
  return highs_rs_lps_basis_condition(rs_, &env, nullptr, nullptr, 0, false,
                                      false);
}

bool HEkk::lpFactorRowCompatible(const HighsInt expectedNumRow) const {
  RustHost host{const_cast<HEkk*>(this)};
  const highs_rs::LpsEnv env = rsEnv(host);
  return highs_rs_lps_lp_factor_row_compatible(rs_, &env, expectedNumRow);
}

std::string HEkk::simplexStrategyToString(
    const HighsInt simplex_strategy) const {
  assert(kSimplexStrategyMin <= simplex_strategy &&
         simplex_strategy <= kSimplexStrategyMax);
  if (simplex_strategy == kSimplexStrategyChoose)
    return "choose simplex solver";
  if (simplex_strategy == kSimplexStrategyDual) return "dual simplex solver";
  if (simplex_strategy == kSimplexStrategyDualPlain)
    return "serial dual simplex solver";
  if (simplex_strategy == kSimplexStrategyDualTasks)
    return "parallel dual simplex solver - SIP";
  if (simplex_strategy == kSimplexStrategyDualMulti)
    return "parallel dual simplex solver - PAMI";
  if (simplex_strategy == kSimplexStrategyPrimal)
    return "primal simplex solver";
  return "Unknown";
}

void HEkk::getUnscaledInfeasibilities(const HighsLp& lp,
                                      HighsInfo& highs_info) const {
  RustHost host{const_cast<HEkk*>(this)};
  const highs_rs::LpsEnv env = rsEnv(host);
  const RsLp v = rsLp(lp);
  highs_rs::UnscaledInfeasibilities u;
  highs_rs_lps_unscaled_infeasibilities(rs_, &env, &v, &u);
  highs_info.num_primal_infeasibilities = u.num_primal_infeasibilities;
  highs_info.max_primal_infeasibility = u.max_primal_infeasibility;
  highs_info.sum_primal_infeasibilities = u.sum_primal_infeasibilities;
  highs_info.num_dual_infeasibilities = u.num_dual_infeasibilities;
  highs_info.max_dual_infeasibility = u.max_dual_infeasibility;
  highs_info.sum_dual_infeasibilities = u.sum_dual_infeasibilities;
  setSolutionStatus(highs_info);
}

HighsInt* HEkk::basicIndex() const {
  HighsInt n;
  return highs_rs_lps_basic_index(rs_, &n);
}

RsMut<HighsInt> HEkk::basicIndexSlice(const bool use) const {
  if (!use) return {nullptr, 0};
  HighsInt n;
  HighsInt* p = highs_rs_lps_basic_index(rs_, &n);
  return {p, size_t(n)};
}

RsMut<int8_t> HEkk::nonbasicSlice(const bool move, const bool use) const {
  if (!use) return {nullptr, 0};
  HighsInt n;
  int8_t* p = highs_rs_lps_nonbasic(rs_, move ? 1 : 0, &n);
  return {p, size_t(n)};
}

void HEkk::resizeBasis(const HighsInt num_tot) {
  highs_rs_lps_resize_basis(rs_, num_tot);
}

void HEkk::appendBasicRows(const HighsInt num_col, const HighsInt num_row,
                           const HighsInt new_num_row) {
  highs_rs_lps_append_basic_rows(rs_, num_col, num_row, new_num_row);
}

void HEkk::flipNonbasicMove(const HighsInt var) {
  highs_rs_lps_flip_nonbasic_move(rs_, var);
}

const double* HEkk::dualEdgeWeights() const {
  return status_.has_dual_steepest_edge_weights
             ? highs_rs_lps_dual_edge_weight(rs_)
             : nullptr;
}

std::vector<double> HEkk::rayValue(const bool primal) const {
  size_t n;
  const double* v = highs_rs_lps_ray_value(rs_, primal, &n);
  return std::vector<double>(v, v + n);
}

void HEkk::setRayValue(const bool primal, const std::vector<double>& value) {
  highs_rs_lps_set_ray_value(rs_, primal, value.data(), value.size());
}

highs_rs::RangingSlices HEkk::rangingSlices() {
  highs_rs::RangingSlices s;
  highs_rs_lps_ranging_slices(rs_, &s);
  return s;
}

highs_rs::LpsEnv HEkk::callEnv() const {
  RustHost host{const_cast<HEkk*>(this)};
  return rsEnv(host);
}

// solveLpSimplex (rust/src/simplex/app.rs): the solver object's data in
// place and the steps on the C++ objects
namespace {

struct RsSimplexApp {
  RsLog log, factor_log;
  void* ctx;
  int64_t (*op)(void*, int, int64_t, void*);
  void* lps;
  RsLp incumbent;
  RsMut<char> model_name;
  HighsModelStatus* model_status;
  HighsInfoStruct* info;
  bool *value_valid, *dual_valid;
  RsVec<double> col_value, col_dual, row_value, row_dual;
  bool *basis_valid, *basis_alien, *basis_useful, *basis_was_alien;
  HighsInt *basis_debug_id, *basis_debug_update_count;
  RsVec<uint8_t> col_status, row_status;
  HighsInt* simplex_strategy;
  double* dual_simplex_cost_perturbation_multiplier;
  HighsInt simplex_unscaled_solution_strategy, cost_scale_factor,
      simplex_dualize_strategy, simplex_permute_strategy;
  RsLpOptions lp_options;
};

uint8_t* statusResize(void* v, size_t n) {
  std::vector<HighsBasisStatus>& x =
      *static_cast<std::vector<HighsBasisStatus>*>(v);
  x.resize(n);
  return reinterpret_cast<uint8_t*>(x.data());
}

RsVec<uint8_t> rsStatusVec(std::vector<HighsBasisStatus>& v) {
  return {&v, statusResize, reinterpret_cast<uint8_t*>(v.data()), v.size()};
}


}  // namespace

extern "C" int highs_rs_solve_lp_simplex(RsSimplexApp* h);

void HEkk::lpBack(HighsLp& lp, const bool matrix) const {
  RsLp v;
  highs_rs_lps_lp_view(rs_, &v);
  HighsScale& scale = lp.scale_;
  scale.strategy = v.scale_strategy;
  scale.has_scaling = v.scale_has_scaling;
  scale.num_col = v.scale_num_col;
  scale.num_row = v.scale_num_row;
  scale.cost = v.scale_cost;
  scale.col.assign(v.scale_col.ptr, v.scale_col.ptr + v.scale_col.len);
  scale.row.assign(v.scale_row.ptr, v.scale_row.ptr + v.scale_row.len);
  lp.is_scaled_ = v.is_scaled;
  if (matrix) {
    HighsSparseMatrix& a = lp.a_matrix_;
    a.format_ = MatrixFormat(v.a.format);
    a.num_col_ = v.a.num_col;
    a.num_row_ = v.a.num_row;
    a.start_.assign(v.a.start.ptr, v.a.start.ptr + v.a.start.len);
    a.p_end_.assign(v.a.p_end.ptr, v.a.p_end.ptr + v.a.p_end.len);
    a.index_.assign(v.a.index.ptr, v.a.index.ptr + v.a.index.len);
    a.value_.assign(v.a.value.ptr, v.a.value.ptr + v.a.value.len);
  }
}

// The steps of solveLpSimplex on the HEkk shell (app.rs ops 1-4, 6-11),
// with `lp` the incumbent LP
int64_t rsSimplexShellOp(HEkk& ekk, HighsProfiling* profiling,
                         HighsOptions& options, HighsCallback& callback,
                         HighsTimer& timer, HighsLp& lp, int code,
                         int64_t arg, void* p) {
  switch (code) {
    case 1:
      // arg: whether the HiGHS basis is valid
      if (profiling) {
        HighsInt profiling_clock = -1;
        if (options.simplex_strategy == kSimplexStrategyPrimal) {
          profiling_clock =
              arg ? kSubSolverPrSimplexBasis : kSubSolverPrSimplexNoBasis;
        } else {
          profiling_clock =
              arg ? kSubSolverDuSimplexBasis : kSubSolverDuSimplexNoBasis;
        }
        profiling->start(profiling_clock);
      }
      return 0;
    case 2:
      if (profiling->sub_solver_) {
        HighsInt profiling_clock = -1;
        HighsProfilingRecord* thread_record =
            profiling->getHighsProfilingRecord();
        if (std::signbit(thread_record->start_time[kSubSolverDuSimplexBasis]))
          profiling_clock = kSubSolverDuSimplexBasis;
        if (std::signbit(
                thread_record->start_time[kSubSolverDuSimplexNoBasis]))
          profiling_clock = kSubSolverDuSimplexNoBasis;
        if (std::signbit(thread_record->start_time[kSubSolverPrSimplexBasis]))
          profiling_clock = kSubSolverPrSimplexBasis;
        if (std::signbit(
                thread_record->start_time[kSubSolverPrSimplexNoBasis]))
          profiling_clock = kSubSolverPrSimplexNoBasis;
        profiling->stop(profiling_clock);
      }
      return 0;
    case 3:
      ekk.initialiseSimplexStats();
      return 0;
    case 4:
      ekk.movedLp(callback, options, timer);
      return 0;
    case 6:
      return int64_t(ekk.solve(arg != 0));
    case 7:
      ekk.clear();
      return 0;
    case 8:
      return ekk.proofOfPrimalInfeasibility();
    case 9:
      ekk.setNlaPointersForLpAndScale(lp);
      return 0;
    case 10:
      ekk.lpBack(lp, arg != 0);
      return 0;
    case 11:
      *static_cast<highs_rs::LpsEnv*>(p) = ekk.callEnv();
      return 0;
  }
  assert(false);
  return 0;
}

static int64_t simplexAppOp(void* ctx, int code, int64_t arg, void* p) {
  HighsLpSolverObject& so = *static_cast<HighsLpSolverObject*>(ctx);
  switch (code) {
    case 5:
      return int64_t(so.ekk_instance_.setBasis(so.basis_));
    case 12:
      so.basis_.debug_origin_name.assign(static_cast<const char*>(p),
                                         size_t(arg));
      return 0;
  }
  return rsSimplexShellOp(so.ekk_instance_, so.profiling_, so.options_,
                          so.callback_, so.timer_, so.lp_, code, arg, p);
}

void RsFactorLogStore::set(const HighsLogOptions& from) {
  output_flag = *from.output_flag;
  log_to_console = *from.log_to_console;
  log_dev_level = *from.log_dev_level;
  log_options.output_flag = &output_flag;
  log_options.log_to_console = &log_to_console;
  log_options.log_dev_level = &log_dev_level;
  log_options.log_stream = from.log_stream;
}

// The options and logs of solveLpSimplex (the data are set by the
// caller), the factor's log in `factor_log`
static void simplexAppOptions(HighsOptions& options, HighsLp& lp,
                              const HighsLogOptions& factor_log,
                              RsSimplexApp& h) {
  h.log = rsLog(options.log_options);
  h.factor_log = rsLog(factor_log);
  h.incumbent = rsLp(lp);
  h.model_name = {const_cast<char*>(lp.model_name_.data()),
                  lp.model_name_.size()};
  h.simplex_strategy = &options.simplex_strategy;
  h.dual_simplex_cost_perturbation_multiplier =
      &options.dual_simplex_cost_perturbation_multiplier;
  h.simplex_unscaled_solution_strategy =
      options.simplex_unscaled_solution_strategy;
  h.cost_scale_factor = options.cost_scale_factor;
  h.simplex_dualize_strategy = options.simplex_dualize_strategy;
  h.simplex_permute_strategy = options.simplex_permute_strategy;
  h.lp_options = rsLpOptions(options);
}

void rsSimplexAppTemplate(HighsOptions& options, HighsLp& lp,
                          RsFactorLogStore& factor_log, void* out) {
  RsSimplexApp& h = *static_cast<RsSimplexApp*>(out);
  h = RsSimplexApp{};
  factor_log.set(options.log_options);
  simplexAppOptions(options, lp, factor_log.log_options, h);
}

HighsStatus solveLpSimplex(HighsLpSolverObject& solver_object) {
  HighsOptions& options = solver_object.options_;
  HighsLp& lp = solver_object.lp_;
  HighsSolution& solution = solver_object.solution_;
  HighsBasis& basis = solver_object.basis_;
  RsFactorLogStore factor_log;
  factor_log.set(options.log_options);
  RsSimplexApp h;
  simplexAppOptions(options, lp, factor_log.log_options, h);
  h.ctx = &solver_object;
  h.op = simplexAppOp;
  h.lps = solver_object.ekk_instance_.rs_;
  h.model_status = &solver_object.model_status_;
  h.info = static_cast<HighsInfoStruct*>(&solver_object.highs_info_);
  h.value_valid = &solution.value_valid;
  h.dual_valid = &solution.dual_valid;
  h.col_value = rsVec(solution.col_value);
  h.col_dual = rsVec(solution.col_dual);
  h.row_value = rsVec(solution.row_value);
  h.row_dual = rsVec(solution.row_dual);
  h.basis_valid = &basis.valid;
  h.basis_alien = &basis.alien;
  h.basis_useful = &basis.useful;
  h.basis_was_alien = &basis.was_alien;
  h.basis_debug_id = &basis.debug_id;
  h.basis_debug_update_count = &basis.debug_update_count;
  h.col_status = rsStatusVec(basis.col_status);
  h.row_status = rsStatusVec(basis.row_status);
  return HighsStatus(highs_rs_solve_lp_simplex(&h));
}

void HighsSimplexStats::report(FILE* file, std::string message) const {
  fprintf(file, "\nSimplex stats: %s\n", message.c_str());
  fprintf(file, "   valid                      = %d\n", this->valid);
  fprintf(file, "   iteration_count            = %d\n",
          static_cast<int>(this->iteration_count));
  fprintf(file, "   num_invert                 = %d\n",
          static_cast<int>(this->num_invert));
  fprintf(file, "   last_invert_num_el         = %d\n",
          static_cast<int>(this->last_invert_num_el));
  fprintf(file, "   last_factored_basis_num_el = %d\n",
          static_cast<int>(this->last_factored_basis_num_el));
  fprintf(file, "   col_aq_density             = %g\n", this->col_aq_density);
  fprintf(file, "   row_ep_density             = %g\n", this->row_ep_density);
  fprintf(file, "   row_ap_density             = %g\n", this->row_ap_density);
  fprintf(file, "   row_DSE_density            = %g\n", this->row_DSE_density);
}

void HighsSimplexStats::initialise(const HighsInt iteration_count_) {
  valid = false;
  iteration_count = -iteration_count_;
  num_invert = 0;
  last_invert_num_el = 0;
  last_factored_basis_num_el = 0;
  col_aq_density = 0;
  row_ep_density = 0;
  row_ap_density = 0;
  row_DSE_density = 0;
}

#endif  // HIGHS_RUST
