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

  static HEkk& e(void* ctx) { return *static_cast<RustHost*>(ctx)->ekk; }

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
    const HighsInt num_tot = ekk.lp_.num_col_ + ekk.lp_.num_row_;
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
      nla_scale_(nullptr),
      original_num_col_(0),
      original_num_row_(0),
      original_num_nz_(0),
      original_offset_(0.0) {
  factor_log_options_.output_flag = &factor_log_data_.output_flag;
  factor_log_options_.log_to_console = &factor_log_data_.log_to_console;
  factor_log_options_.log_dev_level = &factor_log_data_.log_dev_level;
}

HEkk::~HEkk() { highs_rs_lps_free(rs_); }

highs_rs::LpsEnv HEkk::rsEnv(RustHost& host) const {
  highs_rs::LpsEnv env;
  env.lp = rsLp(lp_);
  env.model_name = {const_cast<char*>(lp_.model_name_.data()),
                    lp_.model_name_.size()};
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
  env.host.ctx = &host;
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
  clearEkkDualize();
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
  lp_.clear();
  lp_name_ = "";
}

void HEkk::clearEkkDualize() {
  original_col_cost_.clear();
  original_col_lower_.clear();
  original_col_upper_.clear();
  original_row_lower_.clear();
  original_row_upper_.clear();
  upper_bound_col_.clear();
  upper_bound_row_.clear();
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
  sh_.nla_lp_set = true;
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
  // Move the incumbent LP to EKK
  HighsLp& incumbent_lp = solver_object.lp_;
  this->lp_ = std::move(incumbent_lp);
  incumbent_lp.is_moved_ = true;
  // Update the pointers to the HighsOptions and HighsTimer members of the
  // Highs class, communicated by reference via the HighsLpSolverObject
  this->setPointers(&solver_object.callback_, &solver_object.options_,
                    &solver_object.timer_);
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

HighsStatus HEkk::dualize() {
  assert(lp_.a_matrix_.isColwise());
  original_num_col_ = lp_.num_col_;
  original_num_row_ = lp_.num_row_;
  original_num_nz_ = lp_.a_matrix_.numNz();
  original_offset_ = lp_.offset_;
  original_col_cost_ = lp_.col_cost_;
  original_col_lower_ = lp_.col_lower_;
  original_col_upper_ = lp_.col_upper_;
  original_row_lower_ = lp_.row_lower_;
  original_row_upper_ = lp_.row_upper_;
  // Reserve space for simple dual
  lp_.col_cost_.reserve(original_num_row_);
  lp_.col_lower_.reserve(original_num_row_);
  lp_.col_upper_.reserve(original_num_row_);
  lp_.row_lower_.reserve(original_num_col_);
  lp_.row_upper_.reserve(original_num_col_);
  // Invalidate the original data
  lp_.col_cost_.resize(0);
  lp_.col_lower_.resize(0);
  lp_.col_upper_.resize(0);
  lp_.row_lower_.resize(0);
  lp_.row_upper_.resize(0);
  // The bulk of the constraint matrix of the dual LP is the transpose
  // of the primal constraint matrix. This is obtained row-wise by
  // copying the matrix and flipping the dimensions
  HighsSparseMatrix dual_matrix = lp_.a_matrix_;
  dual_matrix.num_row_ = original_num_col_;
  dual_matrix.num_col_ = original_num_row_;
  dual_matrix.format_ = MatrixFormat::kRowwise;
  // The primal_bound_value vector accumulates the values of the
  // finite bounds on variables - or zero for a free variable - used
  // later to compute the offset and shift for the costs. Many of
  // these components will be zero - all for the case x>=0 - so
  // maintain a list of the nonzeros and corresponding indices. Don't
  // reserve space since they may not be needed
  vector<double> primal_bound_value;
  vector<HighsInt> primal_bound_index;
  const double inf = kHighsInf;
  for (HighsInt iCol = 0; iCol < original_num_col_; iCol++) {
    const double cost = original_col_cost_[iCol];
    const double lower = original_col_lower_[iCol];
    const double upper = original_col_upper_[iCol];
    double primal_bound = inf;
    double row_lower = inf;
    double row_upper = -inf;
    if (lower == upper) {
      // Fixed
      primal_bound = lower;
      // Dual activity a^Ty is free, implying dual for primal column
      // (slack for dual row) is free
      row_lower = -inf;
      row_upper = inf;
    } else if (!highs_isInfinity(-lower)) {
      // Finite lower bound so boxed or lower
      if (!highs_isInfinity(upper)) {
        // Finite upper bound so boxed
        //
        // Treat as lower
        primal_bound = lower;
        // Dual activity a^Ty is bounded above by cost, implying dual
        // for primal column (slack for dual row) is non-negative
        row_lower = -inf;
        row_upper = cost;
        // Treat upper bound as additional constraint
        upper_bound_col_.push_back(iCol);
      } else {
        // Lower (since upper bound is infinite)
        primal_bound = lower;
        // Dual activity a^Ty is bounded above by cost, implying dual
        // for primal column (slack for dual row) is non-negative
        row_lower = -inf;
        row_upper = cost;
      }
    } else if (!highs_isInfinity(upper)) {
      // Upper
      primal_bound = upper;
      // Dual activity a^Ty is bounded below by cost, implying dual
      // for primal column (slack for dual row) is non-positive
      row_lower = cost;
      row_upper = inf;
    } else {
      // FREE
      //
      // Dual activity a^Ty is fixed by cost, implying dual for primal
      // column (slack for dual row) is fixed at zero
      primal_bound = 0;
      row_lower = cost;
      row_upper = cost;
    }
    assert(row_lower < inf);
    assert(row_upper > -inf);
    assert(primal_bound < inf);
    lp_.row_lower_.push_back(row_lower);
    lp_.row_upper_.push_back(row_upper);
    if (primal_bound) {
      primal_bound_value.push_back(primal_bound);
      primal_bound_index.push_back(iCol);
    }
  }
  for (HighsInt iRow = 0; iRow < original_num_row_; iRow++) {
    double lower = original_row_lower_[iRow];
    double upper = original_row_upper_[iRow];
    double col_cost = inf;
    double col_lower = inf;
    double col_upper = -inf;
    if (lower == upper) {
      // Equality constraint
      //
      // Dual variable has primal RHS as cost and is free
      col_cost = lower;
      col_lower = -inf;
      col_upper = inf;
    } else if (!highs_isInfinity(-lower)) {
      // Finite lower bound so boxed or lower
      if (!highs_isInfinity(upper)) {
        // Finite upper bound so boxed
        //
        // Treat as lower
        col_cost = lower;
        col_lower = 0;
        col_upper = inf;
        // Treat upper bound as additional constraint
        upper_bound_row_.push_back(iRow);
      } else {
        // Lower (since upper bound is infinite)
        col_cost = lower;
        col_lower = 0;
        col_upper = inf;
      }
    } else if (!highs_isInfinity(upper)) {
      // Upper
      col_cost = upper;
      col_lower = -inf;
      col_upper = 0;
    } else {
      // FREE
      // Shouldn't get free rows, but handle them anyway
      col_cost = 0;
      col_lower = 0;
      col_upper = 0;
    }
    assert(col_lower < inf);
    assert(col_upper > -inf);
    assert(col_cost < inf);
    lp_.col_cost_.push_back(col_cost);
    lp_.col_lower_.push_back(col_lower);
    lp_.col_upper_.push_back(col_upper);
  }
  vector<HighsInt>& start = lp_.a_matrix_.start_;
  vector<HighsInt>& index = lp_.a_matrix_.index_;
  vector<double>& value = lp_.a_matrix_.value_;
  // Boxed variables and constraints yield extra columns in the dual LP
  HighsSparseMatrix extra_columns;
  extra_columns.ensureColwise();
  extra_columns.num_row_ = original_num_col_;
  HighsInt num_upper_bound_col = upper_bound_col_.size();
  HighsInt num_upper_bound_row = upper_bound_row_.size();
  double one = 1;
  for (HighsInt iX = 0; iX < num_upper_bound_col; iX++) {
    HighsInt iCol = upper_bound_col_[iX];
    const double upper = original_col_upper_[iCol];
    extra_columns.addVec(1, &iCol, &one);
    lp_.col_cost_.push_back(upper);
    lp_.col_lower_.push_back(-inf);
    lp_.col_upper_.push_back(0);
  }

  if (num_upper_bound_row) {
    // Need to identify the submatrix of constraint matrix rows
    // corresponding to those with a row index in
    // upper_bound_row_. When identifying numbers of entries in each
    // row of submatrix, use indirection to get corresponding row
    // index, with a dummy row for rows not in the submatrix.
    HighsInt dummy_row = num_upper_bound_row;
    vector<HighsInt> indirection;
    vector<HighsInt> count;
    indirection.assign(original_num_row_, dummy_row);
    count.assign(num_upper_bound_row + 1, 0);
    HighsInt extra_iRow = 0;
    for (HighsInt iX = 0; iX < num_upper_bound_row; iX++) {
      HighsInt iRow = upper_bound_row_[iX];
      indirection[iRow] = extra_iRow++;
      double upper = original_row_upper_[iRow];
      lp_.col_cost_.push_back(upper);
      lp_.col_lower_.push_back(-inf);
      lp_.col_upper_.push_back(0);
    }
    for (HighsInt iEl = 0; iEl < original_num_nz_; iEl++)
      count[indirection[index[iEl]]]++;
    extra_columns.start_.resize(num_upper_bound_col + num_upper_bound_row + 1);
    for (HighsInt iRow = 0; iRow < num_upper_bound_row; iRow++) {
      extra_columns.start_[num_upper_bound_col + iRow + 1] =
          extra_columns.start_[num_upper_bound_col + iRow] + count[iRow];
      count[iRow] = extra_columns.start_[num_upper_bound_col + iRow];
    }
    HighsInt extra_columns_num_nz =
        extra_columns.start_[num_upper_bound_col + num_upper_bound_row];
    extra_columns.index_.resize(extra_columns_num_nz);
    extra_columns.value_.resize(extra_columns_num_nz);
    for (HighsInt iCol = 0; iCol < original_num_col_; iCol++) {
      for (HighsInt iEl = start[iCol]; iEl < start[iCol + 1]; iEl++) {
        HighsInt iRow = indirection[index[iEl]];
        if (iRow < num_upper_bound_row) {
          HighsInt extra_columns_iEl = count[iRow];
          assert(extra_columns_iEl < extra_columns_num_nz);
          extra_columns.index_[extra_columns_iEl] = iCol;
          extra_columns.value_[extra_columns_iEl] = value[iEl];
          count[iRow]++;
        }
      }
    }
    extra_columns.num_col_ += num_upper_bound_row;
  }
  // Incorporate the cost shift by subtracting A*primal_bound from the
  // cost vector; compute the objective offset
  double delta_offset = 0;
  for (size_t iX = 0; iX < primal_bound_index.size(); iX++) {
    HighsInt iCol = primal_bound_index[iX];
    double multiplier = primal_bound_value[iX];
    delta_offset += multiplier * original_col_cost_[iCol];
    for (HighsInt iEl = start[iCol]; iEl < start[iCol + 1]; iEl++)
      lp_.col_cost_[index[iEl]] -= multiplier * value[iEl];
  }
  if (extra_columns.num_col_) {
    // Incorporate the cost shift by subtracting
    // extra_columns*primal_bound from the cost vector for the extra
    // dual variables
    //
    // Have to scatter the packed primal bound values into a
    // full-length vector
    //
    // ToDo Make this more efficient?
    vector<double> primal_bound;
    primal_bound.assign(original_num_col_, 0);
    for (size_t iX = 0; iX < primal_bound_index.size(); iX++)
      primal_bound[primal_bound_index[iX]] = primal_bound_value[iX];

    for (HighsInt iCol = 0; iCol < extra_columns.num_col_; iCol++) {
      double cost = lp_.col_cost_[original_num_row_ + iCol];
      for (HighsInt iEl = extra_columns.start_[iCol];
           iEl < extra_columns.start_[iCol + 1]; iEl++)
        cost -=
            primal_bound[extra_columns.index_[iEl]] * extra_columns.value_[iEl];
      lp_.col_cost_[original_num_row_ + iCol] = cost;
    }
  }
  lp_.offset_ += delta_offset;
  // Copy the row-wise dual LP constraint matrix and transpose it.
  // ToDo Make this more efficient
  lp_.a_matrix_ = dual_matrix;
  lp_.a_matrix_.ensureColwise();
  // Add the extra columns to the dual LP constraint matrix
  lp_.a_matrix_.addCols(extra_columns);

  HighsInt dual_num_col =
      original_num_row_ + num_upper_bound_col + num_upper_bound_row;
  HighsInt dual_num_row = original_num_col_;
  assert(dual_num_col == (int)lp_.col_cost_.size());
  assert(lp_.a_matrix_.num_col_ == dual_num_col);
  const bool ignore_scaling = true;
  if (!ignore_scaling) {
    // Flip any scale factors
    if (lp_.scale_.has_scaling) {
      std::vector<double> temp_scale = lp_.scale_.row;
      lp_.scale_.row = lp_.scale_.col;
      lp_.scale_.col = temp_scale;
      lp_.scale_.num_col = dual_num_col;
      lp_.scale_.num_row = dual_num_row;
    }
  }
  // Change optimization sense
  if (lp_.sense_ == ObjSense::kMinimize) {
    lp_.sense_ = ObjSense::kMaximize;
  } else {
    lp_.sense_ = ObjSense::kMinimize;
  }
  // Flip LP dimensions
  lp_.num_col_ = dual_num_col;
  lp_.num_row_ = dual_num_row;
  status_.is_dualized = true;
  status_.has_basis = false;
  status_.has_ar_matrix = false;
  status_.has_nla = false;
  highsLogUser(options_->log_options, HighsLogType::kInfo,
               "Solving dual LP with %d columns", (int)dual_num_col);
  if (num_upper_bound_col + num_upper_bound_row) {
    highsLogUser(options_->log_options, HighsLogType::kInfo, " [%d extra from",
                 (int)dual_num_col - original_num_row_);
    if (num_upper_bound_col)
      highsLogUser(options_->log_options, HighsLogType::kInfo,
                   " %d boxed variable(s)", (int)num_upper_bound_col);
    if (num_upper_bound_col && num_upper_bound_row)
      highsLogUser(options_->log_options, HighsLogType::kInfo, " and");
    if (num_upper_bound_row)
      highsLogUser(options_->log_options, HighsLogType::kInfo,
                   " %d boxed constraint(s)", (int)num_upper_bound_row);
    highsLogUser(options_->log_options, HighsLogType::kInfo, "]");
  }
  highsLogUser(options_->log_options, HighsLogType::kInfo, " and %d rows\n",
               (int)dual_num_row);
  //  reportLp(options_->log_options, lp_, HighsLogType::kVerbose);
  return HighsStatus::kOk;
}


HighsStatus HEkk::undualize() {
  if (!this->status_.is_dualized) return HighsStatus::kOk;
  HighsInt dual_num_col = lp_.num_col_;
  // The primal basis from the basis of the dual LP
  const HighsInt num_basic_variables = highs_rs_lps_undualize_basis(
      rs_, dual_num_col, original_num_col_, original_num_row_,
      original_col_lower_.data(), original_col_upper_.data(),
      original_row_lower_.data(), original_row_upper_.data());
  // Change optimization sense
  if (lp_.sense_ == ObjSense::kMinimize) {
    lp_.sense_ = ObjSense::kMaximize;
  } else {
    lp_.sense_ = ObjSense::kMinimize;
  }
  // Flip LP dimensions
  lp_.num_col_ = original_num_col_;
  lp_.num_row_ = original_num_row_;
  // Restore the original offset
  lp_.offset_ = original_offset_;
  // Copy back the costs and bounds
  lp_.col_cost_ = original_col_cost_;
  lp_.col_lower_ = original_col_lower_;
  lp_.col_upper_ = original_col_upper_;
  lp_.row_lower_ = original_row_lower_;
  lp_.row_upper_ = original_row_upper_;
  // The primal constraint matrix is available row-wise as the first
  // original_num_row_ vectors of the dual constraint matrix
  HighsSparseMatrix primal_matrix;
  primal_matrix.start_.resize(original_num_row_ + 1);
  primal_matrix.index_.resize(original_num_nz_);
  primal_matrix.value_.resize(original_num_nz_);

  for (HighsInt iCol = 0; iCol < original_num_row_ + 1; iCol++)
    primal_matrix.start_[iCol] = lp_.a_matrix_.start_[iCol];
  for (HighsInt iEl = 0; iEl < original_num_nz_; iEl++) {
    primal_matrix.index_[iEl] = lp_.a_matrix_.index_[iEl];
    primal_matrix.value_[iEl] = lp_.a_matrix_.value_[iEl];
  }
  primal_matrix.num_col_ = original_num_col_;
  primal_matrix.num_row_ = original_num_row_;
  primal_matrix.format_ = MatrixFormat::kRowwise;
  // Copy the row-wise primal LP constraint matrix and transpose it.
  lp_.a_matrix_ = primal_matrix;
  lp_.a_matrix_.ensureColwise();
  assert(lp_.num_col_ == original_num_col_);
  assert(lp_.num_row_ == original_num_row_);
  assert(lp_.a_matrix_.numNz() == original_num_nz_);
  bool num_basic_variables_ok = num_basic_variables == original_num_row_;
  if (!num_basic_variables_ok)
    printf("HEkk::undualize: Have %d basic variables, not %d\n",
           (int)num_basic_variables, (int)original_num_row_);
  assert(num_basic_variables_ok);

  // Clear the data retained when solving dual LP
  clearEkkDualize();
  status_.is_dualized = false;
  // Now solve with this basis. Should just be a case of reinverting
  // and re-solving for optimal primal and dual values, but
  // numerically marginal LPs will need clean-up
  status_.has_basis = true;
  status_.has_ar_matrix = false;
  status_.has_nla = false;
  status_.has_invert = false;
  HighsInt primal_solve_iteration_count = -iteration_count_;
  HighsStatus return_status = solve();
  primal_solve_iteration_count += iteration_count_;
  highsLogUser(options_->log_options, HighsLogType::kInfo,
               "Solving the primal LP (%s) using the optimal basis of its dual "
               "required %d simplex iterations\n",
               lp_.model_name_.c_str(), (int)primal_solve_iteration_count);
  return return_status;
}

HighsStatus HEkk::permute() {
  assert(1 == 0);
  return HighsStatus::kError;
}

HighsStatus HEkk::unpermute() {
  if (!this->status_.is_permuted) return HighsStatus::kOk;
  assert(1 == 0);
  return HighsStatus::kError;
}

HighsStatus HEkk::solve(const bool force_phase2) {
  // initialiseAnalysis
  analysis_.setup(lp_name_, lp_, *options_, iteration_count_);
  // The solve sets up the simplex NLA for this LP
  snapshotFactorLog();
  setNlaLp(lp_);
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
  this->lp_.num_row_ = lp.num_row_;
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

HighsSolution HEkk::getSolution() {
  HighsSolution solution;
  solution.col_value.resize(lp_.num_col_);
  solution.col_dual.resize(lp_.num_col_);
  solution.row_value.resize(lp_.num_row_);
  solution.row_dual.resize(lp_.num_row_);
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  highs_rs_lps_get_solution(rs_, &env, solution.col_value.data(),
                            solution.col_dual.data(), solution.row_value.data(),
                            solution.row_dual.data());
  solution.value_valid = true;
  solution.dual_valid = true;
  return solution;
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
      rs_, &lp, (int)lp_.sense_,
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
  setNlaLp(lp_);
  RustHost host{this};
  const highs_rs::LpsEnv env = rsEnv(host);
  const HighsStatus status = HighsStatus(
      highs_rs_lps_initialise_basis_and_factor(rs_, &env, only_from_known_basis));
  takeRustOut();
  return status;
}

bool HEkk::lpFactorRowCompatible() const {
  return lpFactorRowCompatible(this->lp_.num_row_);
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
