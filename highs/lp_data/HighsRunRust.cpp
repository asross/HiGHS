/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsRunRust.cpp
 * @brief Highs::run's control flow done by Rust (rust/src/lp_data/run.rs):
 * calledOptimizeModel, runPresolve, runPostsolve, returnFromOptimizeModel
 * and returnFromHighs. Rust calls back here for each step on a C++ object.
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>

#include "Highs.h"
#include "lp_data/HighsInfoDebug.h"
#include "lp_data/HighsSolutionDebug.h"
#include "mip/HighsMipSolver.h"
#include "model/HighsHessianUtils.h"
#include "presolve/ICrashX.h"
#include "simplex/HSimplex.h"

static_assert(sizeof(HighsRunDataStruct) == 48, "HighsRunDataStruct layout");
static_assert(sizeof(HighsModelStatus) == 4, "HighsModelStatus is an int");
static_assert(sizeof(HighsPresolveStatus) == 4,
              "HighsPresolveStatus is an int");

namespace {

struct RsRunStr {
  const char* ptr;
  size_t len;
};

// run.rs: ROptions
struct RsRunOptions {
  RsRunStr solver, run_crossover, presolve;
  bool use_warm_start, icrash, solve_relaxation, allow_unbounded_or_infeasible,
      timeless_log;
  double large_matrix_value, time_limit, primal_feasibility_tolerance,
      mip_feasibility_tolerance;
  HighsInt* highs_debug_level;
  double* objective_bound;
  bool* lp_presolve_requires_basis_postsolve;
  const bool* output_flag;
  const HighsInt* log_dev_level;
};

// run.rs: CHighs
struct RsHighs {
  RsLog log;
  void* ctx;
  int64_t (*op)(void*, int, int64_t, void*, const char*, size_t);
  double (*clock)(void*, int, int);
  HighsModelStatus* model_status;
  HighsPresolveStatus* presolve_status;
  HighsInfoStruct* info;
  HighsRunDataStruct* run_data;
  bool *value_valid, *dual_valid, *basis_valid, *basis_alien, *basis_useful,
      *basis_was_alien, *called_return;
  RsRunOptions o;
};

// run.rs: Facts
struct RsFacts {
  HighsInt num_col, num_row, num_nz;
  bool is_mip, is_qp, is_empty, has_infinite_cost;
  RsRunStr model_name;
};

// run.rs: Op
enum class RunOp {
  kClearSolver = 1,
  kHandleInfCost,
  kExactResizeModel,
  kCompleteSolution,
  kInvalidateInfo,
  kInvalidateRunData,
  kInvalidateBasis,
  kFacts,
  kEnsureColwise,
  kHasLargeValue,
  kDebugAssess,
  kAssessSemiVariables,
  kRelaxSemiVariables,
  kOkHessianDiagonal,
  kCallSolveQp,
  kCallSolveMip,
  kICrash,
  kBasisForSolution,
  kBasisClear,
  kRefineBasis,
  kCallSolveLp,
  kSetEkkLpName,
  kPrepareReducedLp,
  kEkkClear,
  kEkkInvalidate,
  kEkkPivotThreshold,
  kReducedToEmpty,
  kSaveOptions,
  kRestoreOptions,
  kOptionsPrimalSimplex,
  kOptionsCleanup,
  kCopyToPresolve,
  kKktCheck,
  kRecoveredValidity,
  kPostsolveUndo,
  kSetPostsolveStatus,
  kTakeRecoveredSolution,
  kTakeRecoveredBasis,
  kDebugPostsolveSolution,
  kUndoMods,
  kDebugReturn,
  kForceSolutionBasisSize,
  kBasisConsistent,
  kRetainedEkkDataOk,
  kLpDimensionsOk,
  kEkkFactorCompatible,
  kPresolveClear,
  kMipPresolve,
  kPresolveInit,
  kPresolveRun,
  kPresolveLog,
  kPresolveRemoved,
  kClearReducedIntegrality,
  kPresolveTime,
  kLpView,
};

RsRunStr rsRunStr(const std::string& s) { return {s.data(), s.size()}; }

}  // namespace

extern "C" {
int highs_rs_called_optimize_model(const RsHighs* h);
int highs_rs_return_from_optimize_model(const RsHighs* h, int status,
                                        bool undo_mods);
int highs_rs_return_from_highs(const RsHighs* h, int status);
int highs_rs_run_presolve(const RsHighs* h, bool force_lp_presolve,
                          bool force_presolve);
int highs_rs_run_postsolve(const RsHighs* h);
}

// The steps of the run on the Highs object (a friend)
struct HighsRunRust {
  Highs& h;
  // The options saved by kSaveOptions
  std::unique_ptr<HighsOptions> saved_options;
  // An exception thrown by a step (a cancelled task's HighsTask::Interrupt),
  // rethrown once Rust has returned; no step is made after it
  std::exception_ptr pending;

  static int64_t op(void* ctx, int which, int64_t arg, void* p,
                    const char* msg, size_t len) {
    HighsRunRust& r = *static_cast<HighsRunRust*>(ctx);
    if (r.pending) return 0;
    try {
      return r.step(RunOp(which), arg, p, msg, len);
    } catch (...) {
      r.pending = std::current_exception();
      return kAbort;
    }
  }

  static double clock(void* ctx, int which, int action) {
    if (static_cast<HighsRunRust*>(ctx)->pending) return 0;
    Highs& h = static_cast<HighsRunRust*>(ctx)->h;
    HighsTimer& t = h.timer_;
    const HighsInt c = which == 0   ? 0
                       : which == 1 ? t.solve_clock
                       : which == 2 ? t.presolve_clock
                                    : t.postsolve_clock;
    switch (action) {
      case 0:
        return t.read(c);
      case 1:
        t.start(c);
        return 0;
      case 2:
        t.stop(c);
        return 0;
      default:
        return t.running(c) ? 1 : 0;
    }
  }

  RsHighs view() {
    HighsOptions& o = h.options_;
    RsHighs v;
    v.log = rsLog(o.log_options);
    v.ctx = this;
    v.op = op;
    v.clock = clock;
    v.model_status = &h.model_status_;
    v.presolve_status = &h.model_presolve_status_;
    v.info = static_cast<HighsInfoStruct*>(&h.info_);
    v.run_data = static_cast<HighsRunDataStruct*>(&h.run_data_);
    v.value_valid = &h.solution_.value_valid;
    v.dual_valid = &h.solution_.dual_valid;
    v.basis_valid = &h.basis_.valid;
    v.basis_alien = &h.basis_.alien;
    v.basis_useful = &h.basis_.useful;
    v.basis_was_alien = &h.basis_.was_alien;
    v.called_return = &h.called_return_from_optimize_model;
    v.o.solver = rsRunStr(o.solver);
    v.o.run_crossover = rsRunStr(o.run_crossover);
    v.o.presolve = rsRunStr(o.presolve);
    v.o.use_warm_start = o.use_warm_start;
    v.o.icrash = o.icrash;
    v.o.solve_relaxation = o.solve_relaxation;
    v.o.allow_unbounded_or_infeasible = o.allow_unbounded_or_infeasible;
    v.o.timeless_log = o.timeless_log;
    v.o.large_matrix_value = o.large_matrix_value;
    v.o.time_limit = o.time_limit;
    v.o.primal_feasibility_tolerance = o.primal_feasibility_tolerance;
    v.o.mip_feasibility_tolerance = o.mip_feasibility_tolerance;
    v.o.highs_debug_level = &o.highs_debug_level;
    v.o.objective_bound = &o.objective_bound;
    v.o.lp_presolve_requires_basis_postsolve =
        &o.lp_presolve_requires_basis_postsolve;
    v.o.output_flag = o.log_options.output_flag;
    v.o.log_dev_level = o.log_options.log_dev_level;
    return v;
  }

  static int st(HighsStatus s) { return int(s); }
  // run.rs: ABORT
  static constexpr int64_t kAbort = INT64_MIN;

  void rethrow() {
    if (pending) std::rethrow_exception(pending);
  }

  HighsLp& lpOf(int64_t which) {
    return which ? h.presolve_.getReducedProblem() : h.model_.lp_;
  }

  int64_t step(RunOp which, int64_t arg, void* p, const char* m, size_t len) {
    HighsOptions& options = h.options_;
    switch (which) {
      case RunOp::kClearSolver:
        h.clearSolver();
        return 0;
      case RunOp::kHandleInfCost:
        return st(h.handleInfCost());
      case RunOp::kExactResizeModel:
        h.exactResizeModel();
        return 0;
      case RunOp::kCompleteSolution:
        return st(h.completeSolutionFromDiscreteAssignment());
      case RunOp::kInvalidateInfo:
        h.invalidateInfo();
        return 0;
      case RunOp::kInvalidateRunData:
        h.invalidateRunData();
        return 0;
      case RunOp::kInvalidateBasis:
        h.invalidateBasis();
        return 0;
      case RunOp::kFacts: {
        RsFacts& f = *static_cast<RsFacts*>(p);
        const HighsLp& lp = lpOf(arg);
        f.num_col = lp.num_col_;
        f.num_row = lp.num_row_;
        f.num_nz = lp.a_matrix_.numNz();
        f.is_mip = arg ? lp.isMip() : h.model_.isMip();
        f.is_qp = arg ? false : h.model_.isQp();
        f.is_empty = arg ? lp.num_col_ == 0 && lp.num_row_ == 0
                         : h.model_.isEmpty();
        f.has_infinite_cost = lp.has_infinite_cost_;
        f.model_name = rsRunStr(lp.model_name_);
        return 0;
      }
      case RunOp::kEnsureColwise:
        h.model_.lp_.ensureColwise();
        return 0;
      case RunOp::kHasLargeValue:
        return h.model_.lp_.a_matrix_.hasLargeValue(options.large_matrix_value);
      case RunOp::kDebugAssess: {
        HighsStatus return_status = HighsStatus::kOk;
        HighsStatus call_status = assessLp(h.model_.lp_, options);
        assert(call_status == HighsStatus::kOk);
        return_status = interpretCallStatus(options.log_options, call_status,
                                            return_status, "assessLp");
        if (return_status == HighsStatus::kError) return st(return_status);
        if (checkOptions(options.log_options, options.records) !=
            OptionStatus::kOk)
          return st(HighsStatus::kError);
        return st(return_status);
      }
      case RunOp::kAssessSemiVariables:
        return st(assessSemiVariables(h.model_.lp_, options,
                                      *static_cast<bool*>(p)));
      case RunOp::kRelaxSemiVariables: {
        bool made = false;
        relaxSemiVariables(h.model_.lp_, made);
        return made;
      }
      case RunOp::kOkHessianDiagonal:
        return okHessianDiagonal(options, h.model_.hessian_,
                                 h.model_.lp_.sense_);
      case RunOp::kCallSolveQp:
        return st(h.callSolveQp());
      case RunOp::kCallSolveMip:
        return st(h.callSolveMip());
      case RunOp::kICrash:
        return iCrash();
      case RunOp::kBasisForSolution:
        return st(h.basisForSolution());
      case RunOp::kBasisClear:
        h.basis_.clear();
        return 0;
      case RunOp::kRefineBasis:
        refineBasis(h.model_.lp_, h.solution_, h.basis_);
        return 0;
      case RunOp::kCallSolveLp:
        return st(h.callSolveLp(lpOf(arg), std::string(m, len)));
      case RunOp::kSetEkkLpName:
        h.ekk_instance_.lp_name_.assign(m, len);
        return 0;
      case RunOp::kPrepareReducedLp: {
        HighsLp& reduced_lp = h.presolve_.getReducedProblem();
        reduced_lp.origin_name_ = "Reduced LP";
        reduced_lp.setMatrixDimensions();
        if (kAllowDeveloperAssert) {
          assert(assessLp(reduced_lp, options) == HighsStatus::kOk);
        } else {
          reduced_lp.a_matrix_.assessSmallValues(options.log_options,
                                                 options.small_matrix_value);
        }
        return st(cleanBounds(options, reduced_lp));
      }
      case RunOp::kEkkClear:
        h.ekk_instance_.clear();
        return 0;
      case RunOp::kEkkInvalidate:
        h.ekk_instance_.invalidate();
        return 0;
      case RunOp::kEkkPivotThreshold:
        if (!h.ekk_instance_.status_.initialised_for_solve) return 0;
        *static_cast<double*>(p) =
            h.ekk_instance_.info_.factor_pivot_threshold;
        return 1;
      case RunOp::kReducedToEmpty:
        h.solution_.clear();
        h.basis_.clear();
        h.basis_.debug_origin_name = "Presolve to empty";
        h.basis_.valid = true;
        h.basis_.alien = false;
        h.basis_.useful = true;
        h.basis_.was_alien = false;
        h.solution_.value_valid = true;
        h.solution_.dual_valid = true;
        return 0;
      case RunOp::kSaveOptions:
        saved_options.reset(new HighsOptions(options));
        return 0;
      case RunOp::kRestoreOptions:
        options = *saved_options;
        return 0;
      case RunOp::kOptionsPrimalSimplex:
        options.solver = "simplex";
        options.simplex_strategy = kSimplexStrategyPrimal;
        return 0;
      case RunOp::kOptionsCleanup:
        options.solver = kSimplexString;
        options.simplex_strategy = kSimplexStrategyChoose;
        options.simplex_min_concurrency = 1;
        options.simplex_max_concurrency = 1;
        if (arg) options.factor_pivot_threshold = *static_cast<double*>(p);
        return 0;
      case RunOp::kCopyToPresolve:
        h.presolve_.data_.recovered_solution_ = h.solution_;
        h.presolve_.data_.recovered_basis_ = h.basis_;
        return 0;
      case RunOp::kKktCheck:
        h.callLpKktCheck(lpOf(arg), std::string(m, len));
        return 0;
      case RunOp::kRecoveredValidity: {
        const HighsSolution& s = h.presolve_.data_.recovered_solution_;
        return int64_t(s.value_valid) | (int64_t(s.dual_valid) << 1);
      }
      case RunOp::kPostsolveUndo:
        h.presolve_.data_.postSolveStack.undo(
            options, h.presolve_.data_.recovered_solution_,
            h.presolve_.data_.recovered_basis_);
        assert(h.model_.lp_.a_matrix_.isColwise());
        calculateRowValuesQuad(h.model_.lp_,
                               h.presolve_.data_.recovered_solution_);
        if (arg && h.model_.lp_.sense_ == ObjSense::kMaximize)
          h.presolve_.negateReducedLpColDuals();
        return 0;
      case RunOp::kSetPostsolveStatus:
        h.presolve_.postsolve_status_ = HighsPostsolveStatus(arg);
        return 0;
      case RunOp::kTakeRecoveredSolution:
        h.solution_.clear();
        h.solution_ = h.presolve_.data_.recovered_solution_;
        return 0;
      case RunOp::kTakeRecoveredBasis:
        h.basis_.col_status = h.presolve_.data_.recovered_basis_.col_status;
        h.basis_.row_status = h.presolve_.data_.recovered_basis_.row_status;
        h.basis_.debug_origin_name += ": after postsolve";
        return 0;
      case RunOp::kDebugPostsolveSolution:
        return debugHighsSolution("After returning from postsolve", options,
                                  h.model_, h.solution_, h.basis_) ==
               HighsDebugStatus::kLogicalError;
      case RunOp::kUndoMods: {
        HighsStatus return_status = HighsStatus(arg);
        h.restoreInfCost(return_status);
        h.model_.lp_.unapplyMods();
        return st(return_status);
      }
      case RunOp::kDebugReturn:
        return debugReturn();
      case RunOp::kForceSolutionBasisSize:
        h.forceHighsSolutionBasisSize();
        return 0;
      case RunOp::kBasisConsistent:
        return debugHighsBasisConsistent(options, h.model_.lp_, h.basis_) !=
               HighsDebugStatus::kLogicalError;
      case RunOp::kRetainedEkkDataOk:
        return h.ekk_instance_.debugRetainedDataOk(h.model_.lp_) !=
               HighsDebugStatus::kLogicalError;
      case RunOp::kLpDimensionsOk:
        return lpDimensionsOk("returnFromHighs", h.model_.lp_,
                              options.log_options);
      case RunOp::kEkkFactorCompatible:
        if (!h.ekk_instance_.status_.has_nla) return -1;
        return h.ekk_instance_.lpFactorRowCompatible(h.model_.lp_.num_row_);
      case RunOp::kPresolveClear:
        h.presolve_.clear();
        return 0;
      case RunOp::kMipPresolve: {
        HighsMipSolver solver(h.callback_, options, h.model_.lp_, h.solution_);
        solver.setProfiling(h.profiling_);
        solver.timer_.start();
        solver.runMipPresolve(options.presolve_reduction_limit);
        const HighsPresolveStatus status = solver.getPresolveStatus();
        h.presolve_.data_.reduced_lp_ = solver.getPresolvedModel();
        h.presolve_.data_.postSolveStack = solver.getPostsolveStack();
        h.presolve_.presolve_status_ = status;
        return int64_t(status);
      }
      case RunOp::kPresolveInit:
        h.presolve_.init(h.model_.lp_, h.timer_);
        h.presolve_.options_ = &options;
        return 0;
      case RunOp::kPresolveRun:
        return int64_t(h.presolve_.run());
      case RunOp::kPresolveLog:
        h.presolve_log_ = h.presolve_.getPresolveLog();
        return int64_t(h.presolve_.presolve_status_);
      case RunOp::kPresolveRemoved: {
        const HighsInt* removed = static_cast<const HighsInt*>(p);
        h.presolve_.info_.n_cols_removed = removed[0];
        h.presolve_.info_.n_rows_removed = removed[1];
        h.presolve_.info_.n_nnz_removed = removed[2];
        if (arg) h.presolve_.getReducedProblem().clearScale();
        return 0;
      }
      case RunOp::kClearReducedIntegrality:
        h.presolve_.data_.reduced_lp_.integrality_.clear();
        return 0;
      case RunOp::kPresolveTime:
        (arg ? h.presolve_.info_.postsolve_time
             : h.presolve_.info_.presolve_time) = *static_cast<double*>(p);
        return 0;
      case RunOp::kLpView:
        *static_cast<RsLp*>(p) = rsLp(h.model_.lp_);
        return 0;
    }
    assert(false);
    return 0;
  }

  // The iCrash block of calledOptimizeModel: -2 for an error return
  // without returnFromOptimizeModel, otherwise a status to return with
  // it, or 0 (kOk) to carry on
  int64_t iCrash() {
    HighsOptions& options_ = h.options_;
    ICrashStrategy strategy = ICrashStrategy::kICA;
    bool strategy_ok = parseICrashStrategy(options_.icrash_strategy, strategy);
    if (!strategy_ok) {
      highsLogUser(options_.log_options, HighsLogType::kError,
                   "ICrash error: unknown strategy.\n");
      return -2;
    }
    ICrashOptions icrash_options{
        options_.icrash_dualize,         strategy,
        options_.icrash_starting_weight, options_.icrash_iterations,
        options_.icrash_approx_iter,     options_.icrash_exact,
        options_.icrash_breakpoints,     options_.log_options};
    HighsStatus icrash_status =
        callICrash(h.model_.lp_, icrash_options, h.icrash_info_);
    if (icrash_status != HighsStatus::kOk) return st(icrash_status);
    h.solution_.col_value = h.icrash_info_.x_values;
    HighsStatus crossover_status =
        callCrossover(options_, h.model_.lp_, h.basis_, h.solution_,
                      h.model_status_, h.info_, h.callback_);
    highsLogUser(options_.log_options, HighsLogType::kInfo,
                 "Crossover following iCrash has return status of %s, and "
                 "problem status is %s\n",
                 highsStatusToString(crossover_status).c_str(),
                 h.modelStatusToString(h.model_status_).c_str());
    if (crossover_status == HighsStatus::kError) return st(crossover_status);
    assert(options_.simplex_strategy == kSimplexStrategyPrimal);
    return 0;
  }

  // The debug checks of returnFromOptimizeModel: 1 for a logical error
  int64_t debugReturn() {
    HighsOptions& options_ = h.options_;
    bool error = false;
    if (h.solution_.value_valid &&
        debugPrimalSolutionRightSize(options_, h.model_.lp_, h.solution_) ==
            HighsDebugStatus::kLogicalError)
      error = true;
    if (h.solution_.dual_valid &&
        debugDualSolutionRightSize(options_, h.model_.lp_, h.solution_) ==
            HighsDebugStatus::kLogicalError)
      error = true;
    if (h.basis_.valid &&
        debugBasisRightSize(options_, h.model_.lp_, h.basis_) ==
            HighsDebugStatus::kLogicalError)
      error = true;
    if (h.solution_.value_valid &&
        debugHighsSolution("Return from optimizeModel()", options_, h.model_,
                           h.solution_, h.basis_, h.model_status_,
                           h.info_) == HighsDebugStatus::kLogicalError)
      error = true;
    if (debugInfo(options_, h.model_.lp_, h.basis_, h.solution_, h.info_,
                  h.model_status_) == HighsDebugStatus::kLogicalError)
      error = true;
    return error;
  }
};

HighsStatus Highs::calledOptimizeModel() {
  HighsRunRust r{*this, nullptr, nullptr};
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_called_optimize_model(&v));
  r.rethrow();
  return status;
}

HighsStatus Highs::returnFromOptimizeModel(const HighsStatus run_return_status,
                                           const bool undo_mods) {
  HighsRunRust r{*this, nullptr, nullptr};
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_return_from_optimize_model(
      &v, int(run_return_status), undo_mods));
  r.rethrow();
  return status;
}

HighsStatus Highs::returnFromHighs(HighsStatus highs_return_status) {
  HighsRunRust r{*this, nullptr, nullptr};
  const RsHighs v = r.view();
  const HighsStatus status =
      HighsStatus(highs_rs_return_from_highs(&v, int(highs_return_status)));
  r.rethrow();
  return status;
}

HighsPresolveStatus Highs::runPresolve(const bool force_lp_presolve,
                                       const bool force_presolve) {
  HighsRunRust r{*this, nullptr, nullptr};
  const RsHighs v = r.view();
  const HighsPresolveStatus status = HighsPresolveStatus(
      highs_rs_run_presolve(&v, force_lp_presolve, force_presolve));
  r.rethrow();
  return status;
}

HighsPostsolveStatus Highs::runPostsolve() {
  HighsRunRust r{*this, nullptr, nullptr};
  const RsHighs v = r.view();
  const HighsPostsolveStatus status =
      HighsPostsolveStatus(highs_rs_run_postsolve(&v));
  r.rethrow();
  return status;
}

// The basis and tableau queries and setSolution (rust/src/lp_data/query.rs)
extern "C" {
int highs_rs_check_query(const RsLog* log, const char* method,
                         size_t method_len, const char* null_arg,
                         size_t null_len, const char* index_kind,
                         size_t index_kind_len, HighsInt index, HighsInt dim,
                         bool has_invert);
HighsInt highs_rs_reduced_row(RsMut<HighsInt> start, RsMut<HighsInt> index,
                              RsMut<double> value, RsMut<double> binv_row,
                              RsMut<double> row, RsMut<HighsInt> indices);
HighsInt highs_rs_extract_solve(HighsInt count, RsMut<HighsInt> index,
                                RsMut<double> array, RsMut<double> solution,
                                RsMut<HighsInt> indices);
int highs_rs_check_sparse_solution(const RsLog* log, RsMut<HighsInt> index,
                                   RsMut<double> value, RsMut<double> lower,
                                   RsMut<double> upper, double pft);
int highs_rs_new_solution_parts(const RsLog* log, HighsInt num_col,
                                HighsInt num_row, size_t col_value_size,
                                size_t row_dual_size);
}

namespace {
// The checks of a query: a NULL argument, an index out of range, no INVERT
HighsStatus checkQuery(const HighsOptions& options, const char* method,
                       const char* null_arg, const char* index_kind,
                       const HighsInt index, const HighsInt dim,
                       const bool has_invert) {
  const RsLog log = rsLog(options.log_options);
  auto n = [](const char* s) { return s ? strlen(s) : 0; };
  return HighsStatus(highs_rs_check_query(
      &log, method, n(method), null_arg, n(null_arg), index_kind,
      n(index_kind), index, dim, has_invert));
}

template <typename T>
RsMut<T> rsOut(T* p, const size_t n) {
  return {p, p ? n : 0};
}
}  // namespace

HighsStatus Highs::getBasisInverseRow(const HighsInt row, double* row_vector,
                                      HighsInt* row_num_nz,
                                      HighsInt* row_indices) {
  const HighsInt num_row = model_.lp_.num_row_;
  if (checkQuery(options_, "getBasisInverseRow",
                 row_vector ? nullptr : "row_vector", "Row", row, num_row,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  vector<double> rhs(num_row, 0);
  rhs[row] = 1;
  basisSolveInterface(rhs, row_vector, row_num_nz, row_indices, true);
  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisInverseCol(const HighsInt col, double* col_vector,
                                      HighsInt* col_num_nz,
                                      HighsInt* col_indices) {
  const HighsInt num_row = model_.lp_.num_row_;
  if (checkQuery(options_, "getBasisInverseCol",
                 col_vector ? nullptr : "col_vector", "Column", col, num_row,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  vector<double> rhs(num_row, 0);
  rhs[col] = 1;
  basisSolveInterface(rhs, col_vector, col_num_nz, col_indices, false);
  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisSolve(const double* Xrhs, double* solution_vector,
                                 HighsInt* solution_num_nz,
                                 HighsInt* solution_indices) {
  const char* null_arg = !Xrhs              ? "Xrhs"
                         : !solution_vector ? "solution_vector"
                                            : nullptr;
  if (checkQuery(options_, "getBasisSolve", null_arg, nullptr, 0, 0,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  const HighsInt num_row = model_.lp_.num_row_;
  vector<double> rhs(Xrhs, Xrhs + num_row);
  basisSolveInterface(rhs, solution_vector, solution_num_nz, solution_indices,
                      false);
  return HighsStatus::kOk;
}

HighsStatus Highs::getBasisTransposeSolve(const double* Xrhs,
                                          double* solution_vector,
                                          HighsInt* solution_num_nz,
                                          HighsInt* solution_indices) {
  const char* null_arg = !Xrhs              ? "Xrhs"
                         : !solution_vector ? "solution_vector"
                                            : nullptr;
  if (checkQuery(options_, "getBasisTransposeSolve", null_arg, nullptr, 0, 0,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  const HighsInt num_row = model_.lp_.num_row_;
  vector<double> rhs(Xrhs, Xrhs + num_row);
  basisSolveInterface(rhs, solution_vector, solution_num_nz, solution_indices,
                      true);
  return HighsStatus::kOk;
}

HighsStatus Highs::getReducedRow(const HighsInt row, double* row_vector,
                                 HighsInt* row_num_nz, HighsInt* row_indices,
                                 const double* pass_basis_inverse_row_vector) {
  HighsLp& lp = model_.lp_;
  lp.ensureColwise();
  if (checkQuery(options_, "getReducedRow",
                 row_vector ? nullptr : "row_vector", "Row", row, lp.num_row_,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  const HighsInt num_row = lp.num_row_;
  vector<double> basis_inverse_row;
  const double* basis_inverse_row_vector = pass_basis_inverse_row_vector;
  if (basis_inverse_row_vector == NULL) {
    vector<double> rhs(num_row, 0);
    rhs[row] = 1;
    basis_inverse_row.resize(num_row, 0);
    basisSolveInterface(rhs, basis_inverse_row.data(), NULL, NULL, true);
    basis_inverse_row_vector = basis_inverse_row.data();
  }
  const HighsInt num_nz = highs_rs_reduced_row(
      rsMut(lp.a_matrix_.start_), rsMut(lp.a_matrix_.index_),
      rsMut(lp.a_matrix_.value_),
      {const_cast<double*>(basis_inverse_row_vector), size_t(num_row)},
      {row_vector, size_t(lp.num_col_)},
      rsOut(row_num_nz ? row_indices : nullptr, size_t(lp.num_col_)));
  if (row_num_nz) *row_num_nz = num_nz;
  return HighsStatus::kOk;
}

HighsStatus Highs::getReducedColumn(const HighsInt col, double* col_vector,
                                    HighsInt* col_num_nz,
                                    HighsInt* col_indices) {
  HighsLp& lp = model_.lp_;
  lp.ensureColwise();
  if (checkQuery(options_, "getReducedColumn",
                 col_vector ? nullptr : "col_vector", "Column", col,
                 lp.num_col_,
                 ekk_instance_.status_.has_invert) != HighsStatus::kOk)
    return HighsStatus::kError;
  vector<double> rhs(lp.num_row_, 0);
  for (HighsInt el = lp.a_matrix_.start_[col];
       el < lp.a_matrix_.start_[col + 1]; el++)
    rhs[lp.a_matrix_.index_[el]] = lp.a_matrix_.value_[el];
  basisSolveInterface(rhs, col_vector, col_num_nz, col_indices, false);
  return HighsStatus::kOk;
}

HighsStatus Highs::basisSolveInterface(const vector<double>& rhs,
                                       double* solution_vector,
                                       HighsInt* solution_num_nz,
                                       HighsInt* solution_indices,
                                       bool transpose) {
  HighsLp& lp = model_.lp_;
  const HighsInt num_row = lp.num_row_;
  if (num_row == 0) return HighsStatus::kOk;
  assert(ekk_instance_.status_.has_invert);
  ekk_instance_.setNlaPointersForLpAndScale(lp);
  assert(!lp.is_moved_);
  HVector solve_vector;
  solve_vector.setup(num_row);
  solve_vector.clear();
  HighsInt rhs_num_nz = 0;
  for (HighsInt iRow = 0; iRow < num_row; iRow++) {
    if (rhs[iRow]) {
      solve_vector.index[rhs_num_nz++] = iRow;
      solve_vector.array[iRow] = rhs[iRow];
    }
  }
  solve_vector.count = rhs_num_nz;
  const double expected_density = 1;
  if (transpose) {
    ekk_instance_.btran(solve_vector, expected_density);
  } else {
    ekk_instance_.ftran(solve_vector, expected_density);
  }
  const HighsInt num_nz = highs_rs_extract_solve(
      solve_vector.count, rsMut(solve_vector.index), rsMut(solve_vector.array),
      {solution_vector, size_t(num_row)},
      rsOut(solution_indices, size_t(num_row)));
  if (num_nz >= 0) *solution_num_nz = num_nz;
  return HighsStatus::kOk;
}

HighsStatus Highs::setSolution(const HighsSolution& solution) {
  HighsStatus return_status = HighsStatus::kOk;
  const RsLog log = rsLog(options_.log_options);
  const int parts = highs_rs_new_solution_parts(
      &log, model_.lp_.num_col_, model_.lp_.num_row_,
      solution.col_value.size(), solution.row_dual.size());
  const bool new_primal_solution = parts & 1;
  const bool new_dual_solution = parts & 2;
  if (parts) {
    invalidateSolverData();
  } else {
    return_status = HighsStatus::kError;
  }
  if (new_primal_solution) {
    solution_.col_value = solution.col_value;
    if (model_.lp_.num_row_ > 0) {
      solution_.row_value.resize(model_.lp_.num_row_);
      model_.lp_.a_matrix_.ensureColwise();
      return_status = interpretCallStatus(
          options_.log_options, calculateRowValuesQuad(model_.lp_, solution_),
          return_status, "calculateRowValuesQuad");
      if (return_status == HighsStatus::kError) return return_status;
    }
    solution_.value_valid = true;
  }
  if (new_dual_solution) {
    solution_.row_dual = solution.row_dual;
    if (model_.lp_.num_col_ > 0) {
      solution_.col_dual.resize(model_.lp_.num_col_);
      model_.lp_.a_matrix_.ensureColwise();
      return_status = interpretCallStatus(
          options_.log_options, calculateColDualsQuad(model_.lp_, solution_),
          return_status, "calculateColDuals");
      if (return_status == HighsStatus::kError) return return_status;
    }
    solution_.dual_valid = true;
  }
  return returnFromHighs(return_status);
}

HighsStatus Highs::setSolution(const HighsInt num_entries,
                               const HighsInt* index, const double* value) {
  if (model_.lp_.num_col_ == 0) return HighsStatus::kOk;
  const RsLog log = rsLog(options_.log_options);
  const HighsStatus return_status =
      HighsStatus(highs_rs_check_sparse_solution(
          &log, {const_cast<HighsInt*>(index), size_t(num_entries)},
          {const_cast<double*>(value), size_t(num_entries)},
          rsMut(model_.lp_.col_lower_), rsMut(model_.lp_.col_upper_),
          options_.primal_feasibility_tolerance));
  if (return_status == HighsStatus::kError) return return_status;
  HighsSolution new_solution;
  new_solution.col_value.assign(model_.lp_.num_col_, kHighsUndefined);
  for (HighsInt iX = 0; iX < num_entries; iX++)
    new_solution.col_value[index[iX]] = value[iX];
  return interpretCallStatus(options_.log_options, setSolution(new_solution),
                             return_status, "setSolution");
}
#endif
