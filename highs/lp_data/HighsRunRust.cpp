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
#include "io/Filereader.h"
#include "lp_data/HighsInfoDebug.h"
#include "lp_data/HighsModelUtils.h"
#include "lp_data/HighsSolutionDebug.h"
#include "lp_data/HighsSolution.h"
#include "mip/HighsMipSolver.h"
#include "model/HighsHessianUtils.h"
#include "presolve/ICrashX.h"
#include "simplex/HSimplex.h"
#include "util/HighsMatrixPic.h"

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
  HighsInt simplex_strategy;
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
  // drivers.rs
  kMipRun,
  kMipTakeSolution,
  kActiveModifiedUpperBounds,
  kSwapPrimalTolerance,
  kKktFailures,
  kMipFinish,
  kSolutionHasUndefined,
  kSolutionFeasible,
  kSaveColBounds,
  kClearIntegrality,
  kSolutionClear,
  kSwapMipMaxNodes,
  kOptimizeModel,
  kSolutionView,
  kRayRecord,
  kFeasibilityProblem,
  kUnboundednessProblem,
  kHighsRun,
  kCopyRay,
  kComputeDualRay,
  kComputePrimalRay,
  kNeedsMods,
  kReportModelStats,
  kClearPresolve,
  kInitializeMultiThreading,
  kPresolveProfiled,
  kReportPresolveReductions,
  kPresolvedModel,
  kCrossover,
  kPostsolveArgs,
  kPostsolveBasisConsistent,
  kPostsolveSetSolution,
  kPostsolveKkt,
  kPostsolveTakeRecovered,
  kOptionsPostsolveCleanup,
  kSetBasis,
  kSetBasisOrigin,
  kBasisDebug,
  kNewHighsBasis,
  kHessianDims,
  kAssessHessian,
  kHessianClear,
  kCompleteHessian,
  kLogHeader,
  kClearModel,
  kTakeModel,
  kEmptyMatrix,
  kFormatOk,
  kPrepareModelLp,
  kAssessLp,
  kMatrixImages,
  kClearSolver2,
  kTakeHessian,
  kReadModelFile,
  kReadModelPass,
  kReadBasis,
  kWriteModelPrepare,
  kWriteModelLpView,
  kWriteModelCheck,
  kReportWrittenModel,
  kWriteModelFile,
  kWriteBasis,
  kSolutionBasisSizes,
};

// drivers.rs: BasisDebug
struct RsBasisDebug {
  HighsInt id, update_count;
  RsRunStr origin;
};

// drivers.rs: PostsolveArgs
struct RsPostsolveArgs {
  int64_t col_value_size, col_dual_size, row_dual_size;
  bool dual_valid;
  int64_t basis_col_size, basis_row_size;
  bool basis_valid;
};

// drivers.rs: MipResult
struct RsMipResult {
  HighsInt model_status;
  double solution_objective;
  int64_t node_count, total_lp_iterations;
  double dual_bound, gap, primal_dual_integral, row_violation,
      bound_violation, integrality_violation;
};

// drivers.rs: RayRecord
struct RsRayRecord {
  HighsInt index, sign;
  int64_t value_size;
  bool has_invert;
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
  explicit HighsRunRust(Highs& highs) : h(highs) {}
  Highs& h;
  // The options saved by kSaveOptions
  std::unique_ptr<HighsOptions> saved_options;
  // An exception thrown by a step (a cancelled task's HighsTask::Interrupt),
  // rethrown once Rust has returned; no step is made after it
  std::exception_ptr pending;
  // The state of the drivers (drivers.rs) between steps
  HighsLp mip_lp;
  std::unique_ptr<HighsMipSolver> mip_solver;
  double saved_tolerance = 0;
  HighsInt saved_mip_max_nodes = 0;
  std::vector<double> saved_lower, saved_upper, saved_cost;
  std::vector<HighsVarType> saved_integrality;
  HighsHessian saved_hessian;
  std::string saved_presolve;
  bool saved_solve_relaxation = false;
  bool saved_allow_unbounded_or_infeasible = false;
  const HighsSolution* user_solution = nullptr;
  const HighsBasis* user_basis = nullptr;
  HighsModel* user_model = nullptr;
  HighsHessian* user_hessian = nullptr;
  HighsModel read_model;
  HighsBasis read_basis;
  FILE* write_file = nullptr;

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
    v.o.simplex_strategy = o.simplex_strategy;
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
      default:
        return driverStep(which, arg, p, m, len);
    }
    assert(false);
    return 0;
  }

  // The steps of drivers.rs
  int64_t driverStep(RunOp which, int64_t arg, void* p, const char* m,
                     size_t len) {
    HighsOptions& options = h.options_;
    HighsLp& lp = h.model_.lp_;
    switch (which) {
      case RunOp::kMipRun: {
        const bool user_solution = h.solution_.value_valid;
        std::vector<double> user_col_value, user_row_value;
        if (user_solution) {
          user_col_value = std::move(h.solution_.col_value);
          user_row_value = std::move(h.solution_.row_value);
        }
        h.invalidateSolverData();
        if (user_solution) {
          h.solution_.col_value = std::move(user_col_value);
          h.solution_.row_value = std::move(user_row_value);
          h.solution_.value_valid = true;
        }
        const HighsInt log_dev_level = options.log_dev_level;
        assert(lp.a_matrix_.format_ != MatrixFormat::kRowwise);
        const bool has_semi_variables = lp.hasSemiVariables();
        if (has_semi_variables)
          mip_lp = withoutSemiVariables(lp, h.solution_,
                                        options.primal_feasibility_tolerance);
        mip_solver.reset(new HighsMipSolver(
            h.callback_, options, has_semi_variables ? mip_lp : lp,
            h.solution_));
        HighsMipSolver& solver = *mip_solver;
        solver.setProfiling(h.profiling_);
        h.profiling_->start(kSubSolverMip);
        solver.run();
        h.profiling_->stop(kSubSolverMip);
        options.log_dev_level = log_dev_level;
        RsMipResult& r = *static_cast<RsMipResult*>(p);
        r.model_status = HighsInt(solver.modelstatus_);
        r.solution_objective = solver.solution_objective_;
        r.node_count = solver.node_count_;
        r.total_lp_iterations = solver.total_lp_iterations_;
        r.dual_bound = solver.dual_bound_;
        r.gap = solver.gap_;
        r.primal_dual_integral = solver.primal_dual_integral_;
        r.row_violation = solver.row_violation_;
        r.bound_violation = solver.bound_violation_;
        r.integrality_violation = solver.integrality_violation_;
        return 0;
      }
      case RunOp::kMipTakeSolution:
        h.solution_.col_value = mip_solver->solution_;
        h.saved_objective_and_solution_ =
            mip_solver->saved_objective_and_solution_;
        lp.a_matrix_.productQuad(h.solution_.row_value, h.solution_.col_value);
        h.solution_.value_valid = true;
        return 0;
      case RunOp::kActiveModifiedUpperBounds:
        return activeModifiedUpperBounds(options, lp, h.solution_.col_value);
      case RunOp::kSwapPrimalTolerance:
        if (arg == 0) {
          saved_tolerance = options.primal_feasibility_tolerance;
          options.primal_feasibility_tolerance = *static_cast<double*>(p);
        } else {
          options.primal_feasibility_tolerance = saved_tolerance;
        }
        return 0;
      case RunOp::kKktFailures:
        getKktFailures(options, h.model_, h.solution_, h.basis_, h.info_);
        return 0;
      case RunOp::kMipFinish:
        mip_solver.reset();
        return 0;
      case RunOp::kSolutionHasUndefined:
        return h.solution_.hasUndefined();
      case RunOp::kSolutionFeasible: {
        bool valid, integral, feasible;
        HighsStatus status = assessLpPrimalSolution(
            "", options, lp, h.solution_, valid, integral, feasible);
        assert(status != HighsStatus::kError);
        (void)status;
        return feasible;
      }
      case RunOp::kSaveColBounds:
        if (arg == 0) {
          saved_lower = lp.col_lower_;
          saved_upper = lp.col_upper_;
          saved_integrality = lp.integrality_;
        } else {
          lp.col_lower_ = saved_lower;
          lp.col_upper_ = saved_upper;
          lp.integrality_ = saved_integrality;
        }
        return 0;
      case RunOp::kClearIntegrality:
        lp.integrality_.clear();
        return 0;
      case RunOp::kSolutionClear:
        h.solution_.clear();
        return 0;
      case RunOp::kSwapMipMaxNodes:
        if (arg == 0) {
          saved_mip_max_nodes = options.mip_max_nodes;
          options.mip_max_nodes = options.mip_max_start_nodes;
        } else {
          options.mip_max_nodes = saved_mip_max_nodes;
        }
        return 0;
      case RunOp::kOptimizeModel: {
        if (h.profiling_) assert(!h.profiling_->isSubMip());
        const HighsStatus status = h.optimizeModel();
        if (h.profiling_) h.resetProfiling();
        return st(status);
      }
      case RunOp::kSolutionView:
        *static_cast<RsSolution*>(p) = rsSolution(h.solution_);
        return 0;
      case RunOp::kRayRecord: {
        const HighsRayRecord& record = arg ? h.ekk_instance_.primal_ray_record_
                                           : h.ekk_instance_.dual_ray_record_;
        RsRayRecord& r = *static_cast<RsRayRecord*>(p);
        r.index = record.index;
        r.sign = record.sign;
        r.value_size = record.value.size();
        r.has_invert = h.ekk_instance_.status_.has_invert;
        return 0;
      }
      case RunOp::kFeasibilityProblem: {
        const bool is_qp = arg & 1;
        if (arg < 2) {
          saved_cost = lp.col_cost_;
          if (is_qp) saved_hessian = h.model_.hessian_;
          h.getOptionValue("presolve", saved_presolve);
          h.getOptionValue("solve_relaxation", saved_solve_relaxation);
          std::vector<double> zero_costs;
          zero_costs.assign(lp.num_col_, 0);
          HighsRayRecord primal_ray_record =
              h.ekk_instance_.primal_ray_record_.getRayRecord();
          HighsStatus status =
              h.changeColsCost(0, lp.num_col_ - 1, zero_costs.data());
          assert(status == HighsStatus::kOk);
          (void)status;
          h.ekk_instance_.primal_ray_record_.setRayRecord(primal_ray_record);
          if (is_qp) {
            HighsHessian zero_hessian;
            h.passHessian(zero_hessian);
          }
          h.setOptionValue("presolve", kHighsOffString);
          h.setOptionValue("solve_relaxation", true);
        } else {
          lp.col_cost_ = saved_cost;
          if (is_qp) h.model_.hessian_ = saved_hessian;
          h.setOptionValue("presolve", saved_presolve);
          h.setOptionValue("solve_relaxation", saved_solve_relaxation);
        }
        return 0;
      }
      case RunOp::kUnboundednessProblem:
        if (arg == 0) {
          h.getOptionValue("presolve", saved_presolve);
          h.getOptionValue("solve_relaxation", saved_solve_relaxation);
          h.getOptionValue("allow_unbounded_or_infeasible",
                           saved_allow_unbounded_or_infeasible);
          h.setOptionValue("presolve", kHighsOffString);
          h.setOptionValue("solve_relaxation", true);
          h.setOptionValue("allow_unbounded_or_infeasible", false);
        } else {
          h.setOptionValue("presolve", saved_presolve);
          h.setOptionValue("solve_relaxation", saved_solve_relaxation);
          h.setOptionValue("allow_unbounded_or_infeasible",
                           saved_allow_unbounded_or_infeasible);
        }
        return 0;
      case RunOp::kHighsRun:
        return st(h.run());
      case RunOp::kCopyRay: {
        double* value = static_cast<double*>(p);
        const std::vector<double>& ray =
            arg ? h.ekk_instance_.primal_ray_record_.value
                : h.ekk_instance_.dual_ray_record_.value;
        const HighsInt n = arg ? lp.num_col_ : lp.num_row_;
        for (HighsInt i = 0; i < n; i++) value[i] = ray[i];
        return 0;
      }
      case RunOp::kComputeDualRay: {
        double* dual_ray_value = static_cast<double*>(p);
        const HighsInt num_row = lp.num_row_;
        std::vector<double> rhs;
        HighsInt iRow = h.ekk_instance_.dual_ray_record_.index;
        rhs.assign(num_row, 0);
        rhs[iRow] = h.ekk_instance_.dual_ray_record_.sign;
        HighsInt* dual_ray_num_nz = 0;
        h.basisSolveInterface(rhs, dual_ray_value, dual_ray_num_nz, NULL, true);
        h.ekk_instance_.dual_ray_record_.value.resize(num_row);
        for (HighsInt i = 0; i < num_row; i++)
          h.ekk_instance_.dual_ray_record_.value[i] = dual_ray_value[i];
        return 0;
      }
      case RunOp::kComputePrimalRay: {
        double* primal_ray_value = static_cast<double*>(p);
        const HighsInt num_row = lp.num_row_;
        const HighsInt num_col = lp.num_col_;
        HighsInt col = h.ekk_instance_.primal_ray_record_.index;
        assert(h.ekk_instance_.basis_.nonbasicFlag_[col] == kNonbasicFlagTrue);
        std::vector<double> rhs;
        std::vector<double> column;
        column.assign(num_row, 0);
        rhs.assign(num_row, 0);
        lp.ensureColwise();
        HighsInt primal_ray_sign = h.ekk_instance_.primal_ray_record_.sign;
        if (col < num_col) {
          for (HighsInt iEl = lp.a_matrix_.start_[col];
               iEl < lp.a_matrix_.start_[col + 1]; iEl++)
            rhs[lp.a_matrix_.index_[iEl]] =
                primal_ray_sign * lp.a_matrix_.value_[iEl];
        } else {
          rhs[col - num_col] = primal_ray_sign;
        }
        HighsInt* column_num_nz = 0;
        h.basisSolveInterface(rhs, column.data(), column_num_nz, NULL, false);
        for (HighsInt iCol = 0; iCol < num_col; iCol++)
          primal_ray_value[iCol] = 0;
        for (HighsInt iRow = 0; iRow < num_row; iRow++) {
          HighsInt iCol = h.ekk_instance_.basis_.basicIndex_[iRow];
          if (iCol < num_col) primal_ray_value[iCol] = column[iRow];
        }
        if (col < num_col) primal_ray_value[col] = -primal_ray_sign;
        h.ekk_instance_.primal_ray_record_.value.resize(num_col);
        for (HighsInt iCol = 0; iCol < num_col; iCol++)
          h.ekk_instance_.primal_ray_record_.value[iCol] =
              primal_ray_value[iCol];
        return 0;
      }
      case RunOp::kNeedsMods:
        return h.model_.needsMods(options.infinite_cost);
      case RunOp::kReportModelStats:
        h.reportModelStats();
        return 0;
      case RunOp::kClearPresolve:
        h.clearPresolve();
        return 0;
      case RunOp::kInitializeMultiThreading:
        return st(h.initializeMultiThreading());
      case RunOp::kPresolveProfiled: {
        HighsProfiling profiling;
        const bool already_profiling = h.profiling_;
        if (!already_profiling) h.initializeProfiling(&profiling);
        const HighsPresolveStatus status =
            h.runPresolve(options.solve_relaxation, true);
        if (!already_profiling) h.clearProfiling();
        return int64_t(status);
      }
      case RunOp::kReportPresolveReductions:
        reportPresolveReductions(options.log_options, h.model_presolve_status_,
                                 lp, h.presolve_.getReducedProblem());
        return 0;
      case RunOp::kPresolvedModel:
        if (arg == 0) {
          h.presolved_model_ = h.model_;
        } else {
          h.presolved_model_.lp_ = h.presolve_.getReducedProblem();
          h.presolved_model_.lp_.setMatrixDimensions();
        }
        return 0;
      case RunOp::kCrossover:
        if (arg == 0) {
          h.solution_ = *user_solution;
          return st(callCrossover(options, lp, h.basis_, h.solution_,
                                  h.model_status_, h.info_, h.callback_));
        }
        h.info_.objective_function_value =
            lp.objectiveValue(h.solution_.col_value);
        getLpKktFailures(options, lp, h.solution_, h.basis_, h.info_);
        return 0;
      case RunOp::kPostsolveArgs: {
        RsPostsolveArgs& a = *static_cast<RsPostsolveArgs*>(p);
        a.col_value_size = user_solution->col_value.size();
        a.col_dual_size = user_solution->col_dual.size();
        a.row_dual_size = user_solution->row_dual.size();
        a.dual_valid = user_solution->dual_valid;
        a.basis_col_size = user_basis->col_status.size();
        a.basis_row_size = user_basis->row_status.size();
        a.basis_valid = user_basis->valid;
        return 0;
      }
      case RunOp::kPostsolveBasisConsistent:
        return isBasisConsistent(h.presolve_.getReducedProblem(), *user_basis);
      case RunOp::kPostsolveSetSolution: {
        HighsSolution& recovered = h.presolve_.data_.recovered_solution_;
        if (arg == 0) {
          recovered = *user_solution;
          recovered.row_value.assign(h.presolve_.getReducedProblem().num_row_,
                                     0);
          recovered.value_valid = true;
        } else if (arg == 1) {
          recovered.dual_valid = false;
          recovered.col_dual.clear();
          recovered.row_dual.clear();
          h.presolve_.data_.recovered_basis_.valid = false;
        } else {
          recovered.dual_valid = (arg - 2) & 1;
          h.presolve_.data_.recovered_basis_ = *user_basis;
          h.presolve_.data_.recovered_basis_.valid = (arg - 2) & 2;
        }
        return 0;
      }
      case RunOp::kPostsolveKkt:
        if (arg)
          h.info_.objective_function_value =
              computeObjectiveValue(lp, h.solution_);
        assert(!h.model_.isQp());
        getKktFailures(options, false, lp, lp.col_cost_, h.solution_, h.info_,
                       true);
        return 0;
      case RunOp::kPostsolveTakeRecovered:
        h.solution_.clear();
        h.solution_ = h.presolve_.data_.recovered_solution_;
        assert(h.solution_.value_valid);
        if (!h.solution_.dual_valid) {
          h.solution_.col_dual.assign(lp.num_col_, 0);
          h.solution_.row_dual.assign(lp.num_row_, 0);
        }
        h.basis_ = h.presolve_.data_.recovered_basis_;
        h.basis_.debug_origin_name += ": after postsolve";
        return 0;
      case RunOp::kOptionsPostsolveCleanup:
        options.simplex_strategy = kSimplexStrategyChoose;
        options.simplex_min_concurrency = 1;
        options.simplex_max_concurrency = 1;
        return 0;
      case RunOp::kSetBasis: {
        const HighsBasis& basis = *user_basis;
        switch (arg) {
          case 0:
            for (HighsInt iCol = 0; iCol < lp.num_col_; iCol++)
              h.basis_.col_status[iCol] =
                  basis.col_status[iCol] == HighsBasisStatus::kBasic
                      ? HighsBasisStatus::kNonbasic
                      : basis.col_status[iCol];
            h.basis_.alien = false;
            return 0;
          case 1: {
            int64_t* sizes = static_cast<int64_t*>(p);
            sizes[0] = h.basis_.col_status.size();
            sizes[1] = h.basis_.row_status.size();
            sizes[2] = lp.num_col_;
            sizes[3] = lp.num_row_;
            return isBasisRightSize(lp, basis);
          }
          case 2: {
            HighsBasis modifiable_basis = basis;
            modifiable_basis.was_alien = true;
            HighsProfiling profiling;
            const bool already_profiling = h.profiling_;
            if (!already_profiling)
              h.initializeSingleThreadedProfiling(&profiling);
            HighsLpSolverObject solver_object(lp, modifiable_basis, h.solution_,
                                              h.info_, h.ekk_instance_,
                                              h.callback_, options, h.timer_);
            solver_object.setProfiling(h.profiling_);
            HighsStatus return_status =
                formSimplexLpBasisAndFactor(solver_object);
            if (!already_profiling) h.clearProfiling();
            if (return_status != HighsStatus::kOk) return st(return_status);
            h.basis_ = std::move(modifiable_basis);
            return 0;
          }
          case 3:
            return isBasisConsistent(lp, basis);
          default:
            h.basis_ = basis;
            return 0;
        }
      }
      case RunOp::kSetBasisOrigin:
        h.basis_.debug_origin_name.assign(m, len);
        return 0;
      case RunOp::kBasisDebug: {
        RsBasisDebug& d = *static_cast<RsBasisDebug*>(p);
        d.id = h.basis_.debug_id;
        d.update_count = h.basis_.debug_update_count;
        d.origin = rsRunStr(h.basis_.debug_origin_name);
        return 0;
      }
      case RunOp::kNewHighsBasis:
        h.newHighsBasis();
        return 0;
      case RunOp::kHessianDims: {
        HighsInt* d = static_cast<HighsInt*>(p);
        d[0] = h.model_.hessian_.dim_;
        d[1] = d[0] ? h.model_.hessian_.numNz() : 0;
        return 0;
      }
      case RunOp::kAssessHessian:
        return st(assessHessian(h.model_.hessian_, options));
      case RunOp::kHessianClear:
        h.model_.hessian_.clear();
        return 0;
      case RunOp::kCompleteHessian:
        completeHessian(lp.num_col_, h.model_.hessian_);
        return 0;
      case RunOp::kLogHeader:
        h.logHeader();
        return 0;
      case RunOp::kClearModel:
        h.clearModel();
        return 0;
      case RunOp::kTakeModel:
        lp = std::move(user_model->lp_);
        h.model_.hessian_ = std::move(user_model->hessian_);
        lp.origin_name_ = "Original";
        assert(lp.a_matrix_.formatOk());
        return 0;
      case RunOp::kEmptyMatrix:
        lp.a_matrix_.format_ = MatrixFormat::kColwise;
        lp.a_matrix_.start_.assign(lp.num_col_ + 1, 0);
        lp.a_matrix_.index_.clear();
        lp.a_matrix_.value_.clear();
        return 0;
      case RunOp::kFormatOk:
        return arg ? h.model_.hessian_.formatOk() : lp.a_matrix_.formatOk();
      case RunOp::kPrepareModelLp:
        lp.setMatrixDimensions();
        assert(!lp.is_scaled_);
        assert(!lp.is_moved_);
        lp.resetScale();
        return 0;
      case RunOp::kAssessLp:
        return st(assessLp(lp, options));
      case RunOp::kMatrixImages:
        if (options.write_matrix_image)
          writeLpMatrixPicToFile(options, "LpMatrix", lp);
        if (options.write_hessian_image)
          writeHessianPicToFile(options, "Hessian", h.model_.hessian_);
        return 0;
      case RunOp::kClearSolver2:
        return st(h.clearSolver());
      case RunOp::kTakeHessian:
        h.model_.hessian_ = std::move(*user_hessian);
        return 0;
      case RunOp::kReadModelFile: {
        const std::string filename(m, len);
        Filereader* reader =
            Filereader::getFilereader(options.log_options, filename);
        if (reader == NULL) return -1;
        FilereaderRetcode call_code =
            reader->readModelFromFile(options, filename, read_model);
        delete reader;
        return int64_t(call_code);
      }
      case RunOp::kReadModelPass:
        if (arg == 0) {
          read_model.lp_.model_name_.assign(m, len);
          return 0;
        }
        return st(h.passModel(std::move(read_model)));
      case RunOp::kReadBasis:
        if (arg == 0) {
          read_basis = h.basis_;
          return st(readBasisFile(options.log_options, lp, read_basis,
                                  std::string(m, len)));
        } else if (arg == 1) {
          return isBasisConsistent(lp, read_basis);
        }
        h.basis_ = read_basis;
        h.basis_.valid = true;
        h.basis_.useful = true;
        h.newHighsBasis();
        return 0;
      case RunOp::kWriteModelPrepare: {
        HighsLp& model_lp = user_model->lp_;
        model_lp.setMatrixDimensions();
        const HighsStatus call_status = normaliseNames(
            options.log_options, model_lp, HighsFileType(arg));
        assert(call_status != HighsStatus::kError);
        model_lp.ensureColwise();
        return st(call_status);
      }
      case RunOp::kWriteModelLpView:
        *static_cast<RsLp*>(p) = rsLp(user_model->lp_);
        return 0;
      case RunOp::kWriteModelCheck: {
        HighsModel& model = *user_model;
        switch (arg) {
          case 0:
            return model.hessian_.dim_ > 0
                       ? st(assessHessianDimensions(options, model.hessian_))
                       : 0;
          case 1:
            return st(model.lp_.a_matrix_.assessStart(options.log_options));
          case 2:
            return st(
                model.lp_.a_matrix_.assessIndexBounds(options.log_options));
          case 3:
            return model.lp_.col_hash_.hasDuplicate(model.lp_.col_names_);
          default:
            return model.lp_.row_hash_.hasDuplicate(model.lp_.row_names_);
        }
      }
      case RunOp::kReportWrittenModel:
        h.reportModel(*user_model);
        return 0;
      case RunOp::kWriteModelFile: {
        const std::string filename(m, len);
        Filereader* writer =
            Filereader::getFilereader(options.log_options, filename);
        if (arg == 0) {
          delete writer;
          return writer != NULL;
        }
        const HighsStatus status =
            writer->writeModelToFile(options, filename, *user_model);
        delete writer;
        return st(status);
      }
      case RunOp::kWriteBasis:
        if (arg == 0) {
          HighsFileType file_type;
          return st(h.openWriteFile(std::string(m, len), "writeBasis",
                                    write_file, file_type));
        } else if (arg == 1) {
          const HighsStatus call_status =
              normaliseNames(options.log_options, lp);
          assert(call_status != HighsStatus::kError);
          return st(call_status);
        }
        writeBasisFile(write_file, options, lp, h.basis_);
        if (write_file != stdout) fclose(write_file);
        return 0;
      case RunOp::kSolutionBasisSizes:
        if (arg == 0) {
          int64_t* sizes = static_cast<int64_t*>(p);
          sizes[0] = h.solution_.col_value.size();
          sizes[1] = h.solution_.row_value.size();
          sizes[2] = h.solution_.col_dual.size();
          sizes[3] = h.solution_.row_dual.size();
          sizes[4] = h.basis_.col_status.size();
          sizes[5] = h.basis_.row_status.size();
        } else {
          h.solution_.col_value.resize(lp.num_col_, 0);
          h.solution_.row_value.resize(lp.num_row_, 0);
          h.solution_.col_dual.resize(lp.num_col_, 0);
          h.solution_.row_dual.resize(lp.num_row_, 0);
          h.basis_.col_status.resize(lp.num_col_, HighsBasisStatus::kNonbasic);
          h.basis_.row_status.resize(lp.num_row_, HighsBasisStatus::kBasic);
        }
        return 0;
      default:
        break;
    }
    (void)m;
    (void)len;
    assert(false);
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

extern "C" {
int highs_rs_call_solve_mip(const RsHighs* h);
int highs_rs_check_optimality(const RsHighs* h, const char* solver_type,
                              size_t len);
int highs_rs_complete_solution(const RsHighs* h);
int highs_rs_get_ray(const RsHighs* h, bool primal, bool* has_ray,
                     double* value, size_t len);
int highs_rs_presolve(const RsHighs* h);
int highs_rs_crossover(const RsHighs* h);
}

HighsStatus Highs::callSolveMip() {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_call_solve_mip(&v));
  r.rethrow();
  return status;
}

HighsStatus Highs::checkOptimality(const std::string& solver_type) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  return HighsStatus(
      highs_rs_check_optimality(&v, solver_type.data(), solver_type.size()));
}

HighsStatus Highs::completeSolutionFromDiscreteAssignment() {
  assert(model_.isMip() && solution_.value_valid);
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_complete_solution(&v));
  r.rethrow();
  return status;
}

HighsStatus Highs::getDualRayInterface(bool& has_dual_ray,
                                       double* dual_ray_value) {
  assert(!model_.lp_.is_moved_);
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_get_ray(
      &v, false, &has_dual_ray, dual_ray_value, model_.lp_.num_row_));
  r.rethrow();
  return status;
}

HighsStatus Highs::getPrimalRayInterface(bool& has_primal_ray,
                                         double* primal_ray_value) {
  assert(!model_.lp_.is_moved_);
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_get_ray(
      &v, true, &has_primal_ray, primal_ray_value, model_.lp_.num_col_));
  r.rethrow();
  return status;
}

HighsStatus Highs::presolve() {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_presolve(&v));
  r.rethrow();
  return status;
}

HighsStatus Highs::crossover(const HighsSolution& user_solution) {
  HighsRunRust r(*this);
  r.user_solution = &user_solution;
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_crossover(&v));
  r.rethrow();
  return status;
}

extern "C" int highs_rs_call_run_postsolve(const RsHighs* h);
extern "C" int highs_rs_pass_model(const RsHighs* h, int which);
extern "C" bool highs_rs_format_ok(const RsLog* log, bool hessian,
                                   HighsInt num_nz, HighsInt format);

extern "C" int highs_rs_highs_file(const RsHighs* h, int which,
                                   const char* filename, size_t len);
extern "C" void highs_rs_force_solution_basis_size(const RsHighs* h);

HighsStatus Highs::readModel(const std::string& filename) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(
      highs_rs_highs_file(&v, 0, filename.data(), filename.size()));
  r.rethrow();
  return status;
}

HighsStatus Highs::readBasis(const std::string& filename) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(
      highs_rs_highs_file(&v, 1, filename.data(), filename.size()));
  r.rethrow();
  return status;
}

HighsStatus Highs::writeLocalModel(HighsModel& model,
                                   const std::string& filename) {
  HighsRunRust r(*this);
  r.user_model = &model;
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(
      highs_rs_highs_file(&v, 2, filename.data(), filename.size()));
  r.rethrow();
  return status;
}

HighsStatus Highs::writeBasis(const std::string& filename) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(
      highs_rs_highs_file(&v, 3, filename.data(), filename.size()));
  r.rethrow();
  return status;
}

void Highs::forceHighsSolutionBasisSize() {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  highs_rs_force_solution_basis_size(&v);
}

HighsStatus Highs::passModel(HighsModel model) {
  HighsRunRust r(*this);
  r.user_model = &model;
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_pass_model(&v, 0));
  r.rethrow();
  return status;
}

HighsStatus Highs::passHessian(HighsHessian hessian_) {
  HighsRunRust r(*this);
  r.user_hessian = &hessian_;
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_pass_model(&v, 1));
  r.rethrow();
  return status;
}

bool Highs::aFormatOk(const HighsInt num_nz, const HighsInt format) {
  const RsLog log = rsLog(options_.log_options);
  return highs_rs_format_ok(&log, false, num_nz, format);
}

bool Highs::qFormatOk(const HighsInt num_nz, const HighsInt format) {
  const RsLog log = rsLog(options_.log_options);
  return highs_rs_format_ok(&log, true, num_nz, format);
}
extern "C" int highs_rs_set_basis(const RsHighs* h, bool alien,
                                  const char* origin, size_t len);

HighsStatus Highs::setBasis(const HighsBasis& basis,
                            const std::string& origin) {
  HighsRunRust r(*this);
  r.user_basis = &basis;
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(
      highs_rs_set_basis(&v, basis.alien, origin.data(), origin.size()));
  r.rethrow();
  assert(basis_.debug_origin_name != "");
  assert(!basis_.alien || status != HighsStatus::kOk);
  return status;
}

HighsStatus Highs::callRunPostsolve(const HighsSolution& solution,
                                    const HighsBasis& basis) {
  HighsRunRust r(*this);
  r.user_solution = &solution;
  r.user_basis = &basis;
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_call_run_postsolve(&v));
  r.rethrow();
  return status;
}

HighsStatus Highs::calledOptimizeModel() {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_called_optimize_model(&v));
  r.rethrow();
  return status;
}

HighsStatus Highs::returnFromOptimizeModel(const HighsStatus run_return_status,
                                           const bool undo_mods) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status = HighsStatus(highs_rs_return_from_optimize_model(
      &v, int(run_return_status), undo_mods));
  r.rethrow();
  return status;
}

HighsStatus Highs::returnFromHighs(HighsStatus highs_return_status) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsStatus status =
      HighsStatus(highs_rs_return_from_highs(&v, int(highs_return_status)));
  r.rethrow();
  return status;
}

HighsPresolveStatus Highs::runPresolve(const bool force_lp_presolve,
                                       const bool force_presolve) {
  HighsRunRust r(*this);
  const RsHighs v = r.view();
  const HighsPresolveStatus status = HighsPresolveStatus(
      highs_rs_run_presolve(&v, force_lp_presolve, force_presolve));
  r.rethrow();
  return status;
}

HighsPostsolveStatus Highs::runPostsolve() {
  HighsRunRust r(*this);
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

extern "C" double highs_rs_user_scale_solution(
    const HighsUserScaleData* d, RsMut<uint8_t> integrality, bool primal,
    bool dual, RsMut<double> col_value, RsMut<double> row_value,
    RsMut<double> col_dual, RsMut<double> row_dual, double objective,
    double offset);

HighsStatus Highs::userScaleSolution(HighsUserScaleData& data,
                                     bool update_kkt) {
  HighsStatus return_status = HighsStatus::kOk;
  if (!data.user_objective_scale && !data.user_bound_scale)
    return HighsStatus::kOk;
  const HighsLp& lp = this->model_.lp_;
  const bool primal = info_.primal_solution_status != kSolutionStatusNone;
  const bool dual = info_.dual_solution_status != kSolutionStatusNone;
  auto part = [](std::vector<double>& v, const bool use, const HighsInt n) {
    return use ? RsMut<double>{v.data(), size_t(n)} : RsMut<double>{nullptr, 0};
  };
  const double objective_function_value = highs_rs_user_scale_solution(
      &data, rsMut(lp.integrality_), primal, dual,
      part(solution_.col_value, primal, lp.num_col_),
      part(solution_.row_value, primal, lp.num_row_),
      part(solution_.col_dual, dual, lp.num_col_),
      part(solution_.row_dual, dual, lp.num_row_),
      info_.objective_function_value, lp.offset_);
  if (!update_kkt) return return_status;
  info_.objective_function_value = objective_function_value;
  getKktFailures(options_, model_, solution_, basis_, info_);
  return reportKktFailures(model_.lp_, options_, info_,
                           "After removing user scaling")
             ? HighsStatus::kWarning
             : return_status;
}

// Infinite costs, basisForSolution and reportModelStats
// (rust/src/lp_data/model.rs)
struct RsInfCostMods {
  RsMut<HighsInt> index;
  RsMut<double> cost, lower, upper;
  HighsInt num;
};

extern "C" {
int highs_rs_handle_inf_cost(const RsLog* log, double inf_cost, bool minimize,
                             bool is_mip, RsMut<uint8_t> integrality,
                             RsMut<double> cost, RsMut<double> col_lower,
                             RsMut<double> col_upper, RsInfCostMods* m);
void highs_rs_restore_inf_cost(RsMut<HighsInt> index, RsMut<double> saved_cost,
                               RsMut<double> saved_lower,
                               RsMut<double> saved_upper,
                               RsMut<double> col_value,
                               RsMut<uint8_t> col_status, RsMut<double> cost,
                               RsMut<double> col_lower,
                               RsMut<double> col_upper, double* objective);
HighsInt highs_rs_basis_for_solution(
    const RsLog* log, double tol, RsMut<double> col_lower,
    RsMut<double> col_upper, RsMut<double> col_value, RsMut<double> row_lower,
    RsMut<double> row_upper, RsMut<double> row_value,
    RsMut<uint8_t> col_status, RsMut<uint8_t> row_status);
void highs_rs_report_model_stats(const RsLog* log, bool dev, const char* name,
                                 size_t name_len, HighsInt num_col,
                                 HighsInt num_row, HighsInt a_num_nz,
                                 HighsInt hessian_dim, HighsInt q_num_nz,
                                 RsMut<uint8_t> integrality,
                                 RsMut<double> col_lower,
                                 RsMut<double> col_upper);
}

static RsMut<uint8_t> rsBasisStatusOf(std::vector<HighsBasisStatus>& s) {
  return {reinterpret_cast<uint8_t*>(s.data()), s.size()};
}

HighsStatus Highs::handleInfCost() {
  HighsLp& lp = this->model_.lp_;
  if (!lp.has_infinite_cost_) return HighsStatus::kOk;
  const size_t n = lp.num_col_;
  std::vector<HighsInt> index(n);
  std::vector<double> cost(n), lower(n), upper(n);
  RsInfCostMods m = {rsMut(index), rsMut(cost), rsMut(lower), rsMut(upper),
                     0};
  const RsLog log = rsLog(options_.log_options);
  if (HighsStatus(highs_rs_handle_inf_cost(
          &log, options_.infinite_cost, lp.sense_ == ObjSense::kMinimize,
          lp.isMip(), rsMut(lp.integrality_), rsMut(lp.col_cost_),
          rsMut(lp.col_lower_), rsMut(lp.col_upper_), &m)) ==
      HighsStatus::kError)
    return HighsStatus::kError;
  HighsLpMods& mods = lp.mods_;
  mods.save_inf_cost_variable_index.insert(
      mods.save_inf_cost_variable_index.end(), index.begin(),
      index.begin() + m.num);
  mods.save_inf_cost_variable_cost.insert(
      mods.save_inf_cost_variable_cost.end(), cost.begin(),
      cost.begin() + m.num);
  mods.save_inf_cost_variable_lower.insert(
      mods.save_inf_cost_variable_lower.end(), lower.begin(),
      lower.begin() + m.num);
  mods.save_inf_cost_variable_upper.insert(
      mods.save_inf_cost_variable_upper.end(), upper.begin(),
      upper.begin() + m.num);
  lp.has_infinite_cost_ = false;
  return HighsStatus::kOk;
}

void Highs::restoreInfCost(HighsStatus& return_status) {
  HighsLp& lp = this->model_.lp_;
  HighsLpMods& mods = lp.mods_;
  if (mods.save_inf_cost_variable_index.size() == 0) return;
  highs_rs_restore_inf_cost(
      rsMut(mods.save_inf_cost_variable_index),
      rsMut(mods.save_inf_cost_variable_cost),
      rsMut(mods.save_inf_cost_variable_lower),
      rsMut(mods.save_inf_cost_variable_upper),
      solution_.value_valid ? rsMut(solution_.col_value)
                            : RsMut<double>{nullptr, 0},
      basis_.valid ? rsBasisStatusOf(basis_.col_status)
                   : RsMut<uint8_t>{nullptr, 0},
      rsMut(lp.col_cost_), rsMut(lp.col_lower_), rsMut(lp.col_upper_),
      &this->info_.objective_function_value);
  lp.has_infinite_cost_ = true;
  if (this->model_status_ == HighsModelStatus::kInfeasible) {
    this->model_status_ = HighsModelStatus::kUnknown;
    setHighsModelStatusAndClearSolutionAndBasis(this->model_status_);
    return_status = highsStatusFromHighsModelStatus(model_status_);
  }
}

HighsStatus Highs::basisForSolution() {
  HighsLp& lp = model_.lp_;
  assert(!lp.isMip() || options_.solve_relaxation);
  assert(solution_.value_valid);
  invalidateBasis();
  HighsBasis basis;
  basis.col_status.resize(lp.num_col_);
  basis.row_status.resize(lp.num_row_);
  const RsLog log = rsLog(options_.log_options);
  highs_rs_basis_for_solution(
      &log, options_.primal_feasibility_tolerance, rsMut(lp.col_lower_),
      rsMut(lp.col_upper_), rsMut(solution_.col_value), rsMut(lp.row_lower_),
      rsMut(lp.row_upper_), rsMut(solution_.row_value),
      rsBasisStatusOf(basis.col_status), rsBasisStatusOf(basis.row_status));
  return this->setBasis(basis);
}

void Highs::reportModelStats() const {
  const HighsLp& lp = this->model_.lp_;
  const HighsHessian& hessian = this->model_.hessian_;
  const HighsLogOptions& log_options = this->options_.log_options;
  if (!*log_options.output_flag) return;
  const RsLog log = rsLog(log_options);
  highs_rs_report_model_stats(
      &log, *log_options.log_dev_level != 0, lp.model_name_.data(),
      lp.model_name_.size(), lp.num_col_, lp.num_row_, lp.a_matrix_.numNz(),
      hessian.dim_, hessian.dim_ > 0 ? hessian.numNz() : 0,
      rsMut(lp.integrality_), rsMut(lp.col_lower_), rsMut(lp.col_upper_));
}
#endif
