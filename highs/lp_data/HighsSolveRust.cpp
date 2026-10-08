/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsSolveRust.cpp
 * @brief solveLp, solveUnconstrainedLp and
 * assessExcessiveObjectiveBoundScaling of HighsSolve.cpp done by Rust
 * (rust/src/lp_data/solve.rs). The solvers are called back here.
 */
#include "lp_data/HighsRust.h"

#ifdef HIGHS_RUST
#include <exception>

#include "ipm/IpxWrapper.h"
#include "lp_data/HighsSolutionDebug.h"
#include "lp_data/HighsSolve.h"
#include "pdlp/CupdlpWrapper.h"
#include "simplex/HApp.h"

namespace {

struct RsSolveStr {
  const char* ptr;
  size_t len;
};

// solve.rs: CSolve
struct RsSolve {
  RsLog log;
  void* ctx;
  int64_t (*op)(void*, int, const char*, size_t);
  HighsModelStatus* model_status;
  HighsInfoStruct* info;
  const bool *value_valid, *basis_valid;
  HighsInt num_row, num_nz;
  RsSolveStr solver, run_crossover;
  bool run_centring, allow_unbounded_or_infeasible;
  HighsInt highs_debug_level;
  const bool* output_flag;
  const HighsInt* log_dev_level;
  const bool* aborted;
};

// The solver object, and an exception thrown by a step (a cancelled task's
// HighsTask::Interrupt), rethrown once Rust has returned
struct SolveCtx {
  HighsLpSolverObject& solver_object;
  std::exception_ptr pending;
  bool aborted;
};

// solve.rs: SolveOp
enum class SolveOp {
  kDebugAssess = 1,
  kUnconstrained,
  kIpx,
  kPdlp,
  kSimplex,
  kSolutionRightSize,
  kDebugSolution,
};

int64_t solveLpStep(HighsLpSolverObject& solver_object, int which,
                    const char* m, size_t len) {
  HighsOptions& options = solver_object.options_;
  HighsStatus call_status = HighsStatus::kOk;
  switch (SolveOp(which)) {
    case SolveOp::kDebugAssess:
      call_status = assessLp(solver_object.lp_, options);
      assert(call_status == HighsStatus::kOk);
      return int(call_status);
    case SolveOp::kUnconstrained:
      return int(solveUnconstrainedLp(solver_object));
    case SolveOp::kIpx:
      try {
        call_status = solveLpIpx(solver_object);
      } catch (const std::exception& exception) {
        highsLogDev(options.log_options, HighsLogType::kError,
                    "Exception %s in solveLpIpx\n", exception.what());
        call_status = HighsStatus::kError;
      }
      return int(call_status);
    case SolveOp::kPdlp:
      // cuPDLP-C only: HiPDLP is not in Crestline (the solver option
      // rejects "hipdlp")
      solver_object.profiling_->start(kSubSolverPdlp);
      try {
        call_status = solveLpCupdlp(solver_object);
      } catch (const std::exception& exception) {
        highsLogDev(options.log_options, HighsLogType::kError,
                    "Exception %s in solveLpCupdlp\n", exception.what());
        call_status = HighsStatus::kError;
      }
      solver_object.profiling_->stop(kSubSolverPdlp);
      return int(call_status);
    case SolveOp::kSimplex:
      return int(solveLpSimplex(solver_object));
    case SolveOp::kSolutionRightSize:
      return isSolutionRightSize(solver_object.lp_, solver_object.solution_);
    case SolveOp::kDebugSolution:
      return debugHighsLpSolution(std::string(m, len), solver_object) ==
             HighsDebugStatus::kLogicalError;
  }
  assert(false);
  return 0;
}

int64_t solveLpOp(void* ctx, int which, const char* m, size_t len) {
  SolveCtx& c = *static_cast<SolveCtx*>(ctx);
  if (c.aborted) return 0;
  try {
    return solveLpStep(c.solver_object, which, m, len);
  } catch (...) {
    c.pending = std::current_exception();
    c.aborted = true;
    return 0;
  }
}

}  // namespace

extern "C" {
int highs_rs_solve_lp(const RsSolve* c, const char* message, size_t len);
int highs_rs_solve_unconstrained_lp(
    const RsLog* log, bool on, double pft, double dft, const RsLp* lp,
    HighsInfoStruct* info, RsMut<double> col_value, RsMut<double> col_dual,
    RsMut<double> row_value, RsMut<double> row_dual, RsMut<uint8_t> col_status,
    RsMut<uint8_t> row_status);
void highs_rs_assess_excessive_objective_bound_scaling(
    const RsLog* log, const RsLp* lp, RsMut<double> hessian_value,
    HighsInt user_objective_scale, HighsInt user_bound_scale,
    HighsInt* suggested_user_objective_scale,
    HighsInt* suggested_user_bound_scale);
}

HighsStatus solveLp(HighsLpSolverObject& solver_object, const string message) {
  const HighsOptions& options = solver_object.options_;
  SolveCtx ctx{solver_object, nullptr, false};
  RsSolve c;
  c.log = rsLog(options.log_options);
  c.ctx = &ctx;
  c.op = solveLpOp;
  c.model_status = &solver_object.model_status_;
  c.info = static_cast<HighsInfoStruct*>(&solver_object.highs_info_);
  c.value_valid = &solver_object.solution_.value_valid;
  c.basis_valid = &solver_object.basis_.valid;
  c.num_row = solver_object.lp_.num_row_;
  c.num_nz = solver_object.lp_.num_row_ ? solver_object.lp_.a_matrix_.numNz()
                                         : 0;
  c.solver = {options.solver.data(), options.solver.size()};
  c.run_crossover = {options.run_crossover.data(),
                     options.run_crossover.size()};
  c.run_centring = options.run_centring;
  c.allow_unbounded_or_infeasible = options.allow_unbounded_or_infeasible;
  c.highs_debug_level = options.highs_debug_level;
  c.output_flag = options.log_options.output_flag;
  c.log_dev_level = options.log_options.log_dev_level;
  c.aborted = &ctx.aborted;
  const HighsStatus status =
      HighsStatus(highs_rs_solve_lp(&c, message.data(), message.size()));
  if (ctx.pending) std::rethrow_exception(ctx.pending);
  return status;
}

HighsStatus solveUnconstrainedLp(const HighsOptions& options, const HighsLp& lp,
                                 HighsModelStatus& model_status,
                                 HighsInfo& highs_info, HighsSolution& solution,
                                 HighsBasis& basis) {
  resetModelStatusAndHighsInfo(model_status, highs_info);
  assert(lp.num_row_ == 0 || lp.a_matrix_.numNz() == 0);
  if (lp.num_row_ > 0) {
    if (lp.a_matrix_.numNz() > 0) return HighsStatus::kError;
  }
  solution.col_value.assign(lp.num_col_, 0);
  solution.col_dual.assign(lp.num_col_, 0);
  basis.col_status.assign(lp.num_col_, HighsBasisStatus::kNonbasic);
  solution.row_value.assign(lp.num_row_, 0);
  solution.row_dual.assign(lp.num_row_, 0);
  basis.row_status.assign(lp.num_row_, HighsBasisStatus::kBasic);
  const RsLog log = rsLog(options.log_options);
  const RsLp v = rsLp(lp);
  auto status = [](std::vector<HighsBasisStatus>& s) {
    return RsMut<uint8_t>{reinterpret_cast<uint8_t*>(s.data()), s.size()};
  };
  model_status = HighsModelStatus(highs_rs_solve_unconstrained_lp(
      &log, *options.log_options.output_flag,
      options.primal_feasibility_tolerance, options.dual_feasibility_tolerance,
      &v, static_cast<HighsInfoStruct*>(&highs_info), rsMut(solution.col_value),
      rsMut(solution.col_dual), rsMut(solution.row_value),
      rsMut(solution.row_dual), status(basis.col_status),
      status(basis.row_status)));
  solution.value_valid = true;
  solution.dual_valid = true;
  basis.valid = true;
  basis.useful = true;
  return HighsStatus::kOk;
}

void assessExcessiveObjectiveBoundScaling(const HighsLogOptions log_options,
                                          const HighsModel& model,
                                          HighsUserScaleData& user_scale_data) {
  const RsLog log = rsLog(log_options);
  const RsLp v = rsLp(model.lp_);
  const RsMut<double> hessian_value = {
      const_cast<double*>(model.hessian_.value_.data()),
      size_t(model.hessian_.numNz())};
  highs_rs_assess_excessive_objective_bound_scaling(
      &log, &v, hessian_value, user_scale_data.user_objective_scale,
      user_scale_data.user_bound_scale,
      &user_scale_data.suggested_user_objective_scale,
      &user_scale_data.suggested_user_bound_scale);
}
#endif
