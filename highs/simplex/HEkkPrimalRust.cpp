/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkPrimalRust.cpp
 * @brief HEkkPrimal::solve delegating to the Rust port
 * (rust/src/simplex/primal.rs), with the C++ side it calls back
 */
#include "simplex/HEkkPrimal.h"

#ifdef HIGHS_RUST

#include <cassert>

#include "simplex/HEkkRust.h"

namespace {
// Mirrors of the #[repr(C)] items in rust/src/simplex/primal.rs
enum class Op {
  kClearFreshValues = 0,
  kIsUnconstrainedLp,
  kInitialiseSolve,
  kBailout,
  kSolveBailout,
  kReturnFromSolve,
  kInitialiseBound,
  kInitialiseCost,
  kInitialiseNonbasicValueAndMove,
  kComputePrimal,
  kComputeDual,
  kComputeSimplexPrimalInfeasible,
  kComputeSimplexDualInfeasible,
  kComputePrimalObjectiveValue,
  kComputeDualObjectiveValue,
  kResizeBacktrackingEdgeWeight,
  kPutBacktrackingBasisIfInvalid,
  kRebuildRefactor,
  kGetNonsingularInverse,
  kResetSyntheticClock,
  kInitialisePartitionedRowwiseMatrix,
  kClearBadBasisChangeTabooFlag,
  kTabooBadBasisChange,
  kApplyTabooVariableIn,
  kUnapplyTabooVariableIn,
  kIsBadBasisChange,
  kBasisChanged,
  kSetModelStatus,
  kGetModelStatus,
  kSavePrimalPhase1Dual,
  kSavePrimalRay,
  kDualCleanup,
};

enum class Log {
  kNearOptimal = 0,
  kNoBoundPerturbation,
  kOnlyTaboo,
  kFreeColumns,
  kPhase1Start,
  kPhase2NoPerturbation,
  kPhase2Start,
  kReturnPhase1,
  kPhase2Optimal,
  kProblemOptimal,
  kPhase2Unbounded,
  kProblemUnbounded,
  kCleanupShift,
  kRebuildPhase1,
  kChooseRowFailed,
  kDontUseVariableIn,
  kRemoveFreeFailed,
  kMissedBoundShifts,
  kPrimalCorrections,
  kNumericalCheck,
  kShiftBound,
  kPseWeightError,
  kWithoutInvert,
  kLeavingDualInfeasibility,
  kPhase2RowOut,
};

enum ReportKind { kReportIteration = 0, kReportRebuild, kReportAnalysisData };

struct Report {
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

struct Callbacks {
  void* ctx;
  void (*view)(void* ctx, highs_rs::Ekk* out);
  int (*op)(void* ctx, int op, int a, int b, int c);
  void (*report)(void* ctx, int kind, const Report* report);
  void (*log)(void* ctx, int id, const int* i, const double* d);
};
}  // namespace

extern "C" int highs_rs_primal_solve(const Callbacks* cb, bool force_phase2);

bool HEkkPrimal::useRust() const {
  const HighsSimplexAnalysis& analysis = ekk_instance_.analysis_;
  return ekk_instance_.options_->highs_debug_level < kHighsDebugLevelCheap &&
         !analysis.analyse_simplex_summary_data &&
         !analysis.analyse_simplex_runtime_data &&
         !analysis.analyse_simplex_time && !ekk_instance_.debug_solve_report_;
}

HighsStatus HEkkPrimal::solveRust(const bool force_phase2) {
  analysis = &ekk_instance_.analysis_;
  Callbacks cb;
  cb.ctx = this;
  cb.view = [](void* ctx, highs_rs::Ekk* out) {
    *out = static_cast<HEkkPrimal*>(ctx)->ekk_instance_.rustView();
  };
  cb.op = [](void* ctx, int op, int a, int b, int c) -> int {
    HEkkPrimal& primal = *static_cast<HEkkPrimal*>(ctx);
    HEkk& ekk = primal.ekk_instance_;
    HighsSimplexInfo& info = ekk.info_;
    const HighsInt num_tot = ekk.lp_.num_col_ + ekk.lp_.num_row_;
    switch (static_cast<Op>(op)) {
      case Op::kClearFreshValues:
        ekk.clearFreshValues();
        return 0;
      case Op::kIsUnconstrainedLp:
        return ekk.isUnconstrainedLp();
      case Op::kInitialiseSolve:
        ekk.status_.has_primal_objective_value = false;
        ekk.status_.has_dual_objective_value = false;
        ekk.model_status_ = HighsModelStatus::kNotset;
        ekk.solve_bailout_ = false;
        ekk.called_return_from_solve_ = false;
        ekk.exit_algorithm_ = SimplexAlgorithm::kPrimal;
        if (!ekk.status_.has_dual_steepest_edge_weights) {
          // No dual weights to maintain, so ensure that the vectors are
          // assigned since they are used around factorization and when
          // setting up the backtracking information
          ekk.dual_edge_weight_.assign(ekk.lp_.num_row_, 1.0);
          ekk.scattered_dual_edge_weight_.resize(num_tot);
        }
        return 0;
      case Op::kBailout:
        return ekk.bailout();
      case Op::kSolveBailout:
        return ekk.solve_bailout_;
      case Op::kReturnFromSolve:
        return (int)ekk.returnFromSolve(static_cast<HighsStatus>(a));
      case Op::kInitialiseBound:
        ekk.initialiseBound(SimplexAlgorithm::kPrimal, a, b != 0);
        return 0;
      case Op::kInitialiseCost:
        ekk.initialiseCost(SimplexAlgorithm::kPrimal, a);
        return 0;
      case Op::kInitialiseNonbasicValueAndMove:
        ekk.initialiseNonbasicValueAndMove();
        return 0;
      case Op::kComputePrimal:
        ekk.computePrimal();
        return 0;
      case Op::kComputeDual:
        ekk.computeDual();
        return 0;
      case Op::kComputeSimplexPrimalInfeasible:
        ekk.computeSimplexPrimalInfeasible();
        return 0;
      case Op::kComputeSimplexDualInfeasible:
        ekk.computeSimplexDualInfeasible();
        return 0;
      case Op::kComputePrimalObjectiveValue:
        ekk.computePrimalObjectiveValue();
        return 0;
      case Op::kComputeDualObjectiveValue:
        ekk.computeDualObjectiveValue();
        return 0;
      case Op::kResizeBacktrackingEdgeWeight:
        info.backtracking_basis_edge_weight_.resize(num_tot);
        return 0;
      case Op::kPutBacktrackingBasisIfInvalid:
        if (!info.valid_backtracking_basis_) ekk.putBacktrackingBasis();
        return 0;
      case Op::kRebuildRefactor:
        return ekk.rebuildRefactor(a);
      case Op::kGetNonsingularInverse:
        return ekk.getNonsingularInverse(a);
      case Op::kResetSyntheticClock:
        ekk.resetSyntheticClock();
        return 0;
      case Op::kInitialisePartitionedRowwiseMatrix:
        ekk.initialisePartitionedRowwiseMatrix();
        assert(
            ekk.ar_matrix_.debugPartitionOk(ekk.basis_.nonbasicFlag_.data()));
        return 0;
      case Op::kClearBadBasisChangeTabooFlag:
        ekk.clearBadBasisChangeTabooFlag();
        return 0;
      case Op::kTabooBadBasisChange:
        return ekk.tabooBadBasisChange();
      case Op::kApplyTabooVariableIn:
        ekk.applyTabooVariableIn(info.workDual_, 0);
        return 0;
      case Op::kUnapplyTabooVariableIn:
        ekk.unapplyTabooVariableIn(info.workDual_);
        return 0;
      case Op::kIsBadBasisChange:
        return ekk.isBadBasisChange(SimplexAlgorithm::kPrimal, a, b, c);
      case Op::kBasisChanged:
        // The C++ parts of HEkk::updatePivots and HEkk::updateFactor
        ekk.dual_values_valid_ = false;
        ekk.visited_basis_.insert(ekk.basis_.hash);
        assert(!ekk.simplex_nla_.update_.valid_);
        ekk.simplex_nla_.factor_.refactor_info_.clear();
        return 0;
      case Op::kSetModelStatus:
        ekk.model_status_ = static_cast<HighsModelStatus>(a);
        return 0;
      case Op::kGetModelStatus:
        return (int)ekk.model_status_;
      case Op::kSavePrimalPhase1Dual:
        ekk.primal_phase1_dual_ = info.workDual_;
        return 0;
      case Op::kSavePrimalRay:
        ekk.primal_ray_record_.clear();
        ekk.primal_ray_record_.index = a;
        ekk.primal_ray_record_.sign = b;
        return 0;
      case Op::kDualCleanup:
        return (int)primal.cleanupWithDual();
    }
    assert(false);
    return 0;
  };
  cb.report = [](void* ctx, int kind, const Report* r) {
    HEkkPrimal& primal = *static_cast<HEkkPrimal*>(ctx);
    // The data of HEkkPrimal used by iterationAnalysisData
    primal.solve_phase = r->solve_phase;
    primal.edge_weight_mode = static_cast<EdgeWeightMode>(r->edge_weight_mode);
    primal.num_devex_iterations_ = r->num_devex_iterations;
    primal.row_out = r->row_out;
    primal.variable_out = r->variable_out;
    primal.variable_in = r->variable_in;
    primal.rebuild_reason = r->rebuild_reason;
    primal.theta_primal = r->theta_primal;
    primal.theta_dual = r->theta_dual;
    primal.alpha_col = r->alpha_col;
    primal.alpha_row = r->alpha_row;
    primal.numericalTrouble = r->numerical_trouble;
    switch (kind) {
      case kReportIteration:
        primal.iterationAnalysis();
        break;
      case kReportRebuild:
        primal.reportRebuild(r->reason_for_rebuild);
        break;
      default:
        primal.iterationAnalysisData();
    }
  };
  cb.log = [](void* ctx, int id, const int* i, const double* d) {
    HEkk& ekk = static_cast<HEkkPrimal*>(ctx)->ekk_instance_;
    const HighsLogOptions& log_options = ekk.options_->log_options;
    switch (static_cast<Log>(id)) {
      case Log::kNearOptimal:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "Primal feasible and num / max / sum "
                    "dual infeasibilities of "
                    "%" HIGHSINT_FORMAT
                    " / %g "
                    "/ %g, so near-optimal\n",
                    i[0], d[0], d[1]);
        break;
      case Log::kNoBoundPerturbation:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "Near-optimal, so don't use bound perturbation\n");
        break;
      case Log::kOnlyTaboo:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "HEkkPrimal::solve Only basis change is taboo\n");
        break;
      case Log::kFreeColumns:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "HEkkPrimal:: LP has %" HIGHSINT_FORMAT " free columns\n",
                    i[0]);
        break;
      case Log::kPhase1Start:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "primal-phase1-start\n");
        break;
      case Log::kPhase2NoPerturbation:
        highsLogDev(log_options, HighsLogType::kWarning,
                    "Moving to phase 2, but not allowing bound perturbation\n");
        break;
      case Log::kPhase2Start:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "primal-phase2-start\n");
        break;
      case Log::kReturnPhase1:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "primal-return-phase1\n");
        break;
      case Log::kPhase2Optimal:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "primal-phase-2-optimal\n");
        break;
      case Log::kProblemOptimal:
        highsLogDev(log_options, HighsLogType::kDetailed, "problem-optimal\n");
        break;
      case Log::kPhase2Unbounded:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "primal-phase-2-unbounded\n");
        break;
      case Log::kProblemUnbounded:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "problem-primal-unbounded\n");
        break;
      case Log::kCleanupShift:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "primal-cleanup-shift\n");
        break;
      case Log::kRebuildPhase1:
        highsLogDev(
            log_options, HighsLogType::kWarning,
            "HEkkPrimal::rebuild switching back to phase 1 from phase 2\n");
        break;
      case Log::kChooseRowFailed:
        highsLogDev(log_options, HighsLogType::kError,
                    "Primal phase 1 choose row failed\n");
        break;
      case Log::kDontUseVariableIn:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "Chosen entering variable %" HIGHSINT_FORMAT
                    " (Iter = %" HIGHSINT_FORMAT "; Update = %" HIGHSINT_FORMAT
                    ") has computed "
                    "(updated) dual of %10.4g (%10.4g) so don't use it%s%s\n",
                    i[0], i[1], i[2], d[0], d[1], i[3] ? "; too small" : "",
                    i[4] ? "; sign error" : "");
        break;
      case Log::kRemoveFreeFailed:
        highsLogDev(log_options, HighsLogType::kError,
                    "HEkkPrimal::phase1update failed to remove nonbasic free "
                    "column %" HIGHSINT_FORMAT "\n",
                    i[0]);
        break;
      case Log::kMissedBoundShifts:
        highsLogDev(log_options, HighsLogType::kError,
                    "correctPrimal: Missed %d bound shifts\n", i[0]);
        break;
      case Log::kPrimalCorrections:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "phase2CorrectPrimal: num / max / sum primal corrections = "
                    "%" HIGHSINT_FORMAT
                    " / %g / "
                    "%g\n",
                    i[0], d[0], d[1]);
        break;
      case Log::kNumericalCheck:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "Numerical check: Iter %4" HIGHSINT_FORMAT
                    ": alpha_col = %12g, (From %3s alpha_row = "
                    "%12g), aDiff = %12g: measure = %12g\n",
                    i[0], d[0], i[1] ? "Row" : "Col", d[1], d[2], d[3]);
        break;
      case Log::kShiftBound:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "HEkkPrimal::shiftBound Value(%4d) = %10.4g exceeds %s: "
                    "random_value = %g; value = %g; "
                    "feasibility = %g; infeasibility = %g; shift = %g; bound "
                    "= %g; "
                    "new_infeasibility = %g with error %g\n",
                    i[0], d[0], i[1] ? "lower" : "upper", d[1], d[2], d[0],
                    d[3], d[4], d[5], d[6], d[7], d[8]);
        fflush(stdout);
        break;
      case Log::kPseWeightError:
        printf(
            "HEkk::debugPrimalSteepestEdgeWeights Iteration %5d: Checked %2d "
            "weights: "
            "error = %10.4g; norm = %10.4g; relative error = %10.4g\n",
            i[0], i[1], d[0], d[1], d[2]);
        break;
      case Log::kWithoutInvert:
        highsLogDev(log_options, HighsLogType::kError,
                    "HEkkPrimal::solve called without INVERT\n");
        break;
      case Log::kLeavingDualInfeasibility:
        printf("Dual infeasibility %g for leaving column!\n", d[0]);
        break;
      case Log::kPhase2RowOut:
        printf("HEkkPrimal::solvePhase2 row_out = %d solve %d\n", i[0],
               (int)ekk.debug_solve_call_num_);
        fflush(stdout);
        break;
    }
  };
  return static_cast<HighsStatus>(highs_rs_primal_solve(&cb, force_phase2));
}

#endif  // HIGHS_RUST
