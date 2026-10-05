/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file simplex/HEkkDualRust.cpp
 * @brief HEkkDual::solve for the serial strategy delegating to the Rust
 * driver (rust/src/simplex/dual.rs), and the C++ it calls back
 */
#include "simplex/HEkkDual.h"

#ifdef HIGHS_RUST

#include <cassert>
#include <type_traits>

#include "lp_data/HighsLpUtils.h"
#include "simplex/HEkkPrimal.h"
#include "simplex/HEkkRust.h"
#include "simplex/HSimplexDebug.h"
#include "simplex/SimplexTimer.h"

static_assert(sizeof(HighsModelStatus) == sizeof(int),
              "model_status_ is shared with Rust as an i32");
static_assert(sizeof(HighsRandom) == sizeof(uint64_t),
              "random_ is shared with Rust as its 64-bit state");
static_assert(sizeof(std::pair<HighsInt, double>) == 16,
              "workData is shared with Rust as #[repr(C)] (i32, f64)");

namespace highs_rs {
// Mirrors of the #[repr(C)] structs in rust/src/simplex/dual.rs
struct DualEkk {
  // HEkk
  HighsInt* iteration_count;
  int* model_status;
  bool* solve_bailout;
  bool* called_return_from_solve;
  bool* dual_values_valid;
  bool* fresh_unperturbed_dual;
  bool* fresh_dual;
  bool* fresh_primal;
  double* edge_weight_error;
  HighsInt* dual_simplex_cleanup_level;
  HighsInt* dual_simplex_phase1_cleanup_level;
  HighsInt* previous_iteration_cycling_detected;
  uint64_t* random;
  void* basis_records;
  double* nla_build_synthetic_tick;
  // status_
  bool* has_invert;
  bool* has_fresh_invert;
  bool* has_fresh_rebuild;
  bool* has_dual_objective_value;
  bool* has_primal_objective_value;
  bool* has_dual_steepest_edge_weights;
  bool* has_ar_matrix;
  // info_
  HighsInt* dual_phase1_iteration_count;
  HighsInt* dual_phase2_iteration_count;
  bool* allow_cost_shifting;
  bool* allow_cost_perturbation;
  bool* backtracking;
  bool* valid_backtracking_basis;
  bool* store_squared_primal_infeasibility;
  double* factor_pivot_threshold;
  double* col_bfrt_density;
  double* costly_dse_measure;
  double* costly_dse_frequency;
  HighsInt* num_costly_dse_iteration;
  double* average_log_low_dse_weight_error;
  double* average_log_high_dse_weight_error;
  // info_ values
  HighsInt control_iteration_count0;
  bool allow_dual_steepest_edge_to_devex_switch;
  double dual_steepest_edge_weight_log_error_threshold;
  HighsInt dual_edge_weight_strategy;
  bool run_quiet;
  // options_
  double objective_bound;
  double time_limit;
  HighsInt simplex_iteration_limit;
  HighsInt max_dual_simplex_cleanup_level;
  HighsInt max_dual_simplex_phase1_cleanup_level;
  double dual_simplex_pivot_growth_tolerance;
  HighsInt simplex_dse_exact_init_max_rows;
  double small_matrix_value;
  double dual_steepest_edge_weight_error_tolerance;
  bool no_unnecessary_rebuild_refactor;
  double rebuild_refactor_solution_error_tolerance;
  bool dev_log;
  bool iteration_report;
  bool interrupt_callback;
};

struct DualView {
  Ekk ekk;
  HighsInt* devex_index;
  int n_devex_index;
  const HighsInt* num_tot_permutation;
  int n_num_tot_permutation;
};

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

struct DualCallbacks {
  void* ctx;
  void (*refresh)(void*, DualView*);
  int (*start)(void*);
  int (*return_from_solve)(void*, int);
  void (*log)(void*, int, const int*, const double*);
  void (*log_cost_perturbation)(void*, const CostPerturbationReport*);
  void (*set_lp_dual_infeasibility)(void*, const Infeasibility*);
  void (*report_rebuild)(void*, const DualState*, int);
  void (*iteration_report)(void*, const DualState*);
  void (*apply_analysis_data)(void*, const AnalysisData*);
  void (*chuzc_fail)(void*, int, int, const std::pair<HighsInt, double>*,
                     double, double);
  double (*timer_read)(void*);
  bool (*interrupt)(void*);
  double (*factor_solve_error)(void*);
  bool (*get_nonsingular_inverse)(void*, int);
  void (*initialise_partitioned_rowwise_matrix)(void*);
  void (*put_backtracking_basis)(void*);
  bool (*restore_dual_edge_weights)(void*, bool);
  void (*edge_weight_vectors)(void*, bool);
  void (*clear_refactor_info)(void*);
  void (*set_pivot_threshold)(void*, double);
  void (*improve_choose_column_row)(void*, int);
  bool (*proof_of_primal_infeasibility)(void*, int, int);
  void (*save_dual_ray)(void*, int, int);
  int (*primal_cleanup)(void*);
  void (*record_dual_values)(void*);
};

struct DualVectors {
  HVec* row_ep;
  HVec* row_ap;
  HVec* col_aq;
  HVec* col_bfrt;
};
}  // namespace highs_rs

extern "C" int highs_rs_dual_solve(const highs_rs::DualEkk* x,
                                   const highs_rs::DualCallbacks* cb,
                                   const highs_rs::DualVectors* v,
                                   bool force_phase2);

// The messages of the Rust driver: see msg in rust/src/simplex/dual.rs
namespace {
enum DualMessage {
  kNearOptimal = 0,
  kNearOptimalNoPerturbation,
  kNearOptimalUseDevex,
  kComputeDseWeights,
  kCannotCleanup,
  kPhase1Start,
  kPhase1Optimal,
  kRatioTestFailed,
  kExcessivePrimalValues,
  kPhase1NotSolved,
  kPhase1Unbounded,
  kCleaningUpPhase1,
  kPhase1BadPhase,
  kPhase2NoPerturbation,
  kPhase2Start,
  kPhase2FoundFree,
  kPhase2Optimal,
  kProblemOptimal,
  kPhase2NotSolved,
  kProblemPrimalInfeasible,
  kCleanupLevelExceeded,
  kCleanupShift,
  kDseWeightError,
  kSwitchDevexCost,
  kSwitchDevexError,
  kFlips,
  kShifts,
  kShift,
  kPhase1OptimalNotPhase2,
  kPhase1GoPhase2,
  kPhase1FeasibleWrtPhase1,
  kPhase1Return,
  kAlreadyPerturbed,
  kReperturbing,
  kFreeShift,
  kFreeShifts,
  kPossibleLpDualInfeasibility,
  kExactDualInfeasibilities,
  kExactColResidual,
  kExactRowResidual,
  kExactRelativeDelta,
  kObjectiveBoundExceeded,
  kDualUbBailout,
  kBadBasisChange,
};
}  // namespace

struct HEkkDual::RustGlue {
  static HEkkDual& dual(void* ctx) { return *static_cast<HEkkDual*>(ctx); }
  static HEkk& ekk(void* ctx) { return dual(ctx).ekk_instance_; }

  static void refresh(void* ctx, highs_rs::DualView* view) {
    HEkk& e = ekk(ctx);
    view->ekk = e.rustView();
    view->devex_index = e.info_.devex_index_.data();
    view->n_devex_index = e.info_.devex_index_.size();
    view->num_tot_permutation = e.info_.numTotPermutation_.data();
    view->n_num_tot_permutation = e.info_.numTotPermutation_.size();
  }

  static int start(void* ctx) {
    HEkkDual& d = dual(ctx);
    HEkk& e = d.ekk_instance_;
    const HighsLogOptions& log_options = e.options_->log_options;
    e.exit_algorithm_ = SimplexAlgorithm::kDual;
    if (d.debugDualSimplex("Initialise", true) ==
        HighsDebugStatus::kLogicalError)
      return -1;
    // Assumes that the LP has a positive number of rows
    if (e.isUnconstrainedLp()) return -1;
    if (!d.dualInfoOk(e.lp_)) {
      highsLogDev(log_options, HighsLogType::kError,
                  "HPrimalDual::solve has error in dual information\n");
      return -1;
    }
    // Possibly use Li dual steepest edge weights by not storing squared
    // primal infeasibilities
    d.possiblyUseLiDualSteepestEdge();
    assert(e.status_.has_invert);
    if (!e.status_.has_invert) {
      highsLogDev(log_options, HighsLogType::kError,
                  "HDual:: Should enter solve with INVERT\n");
      return -1;
    }
    return 0;
  }

  static int returnFromSolve(void* ctx, int status) {
    return (int)ekk(ctx).returnFromSolve((HighsStatus)status);
  }

  static void log(void* ctx, int id, const int* i, const double* r) {
    HEkk& e = ekk(ctx);
    const HighsLogOptions& log_options = e.options_->log_options;
    const HighsSimplexInfo& info = e.info_;
    switch (id) {
      case kNearOptimal:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "Dual feasible with unperturbed costs and num / max / sum "
                    "primal infeasibilities of "
                    "%" HIGHSINT_FORMAT
                    " / %g "
                    "/ %g, so near-optimal\n",
                    i[0], r[0], r[1]);
        break;
      case kNearOptimalNoPerturbation:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "Near-optimal, so don't use cost perturbation\n");
        break;
      case kNearOptimalUseDevex:
        highsLogDev(
            log_options, HighsLogType::kDetailed,
            "Basis is not logical, but near-optimal, so use Devex rather "
            "than compute steepest edge weights\n");
        break;
      case kComputeDseWeights:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "Basis is not logical, so compute steepest edge weights\n");
        break;
      case kCannotCleanup:
        highsLogDev(log_options, HighsLogType::kWarning,
                    "HEkkDual:: Cannot use level %" HIGHSINT_FORMAT
                    " primal simplex cleanup for %" HIGHSINT_FORMAT
                    " dual infeasibilities\n",
                    i[0], i[1]);
        break;
      case kPhase1Start:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "dual-phase-1-start\n");
        break;
      case kPhase1Optimal:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "dual-phase-1-optimal\n");
        break;
      case kRatioTestFailed:
        highsLogUser(
            log_options, HighsLogType::kError,
            "Dual simplex ratio test failed due to excessive dual values: "
            "consider scaling down the LP objective coefficients\n");
        break;
      case kExcessivePrimalValues:
        highsLogUser(log_options, HighsLogType::kError,
                     "Dual simplex detected excessive primal values: consider "
                     "scaling down the LP bounds\n");
        break;
      case kPhase1NotSolved:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "dual-phase-1-not-solved\n");
        break;
      case kPhase1Unbounded:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "dual-phase-1-unbounded\n");
        break;
      case kCleaningUpPhase1:
        highsLogDev(log_options, HighsLogType::kWarning,
                    "Cleaning up cost perturbation when unbounded in phase 1\n");
        break;
      case kPhase1BadPhase:
        highsLogDev(
            log_options, HighsLogType::kInfo,
            "HEkkDual::solvePhase1 solve_phase == %d (solve call %d; iter %d)\n",
            (int)i[0], (int)e.debug_solve_call_num_, (int)e.iteration_count_);
        break;
      case kPhase2NoPerturbation:
        highsLogDev(log_options, HighsLogType::kWarning,
                    "Moving to phase 2, but not allowing cost perturbation\n");
        break;
      case kPhase2Start:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "dual-phase-2-start\n");
        break;
      case kPhase2FoundFree:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "dual-phase-2-found-free\n");
        break;
      case kPhase2Optimal:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "dual-phase-2-optimal\n");
        break;
      case kProblemOptimal:
        highsLogDev(log_options, HighsLogType::kDetailed, "problem-optimal\n");
        break;
      case kPhase2NotSolved:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "dual-phase-2-not-solved\n");
        break;
      case kProblemPrimalInfeasible:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "problem-primal-infeasible\n");
        break;
      case kCleanupLevelExceeded:
        highsLogDev(log_options, HighsLogType::kError,
                    "Dual simplex cleanup level has exceeded limit of %d\n",
                    (int)i[0]);
        break;
      case kCleanupShift:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "dual-cleanup-shift\n");
        break;
      case kDseWeightError:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "Dual steepest edge weight error is %g\n", r[0]);
        break;
      case kSwitchDevexCost:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "Switch from DSE to Devex after %" HIGHSINT_FORMAT
                    " costly DSE iterations of %" HIGHSINT_FORMAT
                    " with "
                    "densities C_Aq = %11.4g; R_Ep = %11.4g; R_Ap = "
                    "%11.4g; DSE = %11.4g\n",
                    i[0], i[1], r[0], r[1], r[2], r[3]);
        break;
      case kSwitchDevexError:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "Switch from DSE to Devex with log error measure of %g > "
                    "%g = threshold\n",
                    r[0], r[1]);
        break;
      case kFlips:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "Performed num / max / sum = %" HIGHSINT_FORMAT
                    " / %g / %g flip(s) for num / min / max / sum dual "
                    "infeasibility of "
                    "%" HIGHSINT_FORMAT
                    " / %g / %g / %g; objective change = %g\n",
                    i[0], r[0], r[1], i[1], r[2], r[3], r[4], r[5]);
        break;
      case kShifts:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "Performed num / max / sum = %" HIGHSINT_FORMAT
                    " / %g / %g shift(s) for num / max / sum dual "
                    "infeasibility of "
                    "%" HIGHSINT_FORMAT " / %g / %g; objective change = %g\n",
                    i[0], r[0], r[1], i[1], r[2], r[3], r[4]);
        break;
      case kShift: {
        const std::string direction = i[0] ? "  up" : "down";
        highsLogDev(log_options, HighsLogType::kVerbose,
                    "Move %s: cost shift = %g; objective change = %g\n",
                    direction.c_str(), r[0], r[1]);
        break;
      }
      case kPhase1OptimalNotPhase2:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "Optimal in phase 1 but not jumping to phase 2 since "
                    "dual objective is %10.4g: Costs perturbed = "
                    "%" HIGHSINT_FORMAT "\n",
                    r[0], i[0]);
        break;
      case kPhase1GoPhase2:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "LP is dual feasible wrt Phase 2 bounds after removing "
                    "cost perturbations so go to phase 2\n");
        break;
      case kPhase1FeasibleWrtPhase1:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "LP is dual feasible wrt Phase 1 bounds after removing "
                    "cost perturbations: "
                    "dual objective is %10.4g\n",
                    r[0]);
        break;
      case kPhase1Return:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "LP has %d dual feasibilities wrt Phase 1 bounds after "
                    "removing cost perturbations "
                    "so return to phase 1\n",
                    i[0]);
        break;
      case kAlreadyPerturbed:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "Costs are already perturbed in exitPhase1ResetDuals\n");
        break;
      case kReperturbing:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "Re-perturbing costs when optimal in phase 1\n");
        break;
      case kFreeShift:
        highsLogDev(log_options, HighsLogType::kVerbose,
                    "Variable %" HIGHSINT_FORMAT
                    " is free: shift cost to zero dual of %g\n",
                    i[0], r[0]);
        break;
      case kFreeShifts:
        highsLogDev(log_options, HighsLogType::kDetailed,
                    "Performed %" HIGHSINT_FORMAT
                    " cost shift(s) for free variables to zero "
                    "dual values: total = %g\n",
                    i[0], r[0]);
        break;
      case kPossibleLpDualInfeasibility: {
        const std::string lp_dual_status = i[0] ? "infeasible" : "feasible";
        highsLogDev(log_options, HighsLogType::kInfo,
                    "LP is dual %s with dual phase 1 objective %10.4g and num "
                    "/ max / sum dual infeasibilities = %" HIGHSINT_FORMAT
                    " / %9.4g / %9.4g\n",
                    lp_dual_status.c_str(), r[0], i[0], r[1], r[2]);
        break;
      }
      case kExactDualInfeasibilities:
        highsLogDev(log_options, HighsLogType::kInfo,
                    "When computing exact dual objective, the unperturbed "
                    "costs yield num / max / sum dual "
                    "infeasibilities = %d / %g / %g\n",
                    (int)i[0], r[0], r[1]);
        break;
      case kExactColResidual:
        highsLogDev(
            log_options, HighsLogType::kWarning,
            "Col %4" HIGHSINT_FORMAT
            ": ExactDual = %11.4g; WorkDual = %11.4g; Residual = %11.4g\n",
            i[0], r[0], r[1], r[2]);
        break;
      case kExactRowResidual:
        highsLogDev(
            log_options, HighsLogType::kWarning,
            "Row %4" HIGHSINT_FORMAT
            ": ExactDual = %11.4g; WorkDual = %11.4g; Residual = %11.4g\n",
            i[0], r[0], r[1], r[2]);
        break;
      case kExactRelativeDelta:
        highsLogDev(log_options, HighsLogType::kWarning,
                    "||exact dual vector|| = %g; ||delta dual vector|| = %g: "
                    "ratio = %g\n",
                    r[0], r[1], r[2]);
        break;
      case kObjectiveBoundExceeded:
        highsLogDev(
            log_options, HighsLogType::kDetailed,
            "HEkkDual::solvePhase2: %12g = Objective > ObjectiveUB = %12g\n",
            r[0], r[1]);
        break;
      case kDualUbBailout: {
        const std::string action =
            i[0] ? "Have DualUB bailout" : "No   DualUB bailout";
        highsLogDev(log_options, HighsLogType::kInfo,
                    "%s on iteration %" HIGHSINT_FORMAT
                    ": Density %11.4g; Frequency %" HIGHSINT_FORMAT
                    ": "
                    "Residual(Perturbed = %g; Exact = %g)\n",
                    action.c_str(), i[1], r[0], i[2], r[1], r[2]);
        break;
      }
      case kBadBasisChange:
        highsLogDev(log_options, HighsLogType::kWarning,
                    " basis change (%d out; %d in) is bad\n", (int)i[0],
                    (int)i[1]);
        break;
      default:
        assert(false);
    }
    (void)info;
  }

  static void logCostPerturbation(void* ctx,
                                  const highs_rs::CostPerturbationReport* rp) {
    HEkk& e = ekk(ctx);
    const highs_rs::CostPerturbationReport& r = *rp;
    if (!e.options_->output_flag) return;
    const HighsLogOptions& log_options = e.options_->log_options;
    highsLogDev(log_options, HighsLogType::kInfo, "Cost perturbation for %s\n",
                e.lp_.model_name_.c_str());
    highsLogDev(log_options, HighsLogType::kInfo,
                "   Initially have %" HIGHSINT_FORMAT
                " nonzero costs (%3" HIGHSINT_FORMAT "%%)",
                (HighsInt)r.num_original_nonzero_cost, (HighsInt)r.pct0);
    if (r.num_original_nonzero_cost) {
      highsLogDev(log_options, HighsLogType::kInfo,
                  " with min / average / max = %g / %g / %g\n",
                  r.min_abs_cost, r.average_abs_cost, r.max_abs_cost);
    } else {
      highsLogDev(log_options, HighsLogType::kInfo,
                  " but perturb as if max cost was 1\n");
    }
    if (r.large)
      highsLogDev(
          log_options, HighsLogType::kInfo,
          "   Large so set max_abs_cost = sqrt(sqrt(max_abs_cost)) = %g\n",
          r.large_max_abs_cost);
    if (r.small_boxed_rate)
      highsLogDev(log_options, HighsLogType::kInfo,
                  "   Small boxedRate (%g) so set max_abs_cost = "
                  "min(max_abs_cost, 1.0) = "
                  "%g\n",
                  r.boxed_rate, r.small_boxed_max_abs_cost);
    highsLogDev(log_options, HighsLogType::kInfo,
                "   Perturbation column base = %g\n",
                e.cost_perturbation_base_);
    highsLogDev(log_options, HighsLogType::kInfo,
                "   Perturbation row    base = %g\n",
                r.row_cost_perturbation_base);
  }

  static void setLpDualInfeasibility(void* ctx,
                                     const highs_rs::Infeasibility* inf) {
    HighsSimplexAnalysis& analysis = ekk(ctx).analysis_;
    analysis.num_dual_phase_1_lp_dual_infeasibility = inf->num;
    analysis.max_dual_phase_1_lp_dual_infeasibility = inf->max;
    analysis.sum_dual_phase_1_lp_dual_infeasibility = inf->sum;
  }

  static void applyState(HEkkDual& d, const highs_rs::DualState& s) {
    d.solve_phase = s.solve_phase;
    d.edge_weight_mode = (EdgeWeightMode)s.edge_weight_mode;
    d.num_devex_iterations = s.num_devex_iterations;
    d.row_out = s.row_out;
    d.variable_out = s.variable_out;
    d.variable_in = s.variable_in;
    d.rebuild_reason = s.rebuild_reason;
    d.delta_primal = s.delta_primal;
    d.theta_primal = s.theta_primal;
    d.theta_dual = s.theta_dual;
    d.alpha_col = s.alpha_col;
    d.alpha_row = s.alpha_row;
    d.numericalTrouble = s.numerical_trouble;
  }

  static void reportRebuild(void* ctx, const highs_rs::DualState* s,
                            int reason) {
    HEkkDual& d = dual(ctx);
    applyState(d, *s);
    d.reportRebuild(reason);
  }

  static void iterationReport(void* ctx, const highs_rs::DualState* s) {
    HEkkDual& d = dual(ctx);
    applyState(d, *s);
    d.iterationAnalysisData();
    d.analysis->iterationReport();
  }

  // HEkkDual::iterationAnalysisData with the data of an earlier iteration
  static void applyAnalysisData(void* ctx, const highs_rs::AnalysisData* s) {
    HEkkDual& d = dual(ctx);
    HEkk& e = d.ekk_instance_;
    applyState(d, s->state);
    HighsSimplexAnalysis* analysis = d.analysis;
    double cost_scale_factor = pow(2.0, -e.options_->cost_scale_factor);
    HighsSimplexInfo& info = e.info_;
    analysis->simplex_strategy = info.simplex_strategy;
    analysis->edge_weight_mode = d.edge_weight_mode;
    analysis->solve_phase = d.solve_phase;
    analysis->simplex_iteration_count = s->iteration_count;
    analysis->devex_iteration_count = d.num_devex_iterations;
    analysis->pivotal_row_index = d.row_out;
    analysis->leaving_variable = d.variable_out;
    analysis->entering_variable = d.variable_in;
    analysis->rebuild_reason = d.rebuild_reason;
    analysis->reduced_rhs_value = 0;
    analysis->reduced_cost_value = 0;
    analysis->edge_weight = 0;
    analysis->primal_delta = d.delta_primal;
    analysis->primal_step = d.theta_primal;
    analysis->dual_step = d.theta_dual * cost_scale_factor;
    analysis->pivot_value_from_column = d.alpha_col;
    analysis->pivot_value_from_row = d.alpha_row;
    analysis->factor_pivot_threshold = s->factor_pivot_threshold;
    analysis->numerical_trouble = d.numericalTrouble;
    analysis->edge_weight_error = s->edge_weight_error;
    analysis->objective_value = s->updated_dual_objective_value;
    if (d.solve_phase == kSolvePhase2)
      analysis->objective_value *= (HighsInt)e.lp_.sense_;
    analysis->num_primal_infeasibility = s->num_primal_infeasibilities;
    analysis->sum_primal_infeasibility = s->sum_primal_infeasibilities;
    analysis->num_dual_infeasibility = s->num_dual_infeasibilities;
    analysis->sum_dual_infeasibility = s->sum_dual_infeasibilities;
    analysis->col_aq_density = s->col_aq_density;
    analysis->row_ep_density = s->row_ep_density;
    analysis->row_ap_density = s->row_ap_density;
    analysis->row_DSE_density = s->row_dse_density;
    analysis->col_basic_feasibility_change_density =
        info.col_basic_feasibility_change_density;
    analysis->row_basic_feasibility_change_density =
        info.row_basic_feasibility_change_density;
    analysis->col_BFRT_density = s->col_bfrt_density;
    analysis->primal_col_density = s->primal_col_density;
    analysis->dual_col_density = s->dual_col_density;
    analysis->num_costly_DSE_iteration = s->num_costly_dse_iteration;
    analysis->costly_DSE_measure = s->costly_dse_measure;
  }

  static void chuzcFail(void* ctx, int kind, int work_count,
                        const std::pair<HighsInt, double>* work_data,
                        double select_theta, double remain_theta) {
    HEkk& e = ekk(ctx);
    const std::vector<std::pair<HighsInt, double>> data(work_data,
                                                        work_data + work_count);
    const HighsInt num_tot = e.lp_.num_col_ + e.lp_.num_row_;
    if (kind == 1) {
      debugDualChuzcFailQuad0(*e.options_, work_count, data, num_tot,
                              e.info_.workDual_.data(), select_theta,
                              remain_theta, true);
    } else {
      debugDualChuzcFailQuad1(*e.options_, work_count, data, num_tot,
                              e.info_.workDual_.data(), select_theta, true);
    }
  }

  static double timerRead(void* ctx) { return ekk(ctx).timer_->read(); }

  // The user interrupt part of HEkk::bailout
  static bool interrupt(void* ctx) {
    HEkk& e = ekk(ctx);
    HighsCallback& callback = *e.callback_;
    callback.clearHighsCallbackOutput();
    callback.data_out.simplex_iteration_count = e.iteration_count_;
    if (callback.callbackAction(kCallbackSimplexInterrupt,
                                "Simplex interrupt")) {
      highsLogDev(e.options_->log_options, HighsLogType::kInfo,
                  "User interrupt\n");
      e.solve_bailout_ = true;
      e.model_status_ = HighsModelStatus::kInterrupt;
    }
    return e.solve_bailout_;
  }

  static double factorSolveError(void* ctx) {
    return ekk(ctx).factorSolveError();
  }

  static bool getNonsingularInverse(void* ctx, int solve_phase) {
    return ekk(ctx).getNonsingularInverse(solve_phase);
  }

  static void initialisePartitionedRowwiseMatrix(void* ctx) {
    HEkk& e = ekk(ctx);
    assert(e.info_.backtracking_);
    e.initialisePartitionedRowwiseMatrix();
    assert(e.ar_matrix_.debugPartitionOk(e.basis_.nonbasicFlag_.data()));
  }

  static void putBacktrackingBasis(void* ctx) {
    ekk(ctx).putBacktrackingBasis();
  }

  static bool restoreDualEdgeWeights(void* ctx, bool near_optimal) {
    return ekk(ctx).restoreDualEdgeWeights(near_optimal);
  }

  static void edgeWeightVectors(void* ctx, bool assign_unit_weights) {
    HEkk& e = ekk(ctx);
    const HighsInt num_row = e.lp_.num_row_;
    const HighsInt num_tot = e.lp_.num_col_ + num_row;
    if (assign_unit_weights) {
      e.dual_edge_weight_.assign(num_row, 1.0);
      e.scattered_dual_edge_weight_.resize(num_tot);
    }
    e.info_.backtracking_basis_edge_weight_.resize(num_tot);
    e.info_.devex_index_.resize(num_tot);
  }

  static void clearRefactorInfo(void* ctx) {
    ekk(ctx).simplex_nla_.factor_.refactor_info_.clear();
  }

  static void setPivotThreshold(void* ctx, double new_pivot_threshold) {
    HEkk& e = ekk(ctx);
    highsLogUser(e.options_->log_options, HighsLogType::kWarning,
                 "   Increasing Markowitz threshold to %g\n",
                 new_pivot_threshold);
    e.info_.factor_pivot_threshold = new_pivot_threshold;
    e.simplex_nla_.setPivotThreshold(new_pivot_threshold);
  }

  // The C++ part of HEkkDual::improveChooseColumnRow: refine row_ep and
  // compute row_ap in quad precision
  static void improveChooseColumnRow(void* ctx, int row_out) {
    HEkkDual& d = dual(ctx);
    d.rs_row_ep_->pull();
    d.rs_row_ap_->pull();
    d.ekk_instance_.unitBtranIterativeRefinement(row_out, d.row_ep);
    const bool quad_precision = true;
    d.ekk_instance_.tableauRowPrice(quad_precision, d.row_ep, d.row_ap);
    d.rs_row_ep_->push();
    d.rs_row_ap_->push();
  }

  static bool proofOfPrimalInfeasibility(void* ctx, int move_out,
                                         int row_out) {
    HEkkDual& d = dual(ctx);
    d.rs_row_ep_->pull();
    const bool proof =
        d.ekk_instance_.proofOfPrimalInfeasibility(d.row_ep, move_out, row_out);
    d.rs_row_ep_->push();
    return proof;
  }

  static void saveDualRay(void* ctx, int row_out, int move_out) {
    HEkk& e = ekk(ctx);
    e.dual_ray_record_.clear();
    e.dual_ray_record_.index = row_out;
    e.dual_ray_record_.sign = move_out;
  }

  // Clean up dual infeasibilities with the primal simplex: returns the
  // status of the call
  static int primalCleanup(void* ctx) {
    HEkkDual& d = dual(ctx);
    HEkk& e = d.ekk_instance_;
    HighsOptions& options = *e.options_;
    HighsSimplexInfo& info = e.info_;
    highsLogDev(options.log_options, HighsLogType::kInfo,
                "HEkkDual:: Using primal simplex to try to clean up num / "
                "max / sum = %" HIGHSINT_FORMAT
                " / %g / %g dual infeasibilities\n",
                info.num_dual_infeasibilities, info.max_dual_infeasibility,
                info.sum_dual_infeasibilities);
    HighsStatus return_status = HighsStatus::kOk;
    // Switch off any bound perturbation
    double save_primal_simplex_bound_perturbation_multiplier =
        info.primal_simplex_bound_perturbation_multiplier;
    info.primal_simplex_bound_perturbation_multiplier = 0;
    HEkkPrimal primal_solver(e);
    HighsStatus call_status = primal_solver.solve(true);
    // Restore any bound perturbation
    info.primal_simplex_bound_perturbation_multiplier =
        save_primal_simplex_bound_perturbation_multiplier;
    assert(e.called_return_from_solve_);
    return_status = interpretCallStatus(options.log_options, call_status,
                                        return_status, "HEkkPrimal::solve");
    // Reset called_return_from_solve_ to be false, since it's called for
    // this solve
    e.called_return_from_solve_ = false;
    if (return_status != HighsStatus::kOk) return (int)return_status;
    if (e.model_status_ == HighsModelStatus::kOptimal &&
        info.num_primal_infeasibilities + info.num_dual_infeasibilities)
      highsLogDev(options.log_options, HighsLogType::kWarning,
                  "HEkkDual:: Primal simplex clean up yields optimality, "
                  "but with %" HIGHSINT_FORMAT
                  " (max %g) primal infeasibilities and %" HIGHSINT_FORMAT
                  " (max %g) dual infeasibilities\n",
                  info.num_primal_infeasibilities,
                  info.max_primal_infeasibility, info.num_dual_infeasibilities,
                  info.max_dual_infeasibility);
    return (int)HighsStatus::kOk;
  }

  static void recordDualValues(void* ctx) {
    HEkk& e = ekk(ctx);
    e.dual_values_valid_ = true;
    e.dual_values_scaled_ = e.lp_.is_scaled_;
    e.dual_values_basis_hash_ = e.basis_.hash;
    e.dual_values_cost_hash_ = e.costHash();
  }
};

bool HEkkDual::rustEligible() const {
  const HEkk& e = ekk_instance_;
  const HighsSimplexAnalysis& a = e.analysis_;
  return e.info_.simplex_strategy == kSimplexStrategyDualPlain &&
         !a.analyse_simplex_summary_data && !a.analyse_simplex_runtime_data &&
         !a.analyse_simplex_time &&
         e.options_->highs_debug_level < kHighsDebugLevelCheap &&
         !e.debug_solve_report_ && !e.time_report_ &&
         !e.simplex_nla_.update_.valid_;
}

HighsStatus HEkkDual::solveRust(const bool force_phase2) {
  HEkk& e = ekk_instance_;
  const HighsOptions& options = *e.options_;
  HighsSimplexInfo& info = e.info_;
  HighsSimplexStatus& status = e.status_;
  highs_rs::DualEkk x;
  x.iteration_count = &e.iteration_count_;
  x.model_status = reinterpret_cast<int*>(&e.model_status_);
  x.solve_bailout = &e.solve_bailout_;
  x.called_return_from_solve = &e.called_return_from_solve_;
  x.dual_values_valid = &e.dual_values_valid_;
  x.fresh_unperturbed_dual = &e.fresh_unperturbed_dual_;
  x.fresh_dual = &e.fresh_dual_;
  x.fresh_primal = &e.fresh_primal_;
  x.edge_weight_error = &e.edge_weight_error_;
  x.dual_simplex_cleanup_level = &e.dual_simplex_cleanup_level_;
  x.dual_simplex_phase1_cleanup_level = &e.dual_simplex_phase1_cleanup_level_;
  x.previous_iteration_cycling_detected =
      &e.previous_iteration_cycling_detected;
  x.random = reinterpret_cast<uint64_t*>(&e.random_);
  x.basis_records = e.basis_records_.p;
  x.nla_build_synthetic_tick = &e.simplex_nla_.build_synthetic_tick_;
  x.has_invert = &status.has_invert;
  x.has_fresh_invert = &status.has_fresh_invert;
  x.has_fresh_rebuild = &status.has_fresh_rebuild;
  x.has_dual_objective_value = &status.has_dual_objective_value;
  x.has_primal_objective_value = &status.has_primal_objective_value;
  x.has_dual_steepest_edge_weights = &status.has_dual_steepest_edge_weights;
  x.has_ar_matrix = &status.has_ar_matrix;
  x.dual_phase1_iteration_count = &info.dual_phase1_iteration_count;
  x.dual_phase2_iteration_count = &info.dual_phase2_iteration_count;
  x.allow_cost_shifting = &info.allow_cost_shifting;
  x.allow_cost_perturbation = &info.allow_cost_perturbation;
  x.backtracking = &info.backtracking_;
  x.valid_backtracking_basis = &info.valid_backtracking_basis_;
  x.store_squared_primal_infeasibility =
      &info.store_squared_primal_infeasibility;
  x.factor_pivot_threshold = &info.factor_pivot_threshold;
  x.col_bfrt_density = &info.col_BFRT_density;
  x.costly_dse_measure = &info.costly_DSE_measure;
  x.costly_dse_frequency = &info.costly_DSE_frequency;
  x.num_costly_dse_iteration = &info.num_costly_DSE_iteration;
  x.average_log_low_dse_weight_error = &info.average_log_low_DSE_weight_error;
  x.average_log_high_dse_weight_error = &info.average_log_high_DSE_weight_error;
  x.control_iteration_count0 = info.control_iteration_count0;
  x.allow_dual_steepest_edge_to_devex_switch =
      info.allow_dual_steepest_edge_to_devex_switch;
  x.dual_steepest_edge_weight_log_error_threshold =
      info.dual_steepest_edge_weight_log_error_threshold;
  x.dual_edge_weight_strategy = info.dual_edge_weight_strategy;
  x.run_quiet = info.run_quiet;
  x.objective_bound = options.objective_bound;
  x.time_limit = options.time_limit;
  x.simplex_iteration_limit = options.simplex_iteration_limit;
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
  const HighsLogOptions& log_options = options.log_options;
  x.dev_log = *log_options.output_flag && *log_options.log_dev_level;
  x.iteration_report =
      *log_options.log_dev_level >= (HighsInt)kIterationReportLogType;
  x.interrupt_callback = e.callback_->user_callback &&
                         e.callback_->active[kCallbackSimplexInterrupt];

  highs_rs::DualCallbacks cb;
  cb.ctx = this;
  cb.refresh = RustGlue::refresh;
  cb.start = RustGlue::start;
  cb.return_from_solve = RustGlue::returnFromSolve;
  cb.log = RustGlue::log;
  cb.log_cost_perturbation = RustGlue::logCostPerturbation;
  cb.set_lp_dual_infeasibility = RustGlue::setLpDualInfeasibility;
  cb.report_rebuild = RustGlue::reportRebuild;
  cb.iteration_report = RustGlue::iterationReport;
  cb.apply_analysis_data = RustGlue::applyAnalysisData;
  cb.chuzc_fail = RustGlue::chuzcFail;
  cb.timer_read = RustGlue::timerRead;
  cb.interrupt = RustGlue::interrupt;
  cb.factor_solve_error = RustGlue::factorSolveError;
  cb.get_nonsingular_inverse = RustGlue::getNonsingularInverse;
  cb.initialise_partitioned_rowwise_matrix =
      RustGlue::initialisePartitionedRowwiseMatrix;
  cb.put_backtracking_basis = RustGlue::putBacktrackingBasis;
  cb.restore_dual_edge_weights = RustGlue::restoreDualEdgeWeights;
  cb.edge_weight_vectors = RustGlue::edgeWeightVectors;
  cb.clear_refactor_info = RustGlue::clearRefactorInfo;
  cb.set_pivot_threshold = RustGlue::setPivotThreshold;
  cb.improve_choose_column_row = RustGlue::improveChooseColumnRow;
  cb.proof_of_primal_infeasibility = RustGlue::proofOfPrimalInfeasibility;
  cb.save_dual_ray = RustGlue::saveDualRay;
  cb.primal_cleanup = RustGlue::primalCleanup;
  cb.record_dual_values = RustGlue::recordDualValues;

  row_ep.next = nullptr;
  row_ap.next = nullptr;
  col_aq.next = nullptr;
  col_BFRT.next = nullptr;
  int return_status;
  {
    highs_rs::HVecCall ep(row_ep);
    highs_rs::HVecCall ap(row_ap);
    highs_rs::HVecCall aq(col_aq);
    highs_rs::HVecCall bfrt(col_BFRT);
    rs_row_ep_ = &ep;
    rs_row_ap_ = &ap;
    const highs_rs::DualVectors v{ep.get(), ap.get(), aq.get(), bfrt.get()};
    return_status = highs_rs_dual_solve(&x, &cb, &v, force_phase2);
    rs_row_ep_ = nullptr;
    rs_row_ap_ = nullptr;
  }
  return (HighsStatus)return_status;
}

#endif  // HIGHS_RUST
