/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsMipSolverData.h"
#include "mip/feasibilityjump.hh"
#include "util/HighsSparseMatrix.h"

#ifdef HIGHS_RUST
// rust/src/mip/feasjump.rs
struct FjRsProblem {
  int num_col;
  int num_row;
  const double* col_lower;
  const double* col_upper;
  const double* col_cost;
  const uint8_t* col_integer;
  const HighsInt* ar_start;
  const HighsInt* ar_index;
  const double* ar_value;
  const double* row_lower;
  const double* row_upper;
  uint32_t seed;
  double equality_tolerance;
  double violation_tolerance;
  uint64_t max_total_effort;
  uint64_t max_effort_since_improvement;
};
extern "C" int highs_rs_feasibility_jump(const FjRsProblem* p, double* x,
                                         int logging_on, void* log_ctx,
                                         void (*log)(void*, int, const char*));
static void fjRsLog(void* ctx, int type, const char* msg) {
  highsLogDev(*static_cast<const HighsLogOptions*>(ctx), HighsLogType(type),
              "%s", msg);
}
#endif

HighsModelStatus HighsMipSolverData::feasibilityJump() {
  // This is the (presolved) model being solved
  const HighsLp* model = this->mipsolver.model_;
  const HighsLogOptions& log_options = mipsolver.options_mip_->log_options;
  double sense_multiplier = static_cast<double>(model->sense_);

#ifdef HIGHSINT64
  // TODO(BenChampion,9999-12-31): make FJ work with 64-bit HighsInt
  highsLogUser(log_options, HighsLogType::kInfo,
               "Feasibility Jump code isn't currently compatible "
               "with a 64-bit HighsInt: skipping Feasibility Jump\n");
  return HighsModelStatus::kNotset;
#else

  bool found_integer_feasible_solution = false;
  std::vector<double> col_value(model->num_col_, 0.0);
  double objective_function_value;

  const bool use_incumbent = !incumbent.empty();

#ifdef HIGHS_RUST
  std::vector<double> fj_lower(model->num_col_), fj_upper(model->num_col_),
      fj_cost(model->num_col_);
  std::vector<uint8_t> fj_integer(model->num_col_);
#else
  // Configure Feasibility Jump and pass it the problem
  auto solver = external_feasibilityjump::FeasibilityJumpSolver(
      log_options,
      /* seed = */ mipsolver.options_mip_->random_seed,
      /* equalityTolerance = */ epsilon,
      /* violationTolerance = */ feastol);
#endif

  for (HighsInt col = 0; col < model->num_col_; ++col) {
    double lower = model->col_lower_[col];
    double upper = model->col_upper_[col];

    assert(model->integrality_[col] == HighsVarType::kContinuous ||
           model->integrality_[col] == HighsVarType::kInteger ||
           model->integrality_[col] == HighsVarType::kImplicitInteger);
    external_feasibilityjump::VarType fjVarType;
    if (model->integrality_[col] == HighsVarType::kContinuous) {
      fjVarType = external_feasibilityjump::VarType::Continuous;
    } else {
      fjVarType = external_feasibilityjump::VarType::Integer;
      lower = std::ceil(lower - feastol);
      upper = std::floor(upper + feastol);
    }

    const bool legal_bounds = lower <= upper && lower < kHighsInf &&
                              upper > -kHighsInf && !std::isnan(lower) &&
                              !std::isnan(upper);
    if (!legal_bounds) {
      highsLogUser(log_options, HighsLogType::kInfo,
                   "HighsMipSolverData::feasibilityJump() has detected "
                   "infeasible/illegal bounds [%g, %g] for "
                   "column %d: MIP is infeasible\n",
                   lower, upper, int(col));
      assert(legal_bounds);
      return HighsModelStatus::kInfeasible;
    }
#ifdef HIGHS_RUST
    fj_lower[col] = lower;
    fj_upper[col] = upper;
    fj_cost[col] = sense_multiplier * model->col_cost_[col];
    fj_integer[col] = fjVarType == external_feasibilityjump::VarType::Integer;
#else
    solver.addVar(fjVarType, lower, upper,
                  sense_multiplier * model->col_cost_[col]);
#endif

    double initial_assignment = 0.0;
    if (use_incumbent && std::isfinite(incumbent[col])) {
      initial_assignment = std::max(lower, std::min(upper, incumbent[col]));
    } else {
      if (std::isfinite(lower)) {
        initial_assignment = lower;
      } else if (std::isfinite(upper)) {
        initial_assignment = upper;
      }
    }
    col_value[col] = initial_assignment;
  }

  HighsSparseMatrix a_matrix;
  a_matrix.createRowwise(model->a_matrix_);

#ifdef HIGHS_RUST
  const HighsInt nnz = a_matrix.numNz();
  const FjRsProblem problem = {int(model->num_col_),
                               int(model->num_row_),
                               fj_lower.data(),
                               fj_upper.data(),
                               fj_cost.data(),
                               fj_integer.data(),
                               a_matrix.start_.data(),
                               a_matrix.index_.data(),
                               a_matrix.value_.data(),
                               model->row_lower_.data(),
                               model->row_upper_.data(),
                               uint32_t(mipsolver.options_mip_->random_seed),
                               epsilon,
                               feastol,
                               uint64_t(nnz) << 10,
                               uint64_t(nnz) << 8};
  const bool logging_on =
      *log_options.output_flag && *log_options.log_dev_level;
  found_integer_feasible_solution =
      highs_rs_feasibility_jump(
          &problem, col_value.data(), logging_on,
          const_cast<HighsLogOptions*>(&log_options), fjRsLog) != 0;
  (void)objective_function_value;
#else
  for (HighsInt row = 0; row < model->num_row_; ++row) {
    bool hasFiniteLower = std::isfinite(model->row_lower_[row]);
    bool hasFiniteUpper = std::isfinite(model->row_upper_[row]);
    if (hasFiniteLower || hasFiniteUpper) {
      HighsInt row_num_nz = a_matrix.start_[row + 1] - a_matrix.start_[row];
      auto row_index = a_matrix.index_.data() + a_matrix.start_[row];
      auto row_value = a_matrix.value_.data() + a_matrix.start_[row];
      if (hasFiniteLower) {
        solver.addConstraint(external_feasibilityjump::RowType::Gte,
                             model->row_lower_[row], row_num_nz, row_index,
                             row_value, /* relax_continuous = */ 0);
      }
      if (hasFiniteUpper) {
        solver.addConstraint(external_feasibilityjump::RowType::Lte,
                             model->row_upper_[row], row_num_nz, row_index,
                             row_value, /* relax_continuous = */ 0);
      }
    }
  }

  const HighsInt nnz = a_matrix.numNz();
  const size_t kMaxTotalEffort = (size_t)nnz << 10;
  const size_t kMaxEffortSinceLastImprovement = (size_t)nnz << 8;

  auto fjControlCallback =
      [=, &col_value, &found_integer_feasible_solution,
       &objective_function_value](external_feasibilityjump::FJStatus status)
      -> external_feasibilityjump::CallbackControlFlow {
    if (status.solution != nullptr) {
      found_integer_feasible_solution = true;
      col_value = std::vector<double>(status.solution,
                                      status.solution + status.numVars);
      objective_function_value =
          model->offset_ + sense_multiplier * status.solutionObjectiveValue;
    }
    if (status.effortSinceLastImprovement > kMaxEffortSinceLastImprovement ||
        status.totalEffort > kMaxTotalEffort) {
      return external_feasibilityjump::CallbackControlFlow::Terminate;
    } else {
      return external_feasibilityjump::CallbackControlFlow::Continue;
    }
  };

  solver.solve(col_value.data(), fjControlCallback);
#endif

  if (found_integer_feasible_solution) {
    // Initial assignments that violate integrality or column bounds can lead to
    // infeasible results. Even if those initial assignments should not occur,
    // use trySolution rather than addIncumbent for an explicit check.
    trySolution(col_value, kSolutionSourceFeasibilityJump);
  }
  return HighsModelStatus::kNotset;
#endif
}
