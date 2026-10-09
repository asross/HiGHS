/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsMipHost.h
 * @brief The MIP solver of a HIGHS_RUST build is Rust's
 * (rust/src/mip/host): Highs::callSolveMip and Highs::runPresolve pass it
 * the model, the options and the Highs object's callback and profiling,
 * and take its result. HighsMipHost.cpp is the Highs object's side: the
 * profiling clocks, the user callback and the improving solution file.
 */
#ifndef MIP_HIGHS_MIP_HOST_H_
#define MIP_HIGHS_MIP_HOST_H_

#include "HConfig.h"

#ifdef HIGHS_RUST
#include <vector>

#include "lp_data/HStruct.h"
#include "lp_data/HighsCallback.h"
#include "lp_data/HighsLp.h"
#include "lp_data/HighsOptions.h"
#include "lp_data/HighsSolution.h"
#include "presolve/HighsPostsolveStack.h"

/// A MIP solve: the result fields of HighsMipSolver that Highs reads
struct HighsMipRun {
  HighsMipRun(HighsCallback& callback, const HighsOptions& options,
              const HighsLp& lp, const HighsSolution& solution,
              HighsProfiling* profiling);

  HighsModelStatus modelstatus_;
  double solution_objective_;
  int64_t node_count_;
  int64_t total_lp_iterations_;
  double dual_bound_;
  double primal_bound_;
  double gap_;
  double primal_dual_integral_;
  double row_violation_;
  double bound_violation_;
  double integrality_violation_;
  std::vector<double> solution_;
  std::vector<HighsObjectiveSolution> saved_objective_and_solution_;
};

/// runMipPresolve: the presolved model and postsolve stack (the model's
/// other members, and its names through the index maps), and the presolve
/// status
HighsPresolveStatus highsMipPresolve(HighsCallback& callback,
                                     const HighsOptions& options,
                                     const HighsLp& lp,
                                     const HighsSolution& solution,
                                     HighsProfiling* profiling,
                                     HighsLp& presolved,
                                     presolve::HighsPostsolveStack& stack);

#endif  // HIGHS_RUST
#endif
