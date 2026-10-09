/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file lp_data/HighsRanging.h
 * @brief
 */
#ifndef LP_DATA_HIGHS_RANGING_H_
#define LP_DATA_HIGHS_RANGING_H_

#include <vector>

#include "HConfig.h"

#ifdef HIGHS_RUST
#include "lp_data/HighsLp.h"
#include "lp_data/HighsOptions.h"
#include "lp_data/HighsStatus.h"
struct HighsEngine;
#else
#include "lp_data/HighsLpSolverObject.h"
#endif

struct HighsRangingRecord {
  std::vector<double> value_;
  std::vector<double> objective_;
  std::vector<HighsInt> in_var_;
  std::vector<HighsInt> ou_var_;
};

struct HighsRanging {
  bool valid = false;
  HighsRangingRecord col_cost_up;
  HighsRangingRecord col_cost_dn;
  HighsRangingRecord col_bound_up;
  HighsRangingRecord col_bound_dn;
  HighsRangingRecord row_bound_up;
  HighsRangingRecord row_bound_dn;
  void invalidate();
  void clear();
};

#ifdef HIGHS_RUST
// getRangingData of a Highs object's LP and solve
HighsStatus getRangingData(HighsRanging& ranging, HighsEngine& ekk,
                           const HighsOptions& options, const HighsLp& lp,
                           const HighsBasis& basis,
                           const HighsSolution& solution,
                           const HighsModelStatus model_status,
                           const double objective_function_value);
#else
HighsStatus getRangingData(HighsRanging& ranging,
                           HighsLpSolverObject& solver_object);
#endif
void writeRangingFile(FILE* file, const HighsLp& lp,
                      const double objective_function_value,
                      const HighsBasis& basis, const HighsSolution& solution,
                      const HighsRanging& ranging, const HighsInt style);
#endif
