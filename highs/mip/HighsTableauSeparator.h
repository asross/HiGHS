/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsTableauSeparator.h
 * @brief Class for separating cuts from the LP tableaux rows
 *
 */

#ifndef MIP_HIGHS_TABLEAU_SEPARATOR_H_
#define MIP_HIGHS_TABLEAU_SEPARATOR_H_

#include "mip/HighsSeparator.h"

/// Helper class to compute single-row relaxations from the current LP
/// relaxation by substituting bounds and aggregating rows
#ifdef HIGHS_RUST
namespace highs_rs {
struct TableauSeparator;
}

/// Runs in Rust (rust/src/mip/cuts/tableau.rs), which holds numTries and
/// the work space
class HighsTableauSeparator : public HighsSeparator {
 private:
  highs_rs::TableauSeparator* rs_;

 public:
  void separateLpSolution(HighsLpRelaxation& lpRelaxation,
                          HighsLpAggregator& lpAggregator,
                          HighsTransformedLp& transLp,
                          HighsCutPool& cutpool) override;

  HighsTableauSeparator(const HighsMipSolver& mipsolver);
  ~HighsTableauSeparator() override;
  HighsTableauSeparator(const HighsTableauSeparator&) = delete;
  HighsTableauSeparator& operator=(const HighsTableauSeparator&) = delete;
};
#else
class HighsTableauSeparator : public HighsSeparator {
 private:
  int64_t numTries;

 public:
  void separateLpSolution(HighsLpRelaxation& lpRelaxation,
                          HighsLpAggregator& lpAggregator,
                          HighsTransformedLp& transLp,
                          HighsCutPool& cutpool) override;

  HighsTableauSeparator(const HighsMipSolver& mipsolver)
      : HighsSeparator(mipsolver, kTableauSepaString), numTries(0) {}
};
#endif

#endif
