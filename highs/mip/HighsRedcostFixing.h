/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsRedcostFixing.h
 * @brief reduced cost fixing using the current cutoff bound
 */

#ifndef HIGHS_REDCOST_FIXING_H_
#define HIGHS_REDCOST_FIXING_H_

#include <map>
#include <utility>
#include <vector>

#include "HighsConflictPool.h"
#include "mip/HighsDomainChange.h"

class HighsDomain;
class HighsMipSolver;
class HighsLpRelaxation;

#ifdef HIGHS_RUST
class HighsPseudocost;
namespace highs_rs {
struct RedcostFixing;
extern "C" {
RedcostFixing* highs_rs_redcost_new();
void highs_rs_redcost_free(RedcostFixing* r);
}
}  // namespace highs_rs

// The lurking bounds are Rust's (rust/src/mip/redcost.rs); this class is a
// handle
class HighsRedcostFixing {
  highs_rs::RedcostFixing* rs_;

 public:
  HighsRedcostFixing() : rs_(highs_rs::highs_rs_redcost_new()) {}
  HighsRedcostFixing(const HighsRedcostFixing&) = delete;
  HighsRedcostFixing& operator=(const HighsRedcostFixing&) = delete;
  HighsRedcostFixing(HighsRedcostFixing&& other) : rs_(other.rs_) {
    other.rs_ = nullptr;
  }
  HighsRedcostFixing& operator=(HighsRedcostFixing&& other) {
    std::swap(rs_, other.rs_);
    return *this;
  }
  ~HighsRedcostFixing() { highs_rs::highs_rs_redcost_free(rs_); }

  highs_rs::RedcostFixing* rust() const { return rs_; }

  std::vector<std::pair<double, HighsDomainChange>> getLurkingBounds(
      const HighsMipSolver& mipsolver, const HighsDomain& globaldom) const;

  void propagateRootRedcost(const HighsMipSolver& mipsolver);

  static void propagateRedCost(const HighsMipSolver& mipsolver,
                               HighsDomain& localdomain, HighsDomain& globaldom,
                               const HighsLpRelaxation& lp,
                               HighsConflictPool& conflictpool,
                               HighsPseudocost& pseudocost, double upper_limit);

  void addRootRedcost(const HighsMipSolver& mipsolver,
                      const std::vector<double>& lpredcost, double lpobjective);
  // of the LP solution's reduced costs in place (computeBasicDegenerateDuals
  // changes them before they are read)
  void addRootRedcost(const HighsMipSolver& mipsolver,
                      const double* lpredcost, double lpobjective);
};
#else
class HighsRedcostFixing {
  std::vector<std::multimap<double, HighsInt>> lurkingColUpper;
  std::vector<std::multimap<double, HighsInt>> lurkingColLower;

 public:
  std::vector<std::pair<double, HighsDomainChange>> getLurkingBounds(
      const HighsMipSolver& mipsolver, const HighsDomain& globaldom) const;

  void propagateRootRedcost(const HighsMipSolver& mipsolver);

  static void propagateRedCost(const HighsMipSolver& mipsolver,
                               HighsDomain& localdomain, HighsDomain& globaldom,
                               const HighsLpRelaxation& lp,
                               HighsConflictPool& conflictpool,
                               HighsPseudocost& pseudocost, double upper_limit);

  void addRootRedcost(const HighsMipSolver& mipsolver,
                      const std::vector<double>& lpredcost, double lpobjective);
};

#endif  // HIGHS_RUST

#endif
