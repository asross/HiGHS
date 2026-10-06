/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsTransformedLp.h
 * @brief LP transformations useful for cutting plane separation. This includes
 * bound substitution with simple and variable bounds, handling of slack
 * variables, flipping the complementation of integers.
 */

#ifndef MIP_HIGHS_TRANSFORMED_LP_H_
#define MIP_HIGHS_TRANSFORMED_LP_H_

#include <vector>

#include "lp_data/HConst.h"
#include "mip/HighsImplications.h"
#include "util/HighsCDouble.h"
#include "util/HighsInt.h"
#include "util/HighsSparseVectorSum.h"

class HighsLpRelaxation;

#ifdef HIGHS_RUST
#include <memory>

class HighsCutPool;
namespace highs_rs {
struct SepaRound;
}
struct HighsSepaRoundCtx;

/// The data of a separation round, held by the Rust port
/// (rust/src/mip/cuts/round.rs); the separators run in Rust on it
/// (mip/HighsSeparationRust.cpp)
class HighsTransformedLp {
 private:
  const HighsDomain& globaldom_;
  std::unique_ptr<HighsSepaRoundCtx> ctx_;
  highs_rs::SepaRound* rs_ = nullptr;

 public:
  HighsTransformedLp(const HighsLpRelaxation& lprelaxation,
                     HighsImplications& implications,
                     const HighsDomain& globaldom);
  ~HighsTransformedLp();
  HighsTransformedLp(const HighsTransformedLp&) = delete;
  HighsTransformedLp& operator=(const HighsTransformedLp&) = delete;

  /// the Rust round, with the cut pool that cuts go to
  highs_rs::SepaRound* rust(HighsCutPool& cutpool);
  HighsSepaRoundCtx& ctx() { return *ctx_; }

  const HighsDomain& getGlobaldom() const { return globaldom_; }
};
#else
/// Helper class to compute single-row relaxations from the current LP
/// relaxation by substituting bounds and aggregating rows
class HighsTransformedLp {
 private:
  const HighsLpRelaxation& lprelaxation;
  const HighsDomain& globaldom_;

  std::vector<std::pair<HighsInt, HighsImplications::VarBound>> bestVub;
  std::vector<std::pair<HighsInt, HighsImplications::VarBound>> bestVlb;
  std::vector<double> simpleLbDist;
  std::vector<double> simpleUbDist;
  std::vector<double> lbDist;
  std::vector<double> ubDist;
  std::vector<double> boundDist;
  enum class BoundType : uint8_t {
    kSimpleUb,
    kSimpleLb,
    kVariableUb,
    kVariableLb,
  };
  std::vector<BoundType> boundTypes;
  // whether the column can make a transformed base row cut off the LP
  // solution (see isFractional)
  std::vector<uint8_t> fractional;
  HighsSparseVectorSum vectorsum;

 public:
  HighsTransformedLp(const HighsLpRelaxation& lprelaxation,
                     HighsImplications& implications,
                     const HighsDomain& globaldom);

  double boundDistance(HighsInt col) const { return boundDist[col]; }

  // A base row cannot give a cut that the LP solution violates unless one of
  // its columns is an integer column at a fractional value, or has a
  // variable bound whose substitution brings one in (or that the solution
  // violates): otherwise the transformed LP solution is a point of the
  // mixed-integer set that every such cut is valid for.
  bool isFractional(HighsInt col) const { return fractional[col]; }

  bool transform(std::vector<double>& vals, std::vector<double>& upper,
                 std::vector<double>& solval, std::vector<HighsInt>& inds,
                 double& rhs, bool& integralPositive, bool preferVbds = false);

  bool untransform(std::vector<double>& vals, std::vector<HighsInt>& inds,
                   double& rhs, bool integral = false);

  const HighsDomain& getGlobaldom() const { return globaldom_; }
};
#endif

#endif
