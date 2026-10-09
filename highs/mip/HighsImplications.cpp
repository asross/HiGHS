/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsImplications.h"

#include "../extern/pdqsort/pdqsort.h"
#include "mip/HighsCliqueTable.h"
#include "mip/HighsMipSolverData.h"
#include "mip/MipTimer.h"
#ifndef HIGHS_RUST


bool HighsImplications::computeImplications(HighsInt col, bool val) {
  HighsDomain& globaldomain = mipsolver.mipdata_->getDomain();
  HighsCliqueTable& cliquetable = mipsolver.mipdata_->cliquetable;
  globaldomain.propagate();
  if (globaldomain.infeasible() || globaldomain.isFixed(col)) return true;

  // record redundant rows for lifting
  assert(globaldomain.getRedundantRows().size() == 0);
  if (storeLiftingOpportunity != nullptr)
    globaldomain.setRecordRedundantRows(true);

  const auto& domchgstack = globaldomain.getDomainChangeStack();
  const auto& domchgreason = globaldomain.getDomainChangeReason();
  size_t changedend = globaldomain.getChangedCols().size();

  HighsInt stackimplicstart = domchgstack.size() + 1;
  HighsInt numImplications = -stackimplicstart;
  if (val)
    globaldomain.changeBound(HighsBoundType::kLower, col, 1);
  else
    globaldomain.changeBound(HighsBoundType::kUpper, col, 0);

  auto storeLiftingOpportunities = [&](HighsInt col, bool val) {
    // use callback to store lifting opportunities
    if (storeLiftingOpportunity != nullptr) {
      for (const auto& elm : globaldomain.getRedundantRows())
        storeLiftingOpportunity(
            elm.key(), col, val ? 1 : 0,
            (val ? -1 : 1) * globaldomain.getRedundantRowValue(elm.key()));
      globaldomain.clearRedundantRows();
      globaldomain.setRecordRedundantRows(false);
    }
  };

  auto doBacktrack = [&](size_t changedend) {
    globaldomain.backtrack();
    globaldomain.clearChangedCols(changedend);
  };

  auto isInfeasible = [&](HighsInt col, bool val) {
    if (!globaldomain.infeasible()) return false;
    storeLiftingOpportunities(col, val);
    doBacktrack(changedend);
    cliquetable.vertexInfeasible(globaldomain, col, val);
    return true;
  };

  if (isInfeasible(col, val)) return true;

  globaldomain.propagate();

  if (isInfeasible(col, val)) return true;

  HighsInt stackimplicend = domchgstack.size();
  numImplications += stackimplicend;
  mipsolver.mipdata_->getPseudoCost().addInferenceObservation(
      col, numImplications, val);

  std::vector<HighsDomainChange> implics;
  implics.reserve(numImplications);

  HighsInt numEntries = mipsolver.mipdata_->cliquetable.getNumEntries();
  HighsInt maxEntries = 100000 + mipsolver.numNonzero();

  for (HighsInt i = stackimplicstart; i < stackimplicend; ++i) {
    if (domchgreason[i].type == HighsDomain::Reason::kCliqueTable &&
        ((domchgreason[i].index >> 1) == col || numEntries >= maxEntries))
      continue;

    implics.push_back(domchgstack[i]);
  }

  // inform caller about lifting opportunities
  storeLiftingOpportunities(col, val);

  // backtrack
  doBacktrack(changedend);

  // add the implications of binary variables to the clique table
  auto binstart = std::partition(implics.begin(), implics.end(),
                                 [&](const HighsDomainChange& a) {
                                   return !globaldomain.isBinary(a.column);
                                 });

  pdqsort(implics.begin(), binstart);

  std::array<HighsCliqueTable::CliqueVar, 2> clique;
  clique[0] = HighsCliqueTable::CliqueVar(col, val);

  for (auto i = binstart; i != implics.end(); ++i) {
    if (i->boundtype == HighsBoundType::kLower)
      clique[1] = HighsCliqueTable::CliqueVar(i->column, 0);
    else
      clique[1] = HighsCliqueTable::CliqueVar(i->column, 1);

    cliquetable.addClique(mipsolver, clique.data(), 2);
    if (globaldomain.infeasible() || globaldomain.isFixed(col)) return true;
  }

  // store variable bounds derived from implications
  for (auto i = implics.begin(); i != binstart; ++i) {
    if (i->boundtype == HighsBoundType::kLower) {
      if (val == 1) {
        if (globaldomain.col_lower_[i->column] != -kHighsInf)
          addVLB(i->column, col,
                 i->boundval - globaldomain.col_lower_[i->column],
                 globaldomain.col_lower_[i->column]);
      } else
        addVLB(i->column,
               col,  // in case the lower bound is infinite the varbound can
                     // still be tightened as soon as a finite upper bound is
                     // known because the offset is finite
               globaldomain.col_lower_[i->column] - i->boundval, i->boundval);
    } else {
      if (val == 1) {
        if (globaldomain.col_upper_[i->column] != kHighsInf)
          addVUB(i->column, col,
                 i->boundval - globaldomain.col_upper_[i->column],
                 globaldomain.col_upper_[i->column]);
      } else
        addVUB(i->column,
               col,  // in case the upper bound is infinite the varbound can
                     // still be tightened as soon as a finite upper bound is
                     // known because the offset is finite
               globaldomain.col_upper_[i->column] - i->boundval, i->boundval);
    }
  }

  HighsInt loc = 2 * col + val;
  implications[loc].computed = true;
  implics.erase(binstart, implics.end());
  if (!implics.empty()) {
    implications[loc].implics = std::move(implics);
    this->numImplications += implications[loc].implics.size();
  }

  return false;
}

static constexpr bool kSkipBadVbds = true;
static constexpr bool kUseDualsForBreakingTies = true;

std::pair<HighsInt, HighsImplications::VarBound> HighsImplications::getBestVub(
    HighsInt col, const HighsSolution& lpSolution, double& bestUb,
    const HighsDomain& globaldom) const {
  std::pair<HighsInt, VarBound> bestVub =
      std::make_pair(-1, VarBound{0.0, kHighsInf});
  double minbestUb = bestUb;
  double bestUbDist = kHighsInf;
  int64_t bestvubnodes = 0;

  auto isVubBetter = [&](double ubDist, int64_t vubNodes, double minVubVal,
                         HighsInt vubCol, const VarBound& vub) {
    if (ubDist < bestUbDist - mipsolver.mipdata_->feastol) return true;
    if (vubNodes > bestvubnodes) return true;
    if (vubNodes < bestvubnodes) return false;
    if (minVubVal < minbestUb - mipsolver.mipdata_->feastol) return true;
    if (kUseDualsForBreakingTies) {
      if (minVubVal > minbestUb + mipsolver.mipdata_->feastol) return false;
      if (lpSolution.col_dual[vubCol] / vub.coef -
              lpSolution.col_dual[bestVub.first] / bestVub.second.coef >
          mipsolver.mipdata_->feastol)
        return true;
    }

    return false;
  };

  double scale = globaldom.col_upper_[col] - globaldom.col_lower_[col];
  if (scale == kHighsInf)
    scale = 1.0;
  else
    scale = 1.0 / scale;

  vubs[col].for_each([&](HighsInt vubCol, const VarBound& vub) {
    if (vub.coef == kHighsInf) return;
    if (globaldom.isFixed(vubCol)) return;
    assert(globaldom.isBinary(vubCol));
    double vubval = lpSolution.col_value[vubCol] * vub.coef + vub.constant;
    double ubDist = std::max(0.0, vubval - lpSolution.col_value[col]);

    double yDist = mipsolver.mipdata_->feastol +
                   (vub.coef > 0 ? 1 - lpSolution.col_value[vubCol]
                                 : lpSolution.col_value[vubCol]);
    // skip variable bound if the distance towards the variable bound constraint
    // is larger than the distance to the point where the binary column is
    // relaxed to it's weakest bound, i.e. 1 if its coefficient is positive.
    // The variable bound constraint has the form x <= ay + b with y binary and
    // hence the norm is sqrt(1 + a^2) and the distance of the variable bound
    // constraint is ay + b - x evaluated at the solution values of x and y
    // divided by the norm.
    double norm2 = 1.0 + vub.coef * vub.coef;
    if (kSkipBadVbds && ubDist * ubDist > yDist * yDist * norm2) return;

    assert(vubCol >= 0 && vubCol < mipsolver.numCol());
    ubDist *= scale;
    if (ubDist <= bestUbDist + mipsolver.mipdata_->feastol) {
      double minvubval = vub.minValue();
      int64_t vubnodes =
          vub.coef > 0 ? mipsolver.mipdata_->nodequeue.numNodesDown(vubCol)
                       : mipsolver.mipdata_->nodequeue.numNodesUp(vubCol);

      if (isVubBetter(ubDist, vubnodes, minvubval, vubCol, vub)) {
        bestUb = vubval;
        minbestUb = minvubval;
        bestVub = std::make_pair(vubCol, vub);
        bestvubnodes = vubnodes;
        bestUbDist = ubDist;
      }
    }
  });

  return bestVub;
}

std::pair<HighsInt, HighsImplications::VarBound> HighsImplications::getBestVlb(
    HighsInt col, const HighsSolution& lpSolution, double& bestLb,
    const HighsDomain& globaldom) const {
  std::pair<HighsInt, VarBound> bestVlb =
      std::make_pair(-1, VarBound{0.0, -kHighsInf});
  double maxbestlb = bestLb;
  double bestLbDist = kHighsInf;
  int64_t bestvlbnodes = 0;

  auto isVlbBetter = [&](double lbDist, int64_t vlbNodes, double maxVlbVal,
                         HighsInt vlbCol, const VarBound& vlb) {
    if (lbDist < bestLbDist - mipsolver.mipdata_->feastol) return true;
    if (vlbNodes > bestvlbnodes) return true;
    if (vlbNodes < bestvlbnodes) return false;
    if (maxVlbVal > maxbestlb + mipsolver.mipdata_->feastol) return true;
    if (kUseDualsForBreakingTies) {
      if (maxVlbVal < maxbestlb - mipsolver.mipdata_->feastol) return false;
      if (lpSolution.col_dual[vlbCol] / vlb.coef -
              lpSolution.col_dual[bestVlb.first] / bestVlb.second.coef <
          -mipsolver.mipdata_->feastol)
        return true;
    }

    return false;
  };

  double scale = globaldom.col_upper_[col] - globaldom.col_lower_[col];
  if (scale == kHighsInf)
    scale = 1.0;
  else
    scale = 1.0 / scale;

  vlbs[col].for_each([&](HighsInt vlbCol, const VarBound& vlb) {
    if (vlb.coef == -kHighsInf) return;
    if (globaldom.isFixed(vlbCol)) return;
    assert(globaldom.isBinary(vlbCol));
    assert(vlbCol >= 0 && vlbCol < mipsolver.numCol());
    double vlbval = lpSolution.col_value[vlbCol] * vlb.coef + vlb.constant;
    double lbDist = std::max(0.0, lpSolution.col_value[col] - vlbval);

    double yDist = mipsolver.mipdata_->feastol +
                   (vlb.coef > 0 ? lpSolution.col_value[vlbCol]
                                 : 1 - lpSolution.col_value[vlbCol]);

    double norm2 = 1.0 + vlb.coef * vlb.coef;
    if (kSkipBadVbds && lbDist * lbDist > yDist * yDist * norm2) return;

    // scale the distance as if the bounded column was scaled to have ub-lb=1
    lbDist *= scale;
    if (lbDist <= bestLbDist + mipsolver.mipdata_->feastol) {
      double maxvlbval = vlb.maxValue();
      int64_t vlbnodes =
          vlb.coef > 0 ? mipsolver.mipdata_->nodequeue.numNodesUp(vlbCol)
                       : mipsolver.mipdata_->nodequeue.numNodesDown(vlbCol);

      if (isVlbBetter(lbDist, vlbnodes, maxvlbval, vlbCol, vlb)) {
        bestLb = vlbval;
        maxbestlb = maxvlbval;
        bestVlb = std::make_pair(vlbCol, vlb);
        bestvlbnodes = vlbnodes;
        bestLbDist = lbDist;
      }
    }
  });

  return bestVlb;
}

bool HighsImplications::runProbing(HighsInt col, HighsInt& numReductions) {
  HighsDomain& globaldomain = mipsolver.mipdata_->getDomain();
  if (globaldomain.isBinary(col) && !implicationsCached(col, 1) &&
      !implicationsCached(col, 0) &&
      mipsolver.mipdata_->cliquetable.getSubstitution(col) == nullptr) {
    bool infeasible = computeImplications(col, 1);
    if (globaldomain.infeasible()) return true;
    if (infeasible) return true;
    if (mipsolver.mipdata_->cliquetable.getSubstitution(col) != nullptr)
      return true;

    infeasible = computeImplications(col, 0);
    if (globaldomain.infeasible()) return true;
    if (infeasible) return true;
    if (mipsolver.mipdata_->cliquetable.getSubstitution(col) != nullptr)
      return true;

    // analyze implications
    const std::vector<HighsDomainChange>& implicsdown =
        getImplications(col, 0, infeasible);
    const std::vector<HighsDomainChange>& implicsup =
        getImplications(col, 1, infeasible);
    HighsInt nimplicsdown = implicsdown.size();
    HighsInt nimplicsup = implicsup.size();
    HighsInt u = 0;
    HighsInt d = 0;

    while (u < nimplicsup && d < nimplicsdown) {
      if (implicsup[u].column < implicsdown[d].column)
        ++u;
      else if (implicsdown[d].column < implicsup[u].column)
        ++d;
      else {
        assert(implicsup[u].column == implicsdown[d].column);
        HighsInt implcol = implicsup[u].column;
        double lbDown = globaldomain.col_lower_[implcol];
        double ubDown = globaldomain.col_upper_[implcol];
        double lbUp = lbDown;
        double ubUp = ubDown;

        do {
          if (implicsdown[d].boundtype == HighsBoundType::kLower)
            lbDown = std::max(lbDown, implicsdown[d].boundval);
          else
            ubDown = std::min(ubDown, implicsdown[d].boundval);
          ++d;
        } while (d < nimplicsdown && implicsdown[d].column == implcol);

        do {
          if (implicsup[u].boundtype == HighsBoundType::kLower)
            lbUp = std::max(lbUp, implicsup[u].boundval);
          else
            ubUp = std::min(ubUp, implicsup[u].boundval);
          ++u;
        } while (u < nimplicsup && implicsup[u].column == implcol);

        if (colsubstituted[implcol] || globaldomain.isFixed(implcol)) continue;

        if (lbDown == ubDown && lbUp == ubUp &&
            std::abs(lbDown - lbUp) > mipsolver.mipdata_->feastol) {
          HighsSubstitution substitution;
          substitution.substcol = implcol;
          substitution.staycol = col;
          substitution.offset = lbDown;
          substitution.scale = lbUp - lbDown;
          substitutions.push_back(substitution);
          colsubstituted[implcol] = true;
          ++numReductions;
        } else if (!mipsolver.mipdata_->parallelLockActive()) {
          double lb = std::min(lbDown, lbUp);
          double ub = std::max(ubDown, ubUp);

          if (lb > globaldomain.col_lower_[implcol]) {
            globaldomain.changeBound(HighsBoundType::kLower, implcol, lb,
                                     HighsDomain::Reason::unspecified());
            ++numReductions;
          }

          if (ub < globaldomain.col_upper_[implcol]) {
            globaldomain.changeBound(HighsBoundType::kUpper, implcol, ub,
                                     HighsDomain::Reason::unspecified());
            ++numReductions;
          }
        }
      }
    }

    return true;
  }

  return false;
}

void HighsImplications::strengthenVarBound(VarBound& vbnd,
                                           HighsInt multiplier) const {
  // try to strengthen variable bound constraint x + a * y <= b by computing
  // MIR cut. since x is a general-integer variable (with integral bounds) and
  // its coefficient is 1.0, it is not shifted (or complemented). similarly,
  // since y is a binary variable (i.e., its lower bound is 0.0), it does not
  // need to be shifted, and we also do not try to complement it.
  if (std::abs(vbnd.coef) == kHighsInf || std::abs(vbnd.constant) == kHighsInf)
    return;
  constexpr double f0min = 0.005;
  constexpr double f0max = 0.995;
  double downrhs = std::floor(multiplier * vbnd.constant);
  double f0 = multiplier * vbnd.constant - downrhs;
  if (f0 < f0min || f0 > f0max) return;
  double downaj = std::floor(-multiplier * vbnd.coef + kHighsTiny);
  double fj = -multiplier * vbnd.coef - downaj;
  vbnd.constant = multiplier * downrhs;
  vbnd.coef = -multiplier * (downaj + std::max(fj - f0, 0.0) / (1.0 - f0));
};

void HighsImplications::addVUB(HighsInt col, HighsInt vubcol, double vubcoef,
                               double vubconstant) {
  addVUB(col, vubcol, vubcoef, vubconstant,
         mipsolver.mipdata_->getDomain().col_upper_[col],
         mipsolver.isColIntegral(col));
}

void HighsImplications::addVUB(HighsInt col, HighsInt vubcol, double vubcoef,
                               double vubconstant, double colupperbound,
                               bool colisintegral) {
  // assume that VUBs do not have infinite coefficients and infinite constant
  // terms since such VUBs effectively evaluate to NaN.
  assert(std::abs(vubcoef) != kHighsInf || std::abs(vubconstant) != kHighsInf);
  if (tooManyVarBounds()) return;

  VarBound vub{vubcoef, vubconstant};

  if (colisintegral) {
    // try to strengthen VUB
    strengthenVarBound(vub, HighsInt{1});
    if (vub.coef == 0.0) return;
  }

  mipsolver.mipdata_->debugSolution.checkVub(col, vubcol, vubcoef, vubconstant);

  double minBound = vub.minValue();
  if (minBound >= colupperbound - mipsolver.mipdata_->feastol) return;

  auto insertresult = vubs[col].insert_or_get(vubcol, vub);

  if (!insertresult.second) {
    VarBound& currentvub = *insertresult.first;
    double currentMinBound = currentvub.minValue();
    if (minBound < currentMinBound - mipsolver.mipdata_->feastol) {
      currentvub.coef = vub.coef;
      currentvub.constant = vub.constant;
    }
  } else
    numVarBounds++;
}

void HighsImplications::addVLB(HighsInt col, HighsInt vlbcol, double vlbcoef,
                               double vlbconstant) {
  addVLB(col, vlbcol, vlbcoef, vlbconstant,
         mipsolver.mipdata_->getDomain().col_lower_[col],
         mipsolver.isColIntegral(col));
}

void HighsImplications::addVLB(HighsInt col, HighsInt vlbcol, double vlbcoef,
                               double vlbconstant, double colllowerbound,
                               bool colisintegral) {
  // assume that VLBs do not have infinite coefficients and infinite constant
  // terms since such VLBs effectively evaluate to NaN.
  assert(std::abs(vlbcoef) != kHighsInf || std::abs(vlbconstant) != kHighsInf);
  if (tooManyVarBounds()) return;

  VarBound vlb{vlbcoef, vlbconstant};

  if (colisintegral) {
    // try to strengthen VLB
    strengthenVarBound(vlb, HighsInt{-1});
    if (vlb.coef == 0.0) return;
  }

  mipsolver.mipdata_->debugSolution.checkVlb(col, vlbcol, vlbcoef, vlbconstant);

  double maxBound = vlb.maxValue();
  if (maxBound <= colllowerbound + mipsolver.mipdata_->feastol) return;

  auto insertresult = vlbs[col].insert_or_get(vlbcol, vlb);

  if (!insertresult.second) {
    VarBound& currentvlb = *insertresult.first;

    double currentMaxBound = currentvlb.maxValue();
    if (maxBound > currentMaxBound + mipsolver.mipdata_->feastol) {
      currentvlb.coef = vlb.coef;
      currentvlb.constant = vlb.constant;
    }
  } else
    numVarBounds++;
}

void HighsImplications::rebuild(HighsInt ncols,
                                const std::vector<HighsInt>& orig2reducedcol,
                                const std::vector<HighsInt>& orig2reducedrow) {
  std::vector<HighsHashTree<HighsInt, VarBound>> oldvubs;
  std::vector<HighsHashTree<HighsInt, VarBound>> oldvlbs;

  oldvlbs.swap(vlbs);
  oldvubs.swap(vubs);

  colsubstituted.clear();
  colsubstituted.shrink_to_fit();
  implications.clear();
  implications.shrink_to_fit();

  implications.resize(2 * ncols);
  colsubstituted.resize(ncols);
  substitutions.clear();
  vubs.clear();
  vubs.shrink_to_fit();
  vubs.resize(ncols);
  vlbs.clear();
  vlbs.shrink_to_fit();
  vlbs.resize(ncols);
  numImplications = 0;
  numVarBounds = 0;
  HighsInt oldncols = oldvubs.size();

  nextCleanupCall = mipsolver.numNonzero();

  for (HighsInt i = 0; i != oldncols; ++i) {
    HighsInt newi = orig2reducedcol[i];

    if (newi == -1 ||
        !mipsolver.mipdata_->postSolveStack.isColLinearlyTransformable(newi))
      continue;

    oldvubs[i].for_each([&](HighsInt vubCol, VarBound vub) {
      HighsInt newVubCol = orig2reducedcol[vubCol];
      if (newVubCol == -1) return;

      if (!mipsolver.mipdata_->getDomain().isBinary(newVubCol) ||
          !mipsolver.mipdata_->postSolveStack.isColLinearlyTransformable(
              newVubCol))
        return;

      addVUB(newi, newVubCol, vub.coef, vub.constant);
    });

    oldvlbs[i].for_each([&](HighsInt vlbCol, VarBound vlb) {
      HighsInt newVlbCol = orig2reducedcol[vlbCol];
      if (newVlbCol == -1) return;

      if (!mipsolver.mipdata_->getDomain().isBinary(newVlbCol) ||
          !mipsolver.mipdata_->postSolveStack.isColLinearlyTransformable(
              newVlbCol))
        return;

      addVLB(newi, newVlbCol, vlb.coef, vlb.constant);
    });

    // todo also add old implications once implications can be added
    // incrementally for now we discard the old implications as they might be
    // weaker then newly computed ones and adding them would block computation
    // of new implications
  }
}

void HighsImplications::buildFrom(const HighsImplications& init) {
  // todo check if this should be done
  HighsInt numcol = mipsolver.numCol();

  for (HighsInt i = 0; i != numcol; ++i) {
    init.vubs[i].for_each([&](HighsInt vubCol, VarBound vub) {
      if (!mipsolver.mipdata_->getDomain().isBinary(vubCol)) return;
      addVUB(i, vubCol, vub.coef, vub.constant);
    });

    init.vlbs[i].for_each([&](HighsInt vlbCol, VarBound vlb) {
      if (!mipsolver.mipdata_->getDomain().isBinary(vlbCol)) return;
      addVLB(i, vlbCol, vlb.coef, vlb.constant);
    });

    // todo also add old implications once implications can be added
    // incrementally for now we discard the old implications as they might be
    // weaker then newly computed ones and adding them would block computation
    // of new implications
  }
}

void HighsImplications::separateImpliedBounds(
    const HighsLpRelaxation& lpRelaxation, const std::vector<double>& sol,
    HighsCutPool& cutpool, double feastol, HighsDomain& globaldom,
    bool thread_safe) {
  std::array<HighsInt, 2> inds;
  std::array<double, 2> vals;
  double rhs;

  HighsInt numboundchgs = 0;

  // first do probing on all candidates that have not been probed yet
  if (!mipsolver.mipdata_->cliquetable.isFull() && !thread_safe) {
    auto oldNumQueries =
        mipsolver.mipdata_->cliquetable.numNeighbourhoodQueries;
    HighsInt oldNumEntries = mipsolver.mipdata_->cliquetable.getNumEntries();

    for (std::pair<HighsInt, double> fracint :
         lpRelaxation.getFractionalIntegers()) {
      HighsInt col = fracint.first;
      if (globaldom.col_lower_[col] != 0.0 ||
          globaldom.col_upper_[col] != 1.0 ||
          (implicationsCached(col, 0) && implicationsCached(col, 1)))
        continue;

      mipsolver.profiling_->start(kMipClockProbingImplications);
      const bool probing_result = runProbing(col, numboundchgs);
      mipsolver.profiling_->stop(kMipClockProbingImplications);
      if (probing_result) {
        if (globaldom.infeasible()) return;
      }

      if (mipsolver.mipdata_->cliquetable.isFull()) break;
    }

    // if (!mipsolver.submip)
    //   printf("numEntries: %d, beforeProbing: %d\n",
    //          mipsolver.mipdata_->cliquetable.getNumEntries(), oldNumEntries);
    HighsInt numNewEntries =
        mipsolver.mipdata_->cliquetable.getNumEntries() - oldNumEntries;

    nextCleanupCall -= std::max(HighsInt{0}, numNewEntries);

    if (nextCleanupCall < 0) {
      // HighsInt oldNumEntries =
      // mipsolver.mipdata_->cliquetable.getNumEntries();
      if (!mipsolver.mipdata_->parallelLockActive())
        mipsolver.mipdata_->cliquetable.runCliqueMerging(globaldom);

      // printf("numEntries: %d, beforeMerging: %d\n",
      //        mipsolver.mipdata_->cliquetable.getNumEntries(), oldNumEntries);
      nextCleanupCall =
          std::min(mipsolver.mipdata_->numCliqueEntriesAfterFirstPresolve,
                   mipsolver.mipdata_->cliquetable.getNumEntries());
      // printf("nextCleanupCall: %d\n", nextCleanupCall);
    }

    if (!mipsolver.mipdata_->parallelLockActive())
      mipsolver.mipdata_->cliquetable.numNeighbourhoodQueries = oldNumQueries;
  }

  for (std::pair<HighsInt, double> fracint :
       lpRelaxation.getFractionalIntegers()) {
    HighsInt col = fracint.first;
    // skip non binary variables
    if (globaldom.col_lower_[col] != 0.0 || globaldom.col_upper_[col] != 1.0)
      continue;

    bool infeas;
    if (implicationsCached(col, 1)) {
      const std::vector<HighsDomainChange>& implics =
          getImplications(col, 1, infeas);
      if (globaldom.infeasible()) return;

      if (infeas) {
        vals[0] = 1.0;
        inds[0] = col;
        cutpool.addCut(mipsolver, inds.data(), vals.data(), 1, 0.0, false, true,
                       false);
        continue;
      }

      HighsInt nimplics = implics.size();
      for (HighsInt i = 0; i < nimplics; ++i) {
        if (implics[i].boundtype == HighsBoundType::kUpper) {
          if (implics[i].boundval + feastol >=
              globaldom.col_upper_[implics[i].column])
            continue;

          vals[0] = 1.0;
          inds[0] = implics[i].column;
          vals[1] =
              globaldom.col_upper_[implics[i].column] - implics[i].boundval;
          inds[1] = col;
          rhs = globaldom.col_upper_[implics[i].column];

        } else {
          if (implics[i].boundval - feastol <=
              globaldom.col_lower_[implics[i].column])
            continue;

          vals[0] = -1.0;
          inds[0] = implics[i].column;
          vals[1] =
              implics[i].boundval - globaldom.col_lower_[implics[i].column];
          inds[1] = col;
          rhs = -globaldom.col_lower_[implics[i].column];
        }

        double viol = sol[inds[0]] * vals[0] + sol[inds[1]] * vals[1] - rhs;

        if (viol > feastol) {
          // printf("added implied bound cut to pool\n");
          cutpool.addCut(mipsolver, inds.data(), vals.data(), 2, rhs,
                         !mipsolver.isColContinuous(implics[i].column), false,
                         false, false);
        }
      }
    }

    if (implicationsCached(col, 0)) {
      const std::vector<HighsDomainChange>& implics =
          getImplications(col, 0, infeas);
      if (globaldom.infeasible()) return;

      if (infeas) {
        vals[0] = -1.0;
        inds[0] = col;
        cutpool.addCut(mipsolver, inds.data(), vals.data(), 1, -1.0, false,
                       true, false);
        continue;
      }

      HighsInt nimplics = implics.size();
      for (HighsInt i = 0; i < nimplics; ++i) {
        if (implics[i].boundtype == HighsBoundType::kUpper) {
          if (implics[i].boundval + feastol >=
              globaldom.col_upper_[implics[i].column])
            continue;

          vals[0] = 1.0;
          inds[0] = implics[i].column;
          vals[1] =
              implics[i].boundval - globaldom.col_upper_[implics[i].column];
          inds[1] = col;
          rhs = implics[i].boundval;
        } else {
          if (implics[i].boundval - feastol <=
              globaldom.col_lower_[implics[i].column])
            continue;

          vals[0] = -1.0;
          inds[0] = implics[i].column;
          vals[1] =
              globaldom.col_lower_[implics[i].column] - implics[i].boundval;
          inds[1] = col;
          rhs = -implics[i].boundval;
        }

        double viol = sol[inds[0]] * vals[0] + sol[inds[1]] * vals[1] - rhs;

        if (viol > feastol) {
          // printf("added implied bound cut to pool\n");
          cutpool.addCut(mipsolver, inds.data(), vals.data(), 2, rhs,
                         !mipsolver.isColContinuous(implics[i].column), false,
                         false, false);
        }
      }
    }
  }
}

void HighsImplications::cleanupVarbounds(HighsInt col) {
  double ub = mipsolver.mipdata_->getDomain().col_upper_[col];
  double lb = mipsolver.mipdata_->getDomain().col_lower_[col];

  if (ub == lb) {
    HighsInt numVubs = 0;
    vubs[col].for_each([&](HighsInt vubCol, VarBound& vub) { numVubs++; });
    HighsInt numVlbs = 0;
    vlbs[col].for_each([&](HighsInt vlbCol, VarBound& vlb) { numVlbs++; });
    numVarBounds -= numVubs + numVlbs;
    vlbs[col].clear();
    vubs[col].clear();
    return;
  }

  std::vector<HighsInt> delVbds;

  vubs[col].for_each([&](HighsInt vubCol, VarBound& vub) {
    bool redundant = false;
    bool infeasible = false;
    cleanupVub(col, vubCol, vub, ub, redundant, infeasible);
    if (redundant) delVbds.push_back(vubCol);
    if (infeasible) return;
  });

  for (HighsInt vubCol : delVbds) vubs[col].erase(vubCol);
  numVarBounds -= delVbds.size();
  delVbds.clear();

  vlbs[col].for_each([&](HighsInt vlbCol, VarBound& vlb) {
    bool redundant = false;
    bool infeasible = false;
    cleanupVlb(col, vlbCol, vlb, lb, redundant, infeasible);
    if (redundant) delVbds.push_back(vlbCol);
    if (infeasible) return;
  });

  for (HighsInt vlbCol : delVbds) vlbs[col].erase(vlbCol);
  numVarBounds -= delVbds.size();
}

void HighsImplications::cleanupVlb(HighsInt col, HighsInt vlbCol,
                                   HighsImplications::VarBound& vlb, double lb,
                                   bool& redundant, bool& infeasible,
                                   bool allowBoundChanges) const {
  // initialize
  redundant = false;
  infeasible = false;

  // return if there is no variable bound
  if (vlbCol == -1) return;

  // check variable lower bound
  mipsolver.mipdata_->debugSolution.checkVlb(col, vlbCol, vlb.coef,
                                             vlb.constant);

  HighsCDouble maxlb = vlb.maxValue();
  HighsCDouble minlb = vlb.minValue();

  if (maxlb <= lb + mipsolver.mipdata_->feastol) {
    // variable bound is redundant
    redundant = true;
  } else if (minlb < lb - mipsolver.mipdata_->epsilon) {
    // coefficient can be tightened
    double newcoef = static_cast<double>(lb - maxlb);
    if (vlb.coef < 0) {
      vlb.coef = newcoef;
    } else {
      vlb.constant = lb;
      vlb.coef = -newcoef;
    }
    // check tightened variable lower bound
    mipsolver.mipdata_->debugSolution.checkVlb(col, vlbCol, vlb.coef,
                                               vlb.constant);
  } else if (allowBoundChanges && minlb > lb + mipsolver.mipdata_->epsilon) {
    mipsolver.mipdata_->getDomain().changeBound(
        HighsBoundType::kLower, col, static_cast<double>(minlb),
        HighsDomain::Reason::unspecified());
    infeasible = mipsolver.mipdata_->getDomain().infeasible();
  }
}

void HighsImplications::cleanupVub(HighsInt col, HighsInt vubCol,
                                   HighsImplications::VarBound& vub, double ub,
                                   bool& redundant, bool& infeasible,
                                   bool allowBoundChanges) const {
  // initialize
  redundant = false;
  infeasible = false;

  // return if there is no variable bound
  if (vubCol == -1) return;

  // check variable upper bound
  mipsolver.mipdata_->debugSolution.checkVub(col, vubCol, vub.coef,
                                             vub.constant);

  HighsCDouble maxub = vub.maxValue();
  HighsCDouble minub = vub.minValue();

  if (minub >= ub - mipsolver.mipdata_->feastol) {
    // variable bound is redundant
    redundant = true;
  } else if (maxub > ub + mipsolver.mipdata_->epsilon) {
    // coefficient can be tightened
    double newcoef = static_cast<double>(ub - minub);
    if (vub.coef > 0) {
      vub.coef = newcoef;
    } else {
      vub.constant = ub;
      vub.coef = -newcoef;
    }
    // check tightened variable upper bound
    mipsolver.mipdata_->debugSolution.checkVub(col, vubCol, vub.coef,
                                               vub.constant);
  } else if (allowBoundChanges && maxub < ub - mipsolver.mipdata_->epsilon) {
    mipsolver.mipdata_->getDomain().changeBound(
        HighsBoundType::kUpper, col, static_cast<double>(maxub),
        HighsDomain::Reason::unspecified());
    infeasible = mipsolver.mipdata_->getDomain().infeasible();
  }
}

void HighsImplications::applyImplications(HighsDomain& domain,
                                          const HighsInt col,
                                          const HighsInt val) {
  assert(domain.isFixed(col));

  auto checkImplication = [&](const HighsDomainChange& domchg) -> bool {
    assert(!domain.infeasible());
    if (domain.isFixed(domchg.column)) return false;
    const bool isint =
        domain.variableType(domchg.column) != HighsVarType::kContinuous;
    // Directly change bounds on all integer columns. Only change continuous
    // columns that fix the column, as changing their domains risks
    // suppressing further bound changes found in propagation, e.g.,
    // change [0, 100] -> [0, 50], propagation could tighten to [0, 48], but
    // such a tightening would not be applied due to min boundRange improvement.
    if (domchg.boundtype == HighsBoundType::kLower) {
      if ((!isint && domchg.boundval >
                         domain.col_upper_[domchg.column] - domain.feastol()) ||
          (isint && domchg.boundval >
                        domain.col_lower_[domchg.column] + domain.feastol())) {
        domain.changeBound(domchg, HighsDomain::Reason::cliqueTable(col, val));
      }
    } else {
      if ((!isint && domchg.boundval <
                         domain.col_lower_[domchg.column] + domain.feastol()) ||
          (isint && domchg.boundval <
                        domain.col_upper_[domchg.column] - domain.feastol())) {
        domain.changeBound(domchg, HighsDomain::Reason::cliqueTable(col, val));
      }
    }
    return domain.infeasible();
  };

  HighsInt loc = 2 * col + val;
  if (implications[loc].computed) {
    for (HighsDomainChange& domchg : implications[loc].implics) {
      if (checkImplication(domchg)) break;
    }
  }
}

#else  // HIGHS_RUST

// The implications are Rust's (rust/src/mip/implications.rs); these are
// the handle's methods and the callbacks into the MIP solver.

#include "mip/HighsCliqueTableRust.h"

namespace highs_rs {
static_assert(sizeof(HighsSubstitution) == 24 &&
                  sizeof(std::pair<HighsInt, double>) == 16 &&
                  sizeof(HighsImplications::VarBound) == 16,
              "Substitution, FracInt, VarBound layouts");

namespace {
struct ImpCtx {
  HighsImplications* self;
  const HighsMipSolver* mip;
  HighsCutPool* cutpool;
};
ImpCtx& impOf(void* p) { return *static_cast<ImpCtx*>(p); }
HighsDomain& impGlobal(void* p) {
  return impOf(p).mip->mipdata_->getDomain();
}
int64_t impCbNodesDown(void* p, HighsInt col) {
  return impOf(p).mip->mipdata_->nodequeue.numNodesDown(col);
}
int64_t impCbNodesUp(void* p, HighsInt col) {
  return impOf(p).mip->mipdata_->nodequeue.numNodesUp(col);
}
void impCbLiftingBegin(void* p) {
  HighsDomain& globaldomain = impGlobal(p);
  assert(globaldomain.getRedundantRows().size() == 0);
  if (impOf(p).self->storeLiftingOpportunity != nullptr)
    globaldomain.setRecordRedundantRows(true);
}
void impCbLiftingStore(void* p, HighsInt col, bool val) {
  HighsDomain& globaldomain = impGlobal(p);
  auto& store = impOf(p).self->storeLiftingOpportunity;
  if (store != nullptr) {
    for (const auto& elm : globaldomain.getRedundantRows())
      store(elm.key(), col, val ? 1 : 0,
            (val ? -1 : 1) * globaldomain.getRedundantRowValue(elm.key()));
    globaldomain.clearRedundantRows();
    globaldomain.setRecordRedundantRows(false);
  }
}
HighsInt impCbDomchgReason(void* p, HighsInt k, HighsInt* index) {
  const HighsDomain::Reason& r = impGlobal(p).getDomainChangeReason()[k];
  *index = r.index;
  return r.type;
}
size_t impCbChangedColsLen(void* p) {
  return impGlobal(p).getChangedCols().size();
}
void impCbBacktrack(void* p, size_t changedend) {
  HighsDomain& globaldomain = impGlobal(p);
  globaldomain.backtrack();
  globaldomain.clearChangedCols(changedend);
}
void impCbVertexInfeasible(void* p, HighsInt col, HighsInt val) {
  impOf(p).mip->mipdata_->cliquetable.vertexInfeasible(impGlobal(p), col,
                                                       val);
}
void impCbInference(void* p, HighsInt col, HighsInt n, bool val) {
  impOf(p).mip->mipdata_->getPseudoCost().addInferenceObservation(col, n, val);
}
HighsInt impCbCliqueNumEntries(void* p) {
  return impOf(p).mip->mipdata_->cliquetable.getNumEntries();
}
void impCbAddClique2(void* p, ClqVar* clique) {
  impOf(p).mip->mipdata_->cliquetable.addClique(*impOf(p).mip, clique, 2);
}
bool impCbCliqueSubstituted(void* p, HighsInt col) {
  return impOf(p).mip->mipdata_->cliquetable.getSubstitution(col) != nullptr;
}
bool impCbParallelLock(void* p) {
  return impOf(p).mip->mipdata_->parallelLockActive();
}
bool impCbCliqueIsFull(void* p) {
  return impOf(p).mip->mipdata_->cliquetable.isFull();
}
int64_t* impCbCliqueNumQueries(void* p) {
  return &impOf(p).mip->mipdata_->cliquetable.numNeighbourhoodQueries;
}
void impCbRunCliqueMerging(void* p) {
  impOf(p).mip->mipdata_->cliquetable.runCliqueMerging(impGlobal(p));
}
HighsInt impCbEntriesAfterFirstPresolve(void* p) {
  return impOf(p).mip->mipdata_->numCliqueEntriesAfterFirstPresolve;
}
void impCbProbingClock(void* p, bool start) {
  if (start)
    impOf(p).mip->profiling_->start(kMipClockProbingImplications);
  else
    impOf(p).mip->profiling_->stop(kMipClockProbingImplications);
}
void impCbAddCut(void* p, HighsInt* inds, double* vals, HighsInt len,
                 double rhs, bool integral, bool propagate) {
  impOf(p).cutpool->addCut(*impOf(p).mip, inds, vals, len, rhs, integral,
                           propagate, false);
}

ImplicsHost implicsHost(ImpCtx& ctx) {
  const HighsMipSolver& mipsolver = *ctx.mip;
  ImplicsHost h;
  h.ctx = &ctx;
  h.feastol = mipsolver.mipdata_->feastol;
  h.epsilon = mipsolver.mipdata_->epsilon;
  h.num_nonzero = mipsolver.numNonzero();
  h.num_nodes_down = impCbNodesDown;
  h.num_nodes_up = impCbNodesUp;
  h.lifting_begin = impCbLiftingBegin;
  h.lifting_store = impCbLiftingStore;
  h.domchg_reason = impCbDomchgReason;
  h.changed_cols_len = impCbChangedColsLen;
  h.backtrack = impCbBacktrack;
  h.vertex_infeasible = impCbVertexInfeasible;
  h.add_inference_observation = impCbInference;
  h.clique_num_entries = impCbCliqueNumEntries;
  h.add_clique2 = impCbAddClique2;
  h.clique_substituted = impCbCliqueSubstituted;
  h.parallel_lock_active = impCbParallelLock;
  h.clique_is_full = impCbCliqueIsFull;
  h.clique_num_queries = impCbCliqueNumQueries;
  h.run_clique_merging = impCbRunCliqueMerging;
  h.num_clique_entries_after_first_presolve = impCbEntriesAfterFirstPresolve;
  h.probing_clock = impCbProbingClock;
  h.add_cut = impCbAddCut;
  return h;
}
}  // namespace
}  // namespace highs_rs

HighsImplications::HighsImplications(const HighsMipSolver& mipsolver)
    : rs_(highs_rs::highs_rs_implics_new(mipsolver.numCol(),
                                         mipsolver.numNonzero())),
      mipsolver(mipsolver),
      substitutions(rs_, 0, highs_rs::highs_rs_implics_vec,
                    highs_rs::highs_rs_implics_vec_clear) {}

HighsImplications::~HighsImplications() {
  highs_rs::highs_rs_implics_free(rs_);
}

bool HighsImplications::tooManyVarBounds() const {
  return highs_rs::highs_rs_implics_get(rs_, 1);
}

void HighsImplications::addVUB(HighsInt col, HighsInt vubcol, double vubcoef,
                               double vubconstant) {
  addVUB(col, vubcol, vubcoef, vubconstant,
         mipsolver.mipdata_->getDomain().col_upper_[col],
         mipsolver.isColIntegral(col));
}

void HighsImplications::addVUB(HighsInt col, HighsInt vubcol, double vubcoef,
                               double vubconstant, double colupperbound,
                               bool colisintegral) {
  assert(std::abs(vubcoef) != kHighsInf || std::abs(vubconstant) != kHighsInf);
  // (the C++ checks after the early returns; a derived bound is valid)
  mipsolver.mipdata_->debugSolution.checkVub(col, vubcol, vubcoef, vubconstant);
  highs_rs::highs_rs_implics_add_vb(rs_, false, col, vubcol, vubcoef,
                                    vubconstant, colupperbound, colisintegral,
                                    mipsolver.mipdata_->feastol);
}

void HighsImplications::addVLB(HighsInt col, HighsInt vlbcol, double vlbcoef,
                               double vlbconstant) {
  addVLB(col, vlbcol, vlbcoef, vlbconstant,
         mipsolver.mipdata_->getDomain().col_lower_[col],
         mipsolver.isColIntegral(col));
}

void HighsImplications::addVLB(HighsInt col, HighsInt vlbcol, double vlbcoef,
                               double vlbconstant, double collowerbound,
                               bool colisintegral) {
  assert(std::abs(vlbcoef) != kHighsInf || std::abs(vlbconstant) != kHighsInf);
  mipsolver.mipdata_->debugSolution.checkVlb(col, vlbcol, vlbcoef, vlbconstant);
  highs_rs::highs_rs_implics_add_vb(rs_, true, col, vlbcol, vlbcoef,
                                    vlbconstant, collowerbound, colisintegral,
                                    mipsolver.mipdata_->feastol);
}

std::pair<HighsInt, HighsImplications::VarBound> HighsImplications::getBestVb(
    bool vlb, HighsInt col, const double* col_value, const double* col_dual,
    size_t n, double& bound, const HighsDomain& globaldom) const {
  highs_rs::ImpCtx ctx{const_cast<HighsImplications*>(this), &mipsolver,
                       nullptr};
  highs_rs::ImplicsHost h = highs_rs::implicsHost(ctx);
  highs_rs::CliqueDom d =
      highs_rs::cliqueDom(const_cast<HighsDomain&>(globaldom));
  highs_rs::ImplicsVarBound vb;
  HighsInt c = highs_rs::highs_rs_implics_best_vb(
      rs_, vlb, &h, &d, col, col_value, col_dual, n, &bound, &vb);
  return {c, HighsImplications::VarBound{vb.coef, vb.constant}};
}

static std::pair<HighsInt, HighsImplications::VarBound> implicsBestVb(
    const HighsImplications& self, bool vlb, HighsInt col,
    const HighsSolution& lpSolution, double& bound,
    const HighsDomain& globaldom) {
  assert(lpSolution.col_dual.size() >= lpSolution.col_value.size());
  return self.getBestVb(vlb, col, lpSolution.col_value.data(),
                        lpSolution.col_dual.data(),
                        lpSolution.col_value.size(), bound, globaldom);
}

bool HighsImplications::runProbing(HighsInt col, HighsInt& numReductions) {
  highs_rs::ImpCtx ctx{this, &mipsolver, nullptr};
  highs_rs::ImplicsHost h = highs_rs::implicsHost(ctx);
  highs_rs::CliqueDom g = highs_rs::cliqueDom(mipsolver.mipdata_->getDomain());
  return highs_rs::highs_rs_implics_run_probing(rs_, &g, &h, col,
                                                &numReductions);
}

void HighsImplications::rebuild(HighsInt ncols,
                                const std::vector<HighsInt>& orig2reducedcol,
                                const std::vector<HighsInt>& orig2reducedrow) {
  std::vector<uint8_t> transformable(ncols);
  for (HighsInt col = 0; col != ncols; ++col)
    transformable[col] =
        mipsolver.mipdata_->postSolveStack.isColLinearlyTransformable(col);
  highs_rs::CliqueDom g = highs_rs::cliqueDom(mipsolver.mipdata_->getDomain());
  highs_rs::highs_rs_implics_rebuild(rs_, &g, ncols, orig2reducedcol.data(),
                                     orig2reducedcol.size(),
                                     transformable.data());
}

void HighsImplications::buildFrom(const HighsImplications& init) {
  highs_rs::CliqueDom g = highs_rs::cliqueDom(mipsolver.mipdata_->getDomain());
  assert(g.num_col == mipsolver.numCol());
  highs_rs::highs_rs_implics_build_from(rs_, &g, init.rs_);
}

void HighsImplications::separateImpliedBounds(
    const HighsLpRelaxation& lpRelaxation, const std::vector<double>& sol,
    HighsCutPool& cutpool, double feastol, HighsDomain& globaldom,
    bool thread_safe) {
  highs_rs::ImpCtx ctx{this, &mipsolver, &cutpool};
  highs_rs::ImplicsHost h = highs_rs::implicsHost(ctx);
  highs_rs::CliqueDom g = highs_rs::cliqueDom(mipsolver.mipdata_->getDomain());
  highs_rs::CliqueDom d = highs_rs::cliqueDom(globaldom);
  const auto& fracints = lpRelaxation.getFractionalIntegers();
  highs_rs::highs_rs_implics_separate(rs_, &g, &h, &d, fracints.data(),
                                      fracints.size(), sol.data(), sol.size(),
                                      feastol, thread_safe);
}

void HighsImplications::cleanupVarbounds(HighsInt col) {
  highs_rs::ImpCtx ctx{this, &mipsolver, nullptr};
  highs_rs::ImplicsHost h = highs_rs::implicsHost(ctx);
  highs_rs::CliqueDom g = highs_rs::cliqueDom(mipsolver.mipdata_->getDomain());
  highs_rs::highs_rs_implics_cleanup_varbounds(rs_, &g, &h, col);
}

static void implicsCleanupVb(const HighsMipSolver& mipsolver, bool vlb,
                             HighsInt col, HighsInt vbCol,
                             HighsImplications::VarBound& vb, double bound,
                             bool& redundant, bool& infeasible,
                             bool allowBoundChanges) {
  highs_rs::CliqueDom g = highs_rs::cliqueDom(mipsolver.mipdata_->getDomain());
  highs_rs::ImplicsVarBound v{vb.coef, vb.constant};
  highs_rs::highs_rs_implics_cleanup_vb(
      vlb, &g, mipsolver.mipdata_->feastol, mipsolver.mipdata_->epsilon, col,
      vbCol, &v, bound, allowBoundChanges, &redundant, &infeasible);
  vb.coef = v.coef;
  vb.constant = v.constant;
}

void HighsImplications::applyImplications(HighsDomain& domain,
                                          const HighsInt col,
                                          const HighsInt val) {
  assert(domain.isFixed(col));
  highs_rs::CliqueDom d = highs_rs::cliqueDom(domain);
  highs_rs::highs_rs_implics_apply(rs_, &d, col, val);
}

#endif  // HIGHS_RUST
