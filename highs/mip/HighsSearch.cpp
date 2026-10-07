/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsSearch.h"

#include <numeric>

#include "lp_data/HConst.h"
#include "mip/HighsCutGeneration.h"
#include "mip/HighsDomainChange.h"
#include "mip/HighsMipSolverData.h"

#ifndef HIGHS_RUST
HighsSearch::HighsSearch(HighsMipWorker& mipworker, HighsPseudocost& pseudocost)
    : mipworker(mipworker),
      mipsolver(mipworker.getMipSolver()),
      lp(nullptr),
      localdom(mipworker.getGlobalDomain()),
      pseudocost(pseudocost) {
  nnodes = 0;
  nleaves = 0;
  treeweight = 0.0;
  depthoffset = 0;
  lpiterations = 0;
  heurlpiterations = 0;
  sblpiterations = 0;
  upper_limit = kHighsInf;
  inheuristic = false;
  inbranching = false;
  countTreeWeight = true;
  childselrule = mipsolver.submip ? ChildSelectionRule::kHybridInferenceCost
                                  : ChildSelectionRule::kRootSol;
  // the infeasibility flag is overwritten and lost when setDomainChangeStack is
  // called. therefore, assert that localdom is not infeasible here.
  assert(!this->localdom.infeasible());
  this->localdom.setDomainChangeStack(std::vector<HighsDomainChange>());
}
#endif  // HIGHS_RUST

double HighsSearch::checkSol(const std::vector<double>& sol,
                             bool& integerfeasible) const {
  HighsCDouble objval = 0.0;
  integerfeasible = true;
  for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
    objval += sol[i] * mipsolver.colCost(i);
    assert(std::isfinite(sol[i]));

    if (!integerfeasible || !mipsolver.isColInteger(i)) continue;

    if (fractionality(sol[i]) > getFeasTol()) {
      integerfeasible = false;
    }
  }

  return double(objval);
}

#ifndef HIGHS_RUST
bool HighsSearch::orbitsValidInChildNode(
    const HighsDomainChange& branchChg) const {
  HighsInt branchCol = branchChg.column;
  // if the variable is integral or we are in an up branch the stabilizer only
  // stays valid if the column has been stabilized
  const NodeData& currNode = nodestack.back();
  if (!currNode.stabilizerOrbits ||
      currNode.stabilizerOrbits->orbitCols.empty() ||
      currNode.stabilizerOrbits->isStabilized(branchCol))
    return true;

  // a down branch stays valid if the variable is binary
  if (branchChg.boundtype == HighsBoundType::kUpper &&
      localdom.isGlobalBinary(branchChg.column))
    return true;

  return false;
}
#endif  // HIGHS_RUST

double HighsSearch::getCutoffBound() const {
  return std::min(getUpperLimit(), upper_limit);
}

void HighsSearch::setRINSNeighbourhood(const std::vector<double>& basesol,
                                       const std::vector<double>& relaxsol) {
  for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
    if (!mipsolver.isColInteger(i)) continue;
    if (localdom.col_lower_[i] == localdom.col_upper_[i]) continue;

    double intval = std::floor(basesol[i] + 0.5);
    if (std::abs(relaxsol[i] - intval) < getFeasTol()) {
      if (localdom.col_lower_[i] < intval)
        localdom.changeBound(HighsBoundType::kLower, i,
                             std::min(intval, localdom.col_upper_[i]),
                             HighsDomain::Reason::unspecified());
      if (localdom.col_upper_[i] > intval)
        localdom.changeBound(HighsBoundType::kUpper, i,
                             std::max(intval, localdom.col_lower_[i]),
                             HighsDomain::Reason::unspecified());
    }
  }
}

void HighsSearch::setRENSNeighbourhood(const std::vector<double>& lpsol) {
  for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
    if (!mipsolver.isColInteger(i)) continue;
    if (localdom.col_lower_[i] == localdom.col_upper_[i]) continue;

    double downval = std::floor(lpsol[i] + getFeasTol());
    double upval = std::ceil(lpsol[i] - getFeasTol());

    if (localdom.col_lower_[i] < downval) {
      localdom.changeBound(HighsBoundType::kLower, i,
                           std::min(downval, localdom.col_upper_[i]),
                           HighsDomain::Reason::unspecified());
      if (localdom.infeasible()) return;
    }
    if (localdom.col_upper_[i] > upval) {
      localdom.changeBound(HighsBoundType::kUpper, i,
                           std::max(upval, localdom.col_lower_[i]),
                           HighsDomain::Reason::unspecified());
      if (localdom.infeasible()) return;
    }
  }
}

#ifndef HIGHS_RUST
void HighsSearch::createNewNode() {
  nodestack.emplace_back();
  nodestack.back().domgchgStackPos = localdom.getDomainChangeStack().size();
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
void HighsSearch::cutoffNode() { nodestack.back().opensubtrees = 0; }

void HighsSearch::setMinReliable(HighsInt minreliable) {
  pseudocost.setMinReliable(minreliable);
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
void HighsSearch::branchDownwards(HighsInt col, double newub,
                                  double branchpoint) {
  NodeData& currnode = nodestack.back();

  assert(currnode.opensubtrees == 2);
  assert(mipsolver.isColIntegral(col));

  currnode.opensubtrees = 1;
  currnode.branching_point = branchpoint;
  currnode.branchingdecision.column = col;
  currnode.branchingdecision.boundval = newub;
  currnode.branchingdecision.boundtype = HighsBoundType::kUpper;

  HighsInt domchgPos = localdom.getDomainChangeStack().size();
  bool passStabilizerToChildNode =
      orbitsValidInChildNode(currnode.branchingdecision);
  localdom.changeBound(currnode.branchingdecision);
  nodestack.emplace_back(
      currnode.lower_bound, currnode.estimate, currnode.nodeBasis,
      passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);
  nodestack.back().domgchgStackPos = domchgPos;
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
void HighsSearch::branchUpwards(HighsInt col, double newlb,
                                double branchpoint) {
  NodeData& currnode = nodestack.back();

  assert(currnode.opensubtrees == 2);
  assert(mipsolver.isColIntegral(col));

  currnode.opensubtrees = 1;
  currnode.branching_point = branchpoint;
  currnode.branchingdecision.column = col;
  currnode.branchingdecision.boundval = newlb;
  currnode.branchingdecision.boundtype = HighsBoundType::kLower;

  HighsInt domchgPos = localdom.getDomainChangeStack().size();
  bool passStabilizerToChildNode =
      orbitsValidInChildNode(currnode.branchingdecision);
  localdom.changeBound(currnode.branchingdecision);
  nodestack.emplace_back(
      currnode.lower_bound, currnode.estimate, currnode.nodeBasis,
      passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);
  nodestack.back().domgchgStackPos = domchgPos;
}
#endif  // HIGHS_RUST

void HighsSearch::addBoundExceedingConflict() {
  if (getUpperLimit() != kHighsInf) {
    double rhs;
    if (lp->computeDualProof(getDomain(), getUpperLimit(), inds, vals, rhs)) {
      if (getDomain().infeasible()) return;
      localdom.conflictAnalysis(inds.data(), vals.data(), inds.size(), rhs,
                                getConflictPool(), mipworker.getGlobalDomain(),
                                pseudocost);

      HighsCutGeneration cutGen(*lp, getCutPool());
      mipsolver.mipdata_->debugSolution.checkCut(inds.data(), vals.data(),
                                                 inds.size(), rhs);
      cutGen.generateConflict(localdom, mipworker.getGlobalDomain(), inds, vals,
                              rhs);
    }
  }
}

void HighsSearch::addInfeasibleConflict() {
  double rhs;
  if (lp->getLpSolver().getModelStatus() == HighsModelStatus::kObjectiveBound)
    lp->performAging();

  if (lp->computeDualInfProof(getDomain(), inds, vals, rhs)) {
    if (getDomain().infeasible()) return;
    // double minactlocal = 0.0;
    // double minactglobal = 0.0;
    // for (HighsInt i = 0; i < int(inds.size()); ++i) {
    //  if (vals[i] > 0.0) {
    //    minactlocal += localdom.col_lower_[inds[i]] * vals[i];
    //    minactglobal += globaldom.col_lower_[inds[i]] * vals[i];
    //  } else {
    //    minactlocal += localdom.col_upper_[inds[i]] * vals[i];
    //    minactglobal += globaldom.col_upper_[inds[i]] * vals[i];
    //  }
    //}
    // HighsInt oldnumcuts = cutpool.getNumCuts();
    localdom.conflictAnalysis(inds.data(), vals.data(), inds.size(), rhs,
                              getConflictPool(), mipworker.getGlobalDomain(),
                              pseudocost);

    HighsCutGeneration cutGen(*lp, getCutPool());
    mipsolver.mipdata_->debugSolution.checkCut(inds.data(), vals.data(),
                                               inds.size(), rhs);
    cutGen.generateConflict(localdom, mipworker.getGlobalDomain(), inds, vals,
                            rhs);

    // if (cutpool.getNumCuts() > oldnumcuts) {
    //  printf(
    //      "added cut from infeasibility proof with local min activity %g, "
    //      "global min activity %g, and rhs %g\n",
    //      minactlocal, minactglobal, rhs);
    //} else {
    //  printf(
    //      "no cut found for infeasibility proof with local min activity %g, "
    //      "global min "
    //      " activity %g, and rhs % g\n ",
    //      minactlocal, minactglobal, rhs);
    //}
    // HighsInt cutind = cutpool.addCut(inds.data(), vals.data(), inds.size(),
    // rhs); localdom.cutAdded(cutind);
  }
}

#ifndef HIGHS_RUST
HighsInt HighsSearch::selectBranchingCandidate(int64_t maxSbIters,
                                               double& downNodeLb,
                                               double& upNodeLb) {
  assert(!lp->getFractionalIntegers().empty());

  std::vector<double> upscore;
  std::vector<double> downscore;
  std::vector<uint8_t> upscorereliable;
  std::vector<uint8_t> downscorereliable;
  std::vector<double> upbound;
  std::vector<double> downbound;

  HighsInt numfrac = lp->getFractionalIntegers().size();
  const auto& fracints = lp->getFractionalIntegers();

  upscore.resize(numfrac, kHighsInf);
  downscore.resize(numfrac, kHighsInf);
  upbound.resize(numfrac, getCurrentLowerBound());
  downbound.resize(numfrac, getCurrentLowerBound());

  upscorereliable.resize(numfrac, 0);
  downscorereliable.resize(numfrac, 0);

  // initialize up and down scores of variables that have a
  // reliable pseudocost so that they do not get evaluated
  for (HighsInt k = 0; k != numfrac; ++k) {
    HighsInt col = fracints[k].first;
    double fracval = fracints[k].second;

    const double lower_residual =
        (fracval - localdom.col_lower_[col]) - getFeasTol();
    const bool lower_ok = lower_residual > 0;
    if (!lower_ok)
      highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kError,
                   "HighsSearch::selectBranchingCandidate Error fracval = %g "
                   "<= %g = %g + %g = "
                   "localdom.col_lower_[col] + getFeasTol(): "
                   "Residual %g\n",
                   fracval, localdom.col_lower_[col] + getFeasTol(),
                   localdom.col_lower_[col], getFeasTol(), lower_residual);

    const double upper_residual =
        (localdom.col_upper_[col] - fracval) - getFeasTol();
    const bool upper_ok = upper_residual > 0;
    if (!upper_ok)
      highsLogUser(mipsolver.options_mip_->log_options, HighsLogType::kError,
                   "HighsSearch::selectBranchingCandidate Error fracval = %g "
                   ">= %g = %g - %g = "
                   "localdom.col_upper_[col] - getFeasTol(): "
                   "Residual %g\n",
                   fracval, localdom.col_upper_[col] - getFeasTol(),
                   localdom.col_upper_[col], getFeasTol(), upper_residual);

    assert(lower_residual > -1e-12 && upper_residual > -1e-12);

    //    assert(fracval > localdom.col_lower_[col] +
    //    getFeasTol()); assert(fracval <
    //    localdom.col_upper_[col] - getFeasTol());

    if (pseudocost.isReliable(col)) {
      upscore[k] = pseudocost.getPseudocostUp(col, fracval);
      downscore[k] = pseudocost.getPseudocostDown(col, fracval);
      upscorereliable[k] = true;
      downscorereliable[k] = true;
    } else {
      int flags = branchingVarReliableAtNodeFlags(col);
      if (flags & kUpReliable) {
        upscore[k] = pseudocost.getPseudocostUp(col, fracval);
        upscorereliable[k] = true;
      }

      if (flags & kDownReliable) {
        downscore[k] = pseudocost.getPseudocostDown(col, fracval);
        downscorereliable[k] = true;
      }
    }
  }

  std::vector<HighsInt> evalqueue;
  evalqueue.resize(numfrac);
  std::iota(evalqueue.begin(), evalqueue.end(), 0);

  auto numNodesUp = [&](HighsInt k) {
    return getNodeQueue().numNodesUp(fracints[k].first);
  };

  auto numNodesDown = [&](HighsInt k) {
    return getNodeQueue().numNodesDown(fracints[k].first);
  };

  double minScore = getFeasTol();

  auto selectBestScore = [&](bool finalSelection) {
    HighsInt best = -1;
    double bestscore = -1.0;
    double bestnodes = -1.0;
    int64_t bestnumnodes = 0;

    double oldminscore = minScore;
    for (HighsInt k : evalqueue) {
      double score;

      if (upscore[k] <= oldminscore) upscorereliable[k] = true;
      if (downscore[k] <= oldminscore) downscorereliable[k] = true;

      double s = 1e-3 * std::min(upscorereliable[k] ? upscore[k] : 0,
                                 downscorereliable[k] ? downscore[k] : 0);
      minScore = std::max(s, minScore);

      if (upscore[k] <= oldminscore || downscore[k] <= oldminscore)
        score = pseudocost.getScore(fracints[k].first,
                                    std::min(upscore[k], oldminscore),
                                    std::min(downscore[k], oldminscore));
      else {
        score = upscore[k] == kHighsInf || downscore[k] == kHighsInf
                    ? finalSelection ? pseudocost.getScore(fracints[k].first,
                                                           fracints[k].second)
                                     : kHighsInf
                    : pseudocost.getScore(fracints[k].first, upscore[k],
                                          downscore[k]);
      }

      assert(score >= 0.0);
      int64_t upnodes = numNodesUp(k);
      int64_t downnodes = numNodesDown(k);
      double nodes = 0;
      int64_t numnodes = upnodes + downnodes;
      if (upnodes != 0 || downnodes != 0)
        nodes =
            (downnodes / (double)(numnodes)) * (upnodes / (double)(numnodes));
      if (score > bestscore || (score > bestscore - getFeasTol() &&
                                std::make_pair(nodes, numnodes) >
                                    std::make_pair(bestnodes, bestnumnodes))) {
        bestscore = score;
        best = k;
        bestnodes = nodes;
        bestnumnodes = numnodes;
      }
    }

    return best;
  };

  HighsLpRelaxation::Playground playground = lp->playground();

  while (true) {
    bool mustStop =
        getStrongBranchingLpIterations() >= maxSbIters || checkLimits();

    HighsInt candidate = selectBestScore(mustStop);

    if ((upscorereliable[candidate] && downscorereliable[candidate]) ||
        mustStop) {
      downNodeLb = downbound[candidate];
      upNodeLb = upbound[candidate];
      return candidate;
    }

    lp->setObjectiveLimit(getUpperLimit());

    HighsInt col = fracints[candidate].first;
    double fracval = fracints[candidate].second;
    double upval = std::ceil(fracval);
    double downval = std::floor(fracval);

    auto analyzeSolution = [&](double objdelta,
                               const std::vector<double>& sol) {
      size_t numChangedCols = localdom.getChangedCols().size();
      HighsInt domchgStackSize = localdom.getDomainChangeStack().size();
      const auto& domchgstack = localdom.getDomainChangeStack();

      for (HighsInt k = 0; k != numfrac; ++k) {
        if (fracints[k].first == col) continue;
        double otherfracval = fracints[k].second;
        double otherdownval = std::floor(fracints[k].second);
        double otherupval = std::ceil(fracints[k].second);
        if (sol[fracints[k].first] <= otherdownval + getFeasTol()) {
          if (localdom.col_upper_[fracints[k].first] >
              otherdownval + getFeasTol()) {
            localdom.changeBound(HighsBoundType::kUpper, fracints[k].first,
                                 otherdownval);
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(
                  getConflictPool(), mipworker.getGlobalDomain(), pseudocost);
              localdom.backtrack();
              localdom.clearChangedCols(numChangedCols);
              continue;
            }
            localdom.propagate();
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(
                  getConflictPool(), mipworker.getGlobalDomain(), pseudocost);
              localdom.backtrack();
              localdom.clearChangedCols(numChangedCols);
              continue;
            }

            HighsInt newStackSize = localdom.getDomainChangeStack().size();

            bool solutionValid = true;
            for (HighsInt j = domchgStackSize + 1; j < newStackSize; ++j) {
              if (domchgstack[j].boundtype == HighsBoundType::kLower) {
                if (domchgstack[j].boundval >
                    sol[domchgstack[j].column] + getFeasTol()) {
                  solutionValid = false;
                  break;
                }
              } else {
                if (domchgstack[j].boundval <
                    sol[domchgstack[j].column] - getFeasTol()) {
                  solutionValid = false;
                  break;
                }
              }
            }

            localdom.backtrack();
            localdom.clearChangedCols(numChangedCols);
            if (!solutionValid) continue;
          }

          if (objdelta <= getFeasTol()) {
            pseudocost.addObservation(fracints[k].first,
                                      otherdownval - otherfracval, objdelta);
            markBranchingVarDownReliableAtNode(fracints[k].first);
          }

          downscore[k] = std::min(downscore[k], objdelta);
        } else if (sol[fracints[k].first] >= otherupval - getFeasTol()) {
          if (localdom.col_lower_[fracints[k].first] <
              otherupval - getFeasTol()) {
            localdom.changeBound(HighsBoundType::kLower, fracints[k].first,
                                 otherupval);

            if (localdom.infeasible()) {
              localdom.conflictAnalysis(
                  getConflictPool(), mipworker.getGlobalDomain(), pseudocost);
              localdom.backtrack();
              localdom.clearChangedCols(numChangedCols);
              continue;
            }
            localdom.propagate();
            if (localdom.infeasible()) {
              localdom.conflictAnalysis(
                  getConflictPool(), mipworker.getGlobalDomain(), pseudocost);
              localdom.backtrack();
              localdom.clearChangedCols(numChangedCols);
              continue;
            }

            HighsInt newStackSize = localdom.getDomainChangeStack().size();

            bool solutionValid = true;
            for (HighsInt j = domchgStackSize + 1; j < newStackSize; ++j) {
              if (domchgstack[j].boundtype == HighsBoundType::kLower) {
                if (domchgstack[j].boundval >
                    sol[domchgstack[j].column] + getFeasTol()) {
                  solutionValid = false;
                  break;
                }
              } else {
                if (domchgstack[j].boundval <
                    sol[domchgstack[j].column] - getFeasTol()) {
                  solutionValid = false;
                  break;
                }
              }
            }

            localdom.backtrack();
            localdom.clearChangedCols(numChangedCols);

            if (!solutionValid) continue;
          }

          if (objdelta <= getFeasTol()) {
            pseudocost.addObservation(fracints[k].first,
                                      otherupval - otherfracval, objdelta);
            markBranchingVarUpReliableAtNode(fracints[k].first);
          }

          upscore[k] = std::min(upscore[k], objdelta);
        }
      }
    };

    auto strongBranch = [&](bool upbranch) -> bool {
      int64_t inferences = -(int64_t)localdom.getDomainChangeStack().size() - 1;
      HighsBoundType boundtype =
          upbranch ? HighsBoundType::kLower : HighsBoundType::kUpper;
      double boundval = upbranch ? upval : downval;
      HighsDomainChange domchg{boundval, col, boundtype};

      bool orbitalFixing =
          nodestack.back().stabilizerOrbits && orbitsValidInChildNode(domchg);
      localdom.changeBound(domchg);
      localdom.propagate();

      if (!localdom.infeasible()) {
        if (orbitalFixing)
          nodestack.back().stabilizerOrbits->orbitalFixing(localdom);
        else
          getSymmetries().propagateOrbitopes(localdom);
      }

      inferences += localdom.getDomainChangeStack().size();
      if (localdom.infeasible()) {
        localdom.conflictAnalysis(getConflictPool(),
                                  mipworker.getGlobalDomain(), pseudocost);
        pseudocost.addCutoffObservation(col, upbranch);
        localdom.backtrack();
        localdom.clearChangedCols();

        if (upbranch) {
          branchDownwards(col, downval, fracval);
        } else {
          branchUpwards(col, upval, fracval);
        }
        nodestack[nodestack.size() - 2].opensubtrees = 0;
        nodestack[nodestack.size() - 2].skipDepthCount = 1;
        depthoffset -= 1;

        return true;
      }

      pseudocost.addInferenceObservation(col, inferences, upbranch);

      int64_t numiters = lp->getNumLpIterations();
      HighsLpRelaxation::Status status = playground.solveLp(localdom);
      numiters = lp->getNumLpIterations() - numiters;
      lpiterations += numiters;
      sblpiterations += numiters;

      if (lp->scaledOptimal(status)) {
        lp->performAging();

        double delta = upbranch ? upval - fracval : downval - fracval;
        bool integerfeasible;
        const std::vector<double>& sol = lp->getSolution().col_value;
        double solobj = checkSol(sol, integerfeasible);

        double objdelta = std::max(solobj - lp->getObjective(), 0.0);
        if (objdelta <= getEpsilon()) objdelta = 0.0;

        if (upbranch) {
          upscore[candidate] = objdelta;
          upscorereliable[candidate] = true;
          markBranchingVarUpReliableAtNode(col);
        } else {
          downscore[candidate] = objdelta;
          downscorereliable[candidate] = true;
          markBranchingVarDownReliableAtNode(col);
        }
        pseudocost.addObservation(col, delta, objdelta);
        analyzeSolution(objdelta, sol);

        if (lp->unscaledPrimalFeasible(status) && integerfeasible) {
          double cutoffbnd = getCutoffBound();
          addIncumbent(lp->getLpSolver().getSolution().col_value, solobj,
                       inheuristic ? kSolutionSourceHeuristic
                                   : kSolutionSourceBranching);

          if (getUpperLimit() < cutoffbnd)
            lp->setObjectiveLimit(getUpperLimit());
        }

        if (lp->unscaledDualFeasible(status)) {
          if (upbranch) {
            upbound[candidate] = solobj;
          } else {
            downbound[candidate] = solobj;
          }
          if (solobj > getOptimalityLimit()) {
            addBoundExceedingConflict();

            bool pruned = solobj > getCutoffBound();
            if (pruned) mipsolver.mipdata_->debugSolution.nodePruned(localdom);

            localdom.backtrack();
            lp->flushDomain(localdom);

            if (upbranch) {
              branchDownwards(col, downval, fracval);
            } else {
              branchUpwards(col, upval, fracval);
            }
            nodestack[nodestack.size() - 2].opensubtrees = pruned ? 0 : 1;
            nodestack[nodestack.size() - 2].other_child_lb = solobj;
            nodestack[nodestack.size() - 2].skipDepthCount = 1;
            depthoffset -= 1;

            return true;
          }
        } else if (solobj > getCutoffBound()) {
          addBoundExceedingConflict();
          localdom.propagate();
          bool infeas = localdom.infeasible();
          if (infeas) {
            localdom.backtrack();
            lp->flushDomain(localdom);

            if (upbranch) {
              branchDownwards(col, downval, fracval);
            } else {
              branchUpwards(col, upval, fracval);
            }
            nodestack[nodestack.size() - 2].opensubtrees = 0;
            nodestack[nodestack.size() - 2].skipDepthCount = 1;
            depthoffset -= 1;

            return true;
          }
        }
      } else if (status == HighsLpRelaxation::Status::kInfeasible) {
        mipsolver.mipdata_->debugSolution.nodePruned(localdom);
        addInfeasibleConflict();
        pseudocost.addCutoffObservation(col, upbranch);
        localdom.backtrack();
        lp->flushDomain(localdom);

        if (upbranch) {
          branchDownwards(col, downval, fracval);
        } else {
          branchUpwards(col, upval, fracval);
        }
        nodestack[nodestack.size() - 2].opensubtrees = 0;
        nodestack[nodestack.size() - 2].skipDepthCount = 1;
        depthoffset -= 1;

        return true;
      } else {
        // printf("todo2\n");
        // in case of an LP error we set the score of this variable to zero to
        // avoid choosing it as branching candidate if possible
        downscore[candidate] = 0.0;
        upscore[candidate] = 0.0;
        downscorereliable[candidate] = 1;
        upscorereliable[candidate] = 1;
        markBranchingVarUpReliableAtNode(col);
        markBranchingVarDownReliableAtNode(col);
      }

      localdom.backtrack();
      lp->flushDomain(localdom);
      return false;
    };

    if (!downscorereliable[candidate] &&
        (upscorereliable[candidate] ||
         std::make_pair(downscore[candidate],
                        pseudocost.getAvgInferencesDown(col)) >=
             std::make_pair(upscore[candidate],
                            pseudocost.getAvgInferencesUp(col)))) {
      // evaluate down branch
      // if (!mipsolver.submip)
      //   printf("down eval col=%d fracval=%g\n", col, fracval);
      if (strongBranch(false)) return -1;
    } else {
      // if (!mipsolver.submip)
      //  printf("up eval col=%d fracval=%g\n", col, fracval);
      // evaluate up branch
      if (strongBranch(true)) return -1;
    }
  }
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
const HighsSearch::NodeData* HighsSearch::getParentNodeData() const {
  if (nodestack.size() <= 1) return nullptr;

  return &nodestack[nodestack.size() - 2];
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
void HighsSearch::currentNodeToQueue(HighsNodeQueue& nodequeue) {
  auto oldchangedcols = localdom.getChangedCols().size();
  bool prune = nodestack.back().lower_bound > getCutoffBound();
  if (!prune) {
    localdom.propagate();
    localdom.clearChangedCols(oldchangedcols);
    prune = localdom.infeasible();
    if (prune)
      localdom.conflictAnalysis(getConflictPool(), mipworker.getGlobalDomain(),
                                pseudocost);
  }
  if (!prune) {
    std::vector<HighsInt> branchPositions;
    auto domchgStack = localdom.getReducedDomainChangeStack(branchPositions);
    double tmpTreeWeight = nodequeue.emplaceNode(
        std::move(domchgStack), std::move(branchPositions),
        std::max(nodestack.back().lower_bound,
                 localdom.getObjectiveLowerBound()),
        nodestack.back().estimate, getCurrentDepth());
    if (countTreeWeight) treeweight += tmpTreeWeight;
  } else {
    mipsolver.mipdata_->debugSolution.nodePruned(localdom);
    if (countTreeWeight) treeweight += std::ldexp(1.0, 1 - getCurrentDepth());
  }
  nodestack.back().opensubtrees = 0;
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
void HighsSearch::openNodesToQueue(HighsNodeQueue& nodequeue) {
  if (nodestack.empty()) return;

  // get the basis of the node highest up in the tree
  std::shared_ptr<const HighsBasis> basis;
  for (NodeData& nodeData : nodestack) {
    if (nodeData.nodeBasis) {
      basis = std::move(nodeData.nodeBasis);
      break;
    }
  }

  if (nodestack.back().opensubtrees == 0) backtrack(false);

  while (!nodestack.empty()) {
    auto oldchangedcols = localdom.getChangedCols().size();
    bool prune = nodestack.back().lower_bound > getCutoffBound();
    if (!prune) {
      localdom.propagate();
      localdom.clearChangedCols(oldchangedcols);
      prune = localdom.infeasible();
      if (prune)
        localdom.conflictAnalysis(getConflictPool(),
                                  mipworker.getGlobalDomain(), pseudocost);
    }
    if (!prune) {
      std::vector<HighsInt> branchPositions;
      auto domchgStack = localdom.getReducedDomainChangeStack(branchPositions);
      double tmpTreeWeight = nodequeue.emplaceNode(
          std::move(domchgStack), std::move(branchPositions),
          std::max(nodestack.back().lower_bound,
                   localdom.getObjectiveLowerBound()),
          nodestack.back().estimate, getCurrentDepth());
      if (countTreeWeight) treeweight += tmpTreeWeight;
    } else {
      mipsolver.mipdata_->debugSolution.nodePruned(localdom);
      if (countTreeWeight) treeweight += std::ldexp(1.0, 1 - getCurrentDepth());
    }
    nodestack.back().opensubtrees = 0;
    backtrack(false);
  }

  lp->flushDomain(localdom);
  if (basis) {
    if ((HighsInt)basis->row_status.size() == lp->numRows())
      lp->setStoredBasis(std::move(basis));
    lp->recoverBasis();
  }
}
#endif  // HIGHS_RUST

void HighsSearch::flushStatistics(HighsMipSolver& mipsolver) {
  mipsolver.mipdata_->num_nodes += nnodes;
  nnodes = 0;

  mipsolver.mipdata_->num_leaves += nleaves;
  nleaves = 0;

  mipsolver.mipdata_->pruned_treeweight += treeweight;
  treeweight = 0;

  mipsolver.mipdata_->total_lp_iterations += lpiterations;
  lpiterations = 0;

  mipsolver.mipdata_->heuristic_lp_iterations += heurlpiterations;
  heurlpiterations = 0;

  mipsolver.mipdata_->sb_lp_iterations += sblpiterations;
  sblpiterations = 0;
}

int64_t HighsSearch::getHeuristicLpIterations() const {
  return heurlpiterations + mipsolver.mipdata_->heuristic_lp_iterations;
}

int64_t HighsSearch::getTotalLpIterations() const {
  return lpiterations + mipsolver.mipdata_->total_lp_iterations;
}

#ifndef HIGHS_RUST
int64_t HighsSearch::getLocalLpIterations() const { return lpiterations; }

int64_t& HighsSearch::getLocalNodes() { return nnodes; }

int64_t& HighsSearch::getLocalLeaves() { return nleaves; }
#endif  // HIGHS_RUST

int64_t HighsSearch::getStrongBranchingLpIterations() const {
  return sblpiterations + mipsolver.mipdata_->sb_lp_iterations;
}

void HighsSearch::resetLocalDomain() {
  this->lp->resetToGlobalDomain(getDomain());
  localdom = getDomain();

#ifndef NDEBUG
  for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
    assert(lp->getLpSolver().getLp().col_lower_[i] == localdom.col_lower_[i] ||
           mipsolver.isColContinuous(i));
    assert(lp->getLpSolver().getLp().col_upper_[i] == localdom.col_upper_[i] ||
           mipsolver.isColContinuous(i));
  }
#endif
}

#ifndef HIGHS_RUST
void HighsSearch::installNode(HighsNodeQueue::OpenNode&& node) {
  localdom.setDomainChangeStack(node.domchgstack, node.branchings);
  bool globalSymmetriesValid = true;
  if (mipsolver.mipdata_->globalOrbits) {
    // if global orbits have been computed we check whether they are still valid
    // in this node
    const auto& domchgstack = localdom.getDomainChangeStack();
    for (HighsInt i : localdom.getBranchingPositions()) {
      HighsInt col = domchgstack[i].column;
      if (getSymmetries().getColumnPosition(col) == -1) continue;

      if (!getDomain().isBinary(col) ||
          (domchgstack[i].boundtype == HighsBoundType::kLower &&
           domchgstack[i].boundval == 1.0)) {
        globalSymmetriesValid = false;
        break;
      }
    }
  }
  nodestack.emplace_back(
      node.lower_bound, node.estimate, nullptr,
      globalSymmetriesValid ? mipsolver.mipdata_->globalOrbits : nullptr);
  subrootsol.clear();
  depthoffset = node.depth - 1;
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
HighsSearch::NodeResult HighsSearch::evaluateNode() {
  assert(!nodestack.empty());
  NodeData& currnode = nodestack.back();
  const NodeData* parent = getParentNodeData();

  const auto& domchgstack = localdom.getDomainChangeStack();

  if (!inheuristic && currnode.lower_bound > getOptimalityLimit())
    return NodeResult::kSubOptimal;

  localdom.propagate();

  if (!inheuristic && !localdom.infeasible()) {
    if (getSymmetries().numPerms > 0 && !currnode.stabilizerOrbits &&
        (parent == nullptr || !parent->stabilizerOrbits ||
         !parent->stabilizerOrbits->orbitCols.empty())) {
      currnode.stabilizerOrbits = getSymmetries().computeStabilizerOrbits(
          localdom, stabilizerOrbitWorkspace);
    }

    if (currnode.stabilizerOrbits)
      currnode.stabilizerOrbits->orbitalFixing(localdom);
    else
      getSymmetries().propagateOrbitopes(localdom);
  }
  if (parent != nullptr) {
    int64_t inferences = domchgstack.size() - (currnode.domgchgStackPos + 1);

    pseudocost.addInferenceObservation(
        parent->branchingdecision.column, inferences,
        parent->branchingdecision.boundtype == HighsBoundType::kLower);
  }

  NodeResult result = NodeResult::kOpen;

  if (localdom.infeasible()) {
    result = NodeResult::kDomainInfeasible;
    localdom.clearChangedCols();
    if (parent != nullptr && parent->lp_objective != -kHighsInf &&
        parent->branching_point != parent->branchingdecision.boundval) {
      bool upbranch =
          parent->branchingdecision.boundtype == HighsBoundType::kLower;
      pseudocost.addCutoffObservation(parent->branchingdecision.column,
                                      upbranch);
    }

    localdom.conflictAnalysis(getConflictPool(), mipworker.getGlobalDomain(),
                              pseudocost);
  } else {
    lp->flushDomain(localdom);
    lp->setObjectiveLimit(getUpperLimit());

#ifndef NDEBUG
    for (HighsInt i = 0; i != mipsolver.numCol(); ++i) {
      assert(lp->getLpSolver().getLp().col_lower_[i] ==
                 localdom.col_lower_[i] ||
             mipsolver.isColContinuous(i));
      assert(lp->getLpSolver().getLp().col_upper_[i] ==
                 localdom.col_upper_[i] ||
             mipsolver.isColContinuous(i));
    }
#endif
    int64_t oldnumiters = lp->getNumLpIterations();
    HighsLpRelaxation::Status status = lp->resolveLp(&localdom);
    lpiterations += lp->getNumLpIterations() - oldnumiters;

    currnode.lower_bound =
        std::max(localdom.getObjectiveLowerBound(), currnode.lower_bound);

    if (localdom.infeasible()) {
      result = NodeResult::kDomainInfeasible;
      localdom.clearChangedCols();
      if (parent != nullptr && parent->lp_objective != -kHighsInf &&
          parent->branching_point != parent->branchingdecision.boundval) {
        bool upbranch =
            parent->branchingdecision.boundtype == HighsBoundType::kLower;
        pseudocost.addCutoffObservation(parent->branchingdecision.column,
                                        upbranch);
      }

      localdom.conflictAnalysis(getConflictPool(), mipworker.getGlobalDomain(),
                                pseudocost);
    } else if (lp->scaledOptimal(status)) {
      lp->storeBasis();
      lp->performAging();

      currnode.nodeBasis = lp->getStoredBasis();
      currnode.estimate = lp->computeBestEstimate(pseudocost);
      currnode.lp_objective = lp->getObjective();

      if (parent != nullptr && parent->lp_objective != -kHighsInf &&
          parent->branching_point != parent->branchingdecision.boundval) {
        double delta =
            parent->branchingdecision.boundval - parent->branching_point;
        double objdelta =
            std::max(0.0, currnode.lp_objective - parent->lp_objective);

        pseudocost.addObservation(parent->branchingdecision.column, delta,
                                  objdelta);
      }

      if (lp->unscaledPrimalFeasible(status)) {
        if (lp->getFractionalIntegers().empty()) {
          double cutoffbnd = getCutoffBound();
          addIncumbent(lp->getLpSolver().getSolution().col_value,
                       lp->getObjective(),
                       inheuristic ? kSolutionSourceHeuristic
                                   : kSolutionSourceEvaluateNode);
          if (getUpperLimit() < cutoffbnd)
            lp->setObjectiveLimit(getUpperLimit());

          if (lp->unscaledDualFeasible(status)) {
            addBoundExceedingConflict();
            result = NodeResult::kBoundExceeding;
          }
        }
      }

      if (result == NodeResult::kOpen) {
        if (lp->unscaledDualFeasible(status)) {
          currnode.lower_bound =
              std::max(currnode.lp_objective, currnode.lower_bound);

          if (currnode.lower_bound > getCutoffBound()) {
            result = NodeResult::kBoundExceeding;
            addBoundExceedingConflict();
          } else if (getUpperLimit() != kHighsInf) {
            if (!inheuristic) {
              double gap = getUpperLimit() - lp->getObjective();
              lp->computeBasicDegenerateDuals(
                  gap + std::max(10 * getFeasTol(), getEpsilon() * gap),
                  localdom, getDomain(), getConflictPool(),
                  mipworker.getPseudocost(), true);
            }
            HighsRedcostFixing::propagateRedCost(
                mipsolver, localdom, mipworker.getGlobalDomain(), *lp,
                getConflictPool(), mipworker.getPseudocost(), getUpperLimit());
            localdom.propagate();
            if (localdom.infeasible()) {
              result = NodeResult::kDomainInfeasible;
              localdom.clearChangedCols();
              if (parent != nullptr && parent->lp_objective != -kHighsInf &&
                  parent->branching_point !=
                      parent->branchingdecision.boundval) {
                bool upbranch = parent->branchingdecision.boundtype ==
                                HighsBoundType::kLower;
                pseudocost.addCutoffObservation(
                    parent->branchingdecision.column, upbranch);
              }

              localdom.conflictAnalysis(
                  getConflictPool(), mipworker.getGlobalDomain(), pseudocost);
            } else if (!localdom.getChangedCols().empty()) {
              return evaluateNode();
            }
          } else {
            if (!inheuristic) {
              lp->computeBasicDegenerateDuals(kHighsInf, localdom, getDomain(),
                                              getConflictPool(),
                                              mipworker.getPseudocost(), true);
              localdom.propagate();
              if (localdom.infeasible()) {
                result = NodeResult::kDomainInfeasible;
                localdom.clearChangedCols();
                if (parent != nullptr && parent->lp_objective != -kHighsInf &&
                    parent->branching_point !=
                        parent->branchingdecision.boundval) {
                  bool upbranch = parent->branchingdecision.boundtype ==
                                  HighsBoundType::kLower;
                  pseudocost.addCutoffObservation(
                      parent->branchingdecision.column, upbranch);
                }

                localdom.conflictAnalysis(
                    getConflictPool(), mipworker.getGlobalDomain(), pseudocost);
              } else if (!localdom.getChangedCols().empty()) {
                return evaluateNode();
              }
            }
          }
        } else if (lp->getObjective() > getCutoffBound()) {
          // the LP is not solved to dual feasibility due to scaling/numerics
          // therefore we compute a conflict constraint as if the LP was bound
          // exceeding and propagate the local domain again. The lp relaxation
          // class will take care to consider the dual multipliers with an
          // increased zero tolerance due to the dual infeasibility when
          // computing the proof conBoundExceedingstraint.
          addBoundExceedingConflict();
          localdom.propagate();
          if (localdom.infeasible()) {
            result = NodeResult::kBoundExceeding;
          }
        }
      }
    } else if (status == HighsLpRelaxation::Status::kInfeasible) {
      if (lp->getLpSolver().getModelStatus() ==
          HighsModelStatus::kObjectiveBound)
        result = NodeResult::kBoundExceeding;
      else
        result = NodeResult::kLpInfeasible;
      addInfeasibleConflict();
      if (parent != nullptr && parent->lp_objective != -kHighsInf &&
          parent->branching_point != parent->branchingdecision.boundval) {
        bool upbranch =
            parent->branchingdecision.boundtype == HighsBoundType::kLower;
        pseudocost.addCutoffObservation(parent->branchingdecision.column,
                                        upbranch);
      }
    }
  }

  if (result != NodeResult::kOpen) {
    mipsolver.mipdata_->debugSolution.nodePruned(localdom);
    treeweight += std::ldexp(1.0, 1 - getCurrentDepth());
    currnode.opensubtrees = 0;
  } else if (!inheuristic) {
    if (currnode.lower_bound > getOptimalityLimit()) {
      result = NodeResult::kSubOptimal;
      addBoundExceedingConflict();
    }
  }

  return result;
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
HighsSearch::NodeResult HighsSearch::branch() {
  assert(localdom.getChangedCols().empty());

  assert(nodestack.back().opensubtrees == 2);
  nodestack.back().branchingdecision.column = -1;
  inbranching = true;

  HighsInt minrel = pseudocost.getMinReliable();
  double childLb = getCurrentLowerBound();
  NodeResult result = NodeResult::kOpen;
  while (nodestack.back().opensubtrees == 2 &&
         lp->scaledOptimal(lp->getStatus()) &&
         !lp->getFractionalIntegers().empty()) {
    int64_t sbmaxiters = 0;
    if (minrel > 0) {
      int64_t sbiters = getStrongBranchingLpIterations();
      // where graph LNS rounds alternate with the tree search (a loose
      // target gap that incumbents close), strong branching starts with a
      // smaller budget: it is expensive there, and the bound rarely matters
      const int64_t sbBase =
          mipsolver.mipdata_->lns_tree_next >= 0 ? 10000 : 100000;
      sbmaxiters =
          sbBase + ((getTotalLpIterations() - getHeuristicLpIterations() -
                     getStrongBranchingLpIterations()) >>
                    1);
      if (sbiters > sbmaxiters) {
        pseudocost.setMinReliable(0);
      } else if (sbiters > (sbmaxiters >> 1)) {
        double reductionratio = (sbiters - (sbmaxiters >> 1)) /
                                (double)(sbmaxiters - (sbmaxiters >> 1));

        HighsInt minrelreduced = int(minrel - reductionratio * (minrel - 1));
        pseudocost.setMinReliable(std::min(minrel, minrelreduced));
      }
    }

    double degeneracyFac = lp->computeLPDegneracy(localdom);
    pseudocost.setDegeneracyFactor(degeneracyFac);
    if (degeneracyFac >= 10.0) pseudocost.setMinReliable(0);
    // if (!mipsolver.submip)
    //  printf("selecting branching cand with minrel=%d\n",
    //         pseudocost.getMinReliable());
    double downNodeLb = getCurrentLowerBound();
    double upNodeLb = getCurrentLowerBound();
    HighsInt branchcand =
        selectBranchingCandidate(sbmaxiters, downNodeLb, upNodeLb);
    // if (!mipsolver.submip)
    //   printf("branching cand returned as %d\n", branchcand);
    NodeData& currnode = nodestack.back();
    childLb = currnode.lower_bound;
    if (branchcand != -1) {
      auto branching = lp->getFractionalIntegers()[branchcand];
      currnode.branchingdecision.column = branching.first;
      currnode.branching_point = branching.second;

      HighsInt col = branching.first;

      switch (childselrule) {
        case ChildSelectionRule::kUp:
          currnode.branchingdecision.boundtype = HighsBoundType::kLower;
          currnode.branchingdecision.boundval =
              std::ceil(currnode.branching_point);
          currnode.other_child_lb = downNodeLb;
          childLb = upNodeLb;
          break;
        case ChildSelectionRule::kDown:
          currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
          currnode.branchingdecision.boundval =
              std::floor(currnode.branching_point);
          currnode.other_child_lb = upNodeLb;
          childLb = downNodeLb;
          break;
        case ChildSelectionRule::kRootSol: {
          double downPrio = pseudocost.getAvgInferencesDown(col) + getEpsilon();
          double upPrio = pseudocost.getAvgInferencesUp(col) + getEpsilon();
          double downVal = std::floor(currnode.branching_point);
          double upVal = std::ceil(currnode.branching_point);
          if (!subrootsol.empty()) {
            double rootsol = subrootsol[col];
            if (rootsol < downVal)
              rootsol = downVal;
            else if (rootsol > upVal)
              rootsol = upVal;

            upPrio *= (1.0 + (currnode.branching_point - rootsol));
            downPrio *= (1.0 + (rootsol - currnode.branching_point));

          } else {
            if (currnode.lp_objective != -kHighsInf)
              subrootsol = lp->getSolution().col_value;
            if (!getRootLpSol().empty()) {
              double rootsol = getRootLpSol()[col];
              if (rootsol < downVal)
                rootsol = downVal;
              else if (rootsol > upVal)
                rootsol = upVal;

              upPrio *= (1.0 + (currnode.branching_point - rootsol));
              downPrio *= (1.0 + (rootsol - currnode.branching_point));
            }
          }
          if (upPrio + getEpsilon() >= downPrio) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval = upVal;
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval = downVal;
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
          break;
        }
        case ChildSelectionRule::kObj:
          if (mipsolver.colCost(col) >= 0) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
          break;
        case ChildSelectionRule::kRandom:
          if (random.bit()) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
          break;
        case ChildSelectionRule::kBestCost: {
          if (pseudocost.getPseudocostUp(col, currnode.branching_point,
                                         getFeasTol()) >
              pseudocost.getPseudocostDown(col, currnode.branching_point,
                                           getFeasTol())) {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          }
          break;
        }
        case ChildSelectionRule::kWorstCost:
          if (pseudocost.getPseudocostUp(col, currnode.branching_point) >=
              pseudocost.getPseudocostDown(col, currnode.branching_point)) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
          break;
        case ChildSelectionRule::kDisjunction: {
          int64_t numnodesup;
          int64_t numnodesdown;
          numnodesup = getNodeQueue().numNodesUp(col);
          numnodesdown = getNodeQueue().numNodesDown(col);
          if (numnodesup > numnodesdown) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval =
                std::ceil(currnode.branching_point);
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else if (numnodesdown > numnodesup) {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval =
                std::floor(currnode.branching_point);
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          } else {
            if (mipsolver.colCost(col) >= 0) {
              currnode.branchingdecision.boundtype = HighsBoundType::kLower;
              currnode.branchingdecision.boundval =
                  std::ceil(currnode.branching_point);
              currnode.other_child_lb = downNodeLb;
              childLb = upNodeLb;
            } else {
              currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
              currnode.branchingdecision.boundval =
                  std::floor(currnode.branching_point);
              currnode.other_child_lb = upNodeLb;
              childLb = downNodeLb;
            }
          }
          break;
        }
        case ChildSelectionRule::kHybridInferenceCost: {
          double upVal = std::ceil(currnode.branching_point);
          double downVal = std::floor(currnode.branching_point);
          double upScore = (1 + pseudocost.getAvgInferencesUp(col)) /
                           pseudocost.getPseudocostUp(
                               col, currnode.branching_point, getFeasTol());
          double downScore = (1 + pseudocost.getAvgInferencesDown(col)) /
                             pseudocost.getPseudocostDown(
                                 col, currnode.branching_point, getFeasTol());

          if (upScore >= downScore) {
            currnode.branchingdecision.boundtype = HighsBoundType::kLower;
            currnode.branchingdecision.boundval = upVal;
            currnode.other_child_lb = downNodeLb;
            childLb = upNodeLb;
          } else {
            currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
            currnode.branchingdecision.boundval = downVal;
            currnode.other_child_lb = upNodeLb;
            childLb = downNodeLb;
          }
        }
      }
      result = NodeResult::kBranched;
      break;
    }

    assert(!localdom.getChangedCols().empty());
    result = evaluateNode();
    if (result == NodeResult::kSubOptimal) break;
  }
  inbranching = false;
  NodeData& currnode = nodestack.back();
  pseudocost.setMinReliable(minrel);
  pseudocost.setDegeneracyFactor(1.0);

  assert(currnode.opensubtrees == 2 || currnode.opensubtrees == 0);

  if (currnode.opensubtrees != 2 || result == NodeResult::kSubOptimal)
    return result;

  if (currnode.branchingdecision.column == -1) {
    double bestscore = -1.0;
    // solution branching failed, so choose any integer variable to branch
    // on in case we have a different solution status could happen due to a
    // fail in the LP solution process
    pseudocost.setDegeneracyFactor(1e6);

    for (HighsInt i : getIntegralCols()) {
      if (localdom.col_upper_[i] - localdom.col_lower_[i] < 0.5) continue;

      double fracval;
      if (localdom.col_lower_[i] != -kHighsInf &&
          localdom.col_upper_[i] != kHighsInf)
        fracval = std::floor(0.5 * (localdom.col_lower_[i] +
                                    localdom.col_upper_[i] + 0.5)) +
                  0.5;
      else if (localdom.col_lower_[i] != -kHighsInf)
        fracval = localdom.col_lower_[i] + 0.5;
      else if (localdom.col_upper_[i] != kHighsInf)
        fracval = localdom.col_upper_[i] - 0.5;
      else
        fracval = 0.5;

      double score = pseudocost.getScore(i, fracval);
      assert(score >= 0.0);

      if (score > bestscore) {
        bestscore = score;
        bool branchUpwards;
        double cost = lp->unscaledDualFeasible(lp->getStatus())
                          ? lp->getSolution().col_dual[i]
                          : mipsolver.colCost(i);
        if (std::fabs(cost) > getFeasTol() && getCutoffBound() < kHighsInf) {
          // branch in direction of worsening cost first in case the column has
          // cost and we do have an upper bound
          branchUpwards = cost > 0;
        } else if (pseudocost.getAvgInferencesUp(i) >
                   pseudocost.getAvgInferencesDown(i) + getFeasTol()) {
          // column does not have (reduced) cost above tolerance so branch in
          // direction of more inferences
          branchUpwards = true;
        } else if (pseudocost.getAvgInferencesUp(i) <
                   pseudocost.getAvgInferencesDown(i) - getFeasTol()) {
          branchUpwards = false;
        } else {
          // number of inferences give a tie, so we branch in the direction that
          // does have a less recent domain change to avoid branching the same
          // integer column into the same direction over and over
          HighsInt colLowerPos;
          HighsInt colUpperPos;
          localdom.getColLowerPos(i, localdom.getNumDomainChanges(),
                                  colLowerPos);
          localdom.getColUpperPos(i, localdom.getNumDomainChanges(),
                                  colUpperPos);
          branchUpwards = colLowerPos <= colUpperPos;
        }
        if (branchUpwards) {
          double upval = std::ceil(fracval);
          currnode.branching_point = upval;
          currnode.branchingdecision.boundtype = HighsBoundType::kLower;
          currnode.branchingdecision.column = i;
          currnode.branchingdecision.boundval = upval;
        } else {
          double downval = std::floor(fracval);
          currnode.branching_point = downval;
          currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
          currnode.branchingdecision.column = i;
          currnode.branchingdecision.boundval = downval;
        }
      }
    }

    pseudocost.setDegeneracyFactor(1);
  }

  if (currnode.branchingdecision.column == -1) {
    if (lp->getStatus() == HighsLpRelaxation::Status::kOptimal) {
      // if the LP was solved to optimality and all columns are fixed, then this
      // particular assignment is not feasible or has a worse objective in the
      // original space, otherwise the node would not be open. Hence we prune
      // this particular assignment
      currnode.opensubtrees = 0;
      result = NodeResult::kLpInfeasible;
      return result;
    }
    lp->setIterationLimit();

    // create a fresh LP only with model rows since all integer columns are
    // fixed, the cutting planes are not required and the LP could not be solved
    // so we want to make it as easy as possible
    //
    // LP relaxation instantiation
    HighsLpRelaxation lpCopy(mipsolver);
    lpCopy.setProfiling(mipsolver.profiling_);
    lpCopy.loadModel();
    lpCopy.getLpSolver().changeColsBounds(0, mipsolver.numCol() - 1,
                                          localdom.col_lower_.data(),
                                          localdom.col_upper_.data());
    // temporarily use the fresh LP for the HighsSearch class
    HighsLpRelaxation* tmpLp = &lpCopy;
    std::swap(tmpLp, lp);

    // reevaluate the node with LP presolve enabled
    lp->getLpSolver().setOptionValue("presolve", kHighsOnString);
    result = evaluateNode();

    if (result == NodeResult::kOpen) {
      // LP still not solved, reevaluate with primal simplex
      lp->getLpSolver().clearSolver();
      lp->getLpSolver().setOptionValue("simplex_strategy",
                                       kSimplexStrategyPrimal);
      result = evaluateNode();
      lp->getLpSolver().setOptionValue("simplex_strategy",
                                       kSimplexStrategyDual);
      if (result == NodeResult::kOpen) {
        // LP still not solved, reevaluate with IPM instead of simplex
        lp->getLpSolver().clearSolver();
        lp->getLpSolver().setOptionValue("solver", "ipm");
        result = evaluateNode();

        if (result == NodeResult::kOpen) {
          highsLogUser(mipsolver.options_mip_->log_options,
                       HighsLogType::kWarning,
                       "Failed to solve node with all integer columns "
                       "fixed. Declaring node infeasible.\n");
          // LP still not solved, give up and declare as infeasible
          currnode.opensubtrees = 0;
          result = NodeResult::kLpInfeasible;
        }
      }
    }

    // restore old lp relaxation
    std::swap(tmpLp, lp);

    return result;
  }

  // finally open a new node with the branching decision added
  // and remember that we have one open subtree left
  HighsInt domchgPos = localdom.getDomainChangeStack().size();

  bool passStabilizerToChildNode =
      orbitsValidInChildNode(currnode.branchingdecision);
  localdom.changeBound(currnode.branchingdecision);
  currnode.opensubtrees = 1;
  nodestack.emplace_back(
      std::max(childLb, currnode.lower_bound), currnode.estimate,
      currnode.nodeBasis,
      passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);
  nodestack.back().domgchgStackPos = domchgPos;

  return NodeResult::kBranched;
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
bool HighsSearch::backtrack(bool recoverBasis) {
  if (nodestack.empty()) return false;
  assert(!nodestack.empty());
  assert(nodestack.back().opensubtrees == 0);
  while (true) {
    while (nodestack.back().opensubtrees == 0) {
      countTreeWeight = true;
      depthoffset += nodestack.back().skipDepthCount;
      if (nodestack.size() == 1) {
        if (recoverBasis && nodestack.back().nodeBasis)
          lp->setStoredBasis(std::move(nodestack.back().nodeBasis));
        nodestack.pop_back();
        localdom.backtrackToGlobal();
        lp->flushDomain(localdom);
        if (recoverBasis) lp->recoverBasis();
        return false;
      }

      nodestack.pop_back();
#ifndef NDEBUG
      HighsDomainChange branchchg =
#endif
          localdom.backtrack();

      if (nodestack.back().opensubtrees != 0) {
        countTreeWeight = nodestack.back().skipDepthCount == 0;
        // repropagate the node, as it may have become infeasible due to
        // conflicts
        HighsInt oldNumDomchgs = localdom.getNumDomainChanges();
        size_t oldNumChangedCols = localdom.getChangedCols().size();
        localdom.propagate();
        if (!localdom.infeasible() &&
            oldNumDomchgs != localdom.getNumDomainChanges()) {
          if (nodestack.back().stabilizerOrbits)
            nodestack.back().stabilizerOrbits->orbitalFixing(localdom);
          else
            getSymmetries().propagateOrbitopes(localdom);
        }
        if (localdom.infeasible()) {
          localdom.clearChangedCols(oldNumChangedCols);
          if (countTreeWeight)
            treeweight += std::ldexp(1.0, -getCurrentDepth());
          nodestack.back().opensubtrees = 0;
        }
      }

      assert(
          (branchchg.boundtype == HighsBoundType::kLower &&
           branchchg.boundval >= nodestack.back().branchingdecision.boundval) ||
          (branchchg.boundtype == HighsBoundType::kUpper &&
           branchchg.boundval <= nodestack.back().branchingdecision.boundval));
      assert(branchchg.boundtype ==
             nodestack.back().branchingdecision.boundtype);
      assert(branchchg.column == nodestack.back().branchingdecision.column);
    }

    NodeData& currnode = nodestack.back();

    assert(currnode.opensubtrees == 1);
    currnode.opensubtrees = 0;
    bool fallbackbranch =
        currnode.branchingdecision.boundval == currnode.branching_point;
    HighsInt domchgPos = localdom.getDomainChangeStack().size();
    if (currnode.branchingdecision.boundtype == HighsBoundType::kLower) {
      currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
      currnode.branchingdecision.boundval =
          std::floor(currnode.branchingdecision.boundval - 0.5);
    } else {
      currnode.branchingdecision.boundtype = HighsBoundType::kLower;
      currnode.branchingdecision.boundval =
          std::ceil(currnode.branchingdecision.boundval + 0.5);
    }

    if (fallbackbranch)
      currnode.branching_point = currnode.branchingdecision.boundval;

    size_t numChangedCols = localdom.getChangedCols().size();
    bool passStabilizerToChildNode =
        orbitsValidInChildNode(currnode.branchingdecision);
    localdom.changeBound(currnode.branchingdecision);
    double nodelb = std::max(currnode.lower_bound, currnode.other_child_lb);
    bool prune = nodelb > getCutoffBound() || localdom.infeasible();
    if (!prune) {
      localdom.propagate();
      prune = localdom.infeasible();
      if (prune)
        localdom.conflictAnalysis(getConflictPool(),
                                  mipworker.getGlobalDomain(), pseudocost);
    }
    if (!prune) {
      getSymmetries().propagateOrbitopes(localdom);
      prune = localdom.infeasible();
    }
    if (!prune && passStabilizerToChildNode && currnode.stabilizerOrbits) {
      currnode.stabilizerOrbits->orbitalFixing(localdom);
      prune = localdom.infeasible();
    }
    if (prune) {
      localdom.backtrack();
      localdom.clearChangedCols(numChangedCols);
      if (countTreeWeight) treeweight += std::ldexp(1.0, -getCurrentDepth());
      continue;
    }
    nodestack.emplace_back(
        nodelb, currnode.estimate, currnode.nodeBasis,
        passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);

    lp->flushDomain(localdom);
    nodestack.back().domgchgStackPos = domchgPos;
    break;
  }

  if (recoverBasis && nodestack.back().nodeBasis) {
    lp->setStoredBasis(nodestack.back().nodeBasis);
    lp->recoverBasis();
  }

  return true;
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
bool HighsSearch::backtrackPlunge(HighsNodeQueue& nodequeue) {
  const std::vector<HighsDomainChange>& domchgstack =
      localdom.getDomainChangeStack();

  if (nodestack.empty()) return false;
  assert(!nodestack.empty());
  assert(nodestack.back().opensubtrees == 0);

  while (true) {
    while (nodestack.back().opensubtrees == 0) {
      countTreeWeight = true;
      depthoffset += nodestack.back().skipDepthCount;

      if (nodestack.size() == 1) {
        if (nodestack.back().nodeBasis)
          lp->setStoredBasis(std::move(nodestack.back().nodeBasis));
        nodestack.pop_back();
        localdom.backtrackToGlobal();
        lp->flushDomain(localdom);
        lp->recoverBasis();
        return false;
      }

      nodestack.pop_back();
#ifndef NDEBUG
      HighsDomainChange branchchg =
#endif
          localdom.backtrack();

      if (nodestack.back().opensubtrees != 0) {
        countTreeWeight = nodestack.back().skipDepthCount == 0;
        // repropagate the node, as it may have become infeasible due to
        // conflicts
        HighsInt oldNumDomchgs = localdom.getNumDomainChanges();
        HighsInt oldNumChangedCols = localdom.getChangedCols().size();
        localdom.propagate();
        if (!localdom.infeasible() &&
            oldNumDomchgs != localdom.getNumDomainChanges()) {
          if (nodestack.back().stabilizerOrbits)
            nodestack.back().stabilizerOrbits->orbitalFixing(localdom);
          else
            getSymmetries().propagateOrbitopes(localdom);
        }
        if (localdom.infeasible()) {
          localdom.clearChangedCols(oldNumChangedCols);
          if (countTreeWeight)
            treeweight += std::ldexp(1.0, -getCurrentDepth());
          nodestack.back().opensubtrees = 0;
        }
      }

      assert(
          (branchchg.boundtype == HighsBoundType::kLower &&
           branchchg.boundval >= nodestack.back().branchingdecision.boundval) ||
          (branchchg.boundtype == HighsBoundType::kUpper &&
           branchchg.boundval <= nodestack.back().branchingdecision.boundval));
      assert(branchchg.boundtype ==
             nodestack.back().branchingdecision.boundtype);
      assert(branchchg.column == nodestack.back().branchingdecision.column);
    }

    NodeData& currnode = nodestack.back();

    assert(currnode.opensubtrees == 1);
    currnode.opensubtrees = 0;
    bool fallbackbranch =
        currnode.branchingdecision.boundval == currnode.branching_point;
    double nodeScore;
    if (currnode.branchingdecision.boundtype == HighsBoundType::kLower) {
      currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
      currnode.branchingdecision.boundval =
          std::floor(currnode.branchingdecision.boundval - 0.5);
      nodeScore = pseudocost.getScoreDown(
          currnode.branchingdecision.column,
          fallbackbranch ? 0.5 : currnode.branching_point);
    } else {
      currnode.branchingdecision.boundtype = HighsBoundType::kLower;
      currnode.branchingdecision.boundval =
          std::ceil(currnode.branchingdecision.boundval + 0.5);
      nodeScore = pseudocost.getScoreUp(
          currnode.branchingdecision.column,
          fallbackbranch ? 0.5 : currnode.branching_point);
    }

    if (fallbackbranch)
      currnode.branching_point = currnode.branchingdecision.boundval;

    HighsInt domchgPos = domchgstack.size();
    size_t numChangedCols = localdom.getChangedCols().size();
    bool passStabilizerToChildNode =
        orbitsValidInChildNode(currnode.branchingdecision);
    localdom.changeBound(currnode.branchingdecision);
    double nodelb = std::max(currnode.lower_bound, currnode.other_child_lb);
    bool prune = nodelb > getCutoffBound() || localdom.infeasible();
    if (!prune) {
      localdom.propagate();
      prune = localdom.infeasible();
      if (prune)
        localdom.conflictAnalysis(getConflictPool(),
                                  mipworker.getGlobalDomain(), pseudocost);
    }
    if (!prune) {
      getSymmetries().propagateOrbitopes(localdom);
      prune = localdom.infeasible();
    }
    if (!prune && passStabilizerToChildNode && currnode.stabilizerOrbits) {
      currnode.stabilizerOrbits->orbitalFixing(localdom);
      prune = localdom.infeasible();
    }
    if (prune) {
      localdom.backtrack();
      localdom.clearChangedCols(numChangedCols);
      if (countTreeWeight) treeweight += std::ldexp(1.0, -getCurrentDepth());
      continue;
    }

    nodelb = std::max(nodelb, localdom.getObjectiveLowerBound());
    bool nodeToQueue = nodelb > getOptimalityLimit();
    // we check if switching to the other branch of an ancestor yields a higher
    // additive branch score than staying in this node and if so we postpone the
    // node and put it to the queue to backtrack further.
    if (!nodeToQueue) {
      for (HighsInt i = nodestack.size() - 2; i >= 0; --i) {
        if (nodestack[i].opensubtrees == 0) continue;

        bool fallbackbranch = nodestack[i].branchingdecision.boundval ==
                              nodestack[i].branching_point;
        double branchpoint =
            fallbackbranch ? 0.5 : nodestack[i].branching_point;
        double ancestorScoreActive;
        double ancestorScoreInactive;
        if (nodestack[i].branchingdecision.boundtype ==
            HighsBoundType::kLower) {
          ancestorScoreInactive = pseudocost.getScoreDown(
              nodestack[i].branchingdecision.column, branchpoint);
          ancestorScoreActive = pseudocost.getScoreUp(
              nodestack[i].branchingdecision.column, branchpoint);
        } else {
          ancestorScoreActive = pseudocost.getScoreDown(
              nodestack[i].branchingdecision.column, branchpoint);
          ancestorScoreInactive = pseudocost.getScoreUp(
              nodestack[i].branchingdecision.column, branchpoint);
        }

        // if (!mipsolver.submip)
        //   printf("nodeScore: %g, ancestorScore: %g\n", nodeScore,
        //   ancestorScore);
        nodeToQueue = ancestorScoreInactive - ancestorScoreActive >
                      nodeScore + getFeasTol();
        break;
      }
    }

    if (nodeToQueue) {
      // if (!mipsolver.submip) printf("node goes to queue\n");
      std::vector<HighsInt> branchPositions;
      auto domchgStack = localdom.getReducedDomainChangeStack(branchPositions);
      double tmpTreeWeight = nodequeue.emplaceNode(
          std::move(domchgStack), std::move(branchPositions), nodelb,
          nodestack.back().estimate, getCurrentDepth() + 1);
      if (countTreeWeight) treeweight += tmpTreeWeight;
      localdom.backtrack();
      localdom.clearChangedCols(numChangedCols);
      continue;
    }
    nodestack.emplace_back(
        nodelb, currnode.estimate, currnode.nodeBasis,
        passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);

    lp->flushDomain(localdom);
    nodestack.back().domgchgStackPos = domchgPos;
    break;
  }

  if (nodestack.back().nodeBasis) {
    lp->setStoredBasis(nodestack.back().nodeBasis);
    lp->recoverBasis();
  }

  return true;
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
bool HighsSearch::backtrackUntilDepth(HighsInt targetDepth) {
  if (nodestack.empty()) return false;
  assert(!nodestack.empty());
  if (getCurrentDepth() >= targetDepth) nodestack.back().opensubtrees = 0;

  while (nodestack.back().opensubtrees == 0) {
    depthoffset += nodestack.back().skipDepthCount;
    nodestack.pop_back();

#ifndef NDEBUG
    HighsDomainChange branchchg =
#endif
        localdom.backtrack();
    if (nodestack.empty()) {
      lp->flushDomain(localdom);
      return false;
    }
    assert(
        (branchchg.boundtype == HighsBoundType::kLower &&
         branchchg.boundval >= nodestack.back().branchingdecision.boundval) ||
        (branchchg.boundtype == HighsBoundType::kUpper &&
         branchchg.boundval <= nodestack.back().branchingdecision.boundval));
    assert(branchchg.boundtype == nodestack.back().branchingdecision.boundtype);
    assert(branchchg.column == nodestack.back().branchingdecision.column);

    if (getCurrentDepth() >= targetDepth) nodestack.back().opensubtrees = 0;
  }

  NodeData& currnode = nodestack.back();
  assert(currnode.opensubtrees == 1);
  currnode.opensubtrees = 0;
  bool fallbackbranch =
      currnode.branchingdecision.boundval == currnode.branching_point;
  if (currnode.branchingdecision.boundtype == HighsBoundType::kLower) {
    currnode.branchingdecision.boundtype = HighsBoundType::kUpper;
    currnode.branchingdecision.boundval =
        std::floor(currnode.branchingdecision.boundval - 0.5);
  } else {
    currnode.branchingdecision.boundtype = HighsBoundType::kLower;
    currnode.branchingdecision.boundval =
        std::ceil(currnode.branchingdecision.boundval + 0.5);
  }

  if (fallbackbranch)
    currnode.branching_point = currnode.branchingdecision.boundval;

  HighsInt domchgPos = localdom.getDomainChangeStack().size();
  bool passStabilizerToChildNode =
      orbitsValidInChildNode(currnode.branchingdecision);
  localdom.changeBound(currnode.branchingdecision);
  nodestack.emplace_back(
      currnode.lower_bound, currnode.estimate, currnode.nodeBasis,
      passStabilizerToChildNode ? currnode.stabilizerOrbits : nullptr);

  lp->flushDomain(localdom);
  nodestack.back().domgchgStackPos = domchgPos;
  if (nodestack.back().nodeBasis &&
      (HighsInt)nodestack.back().nodeBasis->row_status.size() ==
          lp->getLp().num_row_)
    lp->setStoredBasis(nodestack.back().nodeBasis);
  lp->recoverBasis();

  return true;
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
HighsSearch::NodeResult HighsSearch::dive(int64_t nodeLim) {
  reliableatnode.clear();

  do {
    ++nnodes;
    NodeResult result = evaluateNode();

    if (checkLimits(nnodes)) return result;

    if (result != NodeResult::kOpen) return result;

    result = branch();
    if (result != NodeResult::kBranched) return result;
    if (nnodes >= nodeLim) return result;
  } while (true);
}
#endif  // HIGHS_RUST

#ifndef HIGHS_RUST
void HighsSearch::solveDepthFirst(int64_t maxbacktracks) {
  do {
    if (maxbacktracks == 0) break;

    NodeResult result = dive();
    // if a limit was reached the result might be open
    if (result == NodeResult::kOpen) break;

    --maxbacktracks;

  } while (backtrack());
}
#endif  // HIGHS_RUST

double HighsSearch::getFeasTol() const { return mipsolver.mipdata_->feastol; }

double HighsSearch::getUpperLimit() const {
  if (!mipsolver.mipdata_->parallelLockActive()) {
    return mipsolver.mipdata_->upper_limit;
  } else {
    return mipworker.upper_limit;
  }
}

double HighsSearch::getEpsilon() const { return mipsolver.mipdata_->epsilon; }

double HighsSearch::getOptimalityLimit() const {
  if (!mipsolver.mipdata_->parallelLockActive()) {
    return mipsolver.mipdata_->optimality_limit;
  } else {
    return mipworker.optimality_limit;
  }
}

const std::vector<double>& HighsSearch::getRootLpSol() const {
  return mipsolver.mipdata_->rootlpsol;
}

const std::vector<HighsInt>& HighsSearch::getIntegralCols() const {
  return mipsolver.mipdata_->integral_cols;
}

HighsDomain& HighsSearch::getDomain() const {
  return mipworker.getGlobalDomain();
}

HighsConflictPool& HighsSearch::getConflictPool() const {
  return mipworker.getConflictPool();
}

HighsCutPool& HighsSearch::getCutPool() const { return mipworker.getCutPool(); }

const HighsNodeQueue& HighsSearch::getNodeQueue() const {
  return mipsolver.mipdata_->nodequeue;
}

bool HighsSearch::checkLimits(int64_t nodeOffset) const {
  if (mipsolver.mipdata_->parallelLockActive()) {
    return checkLocalLimits();
  };
  return mipsolver.mipdata_->checkLimits(nodeOffset);
}

bool HighsSearch::checkLocalLimits() const {
  if (mipsolver.mipdata_->terminatorActive())
    if (mipsolver.mipdata_->terminatorTerminated()) return true;

  if (!mipsolver.submip && mipworker.upper_bound < kHighsInf &&
      mipsolver.options_mip_->objective_target > -kHighsInf) {
    const double internal_target =
        static_cast<HighsInt>(mipsolver.orig_model_->sense_) *
            mipsolver.options_mip_->objective_target -
        mipsolver.model_->offset_;
    if (mipworker.upper_bound < internal_target) {
      return true;
    }
  }

  if (mipsolver.options_mip_->mip_max_nodes != kHighsIInf &&
      mipsolver.mipdata_->num_nodes + nnodes >=
          mipsolver.options_mip_->mip_max_nodes) {
    return true;
  }

  if (mipsolver.options_mip_->mip_max_leaves != kHighsIInf &&
      mipsolver.mipdata_->num_leaves + nleaves >=
          mipsolver.options_mip_->mip_max_leaves) {
    return true;
  }

  if (mipsolver.options_mip_->time_limit < kHighsInf &&
      mipsolver.timer_.read() >= mipsolver.options_mip_->time_limit) {
    return true;
  }

  return false;
}

HighsSymmetries& HighsSearch::getSymmetries() const {
  return mipsolver.mipdata_->symmetries;
}

bool HighsSearch::addIncumbent(const std::vector<double>& sol, double solobj,
                               const int solution_source,
                               const bool print_display_line) {
  if (mipsolver.mipdata_->parallelLockActive()) {
    return mipworker.addIncumbent(sol, solobj, solution_source);
  } else {
    return mipsolver.mipdata_->addIncumbent(sol, solobj, solution_source,
                                            print_display_line);
  }
}

#ifdef HIGHS_RUST
namespace highs_rs {

static_assert(sizeof(std::pair<HighsInt, double>) == 16,
              "std::pair<HighsInt, double> is FracInt");

// a std::shared_ptr (HighsBasis or StabilizerOrbits) boxed for Rust
struct SharedBox {
  std::shared_ptr<const void> p;
};

template <typename T>
void* boxShared(std::shared_ptr<const T> p) {
  if (!p) return nullptr;
  return new SharedBox{std::move(p)};
}

template <typename T>
const T* unboxed(void* b) {
  return static_cast<const T*>(static_cast<SharedBox*>(b)->p.get());
}

template <typename T>
std::shared_ptr<const T> takeShared(void* b) {
  if (!b) return nullptr;
  SharedBox* box = static_cast<SharedBox*>(b);
  std::shared_ptr<const T> p = std::static_pointer_cast<const T>(box->p);
  delete box;
  return p;
}

// Mirror of CModel
struct SearchModel {
  HighsInt num_col;
  const double* col_cost;
  const uint8_t* integrality;
  const double* root_lp_sol;
  HighsInt num_root_lp_sol;
  const HighsInt* integral_cols;
  HighsInt num_integral_cols;
  int source_heuristic;
  int source_branching;
  int source_evaluate_node;
};

// Mirror of CSearchFns
struct SearchFns {
  void* (*shared_clone)(void*);
  void (*shared_free)(void*);
  void (*change_bound)(void*, HighsDomainChange);
  void (*propagate)(void*);
  bool (*infeasible)(void*);
  HighsDomainChange (*backtrack)(void*);
  void (*backtrack_to_global)(void*);
  const HighsDomainChange* (*stack)(void*, HighsInt*);
  HighsInt (*num_changed_cols)(void*);
  void (*clear_changed_cols)(void*, HighsInt);
  void (*conflict_analysis)(void*);
  void (*bounds)(void*, const double**, const double**);
  double (*objective_lower_bound)(void*);
  HighsInt (*col_pos)(void*, HighsInt, bool);
  bool (*is_binary)(void*, HighsInt, bool);
  void (*set_stack)(void*, const HighsDomainChange*, HighsInt,
                    const HighsInt*, HighsInt);
  const HighsInt* (*branching_positions)(void*, HighsInt*);
  double (*node_to_queue)(void*, void*, double, double, HighsInt);
  void (*lp_flush_domain)(void*);
  void (*lp_set_objective_limit)(void*, double);
  int (*lp_resolve)(void*);
  int64_t (*lp_num_iterations)(void*);
  int (*lp_status)(void*);
  bool (*lp_query)(void*, int, int);
  double (*lp_objective)(void*);
  const double* (*lp_solution)(void*, int, HighsInt*);
  const std::pair<HighsInt, double>* (*lp_frac_ints)(void*, HighsInt*);
  void* (*lp_store_basis)(void*, bool);
  void (*lp_set_stored_basis)(void*, void*);
  void (*lp_recover_basis)(void*);
  HighsInt (*basis_rows)(void*);
  HighsInt (*lp_rows)(void*, int);
  void (*lp_perform_aging)(void*);
  double (*lp_best_estimate)(void*);
  void (*lp_degenerate_duals)(void*, double);
  double (*lp_degeneracy)(void*);
  void* (*playground_new)(void*);
  int (*playground_solve)(void*, void*);
  void (*playground_free)(void*);
  void (*lp_fallback)(void*, int);
  HighsInt (*num_perms)(void*);
  void* (*global_orbits)(void*);
  HighsInt (*column_position)(void*, HighsInt);
  void* (*compute_stabilizer_orbits)(void*);
  bool (*orbits_query)(void*, HighsInt);
  void (*orbital_fixing)(void*, void*);
  void (*propagate_orbitopes)(void*);
  double (*mip_value)(void*, int);
  int64_t (*mip_stat)(void*, int);
  bool (*check_limits)(void*, int64_t);
  void (*add_incumbent)(void*, const double*, HighsInt, double, int);
  void (*add_bound_exceeding_conflict)(void*);
  void (*add_infeasible_conflict)(void*);
  void (*propagate_redcost)(void*);
  void (*log_frac_error)(void*, bool, double, double, double, double, double);
  void (*model)(void*, SearchModel*);
};


extern "C" {
Search* highs_rs_search_new(const SearchFns* fns, void* ctx,
                            const SearchModel* model, bool submip);
void highs_rs_search_free(Search* s);
SearchStats* highs_rs_search_stats(Search* s);
void highs_rs_search_op(Search* s, Pseudocost* ps, const NodeQueue* nq,
                        int which, HighsInt i, double x, double y, int64_t n,
                        void* q);
int highs_rs_search_run(Search* s, Pseudocost* ps, const NodeQueue* nq,
                        int which, HighsInt i, int64_t n, void* q);
double highs_rs_search_value(const Search* s, int which);
HighsInt highs_rs_search_select(Search* s, Pseudocost* ps,
                                const NodeQueue* nq, int64_t maxSbIters,
                                double* downLb, double* upLb);
void highs_rs_search_install(Search* s, Pseudocost* ps, const NodeQueue* nq,
                             const HighsDomainChange* domchgs,
                             HighsInt ndomchgs, const HighsInt* branchings,
                             HighsInt nbranchings, double lower_bound,
                             double estimate, HighsInt depth);
}

// The C++ side of the Rust search
struct SearchAccess {
  static HighsSearch& s(void* p) { return *static_cast<HighsSearch*>(p); }

  static void* sharedClone(void* b) {
    return new SharedBox{static_cast<SharedBox*>(b)->p};
  }
  static void sharedFree(void* b) { delete static_cast<SharedBox*>(b); }

  static void changeBound(void* p, HighsDomainChange d) {
    s(p).localdom.changeBound(d);
  }
  static void propagate(void* p) { s(p).localdom.propagate(); }
  static bool infeasible(void* p) { return s(p).localdom.infeasible(); }
  static HighsDomainChange backtrack(void* p) {
    return s(p).localdom.backtrack();
  }
  static void backtrackToGlobal(void* p) { s(p).localdom.backtrackToGlobal(); }
  static const HighsDomainChange* stack(void* p, HighsInt* n) {
    const auto& st = s(p).localdom.getDomainChangeStack();
    *n = st.size();
    return st.data();
  }
  static HighsInt numChangedCols(void* p) {
    return s(p).localdom.getChangedCols().size();
  }
  static void clearChangedCols(void* p, HighsInt start) {
    if (start < 0)
      s(p).localdom.clearChangedCols();
    else
      s(p).localdom.clearChangedCols(start);
  }
  static void conflictAnalysis(void* p) {
    HighsSearch& x = s(p);
    x.localdom.conflictAnalysis(x.getConflictPool(),
                                x.mipworker.getGlobalDomain(), x.pseudocost);
  }
  static void bounds(void* p, const double** lower, const double** upper) {
    *lower = s(p).localdom.col_lower_.data();
    *upper = s(p).localdom.col_upper_.data();
  }
  static double objectiveLowerBound(void* p) {
    return s(p).localdom.getObjectiveLowerBound();
  }
  static HighsInt colPos(void* p, HighsInt col, bool upper) {
    HighsDomain& d = s(p).localdom;
    HighsInt pos;
    if (upper)
      d.getColUpperPos(col, d.getNumDomainChanges(), pos);
    else
      d.getColLowerPos(col, d.getNumDomainChanges(), pos);
    return pos;
  }
  static bool isBinary(void* p, HighsInt col, bool global) {
    return global ? s(p).getDomain().isBinary(col)
                  : s(p).localdom.isGlobalBinary(col);
  }
  static void setStack(void* p, const HighsDomainChange* d, HighsInt n,
                       const HighsInt* b, HighsInt nb) {
    s(p).localdom.setDomainChangeStack(std::vector<HighsDomainChange>(d, d + n),
                                       std::vector<HighsInt>(b, b + nb));
  }
  static const HighsInt* branchingPositions(void* p, HighsInt* n) {
    const auto& b = s(p).localdom.getBranchingPositions();
    *n = b.size();
    return b.data();
  }
  static double nodeToQueue(void* p, void* q, double lb, double estimate,
                            HighsInt depth) {
    std::vector<HighsInt> branchPositions;
    auto domchgStack =
        s(p).localdom.getReducedDomainChangeStack(branchPositions);
    return static_cast<HighsNodeQueue*>(q)->emplaceNode(
        std::move(domchgStack), std::move(branchPositions), lb, estimate,
        depth);
  }

  static void lpFlushDomain(void* p) { s(p).lp->flushDomain(s(p).localdom); }
  static void lpSetObjectiveLimit(void* p, double x) {
    s(p).lp->setObjectiveLimit(x);
  }
  static int lpResolve(void* p) {
    return int(s(p).lp->resolveLp(&s(p).localdom));
  }
  static int64_t lpNumIterations(void* p) {
    return s(p).lp->getNumLpIterations();
  }
  static int lpStatus(void* p) { return int(s(p).lp->getStatus()); }
  static bool lpQuery(void* p, int which, int status) {
    HighsLpRelaxation& lp = *s(p).lp;
    auto st = HighsLpRelaxation::Status(status);
    switch (which) {
      case 0:
        return lp.scaledOptimal(st);
      case 1:
        return lp.unscaledPrimalFeasible(st);
      case 2:
        return lp.unscaledDualFeasible(st);
      case 3:
        return st == HighsLpRelaxation::Status::kInfeasible;
      case 4:
        return st == HighsLpRelaxation::Status::kOptimal;
      default:
        return lp.getLpSolver().getModelStatus() ==
               HighsModelStatus::kObjectiveBound;
    }
  }
  static double lpObjective(void* p) { return s(p).lp->getObjective(); }
  static const double* lpSolution(void* p, int which, HighsInt* n) {
    const std::vector<double>& v =
        which == 0   ? s(p).lp->getSolution().col_value
        : which == 1 ? s(p).lp->getSolution().col_dual
                     : s(p).lp->getLpSolver().getSolution().col_value;
    *n = v.size();
    return v.data();
  }
  static const std::pair<HighsInt, double>* lpFracInts(void* p, HighsInt* n) {
    const auto& f = s(p).lp->getFractionalIntegers();
    *n = f.size();
    return f.data();
  }
  static void* lpStoreBasis(void* p, bool get) {
    if (!get) {
      s(p).lp->storeBasis();
      return nullptr;
    }
    return boxShared(s(p).lp->getStoredBasis());
  }
  static void lpSetStoredBasis(void* p, void* b) {
    s(p).lp->setStoredBasis(takeShared<HighsBasis>(b));
  }
  static void lpRecoverBasis(void* p) { s(p).lp->recoverBasis(); }
  static HighsInt basisRows(void* b) {
    return unboxed<HighsBasis>(b)->row_status.size();
  }
  static HighsInt lpRows(void* p, int which) {
    return which == 0 ? s(p).lp->numRows() : s(p).lp->getLp().num_row_;
  }
  static void lpPerformAging(void* p) { s(p).lp->performAging(); }
  static double lpBestEstimate(void* p) {
    return s(p).lp->computeBestEstimate(s(p).pseudocost);
  }
  static void lpDegenerateDuals(void* p, double threshold) {
    HighsSearch& x = s(p);
    x.lp->computeBasicDegenerateDuals(threshold, x.localdom, x.getDomain(),
                                      x.getConflictPool(),
                                      x.mipworker.getPseudocost(), true);
  }
  static double lpDegeneracy(void* p) {
    return s(p).lp->computeLPDegneracy(s(p).localdom);
  }
  static void* playgroundNew(void* p) {
    return new HighsLpRelaxation::Playground(s(p).lp->playground());
  }
  static int playgroundSolve(void* p, void* pg) {
    return int(static_cast<HighsLpRelaxation::Playground*>(pg)->solveLp(
        s(p).localdom));
  }
  static void playgroundFree(void* pg) {
    delete static_cast<HighsLpRelaxation::Playground*>(pg);
  }
  static void lpFallback(void* p, int step) {
    HighsSearch& x = s(p);
    switch (step) {
      case 0: {
        x.lp->setIterationLimit();
        // create a fresh LP only with model rows since all integer columns
        // are fixed, the cutting planes are not required and the LP could
        // not be solved so we want to make it as easy as possible
        x.fallbackLp_.reset(new HighsLpRelaxation(x.mipsolver));
        HighsLpRelaxation& lpCopy = *x.fallbackLp_;
        lpCopy.setProfiling(x.mipsolver.profiling_);
        lpCopy.loadModel();
        lpCopy.getLpSolver().changeColsBounds(0, x.mipsolver.numCol() - 1,
                                              x.localdom.col_lower_.data(),
                                              x.localdom.col_upper_.data());
        // temporarily use the fresh LP for the search
        x.fallbackSwapped_ = x.lp;
        x.lp = &lpCopy;
        // reevaluate the node with LP presolve enabled
        x.lp->getLpSolver().setOptionValue("presolve", kHighsOnString);
        break;
      }
      case 1:
        // LP still not solved, reevaluate with primal simplex
        x.lp->getLpSolver().clearSolver();
        x.lp->getLpSolver().setOptionValue("simplex_strategy",
                                           kSimplexStrategyPrimal);
        break;
      case 2:
        x.lp->getLpSolver().setOptionValue("simplex_strategy",
                                           kSimplexStrategyDual);
        break;
      case 3:
        // LP still not solved, reevaluate with IPM instead of simplex
        x.lp->getLpSolver().clearSolver();
        x.lp->getLpSolver().setOptionValue("solver", "ipm");
        break;
      case 4:
        highsLogUser(x.mipsolver.options_mip_->log_options,
                     HighsLogType::kWarning,
                     "Failed to solve node with all integer columns "
                     "fixed. Declaring node infeasible.\n");
        break;
      default:
        // restore old lp relaxation
        x.lp = x.fallbackSwapped_;
        x.fallbackLp_.reset();
    }
  }
  static HighsInt numPerms(void* p) { return s(p).getSymmetries().numPerms; }
  static void* globalOrbits(void* p) {
    return boxShared(s(p).mipsolver.mipdata_->globalOrbits);
  }
  static HighsInt columnPosition(void* p, HighsInt col) {
    return s(p).getSymmetries().getColumnPosition(col);
  }
  static void* computeStabilizerOrbits(void* p) {
    return boxShared(s(p).getSymmetries().computeStabilizerOrbits(
        s(p).localdom, s(p).stabilizerOrbitWorkspace));
  }
  static bool orbitsQuery(void* b, HighsInt col) {
    const StabilizerOrbits* o = unboxed<StabilizerOrbits>(b);
    return col < 0 ? o->orbitCols.empty() : o->isStabilized(col);
  }
  static void orbitalFixing(void* p, void* b) {
    unboxed<StabilizerOrbits>(b)->orbitalFixing(s(p).localdom);
  }
  static void propagateOrbitopes(void* p) {
    s(p).getSymmetries().propagateOrbitopes(s(p).localdom);
  }
  static double mipValue(void* p, int which) {
    switch (which) {
      case 0:
        return s(p).getFeasTol();
      case 1:
        return s(p).getEpsilon();
      case 2:
        return s(p).getUpperLimit();
      default:
        return s(p).getOptimalityLimit();
    }
  }
  static int64_t mipStat(void* p, int which) {
    const HighsMipSolverData& d = *s(p).mipsolver.mipdata_;
    switch (which) {
      case 0:
        return d.heuristic_lp_iterations;
      case 1:
        return d.total_lp_iterations;
      case 2:
        return d.sb_lp_iterations;
      default:
        return d.lns_tree_next >= 0;
    }
  }
  static bool checkLimits(void* p, int64_t offset) {
    return s(p).checkLimits(offset);
  }
  static void addIncumbent(void* p, const double* sol, HighsInt n, double obj,
                           int source) {
    s(p).addIncumbent(std::vector<double>(sol, sol + n), obj, source);
  }
  static void addBoundExceedingConflict(void* p) {
    s(p).addBoundExceedingConflict();
  }
  static void addInfeasibleConflict(void* p) { s(p).addInfeasibleConflict(); }
  static void propagateRedcost(void* p) {
    HighsSearch& x = s(p);
    HighsRedcostFixing::propagateRedCost(
        x.mipsolver, x.localdom, x.mipworker.getGlobalDomain(), *x.lp,
        x.getConflictPool(), x.mipworker.getPseudocost(), x.getUpperLimit());
  }
  static void logFracError(void* p, bool upper, double fracval, double bound,
                           double colbound, double feastol, double residual) {
    if (!upper)
      highsLogUser(s(p).mipsolver.options_mip_->log_options,
                   HighsLogType::kError,
                   "HighsSearch::selectBranchingCandidate Error fracval = %g "
                   "<= %g = %g + %g = "
                   "localdom.col_lower_[col] + getFeasTol(): "
                   "Residual %g\n",
                   fracval, bound, colbound, feastol, residual);
    else
      highsLogUser(s(p).mipsolver.options_mip_->log_options,
                   HighsLogType::kError,
                   "HighsSearch::selectBranchingCandidate Error fracval = %g "
                   ">= %g = %g - %g = "
                   "localdom.col_upper_[col] - getFeasTol(): "
                   "Residual %g\n",
                   fracval, bound, colbound, feastol, residual);
  }

  static void modelData(void* p, SearchModel* m) {
    fillModel(s(p).mipsolver, m);
  }

  static void fillModel(const HighsMipSolver& mipsolver, SearchModel* m) {
    const HighsLp& model = *mipsolver.model_;
    const HighsMipSolverData& d = *mipsolver.mipdata_;
    m->num_col = mipsolver.numCol();
    m->col_cost = model.col_cost_.data();
    m->integrality =
        reinterpret_cast<const uint8_t*>(model.integrality_.data());
    m->root_lp_sol = d.rootlpsol.data();
    m->num_root_lp_sol = d.rootlpsol.size();
    m->integral_cols = d.integral_cols.data();
    m->num_integral_cols = d.integral_cols.size();
    m->source_heuristic = kSolutionSourceHeuristic;
    m->source_branching = kSolutionSourceBranching;
    m->source_evaluate_node = kSolutionSourceEvaluateNode;
  }

  static const SearchFns fns;
};

const SearchFns SearchAccess::fns = {
    sharedClone,
    sharedFree,
    changeBound,
    propagate,
    infeasible,
    backtrack,
    backtrackToGlobal,
    stack,
    numChangedCols,
    clearChangedCols,
    conflictAnalysis,
    bounds,
    objectiveLowerBound,
    colPos,
    isBinary,
    setStack,
    branchingPositions,
    nodeToQueue,
    lpFlushDomain,
    lpSetObjectiveLimit,
    lpResolve,
    lpNumIterations,
    lpStatus,
    lpQuery,
    lpObjective,
    lpSolution,
    lpFracInts,
    lpStoreBasis,
    lpSetStoredBasis,
    lpRecoverBasis,
    basisRows,
    lpRows,
    lpPerformAging,
    lpBestEstimate,
    lpDegenerateDuals,
    lpDegeneracy,
    playgroundNew,
    playgroundSolve,
    playgroundFree,
    lpFallback,
    numPerms,
    globalOrbits,
    columnPosition,
    computeStabilizerOrbits,
    orbitsQuery,
    orbitalFixing,
    propagateOrbitopes,
    mipValue,
    mipStat,
    checkLimits,
    addIncumbent,
    addBoundExceedingConflict,
    addInfeasibleConflict,
    propagateRedcost,
    logFracError,
    modelData,
};

static Search* newSearch(HighsSearch* s, const HighsMipSolver& mipsolver) {
  SearchModel m;
  SearchAccess::fillModel(mipsolver, &m);
  return highs_rs_search_new(&SearchAccess::fns, s, &m, mipsolver.submip);
}
}  // namespace highs_rs

HighsSearch::HighsSearch(HighsMipWorker& mipworker, HighsPseudocost& pseudocost)
    : rs_(highs_rs::newSearch(this, mipworker.getMipSolver())),
      st_(highs_rs::highs_rs_search_stats(rs_)),
      mipworker(mipworker),
      mipsolver(mipworker.getMipSolver()),
      lp(nullptr),
      localdom(mipworker.getGlobalDomain()),
      pseudocost(pseudocost),
      nnodes(st_->nnodes),
      nleaves(st_->nleaves),
      lpiterations(st_->lpiterations),
      heurlpiterations(st_->heurlpiterations),
      sblpiterations(st_->sblpiterations),
      upper_limit(st_->upper_limit),
      treeweight(st_->treeweight),
      depthoffset(st_->depthoffset),
      inbranching(st_->inbranching),
      inheuristic(st_->inheuristic),
      countTreeWeight(st_->countTreeWeight) {
  // the infeasibility flag is overwritten and lost when setDomainChangeStack is
  // called. therefore, assert that localdom is not infeasible here.
  assert(!this->localdom.infeasible());
  this->localdom.setDomainChangeStack(std::vector<HighsDomainChange>());
}

HighsSearch::~HighsSearch() { highs_rs::highs_rs_search_free(rs_); }

void HighsSearch::op(int which, HighsInt i, double x, double y, int64_t n,
                     HighsNodeQueue* q) {
  highs_rs::highs_rs_search_op(rs_, pseudocost.rust(),
                               mipsolver.mipdata_->nodequeue.rust(), which, i,
                               x, y, n, q);
}

int HighsSearch::run(int which, HighsInt i, int64_t n, HighsNodeQueue* q) {
  return highs_rs::highs_rs_search_run(rs_, pseudocost.rust(),
                                       mipsolver.mipdata_->nodequeue.rust(),
                                       which, i, n, q);
}

double HighsSearch::getCurrentEstimate() const {
  return highs_rs::highs_rs_search_value(rs_, 0);
}

double HighsSearch::getCurrentLowerBound() const {
  return highs_rs::highs_rs_search_value(rs_, 1);
}

HighsInt HighsSearch::selectBranchingCandidate(int64_t maxSbIters,
                                               double& downNodeLb,
                                               double& upNodeLb) {
  return highs_rs::highs_rs_search_select(
      rs_, pseudocost.rust(), mipsolver.mipdata_->nodequeue.rust(),
      maxSbIters, &downNodeLb, &upNodeLb);
}

void HighsSearch::installNode(HighsNodeQueue::OpenNode&& node) {
  highs_rs::highs_rs_search_install(
      rs_, pseudocost.rust(), mipsolver.mipdata_->nodequeue.rust(),
      node.domchgstack.data(), node.domchgstack.size(), node.branchings.data(),
      node.branchings.size(), node.lower_bound, node.estimate, node.depth);
}
#endif  // HIGHS_RUST
