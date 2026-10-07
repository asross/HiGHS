/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#include "mip/HighsSeparation.h"

#include <algorithm>
#include <cassert>
#include <queue>

#include "mip/HighsCliqueTable.h"
#include "mip/HighsDomain.h"
#include "mip/HighsImplications.h"
#include "mip/HighsLpAggregator.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsModkSeparator.h"
#include "mip/HighsPathSeparator.h"
#include "mip/HighsTableauSeparator.h"
#include "mip/HighsTransformedLp.h"

HighsSeparation::HighsSeparation(HighsMipWorker& mipworker)
    : mipworker_(mipworker) {
  /*
  if (mipworker.mipsolver_.profiling_->mip_) {
    implBoundClock =
        mipworker.mipsolver_.profiling_->getSepaClockIndex(kImplboundSepaString);
    cliqueClock =
        mipworker.mipsolver_.profiling_->getSepaClockIndex(kCliqueSepaString);
  }
  */
  implBoundClock = 990;
  cliqueClock = 991;
  const HighsMipSolver& mipsolver = mipworker.getMipSolver();
  separators.emplace_back(new HighsTableauSeparator(mipsolver));
  separators.emplace_back(new HighsPathSeparator(mipsolver));
  separators.emplace_back(new HighsModkSeparator(mipsolver));
}

#ifdef HIGHS_RUST
// The loop runs in Rust (rust/src/mip/separation.rs); these are its steps
namespace highs_rs {
struct SepaFns {
  int64_t (*op)(void*, int, int64_t);
  void* propdomain;
  double rootlpsolobj;
  const double* optimality_limit;
  double feastol;
};
extern "C" {
HighsInt highs_rs_separation_round(const SepaFns* fns, void* ctx, LpRelax* lp,
                                   int* status);
void highs_rs_separation_separate(const SepaFns* fns, void* ctx, LpRelax* lp);
}
}  // namespace highs_rs

struct HighsSeparationAccess {
  HighsSeparation& sepa;
  HighsDomain& propdomain;

  static int64_t op(void* p, int which, int64_t arg) {
    HighsSeparationAccess& a = *static_cast<HighsSeparationAccess*>(p);
    HighsSeparation& s = a.sepa;
    HighsDomain& propdomain = a.propdomain;
    HighsLpRelaxation* lp = s.lp;
    HighsMipWorker& w = s.mipworker_;
    HighsMipSolverData& mipdata = *lp->getMipSolver().mipdata_;
    const bool master = &propdomain == &mipdata.getDomain();
    switch (which) {
      case 0:
        return propdomain.infeasible() || w.getGlobalDomain().infeasible();
      case 1:
        propdomain.propagate();
        return propdomain.infeasible();
      case 2:
        // only modify cliquetable for master worker.
        if (master) mipdata.cliquetable.cleanupFixed(mipdata.getDomain());
        return w.getGlobalDomain().infeasible();
      case 3:
        propdomain.clearChangedCols();
        return 0;
      case 4:
        return propdomain.getChangedCols().size();
      case 5:
        lp->setObjectiveLimit(w.upper_limit);
        return 0;
      case 6:
      case 7:
        if (master) {
          mipdata.redcostfixing.addRootRedcost(mipdata.mipsolver,
                                               lp->getSolution().col_dual,
                                               lp->getObjective());
          if ((which == 6 ? w.upper_limit : mipdata.upper_limit) != kHighsInf)
            mipdata.redcostfixing.propagateRootRedcost(mipdata.mipsolver);
        }
        return 0;
      case 8:
        if (!mipdata.parallelLockActive())
          lp->getMipSolver().profiling_->start(s.implBoundClock);
        mipdata.implications.separateImpliedBounds(
            *lp, lp->getSolution().col_value, w.getCutPool(), mipdata.feastol,
            w.getGlobalDomain(), mipdata.parallelLockActive());
        if (!mipdata.parallelLockActive())
          lp->getMipSolver().profiling_->stop(s.implBoundClock);
        return 0;
      case 9:
        if (!mipdata.parallelLockActive())
          lp->getMipSolver().profiling_->start(s.cliqueClock);
        mipdata.cliquetable.separateCliques(
            lp->getMipSolver(), lp->getLpSolver().getSolution().col_value,
            w.getCutPool(), mipdata.feastol,
            mipdata.parallelLockActive() ? w.randgen
                                         : mipdata.cliquetable.getRandgen(),
            mipdata.parallelLockActive()
                ? w.getNumNeighbourhoodQueries()
                : mipdata.cliquetable.getNumNeighbourhoodQueries());
        if (!mipdata.parallelLockActive())
          lp->getMipSolver().profiling_->stop(s.cliqueClock);
        return 0;
      case 10:
        if (&propdomain != &w.getGlobalDomain())
          lp->computeBasicDegenerateDuals(mipdata.feastol, propdomain,
                                          w.getGlobalDomain(),
                                          w.getConflictPool(),
                                          w.getPseudocost(), true);
        return 0;
      case 11: {
        HighsTransformedLp transLp(*lp, mipdata.implications,
                                   w.getGlobalDomain());
        if (w.getGlobalDomain().infeasible()) return 1;
        HighsLpAggregator lpAggregator(*lp);
        for (const std::unique_ptr<HighsSeparator>& separator : s.separators) {
          separator->run(*lp, lpAggregator, transLp, w.getCutPool());
          if (w.getGlobalDomain().infeasible()) return 1;
        }
        return 0;
      }
      case 12: {
        const std::vector<double>& sol = lp->getLpSolver().getSolution().col_value;
        w.getCutPool().separate(sol, propdomain, s.cutset, mipdata.feastol,
                                mipdata.cutpools);
        // Also separate the global cut pool
        if (&w.getCutPool() != &mipdata.getCutPool())
          mipdata.getCutPool().separate(sol, propdomain, s.cutset,
                                        mipdata.feastol, mipdata.cutpools,
                                        true);
        return s.cutset.numCuts();
      }
      case 13:
        lp->addCuts(s.cutset);
        return 0;
      case 14:
        if (mipdata.parallelLockActive()) {
          w.getSepaLpIterations() += arg;
        } else {
          mipdata.sepa_lp_iterations += arg;
          mipdata.total_lp_iterations += arg;
        }
        return 0;
      default:
        w.getCutPool().performAging();
        return 0;
    }
  }

  highs_rs::SepaFns fns() {
    const HighsMipSolverData& mipdata = *sepa.lp->getMipSolver().mipdata_;
    return highs_rs::SepaFns{op, &propdomain, mipdata.rootlpsolobj,
                             &sepa.mipworker_.optimality_limit,
                             mipdata.feastol};
  }
};

HighsInt HighsSeparation::separationRound(HighsDomain& propdomain,
                                          HighsLpRelaxation::Status& status) {
  HighsSeparationAccess a{*this, propdomain};
  highs_rs::SepaFns f = a.fns();
  int st = int(status);
  HighsInt ncuts =
      highs_rs::highs_rs_separation_round(&f, &a, lp->rust(), &st);
  status = HighsLpRelaxation::Status(st);
  return ncuts;
}

void HighsSeparation::separate(HighsDomain& propdomain) {
  HighsSeparationAccess a{*this, propdomain};
  highs_rs::SepaFns f = a.fns();
  highs_rs::highs_rs_separation_separate(&f, &a, lp->rust());
}
#else
HighsInt HighsSeparation::separationRound(HighsDomain& propdomain,
                                          HighsLpRelaxation::Status& status) {
  const HighsSolution& sol = lp->getLpSolver().getSolution();

  HighsMipSolverData& mipdata = *lp->getMipSolver().mipdata_;

  auto propagateAndResolve = [&]() {
    if (propdomain.infeasible() || mipworker_.getGlobalDomain().infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      propdomain.clearChangedCols();
      return -1;
    }

    propdomain.propagate();
    if (propdomain.infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      propdomain.clearChangedCols();
      return -1;
    }

    // only modify cliquetable for master worker.
    if (&propdomain == &mipdata.getDomain())
      mipdata.cliquetable.cleanupFixed(mipdata.getDomain());

    if (mipworker_.getGlobalDomain().infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      propdomain.clearChangedCols();
      return -1;
    }

    int numBoundChgs = (int)propdomain.getChangedCols().size();

    while (!propdomain.getChangedCols().empty()) {
      lp->setObjectiveLimit(mipworker_.upper_limit);
      status = lp->resolveLp(&propdomain);
      if (!lp->scaledOptimal(status)) return -1;

      if (&propdomain == &mipdata.getDomain() &&
          lp->unscaledDualFeasible(status)) {
        mipdata.redcostfixing.addRootRedcost(
            mipdata.mipsolver, lp->getSolution().col_dual, lp->getObjective());
        if (mipworker_.upper_limit != kHighsInf)
          mipdata.redcostfixing.propagateRootRedcost(mipdata.mipsolver);
      }
    }

    return numBoundChgs;
  };

  if (!mipdata.parallelLockActive())
    lp->getMipSolver().profiling_->start(implBoundClock);
  mipdata.implications.separateImpliedBounds(
      *lp, lp->getSolution().col_value, mipworker_.getCutPool(),
      mipdata.feastol, mipworker_.getGlobalDomain(),
      mipdata.parallelLockActive());
  if (!mipdata.parallelLockActive())
    lp->getMipSolver().profiling_->stop(implBoundClock);

  HighsInt ncuts = 0;
  HighsInt numboundchgs = propagateAndResolve();
  if (numboundchgs == -1)
    return 0;
  else
    ncuts += numboundchgs;

  if (!mipdata.parallelLockActive())
    lp->getMipSolver().profiling_->start(cliqueClock);
  mipdata.cliquetable.separateCliques(
      lp->getMipSolver(), sol.col_value, mipworker_.getCutPool(),
      mipdata.feastol,
      mipdata.parallelLockActive() ? mipworker_.randgen
                                   : mipdata.cliquetable.getRandgen(),
      mipdata.parallelLockActive()
          ? mipworker_.getNumNeighbourhoodQueries()
          : mipdata.cliquetable.getNumNeighbourhoodQueries());
  if (!mipdata.parallelLockActive())
    lp->getMipSolver().profiling_->stop(cliqueClock);

  numboundchgs = propagateAndResolve();
  if (numboundchgs == -1)
    return 0;
  else
    ncuts += numboundchgs;

  if (&propdomain != &mipworker_.getGlobalDomain())
    lp->computeBasicDegenerateDuals(
        mipdata.feastol, propdomain, mipworker_.getGlobalDomain(),
        mipworker_.getConflictPool(), mipworker_.getPseudocost(), true);

  HighsTransformedLp transLp(*lp, mipdata.implications,
                             mipworker_.getGlobalDomain());
  if (mipworker_.getGlobalDomain().infeasible()) {
    status = HighsLpRelaxation::Status::kInfeasible;
    return 0;
  }
  HighsLpAggregator lpAggregator(*lp);

  for (const std::unique_ptr<HighsSeparator>& separator : separators) {
    separator->run(*lp, lpAggregator, transLp, mipworker_.getCutPool());
    if (mipworker_.getGlobalDomain().infeasible()) {
      status = HighsLpRelaxation::Status::kInfeasible;
      return 0;
    }
  }

  numboundchgs = propagateAndResolve();
  if (numboundchgs == -1)
    return 0;
  else
    ncuts += numboundchgs;

  mipworker_.getCutPool().separate(sol.col_value, propdomain, cutset,
                                   mipdata.feastol, mipdata.cutpools);
  // Also separate the global cut pool
  if (&mipworker_.getCutPool() != &mipdata.getCutPool()) {
    mipdata.getCutPool().separate(sol.col_value, propdomain, cutset,
                                  mipdata.feastol, mipdata.cutpools, true);
  }

  if (cutset.numCuts() > 0) {
    ncuts += cutset.numCuts();
    lp->addCuts(cutset);
    status = lp->resolveLp(&propdomain);
    lp->performAging(true);

    // only for the master domain.
    if (&propdomain == &mipdata.getDomain() &&
        lp->unscaledDualFeasible(status)) {
      mipdata.redcostfixing.addRootRedcost(
          mipdata.mipsolver, lp->getSolution().col_dual, lp->getObjective());
      if (mipdata.upper_limit != kHighsInf)
        mipdata.redcostfixing.propagateRootRedcost(mipdata.mipsolver);
    }
  }

  return ncuts;
}

void HighsSeparation::separate(HighsDomain& propdomain) {
  HighsLpRelaxation::Status status = lp->getStatus();
  const HighsMipSolver& mipsolver = lp->getMipSolver();

  if (lp->scaledOptimal(status) && !lp->getFractionalIntegers().empty()) {
    // double firstobj = lp->getObjective();
    double firstobj = mipsolver.mipdata_->rootlpsolobj;

    while (lp->getObjective() < mipworker_.optimality_limit) {
      double lastobj = lp->getObjective();

      int64_t nlpiters = -lp->getNumLpIterations();
      HighsInt ncuts = separationRound(propdomain, status);
      nlpiters += lp->getNumLpIterations();

      if (mipsolver.mipdata_->parallelLockActive()) {
        mipworker_.getSepaLpIterations() += nlpiters;
      } else {
        mipsolver.mipdata_->sepa_lp_iterations += nlpiters;
        mipsolver.mipdata_->total_lp_iterations += nlpiters;
      }

      // printf("separated %" HIGHSINT_FORMAT " cuts\n", ncuts);

      // printf(
      //     "separation round %" HIGHSINT_FORMAT " at node %" HIGHSINT_FORMAT "
      //     added %" HIGHSINT_FORMAT " cuts objective changed " "from %g to %g,
      //     first obj is %g\n", nrounds, (HighsInt)nnodes, ncuts, lastobj,
      //     lp->getObjective(), firstobj);
      if (ncuts == 0 || !lp->scaledOptimal(status) ||
          lp->getFractionalIntegers().empty())
        break;

      // if the objective improved considerably we continue
      if ((lp->getObjective() - firstobj) <=
          std::max((lastobj - firstobj), mipsolver.mipdata_->feastol) * 1.01)
        break;
    }

    // printf("done separating\n");
  } else {
    // printf("no separation, just aging. status: %" HIGHSINT_FORMAT "\n",
    //        (HighsInt)status);
    lp->performAging(true);

    mipworker_.getCutPool().performAging();
  }
}
#endif  // HIGHS_RUST
