/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsGraphLns.cpp
 * @brief Graph-neighbourhood LNS primal heuristic of HighsPrimalHeuristics
 */
#include "HConfig.h"

#ifndef HIGHS_RUST
#include <algorithm>
#include <cmath>
#include <memory>

#include "../extern/pdqsort/pdqsort.h"
#include "io/HighsIO.h"
#include "mip/HighsDomainChange.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipSolverData.h"
#include "mip/HighsPrimalHeuristics.h"


// Graph-neighbourhood LNS
// -----------------------
// A primal heuristic for models whose LP relaxation has a tight bound but a
// misleading vertex (dispatch, unit commitment, lot sizing, scheduling),
// solved to a loose gap (it only runs for mip_rel_gap >= 1e-3), where a good
// incumbent is what finishes the solve. It works on a copy of the LP
// relaxation and a HighsDomain, with warm LP re-solves.
//
// Decision columns. An integer column whose every row has coefficient +-1
// on it, only integral columns, integral coefficients and integral finite
// sides is confined to an interval with integer endpoints once the other
// integers are fixed, and shares no row with a continuous column, so every LP
// vertex has it integral (e.g. |x_t - x_{t-1}| indicators). Fixing such a
// column is wasted and, worse, fixes it before the columns that determine
// it, so the search does not branch on them; the tree still does if a
// fractional value ever survives.
//
// Quick search (after the first root LP): a dive from the LP point (fix the
// most integral decision columns toward it, a chunk per LP solve, up to the
// first rounding that propagation rules out; an infeasible chunk is undone
// and retried at a quarter of its size), then neighbourhoods: a
// breadth-first search over the variable/constraint graph from a seed
// column collects ~400 decision columns (on time-indexed models a time
// window), the rest are fixed to the incumbent, and if the neighbourhood LP
// (warm from the last LP solved) can beat the incumbent it is dived. It ends
// by polishing the incumbent with a flip search. A concurrent LNS helper
// leaves it to its main solver.
//
// Deep search (after the root cuts, before the sub-MIP heuristics, if the
// quick search brought the incumbent within three times the target gap:
// otherwise the model does not suit graph LNS): the same, but
// neighbourhoods are smaller and searched by a depth-first branch
// and bound with a node limit, their size adapting to what that exhausts;
// among four moves it picks the one closing the most gap per LP iteration:
// - BFS over all rows, searched by branch and bound,
// - BFS over rows with at most four decision columns (chains, e.g. one unit
//   over time), searched by branch and bound,
// - flip search: with all decision columns fixed at the incumbent, flip
//   binaries in order of the gain promised by their reduced cost, and pair a
//   promising flip with an opposite flip in one of its rows (swaps); a move
//   is first checked by propagating its fixings, which rules out most,
// - BFS over all rows, dived.
// Seeds are columns with a promising reduced cost at the incumbent, columns
// where the incumbent disagrees with the LP solution, or random.
//
// The quick search stops after 5 neighbourhoods that do not close 5% of the
// gap, the deep search after 10 that do not close 1% of what separates the
// incumbent from the target gap; both also stop on an LP iteration budget
// (for the quick search on a large model, about one root LP's worth), or
// when the incumbent reaches the target gap. If the deep search pays,
// further rounds alternate with the tree search (HighsMipSolver::run). A
// flip search goes on from where the last one from the same incumbent
// stopped.
void HighsPrimalHeuristics::setupDecisionCols() {
  decisionColsSetUp = true;
  decisioncols.clear();
  const HighsLp& model = *mipsolver.model_;
  const HighsMipSolverData& mipdata = *mipsolver.mipdata_;
  const HighsSparseMatrix& A = model.a_matrix_;
  if (!A.isColwise() || mipdata.ARstart_.empty()) {
    decisioncols = intcols;
    return;
  }
  auto integral = [](double v) { return std::fabs(v - std::round(v)) <= 1e-9; };
  std::vector<uint8_t> implied(model.num_col_, 0);
  for (HighsInt col : intcols) {
    bool ok = true;
    for (HighsInt p = A.start_[col]; ok && p != A.start_[col + 1]; ++p) {
      if (std::fabs(std::fabs(A.value_[p]) - 1.0) > 1e-9) {
        ok = false;
        break;
      }
      HighsInt row = A.index_[p];
      if ((model.row_lower_[row] != -kHighsInf &&
           !integral(model.row_lower_[row])) ||
          (model.row_upper_[row] != kHighsInf &&
           !integral(model.row_upper_[row]))) {
        ok = false;
        break;
      }
      for (HighsInt q = mipdata.ARstart_[row]; q != mipdata.ARstart_[row + 1];
           ++q) {
        HighsInt k = mipdata.ARindex_[q];
        if (!integral(mipdata.ARvalue_[q]) || !mipsolver.isColIntegral(k) ||
            (k != col && implied[k])) {
          ok = false;
          break;
        }
      }
    }
    if (ok)
      implied[col] = 1;
    else
      decisioncols.push_back(col);
  }
}

void HighsPrimalHeuristics::graphLNS(HighsMipWorker& worker,
                                     const std::vector<double>& relaxationsol,
                                     bool deep, int64_t maxLpIters) {
  if (mipsolver.submip && !mipsolver.concurrent_lns_) return;
  if (worker.getGlobalDomain().infeasible()) return;
  if (!decisionColsSetUp) setupDecisionCols();
  if (decisioncols.empty() || relaxationsol.empty()) return;

  HighsMipSolverData& mipdata = *mipsolver.mipdata_;
  const HighsLp& model = *mipsolver.model_;
  const HighsSparseMatrix& A = model.a_matrix_;
  const HighsInt numCol = mipsolver.numCol();
  const double feastol = mipdata.feastol;
  const HighsDomain& globaldom = worker.getGlobalDomain();

  std::vector<uint8_t> isDecision(numCol, 0);
  for (HighsInt col : decisioncols) isDecision[col] = 1;

  // one LP relaxation copy for the whole heuristic; bounds live in a domain
  HighsLpRelaxation lp(worker.getLpRelaxation());
  lp.setMipWorker(worker);
  lp.setProfiling(mipsolver.profiling_);
  lp.setAdjustSymmetricBranchingCol(false);
  const int64_t lpItersStart = lp.getNumLpIterations();
  auto chargeIterations = [&]() {
    worker.getHeurLpIterations() += lp.getNumLpIterations() - lpItersStart;
  };
  auto usable = [&](HighsLpRelaxation::Status st) {
    return st == HighsLpRelaxation::Status::kOptimal ||
           st == HighsLpRelaxation::Status::kUnscaledPrimalFeasible;
  };
  auto solve = [&](HighsDomain& dom) {
    lp.setObjectiveLimit(worker.upper_limit);
    return lp.resolveLp(&dom);
  };
  // an LP solution with every integral column integral is a new incumbent
  auto tryIncumbent = [&](HighsLpRelaxation::Status st) {
    if (!usable(st) || !lp.getFractionalIntegers().empty()) return false;
    return addIncumbent(lp.getLpSolver().getSolution().col_value,
                        lp.getObjective(), kSolutionSourceGraphLns, worker);
  };
  // put `dom` back to `snap` and push the undone bounds back into the LP
  auto restore = [&](HighsDomain& dom, const HighsDomain& snap,
                     size_t stackPos) {
    const auto& stack = dom.getDomainChangeStack();
    std::vector<HighsInt> cols;
    for (size_t i = stackPos; i < stack.size(); ++i)
      cols.push_back(stack[i].column);
    dom = snap;
    for (HighsInt c : cols)
      lp.getLpSolver().changeColBounds(c, dom.col_lower_[c], dom.col_upper_[c]);
  };
  auto fixTo = [&](HighsDomain& dom, HighsInt col, double val) {
    if (dom.col_lower_[col] < val)
      dom.changeBound(HighsBoundType::kLower, col, val,
                      HighsDomain::Reason::unspecified());
    if (dom.col_upper_[col] > val)
      dom.changeBound(HighsBoundType::kUpper, col, val,
                      HighsDomain::Reason::unspecified());
    dom.propagate();
    return !dom.infeasible();
  };

  // fix a column as a branching decision and propagate; if that is
  // infeasible, undo it (backtracking to the decision) and return false
  auto fixTry = [&](HighsDomain& dom, HighsInt col, double val) {
    bool branched = false;
    if (dom.col_lower_[col] < val) {
      dom.changeBound(HighsBoundType::kLower, col, val,
                      HighsDomain::Reason::branching());
      branched = true;
    }
    if (dom.col_upper_[col] > val) {
      dom.changeBound(HighsBoundType::kUpper, col, val,
                      branched ? HighsDomain::Reason::unspecified()
                               : HighsDomain::Reason::branching());
      branched = true;
    }
    if (!branched) return !dom.infeasible();
    dom.propagate();
    if (!dom.infeasible()) return true;
    dom.backtrack();
    return false;
  };

  // Dive: fix the most integral unfixed candidates toward the current LP
  // point, `chunk` per LP re-solve, up to the first one whose rounding
  // propagation rules out (the chunk then shrinks to what was fixed). A
  // chunk whose LP is infeasible is undone, with the simplex iterate (basis,
  // factorization, edge weights) of the LP before it, and retried at a
  // quarter of the size; a single failed fixing is flipped the other way.
  // (Going on from the infeasible LP's iterate cost thousands of simplex
  // iterations per failed chunk on ramp models, and rounded that iterate
  // rather than the LP solution; setting the basis instead costs a
  // refactorization and, below 20000 rows, exact edge weights.)
  auto dive = [&](HighsDomain& dom, std::vector<HighsInt> candidates,
                  HighsInt chunk0) {
    HighsInt chunk = std::max(HighsInt{1}, chunk0);
    const HighsInt maxSolves = 5 * HighsInt(candidates.size()) + 100;
    HighsInt solves = 0;
    bool fallback = false;
    std::vector<std::pair<double, HighsInt>> order;
    // the last usable LP solution
    std::vector<double> sol = lp.getLpSolver().getSolution().col_value;
    while (solves < maxSolves) {
      if (worker.terminatorTerminated() || mipdata.checkLimits()) return false;
      order.clear();
      for (HighsInt col : candidates)
        if (dom.col_lower_[col] < dom.col_upper_[col])
          order.emplace_back(std::fabs(sol[col] - std::round(sol[col])), col);
      if (order.empty()) {
        if (lp.getFractionalIntegers().empty()) return false;
        if (fallback) return false;
        // non-decision integers left fractional: dive on them too
        fallback = true;
        candidates.clear();
        for (const auto& f : lp.getFractionalIntegers())
          candidates.push_back(f.first);
        continue;
      }
      pdqsort(order.begin(), order.end());
      HighsInt nfix = std::min<HighsInt>(chunk, order.size());
      HighsDomain snap = dom;
      const size_t pos = dom.getDomainChangeStack().size();
      // fix the chunk up to the first rounding that conflicts with the
      // fixings before it (which is undone)
      HighsInt nfixed = 0;
      for (HighsInt i = 0; i < nfix; ++i) {
        HighsInt col = order[i].second;
        double val =
            std::min(std::max(std::round(sol[col]), dom.col_lower_[col]),
                     dom.col_upper_[col]);
        if (!fixTry(dom, col, val)) break;
        ++nfixed;
      }
      bool feasible = nfixed > 0;
      if (nfixed < nfix) {
        // (the first one conflicting: try its other value, as below)
        nfix = std::max(HighsInt{1}, nfixed);
        chunk = nfix;
      }
      HighsLpRelaxation::Status st = HighsLpRelaxation::Status::kInfeasible;
      bool saved = false;
      if (feasible) {
        saved = lp.getLpSolver().putIterate() == HighsStatus::kOk;
        st = solve(dom);
        ++solves;
      }
      if (usable(st)) {
        if (tryIncumbent(st)) return true;
        sol = lp.getLpSolver().getSolution().col_value;
        chunk = std::min(chunk0, 2 * chunk);
        continue;
      }
      restore(dom, snap, pos);
      if (saved) lp.getLpSolver().getIterate();
      if (nfix > 1) {
        chunk = std::max(HighsInt{1}, chunk / 4);
        continue;
      }
      // a single fixing failed: try the other rounding direction
      HighsInt col = order[0].second;
      double val = std::min(std::max(std::round(sol[col]), dom.col_lower_[col]),
                            dom.col_upper_[col]);
      double other = sol[col] > val ? val + 1 : val - 1;
      other =
          std::min(std::max(other, dom.col_lower_[col]), dom.col_upper_[col]);
      if (other == val || !fixTo(dom, col, other)) return false;
      st = solve(dom);
      ++solves;
      if (!usable(st)) return false;
      if (tryIncumbent(st)) return true;
      sol = lp.getLpSolver().getSolution().col_value;
    }
    return false;
  };

  // 1. root dive from the LP point
  HighsDomain dom(globaldom);
  lp.getLpSolver().changeColsBounds(0, numCol - 1, dom.col_lower_.data(),
                                    dom.col_upper_.data());
  dom.clearChangedCols();
  HighsLpRelaxation::Status st = solve(dom);
  if (!usable(st)) {
    chargeIterations();
    return;
  }
  // the deep search only needs a dive without an incumbent, if diving can
  // find one
  if (!tryIncumbent(st) &&
      (!deep ||
       (mipdata.incumbent.size() != size_t(numCol) && !lnsDiveFailed))) {
    const int64_t diveIters = lp.getNumLpIterations();
    const bool found = dive(dom, decisioncols,
                            std::max<HighsInt>(20, decisioncols.size() / 12));
    if (!found) lnsDiveFailed = true;
    highsLogDev(mipsolver.options_mip_->log_options, HighsLogType::kVerbose,
                "%s dive %s after %lld LP iterations\n",
                mipsolver.concurrent_lns_ ? "LNS(helper)" : "LNS",
                found ? "found a solution" : "failed",
                (long long)(lp.getNumLpIterations() - diveIters));
  }
  if (mipdata.upper_limit == kHighsInf ||
      mipdata.incumbent.size() != size_t(numCol)) {
    chargeIterations();
    return;
  }

  // 2. neighbourhoods. The quick search dives each neighbourhood once,
  // within the heuristic LP budget. The deep search exhausts smaller
  // neighbourhoods by a depth-first branch and bound with a node limit,
  // within an LP-iteration cap relative to the root LP. Both stop when they
  // stall (see progressSince), or once the incumbent is within the target
  // gap of the current bound (the solve then stops at the root).
  const HighsLogOptions& logOptions = mipsolver.options_mip_->log_options;
  const char* who = mipsolver.concurrent_lns_ ? "LNS(helper)" : "LNS";
  const HighsInt maxStall = deep ? 10 : 5;
  const HighsInt maxIt = deep ? 1000 : 100;
  const HighsInt nodeLimit = 300;
  // neighbourhood sizes in decision columns: dived ones start at 400,
  // ones searched by branch and bound at 64
  const double diveSize0 = std::min<double>(400, decisioncols.size());
  const double dfsSize0 = std::min<double>(64, decisioncols.size());
  const double dfsMinSize = std::min(16.0, dfsSize0);
  const double dfsMaxSize =
      std::max(dfsSize0, std::min<double>(1000.0, decisioncols.size()));
  const double itersFac = deep ? 10.0 : 3.0;
  HighsInt since = 0;
  int64_t heurItersCap =
      int64_t(itersFac * mipdata.total_lp_iterations) + (deep ? 5000 : 1000);
  // on large models, the quick search spends about one root LP's worth of
  // LP iterations: where neighbourhood LPs are expensive (e.g. with
  // ramping), the root cuts then come early, and their bound often closes
  // the gap with the incumbent
  if (!deep)
    heurItersCap =
        std::min(heurItersCap,
                 std::max(mipdata.total_lp_iterations + 1000, int64_t{20000}));
  if (maxLpIters >= 0) heurItersCap = std::min(heurItersCap, maxLpIters);
  // the solver's own test of the target gap (as in evaluateRootLp)
  auto withinGap = [&]() {
    return mipdata.upper_bound < kHighsInf &&
           mipdata.lower_bound > mipdata.optimality_limit;
  };
  // Progress (which resets the stall count): in the quick search, an
  // improvement closing at least 5% of the gap; in the deep search, one
  // closing at least 1% of what separates the incumbent from the target gap.
  // Where the bound stays well short of what the target needs (dispatch
  // ticks whose switching costs leave a gap of over 1% that no relaxation
  // here closes), steps of that size are all there is, and the incumbent
  // at the time limit is what counts: with 5%, the deep search stopped after
  // ten such steps and the tree search found little.
  auto progressSince = [&](double before, double limitBefore) {
    if (!deep)
      return mipdata.upper_bound <
             before - std::max(feastol, 0.05 * (before - mipdata.lower_bound));
    return mipdata.optimality_limit <
           limitBefore -
               std::max(feastol, 0.01 * (limitBefore - mipdata.lower_bound));
  };
  // The neighbourhood search is for a loose target gap (as for dispatch
  // or unit commitment models solved to 1%), when a good incumbent may be
  // all that is needed: then it continues while it keeps improving. A
  // small target gap needs the tree search, and the usual sub-MIP
  // heuristics are better value.
  auto lpBudgetExceeded = [&]() {
    const int64_t lnsIters = lp.getNumLpIterations() - lpItersStart;
    if (deep) return lnsIters > heurItersCap;
    const int64_t heurIters = worker.getHeurLpIterations() + lnsIters;
    return heurIters + mipdata.heuristic_lp_iterations >
               100000 + ((mipdata.total_lp_iterations -
                          mipdata.heuristic_lp_iterations -
                          mipdata.sb_lp_iterations) >>
                         1) ||
           lnsIters > heurItersCap;
  };

  // Depth-first branch and bound over the neighbourhood from the solved LP
  // at its root, using at most nodeLimit LP solves. Branches on the
  // fractional candidate closest to integrality, first toward its rounded
  // value; backtracking flips the deepest open decision first. Returns the
  // number of LP solves, and whether the neighbourhood was exhausted.
  std::vector<uint8_t> inCands(numCol, 0);
  auto searchNeighbourhood = [&](HighsDomain& dom,
                                 const std::vector<HighsInt>& cands,
                                 HighsInt nodeLimit, bool& exhausted) {
    for (HighsInt col : cands) inCands[col] = 1;
    struct Decision {
      HighsDomainChange other;
      bool flipped;
    };
    std::vector<Decision> path;
    HighsInt nodes = 0;
    exhausted = false;
    HighsLpRelaxation::Status st = HighsLpRelaxation::Status::kOptimal;
    bool solved = true;  // the LP is solved on entry
    while (true) {
      if (worker.terminatorTerminated() || mipdata.checkLimits()) break;
      if (!solved) {
        st = solve(dom);
        ++nodes;
      }
      solved = false;
      bool prune = !usable(st) || lp.getObjective() >= worker.upper_limit;
      if (!prune) {
        if (tryIncumbent(st)) {
          if (withinGap()) break;
          prune = true;
        } else {
          const std::vector<double>& sol =
              lp.getLpSolver().getSolution().col_value;
          HighsInt bestCol = -1;
          double bestFrac = kHighsInf;
          for (const auto& f : lp.getFractionalIntegers()) {
            HighsInt col = f.first;
            if (dom.col_lower_[col] == dom.col_upper_[col]) continue;
            double frac = std::fabs(sol[col] - std::round(sol[col]));
            // prefer neighbourhood decision columns over the rest
            if (!inCands[col]) frac += 1.0;
            if (frac < bestFrac) {
              bestFrac = frac;
              bestCol = col;
            }
          }
          if (bestCol == -1 || nodes >= nodeLimit) break;
          const double x = sol[bestCol];
          HighsDomainChange up{std::ceil(x), bestCol, HighsBoundType::kLower};
          HighsDomainChange down{std::floor(x), bestCol,
                                 HighsBoundType::kUpper};
          const bool goUp = x - std::floor(x) >= 0.5;
          dom.changeBound(goUp ? up : down, HighsDomain::Reason::branching());
          path.push_back({goUp ? down : up, false});
          dom.propagate();
          if (!dom.infeasible()) continue;
          prune = true;
        }
      }
      // backtrack to the deepest decision whose other branch is open
      bool open = false;
      while (!path.empty()) {
        dom.backtrack();
        if (path.back().flipped) {
          path.pop_back();
          continue;
        }
        path.back().flipped = true;
        dom.changeBound(path.back().other, HighsDomain::Reason::branching());
        dom.propagate();
        if (dom.infeasible()) continue;
        open = true;
        break;
      }
      if (!open) {
        exhausted = true;
        break;
      }
      if (nodes >= nodeLimit) break;
    }
    while (!path.empty()) {
      dom.backtrack();
      path.pop_back();
    }
    for (HighsInt col : cands) inCands[col] = 0;
    return nodes;
  };

  // Rows with at most this many decision columns: a breadth-first search
  // restricted to them follows chains (e.g. one unit over time) rather
  // than spreading over the coupling rows
  const HighsInt kShortRow = 4;
  std::vector<HighsInt> rowDecisions(mipsolver.numRow(), 0);
  for (HighsInt col : decisioncols)
    for (HighsInt p = A.start_[col]; p != A.start_[col + 1]; ++p)
      ++rowDecisions[A.index_[p]];

  // Two neighbourhood types: 0 = BFS over all rows from one seed, 1 = BFS
  // over short rows from as many seeds as needed. Each adapts its size to
  // what the node limit can search, and the type to use next is the one
  // closing the most gap per LP iteration recently, with some exploration.
  // move types: 0 = BFS neighbourhood over all rows searched by branch and
  // bound, 1 = the same over short rows, 2 = flip search, 3 = BFS
  // neighbourhood over all rows dived once
  const HighsInt kNumTypes = 4;
  std::array<LnsMove, 4>& types = lnsMoves;
  for (HighsInt t = 0; t < kNumTypes; ++t)
    if (types[t].size == 0) types[t].size = t == 3 ? diveSize0 : dfsSize0;
  HighsInt deepTries[4] = {0, 0, 0, 0};

  // Flip search: first-improvement local search from the incumbent with all
  // decision columns fixed at it. Binaries are flipped one at a time in
  // order of the gain their reduced cost promises, each tried by a warm LP
  // solve; a promising flip that does not improve on its own is also tried
  // with an opposite flip of a column sharing a row with it (e.g. swapping
  // two units). After each improvement the gains are recomputed. Returns
  // the number of LP solves.
  std::vector<std::pair<double, HighsInt>> flipCands, partners;
  std::vector<double> cur(numCol), rc;
  HighsDomain screen(globaldom);
  std::vector<uint8_t> isOpen(numCol, 0);
  // at most about one root LP's worth of iterations per flip search: on
  // some models (e.g. with ramping) each flip needs a long re-solve
  const int64_t flipMaxIters =
      std::max<int64_t>(5000, mipdata.firstrootlpiters);
  auto flipsExhausted = [&]() {
    return mipdata.upper_bound == lnsFlipObj && lnsFlipNext == -1;
  };
  auto flipSearch = [&](HighsInt maxSolves) -> HighsInt {
    const std::vector<double>& inc = mipdata.incumbent;
    if (inc.size() != size_t(numCol) || flipsExhausted()) return 0;
    // go on from where the last search from this incumbent stopped
    size_t start = mipdata.upper_bound == lnsFlipObj
                       ? std::max<HighsInt>(0, lnsFlipNext)
                       : 0;
    const int64_t flipStartIters = lp.getNumLpIterations();
    for (HighsInt col : decisioncols)
      cur[col] =
          std::min(std::max(std::round(inc[col]), globaldom.col_lower_[col]),
                   globaldom.col_upper_[col]);
    auto setCol = [&](HighsInt col, double val) {
      lp.getLpSolver().changeColBounds(col, val, val);
    };
    auto flipped = [&](HighsInt col) {
      return cur[col] <= globaldom.col_lower_[col] ? globaldom.col_upper_[col]
                                                   : globaldom.col_lower_[col];
    };
    auto binary = [&](HighsInt col) {
      return globaldom.col_upper_[col] - globaldom.col_lower_[col] == 1.0;
    };
    // the incumbent's decision columns are fixed in the LP, others get
    // their global bounds
    lp.getLpSolver().changeColsBounds(0, numCol - 1,
                                      globaldom.col_lower_.data(),
                                      globaldom.col_upper_.data());
    for (HighsInt col : decisioncols) setCol(col, cur[col]);
    lp.setObjectiveLimit(kHighsInf);
    HighsInt solves = 1;
    if (!usable(lp.resolveLp(nullptr))) return solves;
    // Most moves make the LP infeasible (e.g. against minimum up or down
    // times), which domain propagation of the fixings (all decision columns
    // at the incumbent, the move's flipped) over the model rows usually
    // shows much more cheaply than an LP solve. To share the work of fixing
    // every decision column, the screening domain has a base fixing them
    // all at the incumbent except an open set (the next candidates and
    // their partners), and a move then only fixes the open set. Fixings
    // are undone by backtracking to the first of them.
    screen = globaldom;
    screen.clearPoolPropagation();
    std::vector<HighsInt> openCols;
    bool haveBase = false;
    HighsInt screened = 0;
    auto fix = [&](HighsInt col, double val, bool& branched) {
      if (screen.col_lower_[col] < val) {
        screen.changeBound(HighsBoundType::kLower, col, val,
                           branched ? HighsDomain::Reason::unspecified()
                                    : HighsDomain::Reason::branching());
        branched = true;
      }
      if (screen.col_upper_[col] > val) {
        screen.changeBound(HighsBoundType::kUpper, col, val,
                           branched ? HighsDomain::Reason::unspecified()
                                    : HighsDomain::Reason::branching());
        branched = true;
      }
    };
    auto dropBase = [&]() {
      if (haveBase) screen.backtrack();
      haveBase = false;
      for (HighsInt col : openCols) isOpen[col] = 0;
      openCols.clear();
    };
    auto buildBase = [&](const std::vector<HighsInt>& open) {
      dropBase();
      for (HighsInt col : open)
        if (!isOpen[col]) {
          isOpen[col] = 1;
          openCols.push_back(col);
        }
      bool branched = false;
      for (HighsInt col : decisioncols) {
        if (screen.infeasible()) break;
        if (!isOpen[col]) fix(col, cur[col], branched);
      }
      if (!screen.infeasible()) screen.propagate();
      haveBase = branched;
      // (the incumbent satisfies the base, so this is not expected)
      if (screen.infeasible()) dropBase();
    };
    auto propagationInfeasible = [&](const HighsInt* cols, HighsInt n) {
      const bool inBase =
          haveBase && isOpen[cols[0]] && (n == 1 || isOpen[cols[1]]);
      if (!inBase) dropBase();
      auto inMove = [&](HighsInt col) {
        return col == cols[0] || (n > 1 && col == cols[1]);
      };
      bool branched = false;
      for (HighsInt i = 0; i < n; ++i) fix(cols[i], flipped(cols[i]), branched);
      for (HighsInt col : inBase ? openCols : decisioncols) {
        if (screen.infeasible()) break;
        if (!inMove(col)) fix(col, cur[col], branched);
      }
      if (!screen.infeasible()) screen.propagate();
      const bool infeasible = screen.infeasible();
      if (branched) screen.backtrack();
      return infeasible;
    };
    // try a move: on improvement keep it, otherwise undo it
    auto tryMove = [&](const HighsInt* cols, HighsInt n) {
      // (moves ruled out by propagation, which cost far less than an LP
      // solve, count as one in four)
      if (propagationInfeasible(cols, n)) {
        if (++screened % 4 == 0) ++solves;
        return false;
      }
      for (HighsInt i = 0; i < n; ++i) setCol(cols[i], flipped(cols[i]));
      lp.setObjectiveLimit(worker.upper_limit);
      HighsLpRelaxation::Status mst = lp.resolveLp(nullptr);
      ++solves;
      if (usable(mst) && lp.getObjective() < worker.upper_limit &&
          tryIncumbent(mst)) {
        dropBase();
        for (HighsInt i = 0; i < n; ++i) cur[cols[i]] = flipped(cols[i]);
        return true;
      }
      for (HighsInt i = 0; i < n; ++i) setCol(cols[i], cur[cols[i]]);
      return false;
    };
    auto byGain = [](const std::pair<double, HighsInt>& a,
                     const std::pair<double, HighsInt>& b) {
      return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    const HighsInt kPartners = 3;
    bool improved = true;
    while (improved) {
      improved = false;
      // the LP is solved at the incumbent: gains from its reduced costs
      rc = lp.getLpSolver().getSolution().col_dual;
      auto gain = [&](HighsInt col) {
        return flipped(col) > cur[col] ? -rc[col] : rc[col];
      };
      flipCands.clear();
      for (HighsInt col : decisioncols)
        if (binary(col) && gain(col) > feastol)
          flipCands.emplace_back(gain(col), col);
      pdqsort(flipCands.begin(), flipCands.end(), byGain);
      // partners of j: columns in j's rows whose flip moves the row
      // activity the other way, best promised gain first
      auto findPartners = [&](HighsInt j) {
        const double dj = flipped(j) - cur[j];
        partners.clear();
        for (HighsInt p = A.start_[j]; p != A.start_[j + 1]; ++p) {
          const HighsInt row = A.index_[p];
          const double aj = A.value_[p];
          for (HighsInt q = mipdata.ARstart_[row];
               q != mipdata.ARstart_[row + 1]; ++q) {
            const HighsInt k = mipdata.ARindex_[q];
            if (k == j || !isDecision[k] || !binary(k)) continue;
            if (aj * dj * mipdata.ARvalue_[q] * (flipped(k) - cur[k]) >= 0)
              continue;
            partners.emplace_back(gain(k), k);
          }
        }
        pdqsort(partners.begin(), partners.end(), byGain);
        partners.erase(std::unique(partners.begin(), partners.end(),
                                   [](const std::pair<double, HighsInt>& a,
                                      const std::pair<double, HighsInt>& b) {
                                     return a.second == b.second;
                                   }),
                       partners.end());
      };
      const size_t kScreenBatch = 32;
      size_t batchEnd = start;
      std::vector<HighsInt> open;
      for (size_t c = start; c < flipCands.size(); ++c) {
        if (solves >= maxSolves ||
            lp.getNumLpIterations() - flipStartIters > flipMaxIters ||
            withinGap() || worker.terminatorTerminated() ||
            mipdata.checkLimits()) {
          lnsFlipObj = mipdata.upper_bound;
          lnsFlipNext = c;
          dropBase();
          return solves;
        }
        if (c >= batchEnd) {
          // the next candidates and their partners are open in the base
          batchEnd = std::min(flipCands.size(), c + kScreenBatch);
          open.clear();
          for (size_t b = c; b < batchEnd; ++b) {
            const HighsInt jb = flipCands[b].second;
            open.push_back(jb);
            findPartners(jb);
            for (HighsInt i = 0;
                 i < std::min<HighsInt>(kPartners, partners.size()); ++i)
              open.push_back(partners[i].second);
          }
          buildBase(open);
        }
        const HighsInt j = flipCands[c].second;
        if (tryMove(&j, 1)) {
          improved = true;
          break;
        }
        findPartners(j);
        for (HighsInt i = 0;
             i < std::min<HighsInt>(kPartners, partners.size()) && !improved;
             ++i) {
          if (solves >= maxSolves) {
            lnsFlipObj = mipdata.upper_bound;
            lnsFlipNext = c;
            dropBase();
            return solves;
          }
          const HighsInt pair[2] = {j, partners[i].second};
          improved = tryMove(pair, 2);
        }
        if (improved) break;
      }
      start = 0;
    }
    // no improving flip from this incumbent
    dropBase();
    lnsFlipObj = mipdata.upper_bound;
    lnsFlipNext = -1;
    return solves;
  };

  std::vector<HighsInt> neighbourhood, frontier, next, touchedRows, disagree;
  std::vector<uint8_t> inN(numCol, 0), seenRow(mipsolver.numRow(), 0);
  std::vector<double> promise(numCol, 0.0);
  double promiseTotal = 0;
  for (HighsInt it = 0; since < maxStall && it < maxIt; ++it) {
    mipdata.syncConcurrentLns();
    // a helper crosses its incumbent with the main solver's once that is
    // ready (see HighsMipSolverData::crossoverWithMain)
    if (mipsolver.concurrent_lns_) mipdata.crossoverWithMain(worker);
    if (worker.terminatorTerminated() || mipdata.checkLimits() ||
        lpBudgetExceeded() || withinGap())
      break;
    const std::vector<double>& inc = mipdata.incumbent;

    // the quick search only dives; the deep search tries each type once,
    // then mostly the one closing the most gap per LP iteration recently
    HighsInt type = 3;
    if (deep) {
      // each move once (dives may already be known from the quick search),
      // then the best rate of gap closed per LP iteration plus an
      // exploration bonus (UCB) for moves tried less often
      double maxRate = 0;
      HighsInt numTried = 0;
      for (HighsInt t = 0; t < kNumTypes; ++t) {
        maxRate = std::max(maxRate, types[t].rate);
        numTried += types[t].tried;
      }
      double bestScore = -1;
      for (HighsInt t = 0; t < kNumTypes; ++t) {
        // flips are pointless once none improves this incumbent
        if (t == 2 && flipsExhausted()) continue;
        double score;
        if (types[t].tried == 0 || (deepTries[t] == 0 && t != 3))
          score = kHighsInf;
        else
          score = (maxRate > 0 ? types[t].rate / maxRate : 0) +
                  0.5 * std::sqrt(std::log(double(numTried)) / types[t].tried);
        if (score > bestScore) {
          bestScore = score;
          type = t;
        }
      }
      ++deepTries[type];
    }
    const bool dived = type == 3;
    LnsMove& nt = types[type];
    if (type == 2) {
      const double before = mipdata.upper_bound;
      const double limitBefore = mipdata.optimality_limit;
      const double gapBefore = before - mipdata.lower_bound;
      const int64_t startIters = lp.getNumLpIterations();
      const HighsInt solves = flipSearch(nodeLimit);
      ++nt.tried;
      const bool improved = mipdata.upper_bound < before - feastol;
      if (improved) ++nt.improved;
      const double effort =
          1.0 + static_cast<double>(lp.getNumLpIterations() - startIters);
      const double closed =
          gapBefore > 0 ? (before - mipdata.upper_bound) / gapBefore : 0.0;
      nt.rate = 0.7 * nt.rate + 0.3 * closed / effort;
      highsLogDev(logOptions, HighsLogType::kVerbose,
                  "%s %3d flips: %3d LP solves, objective %.4f, %lld LP "
                  "iterations\n",
                  who, int(it), int(solves), mipdata.upper_bound,
                  (long long)(lp.getNumLpIterations() - lpItersStart));
      if (progressSince(before, limitBefore))
        since = 0;
      else
        ++since;
      continue;
    }
    const HighsInt size = HighsInt(nt.size + 0.5);

    auto pickSeed = [&]() {
      // a column whose reduced cost when fixed at the incumbent promises
      // an improvement from flipping it (50%, chosen with probability
      // proportional to that gain), a decision column where the incumbent
      // disagrees with the root LP solution (35%), otherwise any decision
      // column
      HighsInt seed = -1;
      const double r =
          deep ? randgen.fraction() : 0.5 + 0.5 * randgen.fraction();
      if (r < 0.5 && promiseTotal > 0) {
        double pick = randgen.fraction() * promiseTotal;
        for (HighsInt col : decisioncols) {
          pick -= promise[col];
          if (pick <= 0 && promise[col] > 0) {
            seed = col;
            break;
          }
        }
        if (seed != -1 && inN[seed]) seed = -1;
      }
      if (seed == -1 && r < 0.85) {
        disagree.clear();
        for (HighsInt col : decisioncols)
          if (!inN[col] &&
              globaldom.col_lower_[col] != globaldom.col_upper_[col] &&
              std::fabs(inc[col] - relaxationsol[col]) > 0.5)
            disagree.push_back(col);
        if (!disagree.empty())
          seed = disagree[randgen.integer(disagree.size())];
      }
      auto unusable = [&](HighsInt col) {
        return inN[col] ||
               globaldom.col_lower_[col] == globaldom.col_upper_[col];
      };
      if (seed == -1) {
        for (HighsInt tries = 0; tries < 50 && (seed == -1 || unusable(seed));
             ++tries)
          seed = decisioncols[randgen.integer(decisioncols.size())];
        if (unusable(seed)) seed = -1;
      }
      return seed;
    };

    // BFS over the variable/constraint graph, counting decision columns
    for (HighsInt col : neighbourhood) inN[col] = 0;
    for (HighsInt row : touchedRows) seenRow[row] = 0;
    neighbourhood.clear();
    touchedRows.clear();
    while (HighsInt(neighbourhood.size()) < size) {
      HighsInt seed = pickSeed();
      if (seed == -1) break;
      neighbourhood.push_back(seed);
      inN[seed] = 1;
      frontier.assign(1, seed);
      while (!frontier.empty() && HighsInt(neighbourhood.size()) < size) {
        next.clear();
        for (HighsInt j : frontier) {
          for (HighsInt p = A.start_[j]; p != A.start_[j + 1]; ++p) {
            HighsInt row = A.index_[p];
            if (seenRow[row]) continue;
            if (type == 1 && rowDecisions[row] > kShortRow) continue;
            seenRow[row] = 1;
            touchedRows.push_back(row);
            for (HighsInt q = mipdata.ARstart_[row];
                 q != mipdata.ARstart_[row + 1]; ++q) {
              HighsInt k = mipdata.ARindex_[q];
              if (!isDecision[k] || inN[k]) continue;
              inN[k] = 1;
              neighbourhood.push_back(k);
              next.push_back(k);
              if (HighsInt(neighbourhood.size()) >= size) break;
            }
            if (HighsInt(neighbourhood.size()) >= size) break;
          }
          if (HighsInt(neighbourhood.size()) >= size) break;
        }
        frontier.swap(next);
      }
    }

    // fix everything outside the neighbourhood to the incumbent, warm from
    // the last LP solved, near the incumbent: this leads the search much
    // better than starting each neighbourhood from the root basis
    const int64_t startIters = lp.getNumLpIterations();
    dom = globaldom;
    bool feasible = true;
    for (HighsInt col : decisioncols) {
      if (inN[col]) continue;
      double val = std::min(std::max(std::round(inc[col]), dom.col_lower_[col]),
                            dom.col_upper_[col]);
      if (!fixTo(dom, col, val)) {
        feasible = false;
        break;
      }
    }
    ++nt.tried;
    if (!feasible) {
      ++since;
      continue;
    }
    lp.getLpSolver().changeColsBounds(0, numCol - 1, dom.col_lower_.data(),
                                      dom.col_upper_.data());
    dom.clearChangedCols();
    const double before = mipdata.upper_bound;
    const double limitBefore = mipdata.optimality_limit;
    const double gapBefore = before - mipdata.lower_bound;
    st = solve(dom);
    const int64_t nbLpIters = lp.getNumLpIterations() - startIters;
    HighsInt nodes = 0;
    bool exhausted = true;
    if (usable(st)) {
      // the reduced cost of a column fixed at the incumbent is the
      // first-order gain from flipping it: remember the promising ones
      const std::vector<double>& rc = lp.getLpSolver().getSolution().col_dual;
      for (HighsInt col : decisioncols) {
        if (inN[col]) continue;
        double gain = 0;
        if (inc[col] <= globaldom.col_lower_[col] + 0.5)
          gain = -rc[col];
        else if (inc[col] >= globaldom.col_upper_[col] - 0.5)
          gain = rc[col];
        gain = std::max(0.0, gain);
        promiseTotal += gain - promise[col];
        promise[col] = gain;
      }
      promiseTotal = 0;
      for (HighsInt col : decisioncols) promiseTotal += promise[col];
    }
    const bool pruned = !usable(st) || lp.getObjective() >= worker.upper_limit;
    if (!pruned && !dived)
      nodes = searchNeighbourhood(dom, neighbourhood, nodeLimit, exhausted);
    else if (!pruned && !tryIncumbent(st))
      dive(dom, neighbourhood,
           std::max<HighsInt>(2, neighbourhood.size() / 40));
    // a new incumbent from a neighbourhood often has cheap flips nearby
    if (deep && !pruned && mipdata.upper_bound < before - feastol &&
        !withinGap())
      flipSearch(30);
    const bool improved = mipdata.upper_bound < before - feastol;
    const bool progress = progressSince(before, limitBefore);
    if (!dived) {
      // aim for neighbourhoods that the node limit just about exhausts:
      // grow one that was searched without finding anything, shrink one
      // whose search did not finish
      if (!pruned && exhausted && !improved)
        nt.size = std::min(dfsMaxSize, nt.size * 1.1);
      else if (!exhausted)
        nt.size = std::max(dfsMinSize, nt.size * 0.85);
    } else if (pruned) {
      // the neighbourhood LP cannot beat the incumbent: look wider
      nt.size = std::min(4 * diveSize0, nt.size * 1.25);
    } else if (progress) {
      nt.size = diveSize0;
    } else {
      nt.size = std::max(diveSize0 / 2, nt.size * 0.8);
    }
    // effort in LP iterations rather than time, to stay deterministic
    const double effort =
        1.0 + static_cast<double>(lp.getNumLpIterations() - startIters);
    const double closed =
        gapBefore > 0 ? (before - mipdata.upper_bound) / gapBefore : 0.0;
    nt.rate = 0.7 * nt.rate + 0.3 * closed / effort;
    if (improved) ++nt.improved;
    highsLogDev(logOptions, HighsLogType::kVerbose,
                "%s %3d type %d: %4d columns, %3d nodes%s%s, objective %.4f, "
                "%lld (first LP %lld) LP iterations\n",
                who, int(it), int(type), int(neighbourhood.size()), int(nodes),
                pruned ? ", pruned" : "", exhausted ? ", exhausted" : "",
                mipdata.upper_bound,
                (long long)(lp.getNumLpIterations() - lpItersStart),
                (long long)nbLpIters);
    if (progress)
      since = 0;
    else
      ++since;
  }
  // the quick search ends by polishing the incumbent with a short flip
  // search
  if (!deep && !withinGap() && !lpBudgetExceeded() &&
      !worker.terminatorTerminated() && !mipdata.checkLimits()) {
    const double before = mipdata.upper_bound;
    const HighsInt solves = flipSearch(100);
    highsLogDev(logOptions, HighsLogType::kVerbose,
                "%s flip polish: %d LP solves, objective %.4f -> %.4f\n", who,
                int(solves), before, mipdata.upper_bound);
  }
  highsLogDev(logOptions, HighsLogType::kInfo,
              "%s %s search: improved by %d/%d (BFS) %d/%d (short rows) %d/%d "
              "(flips) %d/%d (dives) neighbourhoods\n",
              who, deep ? "deep" : "quick", int(types[0].improved),
              int(types[0].tried), int(types[1].improved), int(types[1].tried),
              int(types[2].improved), int(types[2].tried),
              int(types[3].improved), int(types[3].tried));
  chargeIterations();
}
#endif  // HIGHS_RUST
