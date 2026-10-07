/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#ifndef HIGHS_PRIMAL_HEURISTICS_H_
#define HIGHS_PRIMAL_HEURISTICS_H_

#include <array>
#include <vector>

#include "HConfig.h"
#include "lp_data/HStruct.h"
#include "lp_data/HighsLp.h"
#include "util/HighsRandom.h"

class HighsMipSolver;
class HighsMipWorker;
class HighsLpRelaxation;

#ifdef HIGHS_RUST
namespace highs_rs {
struct Heuristics;
}

// The heuristics' state and logic are Rust's (rust/src/mip/primal.rs and
// graph_lns.rs); they reach the C++ objects through HighsMipRust.h
class HighsPrimalHeuristics {
 private:
  const HighsMipSolver& mipsolver;
  highs_rs::Heuristics* rs_;

 public:
  HighsPrimalHeuristics(HighsMipSolver& mipsolver);
  ~HighsPrimalHeuristics();
  HighsPrimalHeuristics(const HighsPrimalHeuristics&) = delete;
  HighsPrimalHeuristics& operator=(const HighsPrimalHeuristics&) = delete;

  void setupIntCols();

  void graphLNS(HighsMipWorker& worker,
                const std::vector<double>& relaxationsol, bool deep,
                int64_t maxLpIters = -1);

  // returns the number of integer columns where the solutions differ
  HighsInt crossover(HighsMipWorker& worker, const std::vector<double>& other,
                     double otherObjective, double timeCap);

  void rootReducedCost(HighsMipWorker& worker);

  void RENS(HighsMipWorker& worker, const std::vector<double>& relaxationsol);

  void RINS(HighsMipWorker& worker, const std::vector<double>& relaxationsol);

  void feasibilityPump(HighsMipWorker& worker);

  void centralRounding(HighsMipWorker& worker);

  void flushStatistics(HighsMipSolver& mipsolver, HighsMipWorker& worker);

  bool tryRoundedPoint(HighsMipWorker& worker, const std::vector<double>& point,
                       const int solution_source);

  bool linesearchRounding(HighsMipWorker& worker,
                          const std::vector<double>& point1,
                          const std::vector<double>& point2,
                          const int solution_source);

  void randomizedRounding(HighsMipWorker& worker,
                          const std::vector<double>& relaxationsol);

  void shifting(HighsMipWorker& worker,
                const std::vector<double>& relaxationsol);

  void ziRound(HighsMipWorker& worker,
               const std::vector<double>& relaxationsol);

  bool addIncumbent(const std::vector<double>& sol, double solobj,
                    const int solution_source, HighsMipWorker& worker);

  bool trySolution(const std::vector<double>& solution,
                   const int solution_source, HighsMipWorker& worker);

  HighsInt getHeuristicRandom(const HighsInt sup);
};
#else
class HighsPrimalHeuristics {
 private:
  const HighsMipSolver& mipsolver;
  std::vector<HighsInt> intcols;
  std::vector<HighsInt> decisioncols;
  bool decisionColsSetUp = false;
  // a graph-LNS dive from the LP point found nothing: don't dive again
  bool lnsDiveFailed = false;
  // graph LNS move statistics, kept between its calls so that the deep
  // search starts from what the quick search learnt
  struct LnsMove {
    double size = 0;  // neighbourhood size in decision columns (0: unset)
    double rate = 0;  // smoothed fraction of the gap closed per LP iteration
    HighsInt tried = 0;
    HighsInt improved = 0;
  };
  std::array<LnsMove, 4> lnsMoves;
  // where the graph-LNS flip search from the incumbent of objective
  // lnsFlipObj stopped, so that the next one goes on from there (-1: it
  // found no improving flip)
  double lnsFlipObj = kHighsInf;
  HighsInt lnsFlipNext = 0;
  double successObservations;
  HighsInt numSuccessObservations;
  double infeasObservations;
  HighsInt numInfeasObservations;

  HighsRandom randgen;

 public:
  HighsPrimalHeuristics(HighsMipSolver& mipsolver);

  void setupIntCols();

  void setupDecisionCols();

  // Root dive and LNS from the incumbent: quick dived neighbourhoods
  // (after the first root LP), or neighbourhoods searched by a depth-first
  // branch and bound (deep, after the root cuts), using at most maxLpIters
  // LP iterations if that is not negative
  void graphLNS(HighsMipWorker& worker,
                const std::vector<double>& relaxationsol, bool deep,
                int64_t maxLpIters = -1);

  bool solveSubMip(HighsMipWorker& worker, const HighsLp& lp,
                   const HighsBasis& basis, double fixingRate,
                   std::vector<double> colLower, std::vector<double> colUpper,
                   HighsInt maxleaves, HighsInt maxnodes, HighsInt stallnodes,
                   const HighsSolution* start = nullptr,
                   double timeCap = kHighsInf);

  // returns the number of integer columns where the solutions differ
  HighsInt crossover(HighsMipWorker& worker, const std::vector<double>& other,
                     double otherObjective, double timeCap);

  double determineTargetFixingRate(HighsMipWorker& worker);

  void rootReducedCost(HighsMipWorker& worker);

  void RENS(HighsMipWorker& worker, const std::vector<double>& relaxationsol);

  void RINS(HighsMipWorker& worker, const std::vector<double>& relaxationsol);

  void feasibilityPump(HighsMipWorker& worker);

  void centralRounding(HighsMipWorker& worker);

  void flushStatistics(HighsMipSolver& mipsolver, HighsMipWorker& worker);

  bool tryRoundedPoint(HighsMipWorker& worker, const std::vector<double>& point,
                       const int solution_source);

  bool linesearchRounding(HighsMipWorker& worker,
                          const std::vector<double>& point1,
                          const std::vector<double>& point2,
                          const int solution_source);

  void randomizedRounding(HighsMipWorker& worker,
                          const std::vector<double>& relaxationsol);

  void shifting(HighsMipWorker& worker,
                const std::vector<double>& relaxationsol);

  void ziRound(HighsMipWorker& worker,
               const std::vector<double>& relaxationsol);

  bool addIncumbent(const std::vector<double>& sol, double solobj,
                    const int solution_source, HighsMipWorker& worker);

  bool trySolution(const std::vector<double>& solution,
                   const int solution_source, HighsMipWorker& worker);

  HighsInt getNumSuccessObservations(HighsMipWorker& worker) const;

  HighsInt getNumInfeasObservations(HighsMipWorker& worker) const;

  double getSuccessObservations(HighsMipWorker& worker) const;

  double getInfeasObservations(HighsMipWorker& worker) const;

  HighsInt getHeuristicRandom(const HighsInt sup) {
    return randgen.integer(sup);
  }
};

#endif  // HIGHS_RUST

#endif
