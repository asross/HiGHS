/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef HIGHS_MIP_SOLVER_DATA_H_
#define HIGHS_MIP_SOLVER_DATA_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "mip/HighsCliqueTable.h"
#include "mip/HighsConflictPool.h"
#include "mip/HighsCutPool.h"
#include "mip/HighsDebugSol.h"
#include "mip/HighsDomain.h"
#include "mip/HighsImplications.h"
#include "mip/HighsLpRelaxation.h"
#include "mip/HighsMipWorker.h"
#include "mip/HighsNodeQueue.h"
#include "mip/HighsObjectiveFunction.h"
#include "mip/HighsPrimalHeuristics.h"
#include "mip/HighsPseudocost.h"
#include "mip/HighsRedcostFixing.h"
#include "mip/HighsRsSpan.h"
#include "mip/HighsSearch.h"
#include "mip/HighsSeparation.h"
#include "parallel/HighsParallel.h"
#include "presolve/HighsPostsolveStack.h"
#include "presolve/HighsSymmetry.h"
#include "util/HighsTimer.h"

#ifdef HIGHS_RUST
namespace highs_rs {
struct ConcurrentMain;
// rust/src/mip/concurrent.rs: stops and frees the main solver's helper
extern "C" void highs_rs_concurrent_lns_stop(ConcurrentMain** main);
// rust/src/mip/mip_data.rs MipVecs: HighsMipSolverData's vectors
struct MipVecs {
  HighsRsArray<double> incumbent;
  HighsRsArray<double> firstlpsol;
  HighsRsArray<double> rootlpsol;
  HighsRsArray<double> analytic_center;
  HighsRsArray<HighsInt> ar_start;
  HighsRsArray<HighsInt> ar_index;
  HighsRsArray<double> ar_value;
  HighsRsArray<double> max_abs_row_coef;
  HighsRsArray<uint8_t> row_integral;
  HighsRsArray<HighsInt> uplocks;
  HighsRsArray<HighsInt> downlocks;
  HighsRsArray<HighsInt> integer_cols;
  HighsRsArray<HighsInt> implint_cols;
  HighsRsArray<HighsInt> integral_cols;
  HighsRsArray<HighsInt> continuous_cols;
};
extern "C" {
MipVecs* highs_rs_mip_vecs_new();
void highs_rs_mip_vecs_free(MipVecs* v);
// vector `which` (mip_data.rs mod vec) = the n elements of data
void highs_rs_mip_vecs_set(MipVecs* v, int which, const void* data,
                           HighsInt n);
}
// owns the vectors; declared first in HighsMipSolverData, freed last
struct MipVecsOwner {
  MipVecs* p = highs_rs_mip_vecs_new();
  MipVecsOwner() = default;
  MipVecsOwner(const MipVecsOwner&) = delete;
  ~MipVecsOwner() { highs_rs_mip_vecs_free(p); }
};
}  // namespace highs_rs
#else
// Incumbents exchanged between the MIP solver and a concurrent LNS
// helper: a second MIP solver instance on a copy of the presolved model,
// run in its own thread, that only does the root LP, cuts and graph LNS
struct HighsConcurrentLns {
  std::mutex mutex;
  std::vector<double> solution;  // best solution offered so far
  double objective = kHighsInf;
  std::atomic<int64_t> version{0};
  std::atomic<bool> stop{false};
  // the main solver's lower bound, and whether the helper has found a
  // solution within the target gap of it (the main solver then stops)
  std::atomic<double> mainLowerBound{-kHighsInf};
  std::atomic<bool> targetReached{false};
  // the helper's lower bound (its root LP with its own cuts), which the
  // main solver takes during its root node
  std::atomic<double> helperLowerBound{-kHighsInf};
  // the helper's root cuts (rows a.x <= rhs), written once before
  // rootCutsReady is set
  std::atomic<bool> rootCutsReady{false};
  std::vector<HighsInt> cutStart{0}, cutIndex;
  std::vector<double> cutValue, cutRhs;
  std::vector<uint8_t> cutIntegral;
  std::thread thread;
  // Until the helper has crossed its incumbent with the main solver's
  // (HighsPrimalHeuristics::crossover), at the end of its own quick
  // search, the two search independently: neither takes the other's
  // incumbents. Each one's best is kept here for that ([0]: main solver)
  std::atomic<bool> independent{false};
  std::atomic<bool> mainQuickDone{false};
  // the main solver's settings of the heuristics that the helper turns off
  bool runRins = true, runRens = true, runRootReducedCost = true;
  std::vector<double> ownSolution[2];
  double ownObjective[2] = {kHighsInf, kHighsInf};
  // the crossover, for the main solver's log: 1 running, 2 done (then the
  // main solver logs it and sets 3)
  std::atomic<int> crossoverState{0};
  HighsInt crossoverDiffer = 0;
  double crossoverBefore = kHighsInf, crossoverAfter = kHighsInf;

  void offer(const std::vector<double>& sol, double obj) {
    std::lock_guard<std::mutex> lock(mutex);
    if (obj >= objective) return;
    objective = obj;
    solution = sol;
    ++version;
  }
  // Copies the best solution into sol if it is newer than seen and better
  // than obj
  bool take(int64_t& seen, double obj, std::vector<double>& sol) {
    if (version.load() == seen) return false;
    std::lock_guard<std::mutex> lock(mutex);
    seen = version.load();
    if (objective >= obj) return false;
    sol = solution;
    return true;
  }
  void offerOwn(int who, const std::vector<double>& sol, double obj) {
    std::lock_guard<std::mutex> lock(mutex);
    if (obj >= ownObjective[who]) return;
    ownObjective[who] = obj;
    ownSolution[who] = sol;
  }
  bool ownBest(int who, std::vector<double>& sol, double& obj) {
    std::lock_guard<std::mutex> lock(mutex);
    if (ownSolution[who].empty()) return false;
    sol = ownSolution[who];
    obj = ownObjective[who];
    return true;
  }
};
#endif  // HIGHS_RUST

struct HighsPrimaDualIntegral {
  double value;
  double prev_lb;
  double prev_ub;
  double prev_gap;
  double prev_time;
  void initialise();
};

// The scalars of HighsMipSolverData in one block (same layout as
// MipScalars in rust/src/mip/glue.rs), which the Rust port of the solver
// reads and writes in place; HighsMipSolverData keeps references to them
// under their old names
struct HighsMipScalars {
  double feastol = 0.0;
  double epsilon = 0.0;
  double heuristic_effort = 0.0;
  int64_t dispfreq = 0;
  double firstlpsolobj = -kHighsInf;
  double rootlpsolobj = -kHighsInf;
  HighsInt numintegercols = 0;
  HighsInt maxTreeSizeLog2 = 0;
  HighsCDouble pruned_treeweight = 0;
  double avgrootlpiters = 0.0;
  double disptime = 0.0;
  double last_disptime = 0.0;
  int64_t firstrootlpiters = 0;
  int64_t num_nodes = 0;
  int64_t num_leaves = 0;
  int64_t num_leaves_before_run = 0;
  int64_t num_nodes_before_run = 0;
  int64_t total_repair_lp = 0;
  int64_t total_repair_lp_feasible = 0;
  int64_t total_repair_lp_iterations = 0;
  int64_t total_lp_iterations = 0;
  int64_t heuristic_lp_iterations = 0;
  int64_t sepa_lp_iterations = 0;
  int64_t sb_lp_iterations = 0;
  int64_t total_lp_iterations_before_run = 0;
  int64_t heuristic_lp_iterations_before_run = 0;
  int64_t sepa_lp_iterations_before_run = 0;
  int64_t sb_lp_iterations_before_run = 0;
  int64_t num_disp_lines = 0;
  HighsInt numImprovingSols = 0;
  double lower_bound = -kHighsInf;
  double upper_bound = kHighsInf;
  double upper_limit = kHighsInf;
  double optimality_limit = kHighsInf;
  HighsInt numRestarts = 0;
  HighsInt numRestartsRoot = 0;
  HighsInt numCliqueEntriesAfterPresolve = 0;
  HighsInt numCliqueEntriesAfterFirstPresolve = 0;
  int64_t lns_tree_next = -1;
  int64_t lns_tree_wait = 0;
  int64_t lns_quick_lp_iterations = 0;
  int64_t concurrent_lns_seen = 0;
  HighsPrimaDualIntegral primal_dual_integral;
  bool cliquesExtracted = false;
  bool rowMatrixSet = false;
  bool analyticCenterComputed = false;
  bool detectSymmetries = false;
  bool lns_quick_improved = false;
  bool crossoverStartLogged = false;
  bool rootCutsImported = false;
#ifdef HIGHS_RUST
  // the main solver's concurrent LNS helper (rust/src/mip/concurrent.rs)
  highs_rs::ConcurrentMain* concurrent_lns = nullptr;
#endif
};

enum MipSolutionSource : int {
  kSolutionSourceNone = -1,
  kSolutionSourceMin = kSolutionSourceNone,
  //  kSolutionSourceInitial, // 0
  kSolutionSourceBranching,           // B
  kSolutionSourceCentralRounding,     // C
  kSolutionSourceFeasibilityPump,     // F
  kSolutionSourceGraphLns,            // G
  kSolutionSourceHeuristic,           // H
  kSolutionSourceShifting,            // I
  kSolutionSourceFeasibilityJump,     // J
  kSolutionSourceSubMip,              // L
  kSolutionSourceEmptyMip,            // P
  kSolutionSourceRandomizedRounding,  // R
  kSolutionSourceSolveLp,             // S
  kSolutionSourceEvaluateNode,        // T
  kSolutionSourceUnbounded,           // U
  kSolutionSourceUserSolution,        // X
  kSolutionSourceHighsSolution,       // Y
  kSolutionSourceZiRound,             // Z
  kSolutionSourceTrivialL,            // l
  kSolutionSourceTrivialP,            // p
  kSolutionSourceTrivialU,            // u
  kSolutionSourceTrivialZ,            // z
  kSolutionSourceCleanup,
  kSolutionSourceCount
};

struct HighsMipSolverData {
#ifdef HIGHS_RUST
  highs_rs::MipVecsOwner rsv_;
#endif
  HighsMipSolver& mipsolver;
  HighsMipScalars sc_;

  std::deque<HighsLpRelaxation> lps;
  std::deque<HighsCutPool> cutpools;
  std::deque<HighsConflictPool> conflictpools;
  std::deque<HighsDomain> domains;
  std::deque<HighsPseudocost> pseudocosts;
  std::deque<HighsMipWorker> workers;
  bool parallel_lock;

  HighsPrimalHeuristics heuristics;
  HighsCliqueTable cliquetable;
  HighsImplications implications;
  HighsRedcostFixing redcostfixing;
  HighsObjectiveFunction objectiveFunction;
  presolve::HighsPostsolveStack postSolveStack;
  HighsPresolveStatus presolve_status;
  HighsLp presolvedModel;
  bool& cliquesExtracted;
  bool& rowMatrixSet;
  bool& analyticCenterComputed;
  HighsModelStatus analyticCenterStatus;
  // set when graph LNS suits the model, which then has no use for the
  // analytic centre: a computation not started yet is skipped
  std::atomic<bool> skipAnalyticCenter{false};
  bool& detectSymmetries;
  HighsInt& numRestarts;
  HighsInt& numRestartsRoot;
  HighsInt& numCliqueEntriesAfterPresolve;
  HighsInt& numCliqueEntriesAfterFirstPresolve;

#ifdef HIGHS_RUST
  // Rust's (MipVecs), in place
  HighsRsArray<HighsInt>& ARstart_;
  HighsRsArray<HighsInt>& ARindex_;
  HighsRsArray<double>& ARvalue_;
  HighsRsArray<double>& maxAbsRowCoef;
  HighsRsArray<uint8_t>& rowintegral;
  HighsRsArray<HighsInt>& uplocks;
  HighsRsArray<HighsInt>& downlocks;
  HighsRsArray<HighsInt>& integer_cols;
  HighsRsArray<HighsInt>& implint_cols;
  HighsRsArray<HighsInt>& integral_cols;
  HighsRsArray<HighsInt>& continuous_cols;
#else
  std::vector<HighsInt> ARstart_;
  std::vector<HighsInt> ARindex_;
  std::vector<double> ARvalue_;
  std::vector<double> maxAbsRowCoef;
  std::vector<uint8_t> rowintegral;
  std::vector<HighsInt> uplocks;
  std::vector<HighsInt> downlocks;
  std::vector<HighsInt> integer_cols;
  std::vector<HighsInt> implint_cols;
  std::vector<HighsInt> integral_cols;
  std::vector<HighsInt> continuous_cols;
#endif

  HighsSymmetries symmetries;
  std::shared_ptr<const StabilizerOrbits> globalOrbits;

  double& feastol;
  double& epsilon;
  double& heuristic_effort;
  int64_t& dispfreq;
#ifdef HIGHS_RUST
  HighsRsArray<double>& analyticCenter;
  HighsRsArray<double>& firstlpsol;
  HighsRsArray<double>& rootlpsol;
#else
  std::vector<double> analyticCenter;
  std::vector<double> firstlpsol;
  std::vector<double> rootlpsol;
#endif
  double& firstlpsolobj;
  HighsBasis firstrootbasis;
  double& rootlpsolobj;
  HighsInt& numintegercols;
  HighsInt& maxTreeSizeLog2;

  HighsCDouble& pruned_treeweight;
  double& avgrootlpiters;
  double& disptime;
  double& last_disptime;
  int64_t& firstrootlpiters;
  int64_t& num_nodes;
  int64_t& num_leaves;
  int64_t& num_leaves_before_run;
  int64_t& num_nodes_before_run;
  int64_t& total_repair_lp;
  int64_t& total_repair_lp_feasible;
  int64_t& total_repair_lp_iterations;
  int64_t& total_lp_iterations;
  int64_t& heuristic_lp_iterations;
  int64_t& sepa_lp_iterations;
  int64_t& sb_lp_iterations;
  int64_t& total_lp_iterations_before_run;
  int64_t& heuristic_lp_iterations_before_run;
  int64_t& sepa_lp_iterations_before_run;
  int64_t& sb_lp_iterations_before_run;
  int64_t& num_disp_lines;

  HighsInt& numImprovingSols;
  double& lower_bound;
  double& upper_bound;
  double& upper_limit;
  double& optimality_limit;
#ifdef HIGHS_RUST
  HighsRsArray<double>& incumbent;
#else
  std::vector<double> incumbent;
#endif

  HighsNodeQueue nodequeue;

  HighsPrimaDualIntegral& primal_dual_integral;

  HighsDebugSol debugSolution;
#ifdef HIGHS_RUST
  // transformNewIntegerFeasibleSolution's solution in the original space
  HighsSolution rsScratch_;
#endif

  HighsMipSolverData(HighsMipSolver& mipsolver);
#ifdef HIGHS_RUST
  ~HighsMipSolverData() {
    highs_rs::highs_rs_concurrent_lns_stop(&sc_.concurrent_lns);
  }
#else
  ~HighsMipSolverData() { stopConcurrentLns(); }
#endif

  // The main solver owns its concurrent LNS helper; the helper reaches the
  // same pool through mipsolver.concurrent_lns_
  // Graph LNS rounds during the tree search: the next one once the total
  // LP iterations reach lns_tree_next (-1: none), after a wait of
  // lns_tree_wait iterations of tree search
  int64_t& lns_tree_next;
  int64_t& lns_tree_wait;
  // whether the quick graph-LNS search improved the incumbent (kept over
  // restarts): if not, the neighbourhood search does not suit the model
  bool& lns_quick_improved;
  // the LP iterations of the quick graph-LNS search
  int64_t& lns_quick_lp_iterations;

  int64_t& concurrent_lns_seen;
  bool& crossoverStartLogged;
  bool useConcurrentHelper() const;
#ifndef HIGHS_RUST
  std::unique_ptr<HighsConcurrentLns> concurrent_lns;
  void startConcurrentLns();
  // in a concurrent LNS helper: cross its incumbent with the main solver's
  void crossoverWithMain(HighsMipWorker& worker);
  void syncConcurrentLns();
  void stopConcurrentLns();
  void publishRootCuts();
  bool importRootCuts(HighsMipWorker& worker);
#endif
  bool& rootCutsImported;

  bool solutionRowFeasible(const std::vector<double>& solution) const;
  HighsModelStatus feasibilityJump();
  HighsModelStatus trivialHeuristics();

  void startAnalyticCenterComputation(
      const highs::parallel::TaskGroup& taskGroup);
  void finishAnalyticCenterComputation(
      const highs::parallel::TaskGroup& taskGroup);

  struct SymmetryDetectionData {
    HighsSymmetryDetection symDetection;
    HighsSymmetries symmetries;
    double detectionTime = 0.0;
  };

#ifdef HIGHS_RUST
  // the locals of evaluateRootNode, run in Rust (destroyed in reverse order
  // of declaration, as in the C++)
  struct RsRootCtx {
    std::unique_ptr<SymmetryDetectionData> symData;
    highs::parallel::TaskGroup tg;
    std::unique_ptr<HighsSeparation> sepa;
    HighsLpRelaxation::Status sepaStatus = HighsLpRelaxation::Status::kNotSet;
  };
  std::unique_ptr<RsRootCtx> rsRoot_;
  // the locals of performRestart, run in Rust
  struct RsRestartCtx {
    HighsBasis root_basis;
    HighsPseudocostInitialization pscostinit;
    RsRestartCtx(const HighsPseudocost& pscost, HighsInt maxCount,
                 const presolve::HighsPostsolveStack& postsolveStack)
        : pscostinit(pscost, maxCount, postsolveStack) {}
  };
  std::unique_ptr<RsRestartCtx> rsRestart_;
  // the task group of HighsMipSolver::run, run in Rust
  struct RsRunCtx {
    highs::parallel::TaskGroup tg;
  };
  std::unique_ptr<RsRunCtx> rsRun_;
#endif

  void startSymmetryDetection(const highs::parallel::TaskGroup& taskGroup,
                              std::unique_ptr<SymmetryDetectionData>& symData);
  void finishSymmetryDetection(const highs::parallel::TaskGroup& taskGroup,
                               std::unique_ptr<SymmetryDetectionData>& symData);

  void updatePrimalDualIntegral(const double from_lower_bound,
                                const double to_lower_bound,
                                const double from_upper_bound,
                                const double to_upper_bound,
                                const bool check_bound_change = true,
                                const bool check_prev_data = true);
  double limitsToGap(const double use_lower_bound, const double use_upper_bound,
                     double& lb, double& ub) const;

  double computeNewUpperLimit(double upper_bound, double mip_abs_gap,
                              double mip_rel_gap) const;
  bool moreHeuristicsAllowed() const;
  void removeFixedIndices();
  void init();
  void basisTransfer();
  void checkObjIntegrality();
  void runMipPresolve(const HighsInt presolve_reduction_limit);
  void setupDomainPropagation();
  void saveReportMipSolution(const double new_upper_limit = -kHighsInf);
  void runSetup();
  double transformNewIntegerFeasibleSolution(
      const std::vector<double>& sol,
      const bool possibly_store_as_new_incumbent = true);
  double percentageInactiveIntegers() const;
  void performRestart();
  bool checkSolution(const std::vector<double>& solution) const;
  std::vector<std::tuple<HighsInt, HighsInt, double>> getInfeasibleRows(
      const std::vector<double>& solution) const;
  bool trySolution(const std::vector<double>& solution,
                   const int solution_source = kSolutionSourceNone);
  bool rootSeparationRound(HighsMipWorker& worker, HighsSeparation& sepa,
                           HighsInt& ncuts, HighsLpRelaxation::Status& status);
  HighsLpRelaxation::Status evaluateRootLp(HighsMipWorker& worker);

  void evaluateRootNode(HighsMipWorker& worker);

  bool addIncumbent(const std::vector<double>& sol, double solobj,
                    const int solution_source,
                    const bool print_display_line = true,
                    const bool is_user_solution = false);

#ifndef HIGHS_RUST
  const std::vector<double>& getSolution() const;
#endif

  std::string solutionSourceToString(const int solution_source,
                                     const bool code = true) const;
  void printSolutionSourceKey() const;
  void printDisplayLine(const int solution_source = kSolutionSourceNone);

  void getRow(HighsInt row, HighsInt& rowlen, const HighsInt*& rowinds,
              const double*& rowvals) const {
    HighsInt start = ARstart_[row];
    rowlen = ARstart_[row + 1] - start;
    rowinds = ARindex_.data() + start;
    rowvals = ARvalue_.data() + start;
  }

  bool checkLimits(int64_t nodeOffset = 0) const;
  void limitsToBounds(double& dual_bound, double& primal_bound,
                      double& mip_rel_gap) const;
  void updateLowerBound(double new_lower_bound,
                        const bool check_bound_change = true,
                        const bool check_prev_data = true);
  void setCallbackDataOut(const double mipsolver_objective_value) const;
  bool interruptFromCallbackWithData(const int callback_type,
                                     const double mipsolver_objective_value,
                                     const std::string message = "") const;
  void queryExternalSolution(
      const double mipsolver_objective_value,
      const ExternalMipSolutionQueryOrigin external_solution_query_origin);

  HighsInt terminatorConcurrency() const;
  bool terminatorActive() const { return terminatorConcurrency() > 0; }
  HighsInt terminatorMyInstance() const;
  void terminatorTerminate();
  bool terminatorTerminated() const;
  void terminatorReport() const;

  bool parallelLockActive() const {
    return (parallel_lock && hasMultipleWorkers());
  }

  bool hasMultipleWorkers() const { return workers.size() > 1; }

  HighsDomain& getDomain() { return domains[0]; }
  HighsConflictPool& getConflictPool() { return conflictpools[0]; }
  HighsCutPool& getCutPool() { return cutpools[0]; }
  HighsLpRelaxation& getLp() { return lps[0]; }
  HighsPseudocost& getPseudoCost() { return pseudocosts[0]; }
  const HighsDomain& getDomain() const { return domains[0]; }
  const HighsConflictPool& getConflictPool() const { return conflictpools[0]; }
  const HighsCutPool& getCutPool() const { return cutpools[0]; }
  const HighsLpRelaxation& getLp() const { return lps[0]; }
  const HighsPseudocost& getPseudoCost() const { return pseudocosts[0]; }
};

#endif
