/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsMipHost.cpp
 * @brief The Highs object's side of the Rust MIP solver
 * (rust/src/mip/host), whose solve and presolve are its engine's
 * (rust/src/lp_data/top.rs): what the solver calls on the Highs object
 * (rust/src/mip/host/mod.rs HighsFns): the HighsProfiling clocks, the user
 * callback and the improving solution file
 */
#include "mip/HighsMipHost.h"

#ifdef HIGHS_RUST
#include <cstdio>
#include <string>

#include "lp_data/HighsLpHandle.h"
#include "lp_data/HighsLpUtils.h"
#include "lp_data/HighsModelUtils.h"
#include "lp_data/HighsRust.h"
#include "mip/MipTimer.h"

namespace {

// rust/src/mip/setup.rs CallbackOut
struct RsCallbackOut {
  double running_time;
  double objective_function_value;
  int64_t mip_node_count;
  int64_t mip_total_lp_iterations;
  double mip_primal_bound;
  double mip_dual_bound;
  double mip_gap;
  int external_solution_query_origin;
  bool clear_output;
  bool clear_input;
  int solution;
};

// rust/src/mip/host/mod.rs HighsFns
struct RsHighsFns {
  double (*profiling)(void*, int, int, int64_t);
  bool (*callback)(void*, int, int, const RsCallbackOut*, const double*, int,
                   const char*, int);
  const double* (*user_solution)(void*, int*);
  void (*cut_pool_output)(void*, int, int, const double*, const double*,
                          const HighsInt*, const HighsInt*, const double*, int);
  void (*improving_file)(void*, int, const double*, int);
};

}  // namespace

extern "C" {
void highs_rs_mip_register(const RsHighsFns* f);
}

namespace {

// The solve's context: the Highs object's callback, options and model, and
// the improving solution file
struct MipHost {
  HighsCallback* callback;
  const HighsOptions* options;
  const HighsLp* lp;
  FILE* improving;
  // The LP of a model with semi-variables (its names for the improving
  // solution file)
  HighsLp semi_lp;
};

// The clock ids of the clock indices of the Rust solver (root.rs and
// driver.rs mod clk, host/mod.rs mod prof)
const HighsInt kMipClocks[] = {
    kMipClockEvaluateRootNode0,
    kMipClockEvaluateRootNode1,
    kMipClockEvaluateRootNode2,
    kMipClockStartSymmetryDetection,
    kMipClockStartAnalyticCentreComputation,
    kMipClockEvaluateRootLp,
    kMipClockSeparateLpCuts,
    kMipClockRandomizedRounding,
    kMipClockPerformRestart,
    kMipClockRootSeparation,
    kMipClockFinishAnalyticCentreComputation,
    kMipClockRootCentralRounding,
    kMipClockRootSeparationRound0,
    kMipClockRootHeuristicsReducedCost,
    kMipClockRootSeparationRound1,
    kMipClockRootHeuristicsRens,
    kMipClockRootSeparationRound2,
    kMipClockRootFeasibilityPump,
    kMipClockRootSeparationRound3,
    kMipClockRootSeparationRound,
    kMipClockRootSeparationFinishAnalyticCentreComputation,
    kMipClockRootSeparationCentralRounding,
    kMipClockRootSeparationEvaluateRootLp,
    kPresolveTime,
    kMipClockInit,
    kMipClockRunPresolve,
    kSolveTime,
    kMipClockRunSetup,
    kMipClockTrivialHeuristics,
    kMipClockFeasibilityJump,
    kMipClockEvaluateRootNode,
    kMipClockPerformAging0,
    kMipClockSearch,
    kMipClockUpdateLocalDomain,
    kMipClockEvaluateNode1,
    kMipClockNodePrunedLoop,
    kMipClockNodeSearchSeparation,
    kMipClockBacktrackPlunge,
    kMipClockDiveEvaluateNode,
    kMipClockDivePrimalHeuristics,
    kMipClockDiveRandomizedRounding,
    kMipClockDiveRens,
    kMipClockDiveRins,
    kMipClockTheDive,
    kMipClockOpenNodesToQueue0,
    kMipClockDomainPropgate,
    kMipClockPruneInfeasibleNodes,
    kPostsolveTime,
    kMipClockSubMipSolve,
    kSubSolverSubMip,
    kSubSolverIpxAc,
    kMipClockProbingImplications,
    kMipClockProbingPresolve,
    kMipClockEnumerationPresolve,
};

HighsInt recordType(int64_t r) { return r == 0 ? kMipRecord : kSubMipRecord; }

double cbProfiling(void* p, int code, int clock, int64_t arg) {
  HighsProfiling& prof = *static_cast<HighsProfiling*>(p);
  const HighsInt k = kMipClocks[clock];
  switch (code) {
    case 0:
      prof.start(k);
      return 0;
    case 1:
      prof.stop(k);
      return 0;
    case 2:
      return prof.running(k);
    case 3:
      return prof.read(k, recordType(arg));
    case 4:
      return prof.numCall(k, recordType(arg));
    case 5:
      return prof.mip_;
    case 6:
      return prof.isSubMip();
    case 7:
      prof.setSubMip(arg != 0);
      return 0;
    case 8:
      return prof.sub_solver_;
    case 9:
      return prof.myThread();
    case 10:
      prof.start(k, true);
      return 0;
    default: {
      static const char* const kModel[] = {"LP0", "LP1", "LP2", "LP3", "MIP"};
      prof.solveCall(kModel[arg], clock != 0);
      return 0;
    }
  }
}

bool cbCallback(void* ctx, int which, int type, const RsCallbackOut* out,
                const double* sol, int n, const char* message, int len) {
  HighsCallback& cb = *static_cast<MipHost*>(ctx)->callback;
  if (!out) {
    switch (which) {
      case 0:
        return cb.callbackActive(type);
      case 1:
        return bool(cb.user_callback);
      default:
        return cb.data_in.user_has_solution;
    }
  }
  if (out->clear_output) cb.clearHighsCallbackOutput();
  if (sol) cb.data_out.mip_solution.assign(sol, sol + n);
  cb.data_out.running_time = out->running_time;
  cb.data_out.objective_function_value = out->objective_function_value;
  cb.data_out.mip_node_count = out->mip_node_count;
  cb.data_out.mip_total_lp_iterations = out->mip_total_lp_iterations;
  cb.data_out.mip_primal_bound = out->mip_primal_bound;
  cb.data_out.mip_dual_bound = out->mip_dual_bound;
  cb.data_out.mip_gap = out->mip_gap;
  if (out->external_solution_query_origin >= 0)
    cb.data_out.external_solution_query_origin =
        ExternalMipSolutionQueryOrigin(out->external_solution_query_origin);
  if (out->clear_input) cb.clearHighsCallbackInput();
  return cb.callbackAction(type, std::string(message, len));
}

const double* cbUserSolution(void* ctx, int* n) {
  const std::vector<double>& s =
      static_cast<MipHost*>(ctx)->callback->data_in.user_solution;
  *n = s.size();
  return s.data();
}

void cbCutPoolOutput(void* ctx, int num_col, int num_cut, const double* lower,
                     const double* upper, const HighsInt* start,
                     const HighsInt* index, const double* value, int nnz) {
  HighsCallback& cb = *static_cast<MipHost*>(ctx)->callback;
  cb.clearHighsCallbackOutput();
  HighsCallbackOutput& out = cb.data_out;
  out.cutpool_num_col = num_col;
  out.cutpool_num_cut = num_cut;
  out.cutpool_lower.assign(lower, lower + num_cut);
  out.cutpool_upper.assign(upper, upper + num_cut);
  out.cutpool_start.assign(start, start + num_cut + 1);
  out.cutpool_index.assign(index, index + nnz);
  out.cutpool_value.assign(value, value + nnz);
}

void cbImprovingFile(void* ctx, int op, const double* sol, int n) {
  MipHost& h = *static_cast<MipHost*>(ctx);
  switch (op) {
    case 0:
      h.improving = fopen(h.options->mip_improving_solution_file.c_str(), "w");
      return;
    case 1:
      if (h.improving) {
        const std::vector<double> solution(sol, sol + n);
        writeLpObjective(h.improving, h.options->log_options, *h.lp, solution);
        writePrimalSolution(
            h.improving, h.options->log_options, *h.lp, solution,
            h.options->mip_improving_solution_report_sparse);
      }
      return;
    default:
      if (h.improving) fclose(h.improving);
      h.improving = nullptr;
  }
}

const RsHighsFns kHighsFns = {cbProfiling, cbCallback, cbUserSolution,
                              cbCutPoolOutput, cbImprovingFile};

}  // namespace


void* highsMipHostNew(HighsCallback& callback, const HighsOptions& options,
                      const HighsLp& lp, const bool semi) {
  highs_rs_mip_register(&kHighsFns);
  MipHost* host = new MipHost{&callback, &options, &lp, nullptr, HighsLp()};
  if (semi) {
    HighsSolution solution;
    host->semi_lp = withoutSemiVariables(lp, solution,
                                         options.primal_feasibility_tolerance);
    host->lp = &host->semi_lp;
  }
  return host;
}

void highsMipHostFree(void* host) { delete static_cast<MipHost*>(host); }


#endif  // HIGHS_RUST
