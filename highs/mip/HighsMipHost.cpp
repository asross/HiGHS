/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                       */
/*    This file is part of the HiGHS linear optimization suite           */
/*                                                                       */
/*    Available as open-source under the MIT License                     */
/*                                                                       */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/**@file mip/HighsMipHost.cpp
 * @brief The Highs object's side of the Rust MIP solver
 * (rust/src/mip/host): the solve's entry and result, and what the solver
 * calls on the Highs object (rust/src/mip/host/mod.rs HighsFns): the
 * HighsProfiling clocks, the user callback and the improving solution file
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

// rust/src/mip/host/entry.rs MipIn
struct RsMipIn {
  void* host;
  void* profiling;
  RsLog log;
  highs_rs::LpHandle* opts;
  const RsLp* lp;
  const char* model_name;
  size_t model_name_len;
  const double* col_value;
  size_t num_col_value;
  const double* row_value;
  size_t num_row_value;
  bool value_valid;
  HighsInt presolve_reduction_limit;
};

// rust/src/mip/host/entry.rs MipOut
struct RsMipOut {
  int model_status;
  double solution_objective;
  int64_t node_count;
  int64_t total_lp_iterations;
  double dual_bound;
  double primal_bound;
  double gap;
  double primal_dual_integral;
  double row_violation;
  double bound_violation;
  double integrality_violation;
  const double* solution;
  size_t num_solution;
  size_t num_saved;
  int presolve_status;
  RsLp presolved;
  const char* presolved_name;
  size_t presolved_name_len;
  const char* data;
  size_t data_len;
  const void* reductions;
  size_t num_reductions;
  const HighsInt* orig_col_index;
  size_t num_col;
  const HighsInt* orig_row_index;
  size_t num_row;
  const uint8_t* linearly_transformable;
  size_t num_lt;
  HighsInt orig_num_col;
  HighsInt orig_num_row;
  void* solver;
};

}  // namespace

extern "C" {
void highs_rs_mip_register(const RsHighsFns* f);
RsMipOut* highs_rs_mip_solve(const RsMipIn* in);
RsMipOut* highs_rs_mip_presolve(const RsMipIn* in);
const double* highs_rs_mip_saved(const RsMipOut* o, size_t k,
                                 double* objective, size_t* n);
void highs_rs_mip_out_free(RsMipOut* o);
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

// The solve's input; `lpv` and `opts` live as long as it
RsMipIn mipIn(MipHost& host, HighsProfiling* profiling, const RsLp& lpv,
              highs_rs::LpHandle* opts, const HighsSolution& solution) {
  highs_rs_mip_register(&kHighsFns);
  const HighsLp& lp = *host.lp;
  RsMipIn in;
  in.host = &host;
  in.profiling = profiling;
  in.log = rsLog(host.options->log_options);
  in.opts = opts;
  in.lp = &lpv;
  in.model_name = lp.model_name_.data();
  in.model_name_len = lp.model_name_.size();
  in.value_valid = solution.value_valid;
  in.col_value = solution.col_value.data();
  in.num_col_value = solution.col_value.size();
  in.row_value = solution.row_value.data();
  in.num_row_value = solution.row_value.size();
  in.presolve_reduction_limit = host.options->presolve_reduction_limit;
  return in;
}

template <typename T, typename V>
void take(std::vector<T>& vec, const V& r) {
  vec.assign(r.ptr, r.ptr + r.len);
}

}  // namespace

HighsMipRun::HighsMipRun(HighsCallback& callback, const HighsOptions& options,
                         const HighsLp& lp, const HighsSolution& solution,
                         HighsProfiling* profiling) {
  MipHost host{&callback, &options, &lp, nullptr};
  HighsLpHandle opts;
  rsSyncOptions(opts.p, options);
  HighsLp& lpm = const_cast<HighsLp&>(lp);
  const RsLp lpv = rsLp(lpm);
  const RsMipIn in = mipIn(host, profiling, lpv, opts.p, solution);
  RsMipOut* out = highs_rs_mip_solve(&in);
  modelstatus_ = HighsModelStatus(out->model_status);
  solution_objective_ = out->solution_objective;
  node_count_ = out->node_count;
  total_lp_iterations_ = out->total_lp_iterations;
  dual_bound_ = out->dual_bound;
  primal_bound_ = out->primal_bound;
  gap_ = out->gap;
  primal_dual_integral_ = out->primal_dual_integral;
  row_violation_ = out->row_violation;
  bound_violation_ = out->bound_violation;
  integrality_violation_ = out->integrality_violation;
  solution_.assign(out->solution, out->solution + out->num_solution);
  for (size_t k = 0; k < out->num_saved; ++k) {
    HighsObjectiveSolution record;
    size_t n;
    const double* v = highs_rs_mip_saved(out, k, &record.objective, &n);
    record.col_value.assign(v, v + n);
    saved_objective_and_solution_.push_back(std::move(record));
  }
  highs_rs_mip_out_free(out);
}

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

HighsPresolveStatus highsMipPresolve(HighsCallback& callback,
                                     const HighsOptions& options,
                                     const HighsLp& lp,
                                     const HighsSolution& solution,
                                     HighsProfiling* profiling,
                                     HighsLp& presolved,
                                     presolve::HighsPostsolveStack& stack) {
  MipHost host{&callback, &options, &lp, nullptr};
  HighsLpHandle opts;
  rsSyncOptions(opts.p, options);
  HighsLp& lpm = const_cast<HighsLp&>(lp);
  const RsLp lpv = rsLp(lpm);
  const RsMipIn in = mipIn(host, profiling, lpv, opts.p, solution);
  RsMipOut* out = highs_rs_mip_presolve(&in);
  const HighsPresolveStatus status = HighsPresolveStatus(out->presolve_status);
  // the presolved model: the model's other members, the presolve's data
  presolved = lp;
  const RsLp& v = out->presolved;
  presolved.num_col_ = v.num_col;
  presolved.num_row_ = v.num_row;
  take(presolved.col_cost_, v.col_cost);
  take(presolved.col_lower_, v.col_lower);
  take(presolved.col_upper_, v.col_upper);
  take(presolved.row_lower_, v.row_lower);
  take(presolved.row_upper_, v.row_upper);
  HighsSparseMatrix& a = presolved.a_matrix_;
  a.format_ = MatrixFormat(v.a.format);
  a.num_col_ = v.a.num_col;
  a.num_row_ = v.a.num_row;
  take(a.start_, v.a.start);
  take(a.p_end_, v.a.p_end);
  take(a.index_, v.a.index);
  take(a.value_, v.a.value);
  presolved.sense_ = ObjSense(v.sense);
  presolved.offset_ = v.offset;
  const HighsVarType* integrality =
      reinterpret_cast<const HighsVarType*>(v.integrality.ptr);
  presolved.integrality_.assign(integrality, integrality + v.integrality.len);
  presolved.model_name_.assign(out->presolved_name, out->presolved_name_len);
  // the names follow the index maps
  if (lp.col_names_.size() > 0) {
    std::vector<std::string> names(out->num_col);
    for (size_t i = 0; i != out->num_col; ++i)
      names[i] = lp.col_names_[out->orig_col_index[i]];
    presolved.col_names_ = std::move(names);
  }
  if (lp.row_names_.size() > 0) {
    std::vector<std::string> names(out->num_row);
    for (size_t i = 0; i != out->num_row; ++i)
      names[i] = lp.row_names_[out->orig_row_index[i]];
    presolved.row_names_ = std::move(names);
  }
  stack.rustSet(out->data, out->data_len, out->reductions, out->num_reductions,
                out->orig_col_index, out->num_col, out->orig_row_index,
                out->num_row, out->linearly_transformable, out->num_lt,
                out->orig_num_col, out->orig_num_row);
  highs_rs_mip_out_free(out);
  return status;
}

#endif  // HIGHS_RUST
